#!/usr/bin/env python3
"""Train DrainLite on the fixed native ITZI-SWMM connected labels."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import uniform_filter
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance

from publication_plot_style import add_panel_labels, configure_publication_style


configure_publication_style()

from train_drainlite_residual import (
    CELL_AREA_M2,
    DRAINAGE_FEATURES,
    FEATURE_GROUPS,
    SUPERSET_FEATURES,
    WALL_HEIGHT,
    choose_pixels,
    csi,
    feature_matrix,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_NAME = "region1_20m_connected_swmm_v1"
DEFAULT_OUTPUT_NAME = "drainlite_connected_residual"
FLOOD_DIR = ROOT / "LarNO-main" / "benchmark" / "urbanflood" / "flood" / DEFAULT_DATASET_NAME
GEO_DIR = ROOT / "LarNO-main" / "benchmark" / "urbanflood" / "geodata" / DEFAULT_DATASET_NAME
CONNECTED_METRICS = ROOT / "extended_study" / "output" / "connected_itzi_swmm" / "connected_itzi_swmm_metrics.csv"
OUT = ROOT / "extended_study" / "output" / DEFAULT_OUTPUT_NAME

DEFAULT_TRAIN = ["event1", "event20", "event65", "event66", "event67"]
DEFAULT_TEST = ["event68", "event69", "event70"]


def configure_paths(dataset_name: str, output_name: str, connected_metrics: str | None = None) -> None:
    global FLOOD_DIR, GEO_DIR, CONNECTED_METRICS, OUT
    FLOOD_DIR = ROOT / "LarNO-main" / "benchmark" / "urbanflood" / "flood" / dataset_name
    GEO_DIR = ROOT / "LarNO-main" / "benchmark" / "urbanflood" / "geodata" / dataset_name
    OUT = ROOT / "extended_study" / "output" / output_name
    if connected_metrics:
        CONNECTED_METRICS = Path(connected_metrics)
    else:
        rebuilt = FLOOD_DIR / "connected_swmm_metrics_rebuilt.csv"
        CONNECTED_METRICS = rebuilt if rebuilt.exists() else CONNECTED_METRICS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--output-name", default=DEFAULT_OUTPUT_NAME)
    parser.add_argument("--connected-metrics", default=None)
    parser.add_argument("--train-events", nargs="+", default=DEFAULT_TRAIN)
    parser.add_argument("--test-events", nargs="+", default=DEFAULT_TEST)
    parser.add_argument("--pixels-per-step", type=int, default=2500)
    parser.add_argument("--max-iter", type=int, default=350)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-leaf-nodes", type=int, default=31)
    parser.add_argument("--l2-regularization", type=float, default=0.01)
    parser.add_argument("--random-state", type=int, default=670)
    parser.add_argument("--predict-chunk", type=int, default=60000)
    parser.add_argument("--importance-samples", type=int, default=8000)
    parser.add_argument("--importance-repeats", type=int, default=3)
    parser.add_argument(
        "--residual-deadband-mm",
        type=float,
        default=0.0,
        help="Set predicted signed residuals with absolute value below this threshold to zero.",
    )
    return parser.parse_args()


def load_static() -> dict[str, np.ndarray]:
    dem = np.load(GEO_DIR / "dem.npy").astype(np.float32)
    h, w = dem.shape
    rr, cc = np.indices((h, w), dtype=np.float32)
    cell_size_m = float(np.sqrt(CELL_AREA_M2))
    gy, gx = np.gradient(np.nan_to_num(dem, nan=WALL_HEIGHT), cell_size_m, cell_size_m)
    pipe = np.load(GEO_DIR / "pipe_mask.npy").astype(np.float32)
    inlet = np.load(GEO_DIR / "drain_inlet_mask.npy").astype(np.float32)
    capacity = np.load(GEO_DIR / "pipe_capacity.npy").astype(np.float32)
    static = {
        "dem_m": np.nan_to_num(dem, nan=WALL_HEIGHT).astype(np.float32),
        "active_mask": np.isfinite(dem) & (dem < WALL_HEIGHT),
        "dem_slope": np.sqrt(gx * gx + gy * gy).astype(np.float32),
        "row_norm": (rr / max(h - 1, 1)).astype(np.float32),
        "col_norm": (cc / max(w - 1, 1)).astype(np.float32),
        "pipe_density_3x3": uniform_filter(pipe, size=3, mode="nearest").astype(np.float32),
        "pipe_density_7x7": uniform_filter(pipe, size=7, mode="nearest").astype(np.float32),
        "inlet_density_7x7": uniform_filter(inlet, size=7, mode="nearest").astype(np.float32),
        "capacity_density_7x7": uniform_filter(capacity, size=7, mode="nearest").astype(np.float32),
    }
    for name in DRAINAGE_FEATURES:
        static[name] = np.nan_to_num(np.load(GEO_DIR / f"{name}.npy").astype(np.float32), nan=0.0)
    return static


def read_connected_swmm() -> dict[str, dict[str, float]]:
    rows: dict[str, dict[str, float]] = {}
    if not CONNECTED_METRICS.exists():
        return rows
    with CONNECTED_METRICS.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            event = row["event"]
            parsed: dict[str, float] = {}
            for key, value in row.items():
                if key == "event":
                    continue
                try:
                    parsed[key] = float(value)
                except Exception:
                    pass
            rows[event] = {
                "precip_mm": 0.0,
                "routing_inflow_m3": parsed.get("external_inflow_m3", 0.0),
                "external_outflow_m3": parsed.get("external_outflow_m3", 0.0),
                "routing_final_storage_m3": parsed.get("final_stored_m3", 0.0),
                "routing_continuity_error_pct": parsed.get("continuity_error_pct", 0.0),
                "outfall_volume_m3": parsed.get("external_outflow_m3", 0.0),
                "flooded_node_count": 0.0,
                "node_flooding_volume_m3": parsed.get("flooding_loss_m3", 0.0),
                **parsed,
            }
    return rows


def load_event(event: str) -> dict[str, np.ndarray]:
    event_dir = FLOOD_DIR / event
    surface = np.load(event_dir / "h_itzi_surface.npy").astype(np.float32)
    connected = np.load(event_dir / "h_itzi_swmm_connected.npy").astype(np.float32)
    mike = np.load(event_dir / "h_mike_ref.npy").astype(np.float32)
    rainfall = np.load(event_dir / "rainfall.npy").astype(np.float32)
    return {
        "surface": surface,
        "official": connected,
        "mike": mike,
        "rainfall": rainfall,
        "cumsum": np.cumsum(rainfall, axis=0, dtype=np.float32),
    }


def choose_residual_pixels(
    rng: np.random.Generator,
    static: dict[str, np.ndarray],
    surface_t: np.ndarray,
    residual_t: np.ndarray,
    n: int,
) -> tuple[np.ndarray, np.ndarray]:
    changed = static["active_mask"] & (np.abs(residual_t) > 1e-5)
    pseudo_target = np.abs(residual_t)
    rows, cols = choose_pixels(rng, static, surface_t, pseudo_target, n)
    if np.any(changed):
        changed_flat = np.flatnonzero(changed)
        extra = rng.choice(changed_flat, size=min(max(n // 3, 1), changed_flat.size), replace=False)
        current = np.ravel_multi_index((rows, cols), static["active_mask"].shape)
        idx = np.unique(np.concatenate([current, extra]))
        if idx.size > n:
            idx = rng.choice(idx, size=n, replace=False)
        rows, cols = np.unravel_index(idx, static["active_mask"].shape)
    return rows, cols


def build_training_samples(
    events: list[str],
    static: dict[str, np.ndarray],
    swmm: dict[str, dict[str, float]],
    pixels_per_step: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_state)
    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    w_parts: list[np.ndarray] = []
    for event in events:
        arrays = load_event(event)
        for t in range(arrays["surface"].shape[0]):
            residual_t = arrays["official"][t] - arrays["surface"][t]
            rows, cols = choose_residual_pixels(rng, static, arrays["surface"][t], residual_t, pixels_per_step)
            x_parts.append(feature_matrix(event, arrays, static, swmm, t, rows, cols, SUPERSET_FEATURES))
            y = residual_t[rows, cols] * 1000.0
            weights = np.ones_like(y, dtype=np.float32)
            weights += (np.abs(y) > 0.2).astype(np.float32) * 4.0
            weights += (arrays["surface"][t, rows, cols] > 0.03).astype(np.float32) * 2.0
            weights += (static["pipe_density_7x7"][rows, cols] > 0).astype(np.float32) * 1.0
            y_parts.append(y.astype(np.float32))
            w_parts.append(weights)
        print(f"sampled {event}: {sum(len(p) for p in y_parts):,} rows total", flush=True)
    return np.vstack(x_parts), np.concatenate(y_parts), np.concatenate(w_parts)


def train_models(
    x_all: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, HistGradientBoostingRegressor]:
    models: dict[str, HistGradientBoostingRegressor] = {}
    model_dir = OUT / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    col_index = {name: i for i, name in enumerate(SUPERSET_FEATURES)}
    for model_name, features in FEATURE_GROUPS.items():
        cols = [col_index[name] for name in features]
        model = HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=args.learning_rate,
            max_iter=args.max_iter,
            max_leaf_nodes=args.max_leaf_nodes,
            l2_regularization=args.l2_regularization,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=args.random_state,
        )
        print(f"training {model_name}: rows={x_all.shape[0]:,}, features={len(cols)}", flush=True)
        model.fit(x_all[:, cols], y, sample_weight=weights)
        models[model_name] = model
        joblib.dump({"model": model, "features": features}, model_dir / f"{model_name}.joblib")
    return models


def predict_event(
    event: str,
    model: HistGradientBoostingRegressor,
    features: list[str],
    arrays: dict[str, np.ndarray],
    static: dict[str, np.ndarray],
    swmm: dict[str, dict[str, float]],
    chunk_size: int,
    residual_deadband_mm: float,
) -> tuple[np.ndarray, np.ndarray]:
    t_count, h, w = arrays["surface"].shape
    active_rows, active_cols = np.where(static["active_mask"])
    residual_mm = np.zeros((t_count, h, w), dtype=np.float32)
    for t in range(t_count):
        pred_t = np.zeros((h, w), dtype=np.float32)
        for start in range(0, active_rows.size, chunk_size):
            rows = active_rows[start : start + chunk_size]
            cols = active_cols[start : start + chunk_size]
            x = feature_matrix(event, arrays, static, swmm, t, rows, cols, features)
            pred_t[rows, cols] = model.predict(x).astype(np.float32)
        if residual_deadband_mm > 0:
            pred_t[np.abs(pred_t) < residual_deadband_mm] = 0.0
        residual_mm[t] = np.clip(pred_t, -1200.0, 1200.0)
    h_pred = np.maximum(arrays["surface"] + residual_mm / 1000.0, 0.0).astype(np.float32)
    return h_pred, residual_mm


def event_metrics(
    event: str,
    model: str,
    pred: np.ndarray,
    target: np.ndarray,
    surface: np.ndarray,
    mike: np.ndarray,
    active_mask: np.ndarray | None = None,
) -> dict[str, object]:
    if active_mask is None:
        active_mask = np.ones(pred.shape[1:], dtype=bool)
    active_mask = np.asarray(active_mask, dtype=bool)
    if active_mask.shape != pred.shape[1:]:
        raise ValueError(f"active_mask shape {active_mask.shape} does not match prediction shape {pred.shape[1:]}")

    pred_v = pred[:, active_mask]
    target_v = target[:, active_mask]
    surface_v = surface[:, active_mask]
    mike_v = mike[:, active_mask]
    err = pred_v - target_v
    target_residual = target_v - surface_v
    pred_residual = pred_v - surface_v
    final_surcharge = np.maximum(pred_v[-1] - surface_v[-1], 0.0)
    final_reduction = np.maximum(surface_v[-1] - pred_v[-1], 0.0)
    target_final_surcharge = np.maximum(target_v[-1] - surface_v[-1], 0.0)
    target_final_reduction = np.maximum(surface_v[-1] - target_v[-1], 0.0)
    pred_hmax = pred_v.max(axis=1)
    target_hmax = target_v.max(axis=1)
    mike_hmax = mike_v.max(axis=1)
    pred_peak_t = int(np.argmax(pred_hmax))
    target_peak_t = int(np.argmax(target_hmax))
    pred_volume = pred_v.sum(axis=1) * CELL_AREA_M2
    target_volume = target_v.sum(axis=1) * CELL_AREA_M2
    target_reduction_m3 = float(np.sum(target_final_reduction) * CELL_AREA_M2)
    predicted_reduction_m3 = float(np.sum(final_reduction) * CELL_AREA_M2)
    return {
        "event": event,
        "model": model,
        "evaluation_active_cells": int(active_mask.sum()),
        "mae_m": float(np.mean(np.abs(err))),
        "rmse_m": float(np.sqrt(np.mean(err * err))),
        "csi_0p03": csi(pred_v, target_v, 0.03),
        "csi_0p15": csi(pred_v, target_v, 0.15),
        "peak_pred_m": float(np.max(pred_hmax)),
        "peak_target_m": float(np.max(target_hmax)),
        "peak_error_m": float(np.max(pred_hmax) - np.max(target_hmax)),
        "abs_peak_error_m": float(abs(np.max(pred_hmax) - np.max(target_hmax))),
        # Saved frames represent the end of each five-minute forcing interval.
        "peak_time_pred_h": (pred_peak_t + 1) * 5.0 / 60.0,
        "peak_time_target_h": (target_peak_t + 1) * 5.0 / 60.0,
        "peak_time_error_h": (pred_peak_t - target_peak_t) * 5.0 / 60.0,
        "abs_peak_time_error_h": abs(pred_peak_t - target_peak_t) * 5.0 / 60.0,
        "signed_residual_mae_m": float(np.mean(np.abs(pred_residual - target_residual))),
        "peak_volume_error_m3": float(pred_volume[target_peak_t] - target_volume[target_peak_t]),
        "final_volume_error_m3": float(pred_volume[-1] - target_volume[-1]),
        "drainage_reduction_final_m3": predicted_reduction_m3,
        "target_reduction_final_m3": target_reduction_m3,
        "final_reduction_capture_pct": (
            100.0 * predicted_reduction_m3 / target_reduction_m3 if target_reduction_m3 > 0 else math.nan
        ),
        "local_surcharge_final_m3": float(np.sum(final_surcharge) * CELL_AREA_M2),
        "target_surcharge_final_m3": float(np.sum(target_final_surcharge) * CELL_AREA_M2),
        "mae_to_mike_m": float(np.mean(np.abs(pred_v - mike_v))),
        "peak_error_to_mike_m": float(np.max(pred_hmax) - np.max(mike_hmax)),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def signed_extreme_residual(residual_mm: np.ndarray) -> np.ndarray:
    idx = np.argmax(np.abs(residual_mm), axis=0)
    rr, cc = np.indices(residual_mm.shape[1:])
    return residual_mm[idx, rr, cc] / 1000.0


def make_peak_map(
    event: str,
    arrays: dict[str, np.ndarray],
    pred: np.ndarray,
    residual_mm: np.ndarray,
    path: Path,
    water_vmax: float | None = None,
    error_limit: float | None = None,
    residual_limit: float | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mike_peak = arrays["mike"].max(axis=0)
    surface_peak = arrays["surface"].max(axis=0)
    target_peak = arrays["official"].max(axis=0)
    pred_peak = pred.max(axis=0)
    error_peak = pred_peak - target_peak
    active = np.load(GEO_DIR / "dem.npy").astype(np.float32) < WALL_HEIGHT
    target_peak_index = np.argmax(arrays["official"], axis=0)
    rr, cc = np.indices(target_peak_index.shape)
    residual_peak = residual_mm[target_peak_index, rr, cc] / 1000.0
    vmax = water_vmax or max(
        float(mike_peak[active].max()),
        float(surface_peak[active].max()),
        float(target_peak[active].max()),
        float(pred_peak[active].max()),
    )
    error_lim = error_limit or max(0.02, float(np.nanpercentile(np.abs(error_peak[active]), 99.5)))
    residual_lim = residual_limit or max(0.02, float(np.nanpercentile(np.abs(residual_peak[active]), 99.5)))
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.85))
    panels = [
        ("MIKE reference", mike_peak, "viridis", 0, vmax, "Water depth (m)"),
        ("Surface only", surface_peak, "viridis", 0, vmax, "Water depth (m)"),
        ("ITZI-SWMM label", target_peak, "viridis", 0, vmax, "Water depth (m)"),
        ("DrainLite prediction", pred_peak, "viridis", 0, vmax, "Water depth (m)"),
        ("Prediction error", error_peak, "coolwarm", -error_lim, error_lim, "Depth difference (m)"),
        ("Residual at coupled peak", residual_peak, "coolwarm", -residual_lim, residual_lim, "Depth difference (m)"),
    ]
    for ax, (title, data, cmap, vmin, vmax_i, colorbar_label) in zip(axes.flat, panels):
        shown = np.ma.masked_where(~active, data)
        im = ax.imshow(shown, cmap=cmap, vmin=vmin, vmax=vmax_i, origin="upper", interpolation="nearest")
        ax.set_title(title)
        ax.axis("off")
        cb = fig.colorbar(im, ax=ax, fraction=0.038, pad=0.02, extend="both" if vmin < 0 else "neither")
        cb.set_label(colorbar_label, fontsize=6.8)
        cb.ax.tick_params(labelsize=6.5)
    add_panel_labels(axes.flat, x=0.0, y=1.04)
    fig.suptitle(f"{event}: peak-depth fields and drainage residual")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def make_static_figures(static: dict[str, np.ndarray]) -> None:
    fig_dir = OUT / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.1, 5.0))
    dem = np.ma.masked_where(~static["active_mask"], static["dem_m"])
    height, width = dem.shape
    extent = (0.0, width * 0.02, height * 0.02, 0.0)
    image = ax.imshow(dem, cmap="terrain", interpolation="nearest", origin="upper", extent=extent)
    pipe_overlay = np.ma.masked_where(static["pipe_mask"] <= 0, static["pipe_mask"])
    ax.imshow(
        pipe_overlay,
        cmap="gray_r",
        vmin=0,
        vmax=1,
        interpolation="nearest",
        origin="upper",
        extent=extent,
        alpha=0.78,
    )
    inlet_y, inlet_x = np.where(static["drain_inlet_mask"] > 0)
    out_y, out_x = np.where(static["drain_outfall_mask"] > 0)
    if inlet_y.size:
        ax.scatter((inlet_x + 0.5) * 0.02, (inlet_y + 0.5) * 0.02, s=3.0, c="#0072B2", label="Inlet/manhole", alpha=0.75, linewidths=0)
    if out_y.size:
        ax.scatter((out_x + 0.5) * 0.02, (out_y + 0.5) * 0.02, s=22, c="#D55E00", edgecolors="white", linewidths=0.45, label="Outfall", zorder=5)
    cb = fig.colorbar(image, ax=ax, fraction=0.032, pad=0.02)
    cb.set_label("Elevation (m)")
    ax.set_title("Road-aligned conceptual drainage network")
    ax.set_xlabel("Grid x distance (km)")
    ax.set_ylabel("Grid y distance (km)")
    bar_y = height * 0.02 - 0.18
    ax.plot([0.18, 1.18], [bar_y, bar_y], color="black", linewidth=2.2, solid_capstyle="butt")
    ax.text(0.68, bar_y - 0.08, "1 km", ha="center", va="bottom", fontsize=7.2)
    ax.legend(loc="upper left", frameon=True, framealpha=0.88, facecolor="white", edgecolor="0.7")
    fig.tight_layout()
    fig.savefig(fig_dir / "connected_static_dem_network.png")
    plt.close(fig)


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_model: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_model.setdefault(str(row["model"]), []).append(row)
    surface_mae = np.mean([float(r["mae_m"]) for r in by_model.get("surface_only", [])])
    base_mae = np.mean([float(r["mae_m"]) for r in by_model.get("base", [])])
    out = []
    for model, model_rows in by_model.items():
        mae = np.mean([float(r["mae_m"]) for r in model_rows])
        out.append(
            {
                "model": model,
                "n_events": len(model_rows),
                "mae_mm": mae * 1000.0,
                "rmse_mm": np.mean([float(r["rmse_m"]) for r in model_rows]) * 1000.0,
                "csi_0p03": np.mean([float(r["csi_0p03"]) for r in model_rows]),
                "csi_0p15": np.mean([float(r["csi_0p15"]) for r in model_rows]),
                "peak_error_mm": np.mean([float(r["peak_error_m"]) for r in model_rows]) * 1000.0,
                "signed_residual_mae_mm": np.mean([float(r["signed_residual_mae_m"]) for r in model_rows]) * 1000.0,
                "mae_to_mike_mm": np.mean([float(r["mae_to_mike_m"]) for r in model_rows]) * 1000.0,
                "mae_improvement_vs_surface_pct": 100.0 * (surface_mae - mae) / surface_mae if surface_mae > 0 else math.nan,
                "mae_improvement_vs_base_pct": 100.0 * (base_mae - mae) / base_mae if base_mae > 0 else math.nan,
            }
        )
    return sorted(out, key=lambda r: float(r["mae_mm"]))


def build_importance_sample(
    events: list[str],
    static: dict[str, np.ndarray],
    swmm: dict[str, dict[str, float]],
    n_samples: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_state + 99)
    per_step = max(100, n_samples // max(len(events) * 72, 1))
    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    for event in events:
        arrays = load_event(event)
        for t in range(arrays["surface"].shape[0]):
            residual_t = arrays["official"][t] - arrays["surface"][t]
            rows, cols = choose_residual_pixels(rng, static, arrays["surface"][t], residual_t, per_step)
            x_parts.append(feature_matrix(event, arrays, static, swmm, t, rows, cols, SUPERSET_FEATURES))
            y_parts.append(residual_t[rows, cols] * 1000.0)
    x = np.vstack(x_parts)
    y = np.concatenate(y_parts)
    if x.shape[0] > n_samples:
        idx = rng.choice(np.arange(x.shape[0]), size=n_samples, replace=False)
        return x[idx], y[idx]
    return x, y


def save_feature_importance(
    models: dict[str, HistGradientBoostingRegressor],
    test_events: list[str],
    static: dict[str, np.ndarray],
    swmm: dict[str, dict[str, float]],
    args: argparse.Namespace,
) -> None:
    x_sup, y = build_importance_sample(test_events, static, swmm, args.importance_samples, args.random_state)
    col_index = {name: i for i, name in enumerate(SUPERSET_FEATURES)}
    for model_name in ["all_static", "swmm_assisted"]:
        if model_name not in models:
            continue
        features = FEATURE_GROUPS[model_name]
        cols = [col_index[name] for name in features]
        result = permutation_importance(
            models[model_name],
            x_sup[:, cols],
            y,
            n_repeats=args.importance_repeats,
            random_state=args.random_state,
            scoring="neg_mean_absolute_error",
        )
        rows = [
            {
                "model": model_name,
                "feature": feature,
                "importance_mean": float(mean),
                "importance_std": float(std),
            }
            for feature, mean, std in zip(features, result.importances_mean, result.importances_std)
        ]
        rows = sorted(rows, key=lambda r: r["importance_mean"], reverse=True)
        write_csv(OUT / "metrics" / f"feature_importance_{model_name}.csv", rows)
        top = rows[:20]
        plt.figure(figsize=(7.1, 4.6))
        plt.barh([r["feature"] for r in reversed(top)], [r["importance_mean"] for r in reversed(top)], color="#0072B2")
        plt.xlabel("Permutation importance (MAE increase, mm)")
        plt.title(f"DrainLite-ConnectedResidual {model_name} feature importance")
        plt.tight_layout()
        plt.savefig(OUT / "figures" / f"feature_importance_{model_name}.png")
        plt.close()


def main() -> int:
    args = parse_args()
    configure_paths(args.dataset_name, args.output_name, args.connected_metrics)
    for d in [OUT / "metrics", OUT / "figures", OUT / "models", OUT / "predictions"]:
        d.mkdir(parents=True, exist_ok=True)
    static = load_static()
    swmm = read_connected_swmm()
    make_static_figures(static)
    x_all, y, weights = build_training_samples(args.train_events, static, swmm, args.pixels_per_step, args.random_state)
    np.save(OUT / "metrics" / "training_target_connected_signed_residual_mm_sample.npy", y)
    models = train_models(x_all, y, weights, args)

    rows: list[dict[str, object]] = []
    for event in args.test_events:
        arrays = load_event(event)
        event_dir = OUT / "predictions" / event
        event_dir.mkdir(parents=True, exist_ok=True)
        rows.append(event_metrics(event, "surface_only", arrays["surface"], arrays["official"], arrays["surface"], arrays["mike"], static["active_mask"]))
        for model_name, model in models.items():
            pred, residual_mm = predict_event(
                event,
                model,
                FEATURE_GROUPS[model_name],
                arrays,
                static,
                swmm,
                args.predict_chunk,
                args.residual_deadband_mm,
            )
            np.save(event_dir / f"h_drainlite_connected_{model_name}.npy", pred)
            np.save(event_dir / f"signed_residual_pred_{model_name}_mm.npy", residual_mm)
            rows.append(event_metrics(event, model_name, pred, arrays["official"], arrays["surface"], arrays["mike"], static["active_mask"]))
            if model_name == "all_static":
                make_peak_map(event, arrays, pred, residual_mm, OUT / "figures" / f"{event}_connected_peak_maps.png")
        print(f"predicted {event}", flush=True)

    write_csv(OUT / "metrics" / "connected_residual_event_metrics.csv", rows)
    write_csv(OUT / "metrics" / "connected_residual_ablation_summary.csv", summarize(rows))
    save_feature_importance(models, args.test_events, static, swmm, args)
    metadata = {
        "model": "LarNO-DrainLite-ConnectedResidual",
        "positioning": "lightweight residual correction for injecting native ITZI-SWMM drainage effects into a surface flood prediction",
        "dataset": args.dataset_name,
        "target": "1000 * (h_itzi_swmm_connected - h_itzi_surface), signed mm",
        "prediction": "h_pred = max(h_itzi_surface + signed_residual_mm / 1000, 0)",
        "train_events": args.train_events,
        "test_events": args.test_events,
        "feature_groups": FEATURE_GROUPS,
        "superset_features": SUPERSET_FEATURES,
        "pixels_per_step": args.pixels_per_step,
        "max_iter": args.max_iter,
        "learning_rate": args.learning_rate,
        "max_leaf_nodes": args.max_leaf_nodes,
        "l2_regularization": args.l2_regularization,
        "random_state": args.random_state,
        "connected_metrics": str(CONNECTED_METRICS),
        "quality_note": "SWMM continuity warnings are inherited from the physical labels and must be disclosed in reports.",
        "workflow": str(ROOT / "extended_study" / "ITZI_SWMM_FIXED_WORKFLOW.md"),
    }
    (OUT / "connected_residual_run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote DrainLite connected residual outputs to {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
