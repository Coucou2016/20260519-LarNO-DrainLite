#!/usr/bin/env python3
"""Train LarNO-DrainLite, a lightweight drainage residual model.

DrainLite learns the reduction from ITZI surface-only depth to the
ITZI + conceptual inlet-sink label.  It is intentionally CPU-friendly:
no LarNO backbone is trained or executed during fitting.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import OrderedDict
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import uniform_filter
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error


ROOT = Path(__file__).resolve().parents[1]
FLOOD_DIR = ROOT / "LarNO-main" / "benchmark" / "urbanflood" / "flood" / "region1_20m_drainage_v1"
GEO_DIR = ROOT / "LarNO-main" / "benchmark" / "urbanflood" / "geodata" / "region1_20m_drainage_v1"
CONFIG_DIR = ROOT / "LarNO-main" / "code" / "urbanflood_larfno" / "configs"
SWMM_CSV = ROOT / "extended_study" / "output" / "swmm_network" / "swmm_summary_metrics.csv"
OUT = ROOT / "extended_study" / "output" / "drainlite_residual"
CELL_AREA_M2 = 20.0 * 20.0
CELL_SIZE_M = 20.0
WALL_HEIGHT = 49.9

DRAINAGE_FEATURES = [
    "drain_inlet_mask",
    "drain_outfall_mask",
    "pipe_mask",
    "pipe_diameter",
    "pipe_slope",
    "pipe_capacity",
    "pipe_cover_depth",
    "distance_to_outfall",
]

BASE_FEATURES = [
    "surface_depth_mm",
    "rainfall_mm",
    "cumsum_rainfall_mm",
    "dem_m",
    "dem_slope",
    "row_norm",
    "col_norm",
    "time_norm",
    "surface_depth_3x3_mean_mm",
    "surface_depth_7x7_mean_mm",
]
MASK_FEATURES = [
    "drain_inlet_mask",
    "drain_outfall_mask",
    "pipe_mask",
    "distance_to_outfall",
    "pipe_density_3x3",
    "pipe_density_7x7",
    "inlet_density_7x7",
]
HYDRAULIC_FEATURES = [
    "pipe_diameter",
    "pipe_slope",
    "pipe_capacity",
    "pipe_cover_depth",
    "distance_to_outfall",
    "capacity_density_7x7",
]
ALL_STATIC_EXTRA = [
    "drain_inlet_mask",
    "drain_outfall_mask",
    "pipe_mask",
    "pipe_diameter",
    "pipe_slope",
    "pipe_capacity",
    "pipe_cover_depth",
    "distance_to_outfall",
    "pipe_density_3x3",
    "pipe_density_7x7",
    "inlet_density_7x7",
    "capacity_density_7x7",
]
SWMM_FEATURES = [
    "swmm_precip_mm",
    "swmm_routing_inflow_m3",
    "swmm_external_outflow_m3",
    "swmm_routing_final_storage_m3",
    "swmm_routing_continuity_error_pct",
    "swmm_outfall_volume_m3",
    "swmm_flooded_node_count",
    "swmm_node_flooding_volume_m3",
]

FEATURE_GROUPS: OrderedDict[str, list[str]] = OrderedDict([
    ("base", BASE_FEATURES),
    ("mask_only", BASE_FEATURES + MASK_FEATURES),
    ("hydraulic_only", BASE_FEATURES + HYDRAULIC_FEATURES),
    ("all_static", BASE_FEATURES + ALL_STATIC_EXTRA),
    ("swmm_assisted", BASE_FEATURES + ALL_STATIC_EXTRA + SWMM_FEATURES),
])
SUPERSET_FEATURES = list(OrderedDict.fromkeys(FEATURE_GROUPS["swmm_assisted"]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pixels-per-step", type=int, default=2000)
    parser.add_argument("--max-iter", type=int, default=400)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-leaf-nodes", type=int, default=31)
    parser.add_argument("--l2-regularization", type=float, default=0.01)
    parser.add_argument("--random-state", type=int, default=666)
    parser.add_argument("--importance-samples", type=int, default=6000)
    parser.add_argument("--importance-repeats", type=int, default=3)
    parser.add_argument("--predict-chunk", type=int, default=60000)
    parser.add_argument("--train-list", default="region1_drainage_train.txt")
    parser.add_argument("--test-list", default="region1_drainage_test.txt")
    return parser.parse_args()


def read_event_list(name: str) -> list[str]:
    return [
        line.strip()
        for line in (CONFIG_DIR / name).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def read_swmm() -> dict[str, dict[str, float]]:
    rows: dict[str, dict[str, float]] = {}
    if not SWMM_CSV.exists():
        return rows
    with SWMM_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            event = row["event"]
            rows[event] = {}
            for key, value in row.items():
                if key == "event":
                    continue
                try:
                    rows[event][key] = float(value)
                except Exception:
                    rows[event][key] = 0.0
    return rows


def load_static() -> dict[str, np.ndarray]:
    dem = np.load(GEO_DIR / "dem.npy").astype(np.float32)
    h, w = dem.shape
    rr, cc = np.indices((h, w), dtype=np.float32)
    gy, gx = np.gradient(np.nan_to_num(dem, nan=WALL_HEIGHT), CELL_SIZE_M, CELL_SIZE_M)
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


def load_event(event: str) -> dict[str, np.ndarray]:
    event_dir = FLOOD_DIR / event
    surface = np.load(event_dir / "h_itzi_surface.npy").astype(np.float32)
    sink = np.load(event_dir / "h_itzi_sink.npy").astype(np.float32)
    mike = np.load(event_dir / "h_mike_ref.npy").astype(np.float32)
    rainfall = np.load(event_dir / "rainfall.npy").astype(np.float32)
    return {
        "surface": surface,
        "sink": sink,
        "mike": mike,
        "rainfall": rainfall,
        "cumsum": np.cumsum(rainfall, axis=0, dtype=np.float32),
    }


def choose_pixels(
    rng: np.random.Generator,
    static: dict[str, np.ndarray],
    surface_t: np.ndarray,
    target_reduction_t: np.ndarray,
    n: int,
) -> tuple[np.ndarray, np.ndarray]:
    active = static["active_mask"]
    wet = active & (surface_t > 0.001)
    positive = active & (target_reduction_t > 1e-5)
    near_pipe = active & (
        (static["pipe_density_7x7"] > 0)
        | (static["inlet_density_7x7"] > 0)
        | (static["distance_to_outfall"] < 250.0)
    )

    chosen: list[np.ndarray] = []
    quotas = [
        (positive, int(n * 0.35)),
        (wet, int(n * 0.25)),
        (near_pipe, int(n * 0.25)),
        (active, n),
    ]
    for mask, quota in quotas:
        if quota <= 0:
            continue
        flat = np.flatnonzero(mask)
        if flat.size == 0:
            continue
        take = min(quota, flat.size)
        chosen.append(rng.choice(flat, size=take, replace=False))
    if not chosen:
        flat = np.flatnonzero(active)
        chosen = [rng.choice(flat, size=min(n, flat.size), replace=False)]
    idx = np.unique(np.concatenate(chosen))
    if idx.size > n:
        idx = rng.choice(idx, size=n, replace=False)
    elif idx.size < n:
        active_flat = np.flatnonzero(active)
        extra = rng.choice(active_flat, size=min(n - idx.size, active_flat.size), replace=False)
        idx = np.unique(np.concatenate([idx, extra]))
        if idx.size > n:
            idx = rng.choice(idx, size=n, replace=False)
    return np.unravel_index(idx, active.shape)


def swmm_feature_values(event: str, swmm: dict[str, dict[str, float]]) -> dict[str, float]:
    row = swmm.get(event, {})
    return {
        "swmm_precip_mm": row.get("precip_mm", 0.0),
        "swmm_routing_inflow_m3": row.get("routing_inflow_m3", 0.0) / 1_000_000.0,
        "swmm_external_outflow_m3": row.get("external_outflow_m3", 0.0) / 100_000.0,
        "swmm_routing_final_storage_m3": row.get("routing_final_storage_m3", 0.0) / 1_000_000.0,
        "swmm_routing_continuity_error_pct": row.get("routing_continuity_error_pct", 0.0),
        "swmm_outfall_volume_m3": row.get("outfall_volume_m3", 0.0) / 100_000.0,
        "swmm_flooded_node_count": row.get("flooded_node_count", 0.0) / 1000.0,
        "swmm_node_flooding_volume_m3": row.get("node_flooding_volume_m3", 0.0) / 1_000_000.0,
    }


def feature_matrix(
    event: str,
    arrays: dict[str, np.ndarray],
    static: dict[str, np.ndarray],
    swmm: dict[str, dict[str, float]],
    t: int,
    rows: np.ndarray,
    cols: np.ndarray,
    features: list[str],
) -> np.ndarray:
    surface_t = arrays["surface"][t]
    surf3 = uniform_filter(surface_t, size=3, mode="nearest")
    surf7 = uniform_filter(surface_t, size=7, mode="nearest")
    scalar_swmm = swmm_feature_values(event, swmm)
    values: dict[str, np.ndarray] = {
        "surface_depth_mm": surface_t[rows, cols] * 1000.0,
        "rainfall_mm": arrays["rainfall"][t, rows, cols],
        "cumsum_rainfall_mm": arrays["cumsum"][t, rows, cols],
        "dem_m": static["dem_m"][rows, cols],
        "dem_slope": static["dem_slope"][rows, cols],
        "row_norm": static["row_norm"][rows, cols],
        "col_norm": static["col_norm"][rows, cols],
        "time_norm": np.full(rows.shape, t / 71.0, dtype=np.float32),
        "surface_depth_3x3_mean_mm": surf3[rows, cols] * 1000.0,
        "surface_depth_7x7_mean_mm": surf7[rows, cols] * 1000.0,
    }
    for name in DRAINAGE_FEATURES:
        values[name] = static[name][rows, cols]
    for name in ["pipe_density_3x3", "pipe_density_7x7", "inlet_density_7x7", "capacity_density_7x7"]:
        values[name] = static[name][rows, cols]
    for name, value in scalar_swmm.items():
        values[name] = np.full(rows.shape, value, dtype=np.float32)
    return np.column_stack([values[name] for name in features]).astype(np.float32, copy=False)


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
            surface_t = arrays["surface"][t]
            reduction_t = np.maximum(arrays["surface"][t] - arrays["sink"][t], 0.0)
            rows, cols = choose_pixels(rng, static, surface_t, reduction_t, pixels_per_step)
            x = feature_matrix(event, arrays, static, swmm, t, rows, cols, SUPERSET_FEATURES)
            y = reduction_t[rows, cols] * 1000.0
            weights = np.ones_like(y, dtype=np.float32)
            weights += (y > 0.5).astype(np.float32) * 4.0
            weights += (surface_t[rows, cols] > 0.03).astype(np.float32) * 2.0
            x_parts.append(x)
            y_parts.append(y.astype(np.float32))
            w_parts.append(weights)
        print(f"sampled {event}: {sum(len(p) for p in y_parts):,} rows total")
    return np.vstack(x_parts), np.concatenate(y_parts), np.concatenate(w_parts)


def train_models(
    x_all: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, HistGradientBoostingRegressor]:
    models: dict[str, HistGradientBoostingRegressor] = {}
    col_index = {name: i for i, name in enumerate(SUPERSET_FEATURES)}
    model_dir = OUT / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
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
        print(f"training {model_name}: rows={x_all.shape[0]:,}, features={len(cols)}")
        model.fit(x_all[:, cols], y, sample_weight=weights)
        models[model_name] = model
        joblib.dump({"model": model, "features": features}, model_dir / f"{model_name}.joblib")
    return models


def csi(pred: np.ndarray, target: np.ndarray, threshold: float) -> float:
    p = pred >= threshold
    t = target >= threshold
    tp = float(np.logical_and(p, t).sum())
    fp = float(np.logical_and(p, ~t).sum())
    fn = float(np.logical_and(~p, t).sum())
    denom = tp + fp + fn
    return 1.0 if denom == 0 else tp / denom


def event_metrics(
    event: str,
    model: str,
    pred: np.ndarray,
    target: np.ndarray,
    surface: np.ndarray,
    target_name: str,
) -> dict[str, float | str]:
    err = pred - target
    target_vol = target.sum(axis=(1, 2)) * CELL_AREA_M2
    peak_t = int(np.argmax(target_vol))
    target_reduction = np.maximum(surface - target, 0.0)
    pred_reduction = np.maximum(surface - pred, 0.0)
    actual_reduction_volume = float(target_reduction.sum() * CELL_AREA_M2)
    pred_reduction_volume = float(pred_reduction.sum() * CELL_AREA_M2)
    capture_pct = 100.0 * pred_reduction_volume / actual_reduction_volume if actual_reduction_volume > 0 else math.nan
    return {
        "event": event,
        "model": model,
        "target": target_name,
        "mae_m": float(np.mean(np.abs(err))),
        "rmse_m": float(np.sqrt(np.mean(err * err))),
        "csi_0p03": csi(pred, target, 0.03),
        "csi_0p15": csi(pred, target, 0.15),
        "peak_pred_m": float(np.max(pred)),
        "peak_target_m": float(np.max(target)),
        "peak_error_m": float(np.max(pred) - np.max(target)),
        "mean_error_m": float(np.mean(err)),
        "peak_volume_error_m3": float((pred[peak_t].sum() - target[peak_t].sum()) * CELL_AREA_M2),
        "peak_area_error_m2_0p03": float(((pred[peak_t] >= 0.03).sum() - (target[peak_t] >= 0.03).sum()) * CELL_AREA_M2),
        "reduction_mae_m": float(np.mean(np.abs(pred_reduction - target_reduction))),
        "reduction_capture_pct": capture_pct,
        "reduction_capture_error_pct": float(capture_pct - 100.0) if not math.isnan(capture_pct) else math.nan,
    }


def predict_event(
    event: str,
    model_name: str,
    model: HistGradientBoostingRegressor,
    features: list[str],
    arrays: dict[str, np.ndarray],
    static: dict[str, np.ndarray],
    swmm: dict[str, dict[str, float]],
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    t_count, h, w = arrays["surface"].shape
    active_rows, active_cols = np.where(static["active_mask"])
    reduction_mm = np.zeros((t_count, h, w), dtype=np.float32)
    for t in range(t_count):
        pred_t = np.zeros((h, w), dtype=np.float32)
        for start in range(0, active_rows.size, chunk_size):
            rows = active_rows[start:start + chunk_size]
            cols = active_cols[start:start + chunk_size]
            x = feature_matrix(event, arrays, static, swmm, t, rows, cols, features)
            pred = model.predict(x).astype(np.float32)
            max_reduction = arrays["surface"][t, rows, cols] * 1000.0
            pred = np.clip(pred, 0.0, max_reduction)
            pred_t[rows, cols] = pred
        reduction_mm[t] = pred_t
    h_pred = np.maximum(arrays["surface"] - reduction_mm / 1000.0, 0.0).astype(np.float32)
    return h_pred, reduction_mm


def save_predictions_and_metrics(
    models: dict[str, HistGradientBoostingRegressor],
    test_events: list[str],
    static: dict[str, np.ndarray],
    swmm: dict[str, dict[str, float]],
    args: argparse.Namespace,
) -> list[dict[str, float | str]]:
    pred_root = OUT / "predictions"
    pred_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float | str]] = []
    for event in test_events:
        arrays = load_event(event)
        event_dir = pred_root / event
        event_dir.mkdir(parents=True, exist_ok=True)
        surface = arrays["surface"].astype(np.float32)
        for target_name, target in [("itzi_sink", arrays["sink"]), ("mike_ref", arrays["mike"])]:
            rows.append(event_metrics(event, "surface_only", surface, target, surface, target_name))
            rows.append(event_metrics(event, "itzi_sink_label", arrays["sink"], target, surface, target_name))
        for model_name, model in models.items():
            h_pred, reduction_mm = predict_event(
                event,
                model_name,
                model,
                FEATURE_GROUPS[model_name],
                arrays,
                static,
                swmm,
                args.predict_chunk,
            )
            np.save(event_dir / f"h_drainlite_{model_name}.npy", h_pred)
            np.save(event_dir / f"reduction_pred_{model_name}_mm.npy", reduction_mm)
            for target_name, target in [("itzi_sink", arrays["sink"]), ("mike_ref", arrays["mike"])]:
                rows.append(event_metrics(event, model_name, h_pred, target, surface, target_name))
            if model_name == "all_static":
                make_peak_map(event, arrays, h_pred, reduction_mm, OUT / "figures" / f"{event}_peak_maps.png")
        print(f"predicted {event}")
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def aggregate_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    sink_rows = [r for r in rows if r["target"] == "itzi_sink"]
    by_model: dict[str, list[dict[str, object]]] = {}
    for row in sink_rows:
        by_model.setdefault(str(row["model"]), []).append(row)
    surface_mae = np.mean([float(r["mae_m"]) for r in by_model.get("surface_only", [])])
    base_mae = np.mean([float(r["mae_m"]) for r in by_model.get("base", [])])
    summary = []
    for model, model_rows in by_model.items():
        mae = np.mean([float(r["mae_m"]) for r in model_rows])
        summary.append({
            "model": model,
            "target": "itzi_sink",
            "n_events": len(model_rows),
            "mae_mm": mae * 1000.0,
            "rmse_mm": np.mean([float(r["rmse_m"]) for r in model_rows]) * 1000.0,
            "csi_0p03": np.mean([float(r["csi_0p03"]) for r in model_rows]),
            "csi_0p15": np.mean([float(r["csi_0p15"]) for r in model_rows]),
            "peak_error_mm": np.mean([float(r["peak_error_m"]) for r in model_rows]) * 1000.0,
            "reduction_mae_mm": np.mean([float(r["reduction_mae_m"]) for r in model_rows]) * 1000.0,
            "mae_improvement_vs_surface_pct": 100.0 * (surface_mae - mae) / surface_mae if surface_mae > 0 else math.nan,
            "mae_improvement_vs_base_pct": 100.0 * (base_mae - mae) / base_mae if base_mae > 0 else math.nan,
        })
    return sorted(summary, key=lambda r: float(r["mae_mm"]))


def make_static_figures(static: dict[str, np.ndarray], swmm: dict[str, dict[str, float]]) -> None:
    fig_dir = OUT / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(11, 8))
    dem = np.ma.masked_where(~static["active_mask"], static["dem_m"])
    plt.imshow(dem, cmap="terrain")
    plt.contour(static["pipe_mask"], levels=[0.5], colors="black", linewidths=0.35)
    inlet_y, inlet_x = np.where(static["drain_inlet_mask"] > 0)
    out_y, out_x = np.where(static["drain_outfall_mask"] > 0)
    if inlet_y.size:
        plt.scatter(inlet_x, inlet_y, s=2, c="#0b5fff", label="inlet", alpha=0.8)
    if out_y.size:
        plt.scatter(out_x, out_y, s=18, c="#d62728", label="outfall", alpha=0.9)
    plt.title("DEM with road-aligned conceptual pipe network")
    plt.axis("off")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(fig_dir / "static_dem_network.png", dpi=180)
    plt.close()

    if swmm:
        events = list(swmm)
        continuity = [swmm[e].get("routing_continuity_error_pct", 0.0) for e in events]
        outfall = [swmm[e].get("outfall_volume_m3", 0.0) / 1000.0 for e in events]
        flooding = [swmm[e].get("node_flooding_volume_m3", 0.0) / 1000.0 for e in events]
        x = np.arange(len(events))
        fig, ax1 = plt.subplots(figsize=(12, 5))
        ax1.bar(x - 0.18, outfall, width=0.36, label="outfall volume (10^3 m3)", color="#4477aa")
        ax1.bar(x + 0.18, flooding, width=0.36, label="node ponding volume (10^3 m3)", color="#cc6677")
        ax1.set_ylabel("Volume (10^3 m3)")
        ax2 = ax1.twinx()
        ax2.plot(x, continuity, color="#228833", marker="o", label="routing continuity error (%)")
        ax2.set_ylabel("Continuity error (%)")
        ax1.set_xticks(x)
        ax1.set_xticklabels([e.replace("event", "E") for e in events], rotation=45, ha="right")
        ax1.set_title("SWMM standalone dynamic-wave summary")
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc="upper left")
        fig.tight_layout()
        fig.savefig(fig_dir / "swmm_summary.png", dpi=180)
        plt.close(fig)


def make_peak_map(event: str, arrays: dict[str, np.ndarray], pred: np.ndarray, reduction_mm: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mike_peak = arrays["mike"].max(axis=0)
    surface_peak = arrays["surface"].max(axis=0)
    sink_peak = arrays["sink"].max(axis=0)
    pred_peak = pred.max(axis=0)
    error_peak = pred_peak - sink_peak
    reduction_peak = reduction_mm.max(axis=0) / 1000.0
    vmax = max(float(np.nanmax(mike_peak)), float(np.nanmax(surface_peak)), float(np.nanmax(sink_peak)), float(np.nanmax(pred_peak)))
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    panels = [
        ("MIKE reference peak", mike_peak, "viridis", 0, vmax),
        ("ITZI surface-only peak", surface_peak, "viridis", 0, vmax),
        ("ITZI + sink label peak", sink_peak, "viridis", 0, vmax),
        ("DrainLite all_static peak", pred_peak, "viridis", 0, vmax),
        ("DrainLite - sink error", error_peak, "coolwarm", -max(0.2, vmax / 3), max(0.2, vmax / 3)),
        ("Predicted reduction", reduction_peak, "magma", 0, max(0.05, float(np.nanmax(reduction_peak)))),
    ]
    for ax, (title, data, cmap, vmin, vmax_i) in zip(axes.flat, panels):
        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax_i)
        ax.set_title(title)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.038, pad=0.02)
    fig.suptitle(event)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def build_importance_sample(
    events: list[str],
    static: dict[str, np.ndarray],
    swmm: dict[str, dict[str, float]],
    n_samples: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_state + 1000)
    x_parts = []
    y_parts = []
    per_event_step = max(20, n_samples // max(len(events) * 72, 1))
    for event in events:
        arrays = load_event(event)
        for t in range(72):
            reduction_t = np.maximum(arrays["surface"][t] - arrays["sink"][t], 0.0)
            rows, cols = choose_pixels(rng, static, arrays["surface"][t], reduction_t, per_event_step)
            x_parts.append(feature_matrix(event, arrays, static, swmm, t, rows, cols, SUPERSET_FEATURES))
            y_parts.append(reduction_t[rows, cols] * 1000.0)
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
    metrics_dir = OUT / "metrics"
    fig_dir = OUT / "figures"
    x_sup, y = build_importance_sample(test_events, static, swmm, args.importance_samples, args.random_state)
    col_index = {name: i for i, name in enumerate(SUPERSET_FEATURES)}
    for model_name in ["all_static", "swmm_assisted"]:
        if model_name not in models:
            continue
        features = FEATURE_GROUPS[model_name]
        cols = [col_index[name] for name in features]
        x = x_sup[:, cols]
        result = permutation_importance(
            models[model_name],
            x,
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
        write_csv(metrics_dir / f"feature_importance_{model_name}.csv", rows)

        top = rows[:20]
        plt.figure(figsize=(9, 6))
        plt.barh([r["feature"] for r in reversed(top)], [r["importance_mean"] for r in reversed(top)], color="#4477aa")
        plt.xlabel("Permutation importance (MAE increase, mm)")
        plt.title(f"DrainLite {model_name} feature importance")
        plt.tight_layout()
        plt.savefig(fig_dir / f"feature_importance_{model_name}.png", dpi=180)
        plt.close()


def main() -> int:
    args = parse_args()
    (OUT / "metrics").mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    train_events = read_event_list(args.train_list)
    test_events = read_event_list(args.test_list)
    static = load_static()
    swmm = read_swmm()
    make_static_figures(static, swmm)

    x_all, y, weights = build_training_samples(
        train_events,
        static,
        swmm,
        pixels_per_step=args.pixels_per_step,
        random_state=args.random_state,
    )
    np.save(OUT / "metrics" / "training_target_reduction_mm_sample.npy", y)
    models = train_models(x_all, y, weights, args)
    rows = save_predictions_and_metrics(models, test_events, static, swmm, args)
    write_csv(OUT / "metrics" / "drainlite_event_metrics.csv", rows)
    summary = aggregate_summary(rows)
    write_csv(OUT / "metrics" / "drainlite_ablation_summary.csv", summary)
    save_feature_importance(models, test_events, static, swmm, args)
    metadata = {
        "model": "LarNO-DrainLite",
        "train_events": train_events,
        "test_events": test_events,
        "feature_groups": FEATURE_GROUPS,
        "superset_features": SUPERSET_FEATURES,
        "pixels_per_step": args.pixels_per_step,
        "max_iter": args.max_iter,
        "learning_rate": args.learning_rate,
        "max_leaf_nodes": args.max_leaf_nodes,
        "l2_regularization": args.l2_regularization,
        "random_state": args.random_state,
        "target": "1000 * max(h_itzi_surface - h_itzi_sink, 0), in mm",
        "prediction": "h_drainlite_m = max(h_itzi_surface_m - pred_reduction_mm / 1000, 0)",
        "boundary": "SWMM is standalone 1D dynamic-wave metrics only; not two-way ITZI-SWMM coupling",
    }
    (OUT / "drainlite_run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote DrainLite outputs to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
