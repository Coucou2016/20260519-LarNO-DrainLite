#!/usr/bin/env python3
"""Reviewer-driven DrainLite experiments with aligned drainage rasters.

The experiment fixes three confounders in the earlier analysis: row/column
coordinates are separated from the physical baseline, every feature ablation
uses identical sampled pixels, and negative controls preserve the sampled
locations while disrupting only drainage-feature alignment.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import OrderedDict
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import distance_transform_edt
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance

import train_drainlite_connected_residual as connected
from publication_plot_style import add_panel_labels, configure_publication_style
from train_drainlite_residual import (
    ALL_STATIC_EXTRA,
    BASE_FEATURES,
    HYDRAULIC_FEATURES,
    MASK_FEATURES,
    feature_matrix as base_feature_matrix,
)


configure_publication_style()

ROOT = Path(__file__).resolve().parents[1]
DATASET = "region1_20m_connected_swmm_v2_inf1mmh"
OUTPUT_NAME = "larno_drainlite_major_revision"
EVENTS = ["event1", "event67", "event68", "event69", "event70"]
CELL_SIZE_M = 20.0
COUNT_FEATURES = ["drain_inlet_count", "pipe_segment_count"]
BASE_NO_XY = [name for name in BASE_FEATURES if name not in {"row_norm", "col_norm"}]
ALL_STATIC_EXTENDED = list(OrderedDict.fromkeys(ALL_STATIC_EXTRA + COUNT_FEATURES))
FEATURE_GROUPS: OrderedDict[str, list[str]] = OrderedDict(
    [
        ("base_no_xy", BASE_NO_XY),
        ("base_xy", BASE_FEATURES),
        ("no_xy_masks", BASE_NO_XY + MASK_FEATURES),
        ("no_xy_hydraulic", BASE_NO_XY + HYDRAULIC_FEATURES),
        ("no_xy_all_static", BASE_NO_XY + ALL_STATIC_EXTRA),
        ("no_xy_all_static_count", BASE_NO_XY + ALL_STATIC_EXTENDED),
        ("all_static_xy", BASE_FEATURES + ALL_STATIC_EXTENDED),
        ("no_xy_all_static_shift", BASE_NO_XY + ALL_STATIC_EXTENDED),
        ("no_xy_all_static_shuffle", BASE_NO_XY + ALL_STATIC_EXTENDED),
        ("sink_only_no_xy_all_static_count", BASE_NO_XY + ALL_STATIC_EXTENDED),
    ]
)
SUPERSET_FEATURES = list(
    OrderedDict.fromkeys(BASE_FEATURES + ALL_STATIC_EXTENDED)
)
STATIC_MATRIX_FEATURES = set(ALL_STATIC_EXTENDED) | {
    "pipe_density_3x3",
    "pipe_density_7x7",
    "inlet_density_7x7",
    "capacity_density_7x7",
}
KEY_PREDICTIONS = {
    "base_no_xy",
    "no_xy_all_static_count",
    "all_static_xy",
    "no_xy_all_static_shift",
    "no_xy_all_static_shuffle",
    "sink_only_no_xy_all_static_count",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", default=DATASET)
    parser.add_argument("--output-name", default=OUTPUT_NAME)
    parser.add_argument("--events", nargs="+", default=EVENTS)
    parser.add_argument("--pixels-per-step", type=int, default=700)
    parser.add_argument("--max-iter", type=int, default=220)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-leaf-nodes", type=int, default=31)
    parser.add_argument("--l2-regularization", type=float, default=0.01)
    parser.add_argument("--random-state", type=int, default=671)
    parser.add_argument("--predict-chunk", type=int, default=60000)
    parser.add_argument("--importance-samples", type=int, default=10000)
    parser.add_argument("--importance-repeats", type=int, default=3)
    parser.add_argument("--bootstrap-repeats", type=int, default=10000)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def shifted(array: np.ndarray, dr: int = 1, dc: int = 1, fill: float = 0.0) -> np.ndarray:
    out = np.full_like(array, fill)
    src_r0, src_r1 = max(-dr, 0), min(array.shape[0] - dr, array.shape[0])
    src_c0, src_c1 = max(-dc, 0), min(array.shape[1] - dc, array.shape[1])
    dst_r0, dst_r1 = max(dr, 0), min(array.shape[0] + dr, array.shape[0])
    dst_c0, dst_c1 = max(dc, 0), min(array.shape[1] + dc, array.shape[1])
    if src_r1 > src_r0 and src_c1 > src_c0:
        out[dst_r0:dst_r1, dst_c0:dst_c1] = array[src_r0:src_r1, src_c0:src_c1]
    return out


def static_variants(dataset_name: str, random_state: int) -> dict[str, dict[str, np.ndarray]]:
    connected.configure_paths(dataset_name, OUTPUT_NAME)
    aligned = connected.load_static()
    geo = ROOT / "LarNO-main" / "benchmark" / "urbanflood" / "geodata" / dataset_name
    for name in COUNT_FEATURES:
        aligned[name] = np.load(geo / f"{name}.npy").astype(np.float32)

    shift_static = {name: value.copy() for name, value in aligned.items()}
    shuffle_static = {name: value.copy() for name, value in aligned.items()}
    drainage_fields = sorted(STATIC_MATRIX_FEATURES)
    for name in drainage_fields:
        fill = float(np.nanmax(aligned[name])) if name == "distance_to_outfall" else 0.0
        shift_static[name] = shifted(aligned[name], dr=1, dc=1, fill=fill)

    active_flat = np.flatnonzero(aligned["active_mask"])
    rng = np.random.default_rng(random_state + 1000)
    permutation = rng.permutation(active_flat.size)
    for name in drainage_fields:
        source = aligned[name].ravel()[active_flat]
        target = shuffle_static[name].ravel()
        target[active_flat] = source[permutation]
        shuffle_static[name] = target.reshape(aligned[name].shape)
    return {"aligned": aligned, "shift": shift_static, "shuffle": shuffle_static}


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
    supported = [name for name in features if name not in COUNT_FEATURES]
    base = base_feature_matrix(event, arrays, static, swmm, t, rows, cols, supported)
    columns = {name: base[:, idx] for idx, name in enumerate(supported)}
    for name in COUNT_FEATURES:
        if name in features:
            columns[name] = static[name][rows, cols]
    return np.column_stack([columns[name] for name in features]).astype(np.float32, copy=False)


def sample_locations(
    events: list[str],
    static: dict[str, np.ndarray],
    pixels_per_step: int,
    random_state: int,
) -> dict[str, list[tuple[np.ndarray, np.ndarray]]]:
    out: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    for event_index, event in enumerate(events):
        arrays = connected.load_event(event)
        rng = np.random.default_rng(random_state + 31 * (event_index + 1))
        event_locations = []
        for t in range(arrays["surface"].shape[0]):
            residual_t = arrays["official"][t] - arrays["surface"][t]
            event_locations.append(
                connected.choose_residual_pixels(
                    rng, static, arrays["surface"][t], residual_t, pixels_per_step
                )
            )
        out[event] = event_locations
    return out


def build_event_samples(
    event: str,
    arrays: dict[str, np.ndarray],
    variants: dict[str, dict[str, np.ndarray]],
    swmm: dict[str, dict[str, float]],
    locations: list[tuple[np.ndarray, np.ndarray]],
) -> dict[str, np.ndarray]:
    matrices = {key: [] for key in variants}
    y_signed_parts = []
    y_sink_parts = []
    weight_parts = []
    for t, (rows, cols) in enumerate(locations):
        for key, static in variants.items():
            matrices[key].append(
                feature_matrix(event, arrays, static, swmm, t, rows, cols, SUPERSET_FEATURES)
            )
        residual_mm = (arrays["official"][t, rows, cols] - arrays["surface"][t, rows, cols]) * 1000.0
        weights = np.ones_like(residual_mm, dtype=np.float32)
        weights += (np.abs(residual_mm) > 0.2).astype(np.float32) * 4.0
        weights += (arrays["surface"][t, rows, cols] > 0.03).astype(np.float32) * 2.0
        weights += (variants["aligned"]["pipe_density_7x7"][rows, cols] > 0).astype(np.float32)
        y_signed_parts.append(residual_mm.astype(np.float32))
        y_sink_parts.append(np.maximum(-residual_mm, 0.0).astype(np.float32))
        weight_parts.append(weights)
    return {
        **{key: np.vstack(parts) for key, parts in matrices.items()},
        "y_signed": np.concatenate(y_signed_parts),
        "y_sink": np.concatenate(y_sink_parts),
        "weights": np.concatenate(weight_parts),
    }


def fit_model(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    features: list[str],
    args: argparse.Namespace,
) -> HistGradientBoostingRegressor:
    index = {name: idx for idx, name in enumerate(SUPERSET_FEATURES)}
    cols = [index[name] for name in features]
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
    model.fit(x[:, cols], y, sample_weight=weights)
    return model


def predict_event(
    event: str,
    model: HistGradientBoostingRegressor,
    features: list[str],
    arrays: dict[str, np.ndarray],
    static: dict[str, np.ndarray],
    swmm: dict[str, dict[str, float]],
    chunk_size: int,
    sink_only: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    t_count, height, width = arrays["surface"].shape
    active_rows, active_cols = np.where(static["active_mask"])
    residual_mm = np.zeros((t_count, height, width), dtype=np.float32)
    for t in range(t_count):
        values = np.zeros((height, width), dtype=np.float32)
        for start in range(0, active_rows.size, chunk_size):
            rows = active_rows[start : start + chunk_size]
            cols = active_cols[start : start + chunk_size]
            x = feature_matrix(event, arrays, static, swmm, t, rows, cols, features)
            pred = model.predict(x).astype(np.float32)
            values[rows, cols] = -np.maximum(pred, 0.0) if sink_only else pred
        residual_mm[t] = np.clip(values, -1200.0, 1200.0)
    prediction = np.maximum(arrays["surface"] + residual_mm / 1000.0, 0.0).astype(np.float32)
    return prediction, residual_mm


def condition_metrics(
    event: str,
    model: str,
    pred: np.ndarray,
    arrays: dict[str, np.ndarray],
    active: np.ndarray,
    proximity_m: np.ndarray,
) -> list[dict[str, object]]:
    target = arrays["official"]
    surface = arrays["surface"]
    true_residual = target - surface
    masks: OrderedDict[str, np.ndarray] = OrderedDict(
        [
            ("all_active", np.broadcast_to(active, pred.shape)),
            ("surface_wet_1mm", (surface > 0.001) & active),
            ("union_wet_1mm", ((surface > 0.001) | (target > 0.001)) & active),
            ("target_severe_30mm", (target > 0.03) & active),
            ("residual_active_1mm", (np.abs(true_residual) > 0.001) & active),
            ("negative_residual_1mm", (true_residual < -0.001) & active),
            ("positive_residual_1mm", (true_residual > 0.001) & active),
        ]
    )
    for low, high, label in [(0, 20, "network_0_20m"), (20, 60, "network_20_60m"), (60, 140, "network_60_140m")]:
        masks[label] = np.broadcast_to((proximity_m >= low) & (proximity_m < high) & active, pred.shape)
    masks["network_over_140m"] = np.broadcast_to((proximity_m >= 140) & active, pred.shape)

    rows = []
    for condition, mask in masks.items():
        count = int(mask.sum())
        if count == 0:
            continue
        p = pred[mask]
        t = target[mask]
        s = surface[mask]
        err = p - t
        rows.append(
            {
                "event": event,
                "model": model,
                "condition": condition,
                "n_cell_times": count,
                "mae_mm": float(np.mean(np.abs(err)) * 1000.0),
                "rmse_mm": float(np.sqrt(np.mean(err * err)) * 1000.0),
                "bias_mm": float(np.mean(err) * 1000.0),
                "signed_residual_mae_mm": float(np.mean(np.abs((p - s) - (t - s))) * 1000.0),
                "csi_0p03": connected.csi(p, t, 0.03),
                "csi_0p15": connected.csi(p, t, 0.15),
            }
        )
    return rows


def mike_metrics(
    event: str,
    model: str,
    pred: np.ndarray,
    mike: np.ndarray,
    active: np.ndarray,
) -> dict[str, object]:
    p = pred[:, active]
    m = mike[:, active]
    err = p - m
    p_global_peak = float(p.max())
    m_global_peak = float(m.max())
    p_volume = p.sum(axis=1) * connected.CELL_AREA_M2
    m_volume = m.sum(axis=1) * connected.CELL_AREA_M2
    return {
        "event": event,
        "model": model,
        "mae_mm": float(np.mean(np.abs(err)) * 1000.0),
        "rmse_mm": float(np.sqrt(np.mean(err * err)) * 1000.0),
        "csi_0p03": connected.csi(p, m, 0.03),
        "csi_0p15": connected.csi(p, m, 0.15),
        "global_peak_bias_mm": (p_global_peak - m_global_peak) * 1000.0,
        "global_peak_abs_error_mm": abs(p_global_peak - m_global_peak) * 1000.0,
        "peak_map_mae_mm": float(np.mean(np.abs(pred.max(axis=0)[active] - mike.max(axis=0)[active])) * 1000.0),
        "final_volume_bias_m3": float(p_volume[-1] - m_volume[-1]),
        "final_volume_abs_error_m3": float(abs(p_volume[-1] - m_volume[-1])),
    }


def summarize_event_metrics(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: OrderedDict[str, list[dict[str, object]]] = OrderedDict()
    for row in rows:
        grouped.setdefault(str(row["model"]), []).append(row)
    surface_mae = np.mean([float(row["mae_m"]) for row in grouped["surface_only"]])
    base_mae = np.mean([float(row["mae_m"]) for row in grouped["base_no_xy"]])
    output = []
    for model, model_rows in grouped.items():
        mae = np.mean([float(row["mae_m"]) for row in model_rows])
        output.append(
            {
                "model": model,
                "n_events": len(model_rows),
                "mae_mm": mae * 1000.0,
                "rmse_mm": np.mean([float(row["rmse_m"]) for row in model_rows]) * 1000.0,
                "csi_0p03": np.mean([float(row["csi_0p03"]) for row in model_rows]),
                "csi_0p15": np.mean([float(row["csi_0p15"]) for row in model_rows]),
                "signed_peak_bias_mm": np.mean([float(row["peak_error_m"]) for row in model_rows]) * 1000.0,
                "mean_abs_peak_error_mm": np.mean([float(row["abs_peak_error_m"]) for row in model_rows]) * 1000.0,
                "mean_abs_final_volume_error_m3": np.mean([abs(float(row["final_volume_error_m3"])) for row in model_rows]),
                "improvement_vs_surface_pct": 100.0 * (surface_mae - mae) / surface_mae,
                "improvement_vs_base_no_xy_pct": 100.0 * (base_mae - mae) / base_mae,
            }
        )
    return sorted(output, key=lambda row: float(row["mae_mm"]))


def paired_bootstrap(
    rows: list[dict[str, object]],
    candidate: str,
    reference: str,
    repeats: int,
    random_state: int,
) -> dict[str, object]:
    lookup = {(str(row["event"]), str(row["model"])): float(row["mae_m"]) * 1000.0 for row in rows}
    events = sorted({event for event, model in lookup if model == candidate and (event, reference) in lookup})
    deltas = np.array([lookup[(event, candidate)] - lookup[(event, reference)] for event in events], dtype=float)
    rng = np.random.default_rng(random_state)
    boot = np.mean(deltas[rng.integers(0, len(deltas), size=(repeats, len(deltas)))], axis=1)
    signs = np.array([-1.0, 1.0])
    exact = np.array(np.meshgrid(*([signs] * len(deltas)))).T.reshape(-1, len(deltas))
    exact_means = np.mean(exact * np.abs(deltas), axis=1)
    observed = abs(float(np.mean(deltas)))
    p_value = float(np.mean(np.abs(exact_means) >= observed - 1e-12))
    return {
        "candidate": candidate,
        "reference": reference,
        "n_events": len(events),
        "event_mean_delta_mae_mm": float(np.mean(deltas)),
        "event_median_delta_mae_mm": float(np.median(deltas)),
        "event_sd_delta_mae_mm": float(np.std(deltas, ddof=1)),
        "bootstrap_ci95_low_mm": float(np.quantile(boot, 0.025)),
        "bootstrap_ci95_high_mm": float(np.quantile(boot, 0.975)),
        "wins": int(np.sum(deltas < 0)),
        "losses": int(np.sum(deltas > 0)),
        "ties": int(np.sum(deltas == 0)),
        "exact_sign_flip_p_two_sided": p_value,
        "event_deltas_mm": {event: float(delta) for event, delta in zip(events, deltas)},
    }


def plot_results(
    out: Path,
    event_rows: list[dict[str, object]],
    summary: list[dict[str, object]],
    importance_rows: list[dict[str, object]],
) -> None:
    figures = out / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    display = [
        "surface_only",
        "base_no_xy",
        "base_xy",
        "no_xy_all_static",
        "no_xy_all_static_count",
        "all_static_xy",
        "no_xy_all_static_shift",
        "no_xy_all_static_shuffle",
        "sink_only_no_xy_all_static_count",
    ]
    summary_lookup = {str(row["model"]): row for row in summary}
    labels = [name for name in display if name in summary_lookup]
    means = [float(summary_lookup[name]["mae_mm"]) for name in labels]
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.25), constrained_layout=True)
    axes[0].bar(np.arange(len(labels)), means, color="#2563eb", alpha=0.82)
    for xpos, name in enumerate(labels):
        values = [float(row["mae_m"]) * 1000.0 for row in event_rows if row["model"] == name]
        axes[0].scatter(np.full(len(values), xpos), values, color="#111827", s=11, zorder=3)
    axes[0].set_xticks(np.arange(len(labels)))
    axes[0].set_xticklabels(labels, rotation=55, ha="right", fontsize=7)
    axes[0].set_ylabel("MAE to coupled label (mm)")
    axes[0].set_title("Five-event leave-one-event-out comparison")
    axes[0].grid(axis="y", alpha=0.25)

    compare = ["base_no_xy", "no_xy_all_static_count", "all_static_xy", "sink_only_no_xy_all_static_count"]
    for name, color in zip(compare, ["#6b7280", "#2563eb", "#f97316", "#16a34a"]):
        values = [float(row["mae_m"]) * 1000.0 for row in event_rows if row["model"] == name]
        axes[1].plot(EVENTS[: len(values)], values, marker="o", lw=1.25, label=name, color=color)
    axes[1].set_ylabel("Event MAE (mm)")
    axes[1].set_title("Paired event-level performance")
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=6.8)
    add_panel_labels(axes, x=-0.13, y=1.03)
    fig.savefig(figures / "major_revision_ablation_event_dots.png")
    plt.close(fig)

    if importance_rows:
        by_feature: dict[str, list[float]] = {}
        for row in importance_rows:
            by_feature.setdefault(str(row["feature"]), []).append(float(row["importance_mae_mm"]))
        ranked = sorted(by_feature, key=lambda name: np.mean(by_feature[name]), reverse=True)[:15][::-1]
        means = [np.mean(by_feature[name]) for name in ranked]
        stds = [np.std(by_feature[name], ddof=1) if len(by_feature[name]) > 1 else 0.0 for name in ranked]
        fig, ax = plt.subplots(figsize=(7.1, 4.2), constrained_layout=True)
        ypos = np.arange(len(ranked))
        ax.barh(ypos, means, xerr=stds, color="#2563eb", alpha=0.82, capsize=2)
        for y, name in zip(ypos, ranked):
            ax.scatter(by_feature[name], np.full(len(by_feature[name]), y), color="#111827", s=10, zorder=3)
        ax.set_yticks(ypos)
        ax.set_yticklabels(ranked, fontsize=8)
        ax.set_xlabel("Held-event permutation increase in MAE (mm)")
        ax.set_title("Cross-fold permutation importance; correlated-feature interpretation")
        ax.grid(axis="x", alpha=0.25)
        fig.savefig(figures / "major_revision_cross_fold_importance.png")
        plt.close(fig)


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    out = ROOT / "extended_study" / "output" / args.output_name
    for directory in [out / "metrics", out / "models", out / "predictions", out / "figures"]:
        directory.mkdir(parents=True, exist_ok=True)
    connected.configure_paths(args.dataset_name, args.output_name)
    variants = static_variants(args.dataset_name, args.random_state)
    swmm = connected.read_connected_swmm()
    locations = sample_locations(args.events, variants["aligned"], args.pixels_per_step, args.random_state)
    event_samples = {}
    event_arrays = {}
    for event in args.events:
        print(f"preparing fixed samples: {event}", flush=True)
        arrays = connected.load_event(event)
        event_arrays[event] = arrays
        event_samples[event] = build_event_samples(event, arrays, variants, swmm, locations[event])

    active = variants["aligned"]["active_mask"]
    network_mask = (variants["aligned"]["drain_inlet_mask"] > 0) | (variants["aligned"]["pipe_mask"] > 0)
    proximity_m = distance_transform_edt(~network_mask) * CELL_SIZE_M
    event_rows: list[dict[str, object]] = []
    condition_rows: list[dict[str, object]] = []
    mike_rows: list[dict[str, object]] = []
    importance_rows: list[dict[str, object]] = []

    for fold_index, test_event in enumerate(args.events, start=1):
        train_events = [event for event in args.events if event != test_event]
        print(f"fold {fold_index}/{len(args.events)} test={test_event}", flush=True)
        aligned_x = np.vstack([event_samples[event]["aligned"] for event in train_events])
        shift_x = np.vstack([event_samples[event]["shift"] for event in train_events])
        shuffle_x = np.vstack([event_samples[event]["shuffle"] for event in train_events])
        y_signed = np.concatenate([event_samples[event]["y_signed"] for event in train_events])
        y_sink = np.concatenate([event_samples[event]["y_sink"] for event in train_events])
        weights = np.concatenate([event_samples[event]["weights"] for event in train_events])
        models = {}
        for model_name, features in FEATURE_GROUPS.items():
            source_x = shift_x if model_name.endswith("_shift") else shuffle_x if model_name.endswith("_shuffle") else aligned_x
            target_y = y_sink if model_name.startswith("sink_only") else y_signed
            print(f"  training {model_name}: {source_x.shape[0]:,} rows, {len(features)} features", flush=True)
            models[model_name] = fit_model(source_x, target_y, weights, features, args)
            model_dir = out / "models" / test_event
            model_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(
                {"model": models[model_name], "features": features, "target": "sink" if model_name.startswith("sink_only") else "signed"},
                model_dir / f"{model_name}.joblib",
            )

        arrays = event_arrays[test_event]
        baseline_row = connected.event_metrics(
            test_event, "surface_only", arrays["surface"], arrays["official"], arrays["surface"], arrays["mike"], active
        )
        event_rows.append(baseline_row)
        condition_rows.extend(condition_metrics(test_event, "surface_only", arrays["surface"], arrays, active, proximity_m))
        mike_rows.append(mike_metrics(test_event, "ITZI surface-only", arrays["surface"], arrays["mike"], active))
        mike_rows.append(mike_metrics(test_event, "ITZI-SWMM coupled", arrays["official"], arrays["mike"], active))

        predictions: dict[str, np.ndarray] = {}
        for model_name, model in models.items():
            static_key = "shift" if model_name.endswith("_shift") else "shuffle" if model_name.endswith("_shuffle") else "aligned"
            pred, residual_mm = predict_event(
                test_event,
                model,
                FEATURE_GROUPS[model_name],
                arrays,
                variants[static_key],
                swmm,
                args.predict_chunk,
                sink_only=model_name.startswith("sink_only"),
            )
            predictions[model_name] = pred
            event_rows.append(
                connected.event_metrics(test_event, model_name, pred, arrays["official"], arrays["surface"], arrays["mike"], active)
            )
            if model_name in KEY_PREDICTIONS:
                pred_dir = out / "predictions" / test_event
                pred_dir.mkdir(parents=True, exist_ok=True)
                np.save(pred_dir / f"h_{model_name}.npy", pred)
                if model_name in {"no_xy_all_static_count", "sink_only_no_xy_all_static_count"}:
                    np.save(pred_dir / f"residual_{model_name}_mm.npy", residual_mm)
            if model_name in {"base_no_xy", "no_xy_all_static_count", "all_static_xy", "no_xy_all_static_shift", "no_xy_all_static_shuffle", "sink_only_no_xy_all_static_count"}:
                condition_rows.extend(condition_metrics(test_event, model_name, pred, arrays, active, proximity_m))
            if model_name in {"base_no_xy", "no_xy_all_static_count", "all_static_xy", "sink_only_no_xy_all_static_count"}:
                mike_rows.append(mike_metrics(test_event, model_name, pred, arrays["mike"], active))

        importance_model = models["no_xy_all_static_count"]
        sample = event_samples[test_event]["aligned"]
        y_test = event_samples[test_event]["y_signed"]
        features = FEATURE_GROUPS["no_xy_all_static_count"]
        col_index = {name: idx for idx, name in enumerate(SUPERSET_FEATURES)}
        cols = [col_index[name] for name in features]
        rng = np.random.default_rng(args.random_state + fold_index)
        take = rng.choice(sample.shape[0], size=min(args.importance_samples, sample.shape[0]), replace=False)
        importance = permutation_importance(
            importance_model,
            sample[take][:, cols],
            y_test[take],
            scoring="neg_mean_absolute_error",
            n_repeats=args.importance_repeats,
            random_state=args.random_state + fold_index,
        )
        for feature, mean, std in zip(features, importance.importances_mean, importance.importances_std):
            importance_rows.append(
                {
                    "fold_event": test_event,
                    "feature": feature,
                    "importance_mae_mm": float(mean),
                    "importance_std_mm": float(std),
                    "n_samples": int(take.size),
                }
            )
        write_csv(out / "metrics" / "event_metrics_partial.csv", event_rows)

    summary = summarize_event_metrics(event_rows)
    bootstraps = [
        paired_bootstrap(event_rows, "no_xy_all_static_count", "surface_only", args.bootstrap_repeats, args.random_state),
        paired_bootstrap(event_rows, "no_xy_all_static_count", "base_no_xy", args.bootstrap_repeats, args.random_state + 1),
        paired_bootstrap(event_rows, "all_static_xy", "no_xy_all_static_count", args.bootstrap_repeats, args.random_state + 2),
        paired_bootstrap(event_rows, "no_xy_all_static_shift", "no_xy_all_static_count", args.bootstrap_repeats, args.random_state + 3),
        paired_bootstrap(event_rows, "no_xy_all_static_shuffle", "no_xy_all_static_count", args.bootstrap_repeats, args.random_state + 4),
        paired_bootstrap(event_rows, "sink_only_no_xy_all_static_count", "no_xy_all_static_count", args.bootstrap_repeats, args.random_state + 5),
    ]
    write_csv(out / "metrics" / "event_metrics.csv", event_rows)
    write_csv(out / "metrics" / "ablation_summary.csv", summary)
    write_csv(out / "metrics" / "conditional_metrics.csv", condition_rows)
    write_csv(out / "metrics" / "mike_configuration_screening_comparison.csv", mike_rows)
    write_csv(out / "metrics" / "cross_fold_permutation_importance.csv", importance_rows)
    write_csv(out / "metrics" / "paired_bootstrap_summary.csv", [{k: v for k, v in row.items() if k != "event_deltas_mm"} for row in bootstraps])
    (out / "metrics" / "paired_bootstrap_details.json").write_text(json.dumps(bootstraps, indent=2), encoding="utf-8")
    plot_results(out, event_rows, summary, importance_rows)

    metadata = {
        "dataset": args.dataset_name,
        "events": args.events,
        "evaluation": "five-event leave-one-event-out",
        "feature_groups": FEATURE_GROUPS,
        "fixed_sampling": True,
        "coordinate_mapping": "static rasters rebuilt with the same int() rule as the ITZI coupling runner",
        "negative_controls": {
            "shift": "all drainage fields shifted one 20 m cell south and east; sampling locations unchanged",
            "shuffle": "one fixed active-cell permutation applied jointly to all drainage fields; sampling locations unchanged",
        },
        "positive_residual_caveat": "positive coupled-minus-surface residual is not treated as proof of sewer surcharge",
        "runtime_seconds": time.perf_counter() - started,
        "arguments": vars(args),
    }
    (out / "major_revision_experiment_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"summary": summary, "bootstrap": bootstraps, "runtime_seconds": metadata["runtime_seconds"]}, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
