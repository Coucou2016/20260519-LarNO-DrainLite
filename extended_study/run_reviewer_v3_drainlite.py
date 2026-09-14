#!/usr/bin/env python3
"""Run leakage-controlled DrainLite experiments on reviewer-v3 labels.

The learning target is the signed drainage residual C-B, where B is the
surface-only control with exactly the same inlet-neighbourhood roughness as C,
and C is the native Itzi-SWMM bidirectional simulation.  MIKE arrays are used
only for descriptive external-reference metrics.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import time
from collections import OrderedDict
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import psutil
from scipy.ndimage import distance_transform_edt, uniform_filter
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance

from publication_plot_style import add_panel_labels, configure_publication_style


configure_publication_style()

ROOT = Path(__file__).resolve().parents[1]
CELL_M = 20.0
CELL_AREA_M2 = CELL_M * CELL_M
WALL_HEIGHT_M = 49.9

BASE = [
    "surface_depth_mm",
    "rainfall_mm",
    "cumulative_rainfall_mm",
    "dem_m",
    "dem_slope",
    "time_norm",
    "time_sin",
    "time_cos",
    "surface_depth_3x3_mean_mm",
    "surface_depth_7x7_mean_mm",
]
MASKS = [
    "drain_inlet_mask",
    "pipe_mask",
    "drain_inlet_count",
    "pipe_segment_count",
    "network_degree",
    "pipe_density_3x3",
    "pipe_density_7x7",
    "inlet_density_7x7",
]
HYDRAULICS = [
    "pipe_diameter",
    "pipe_slope",
    "pipe_capacity",
    "pipe_cover_depth",
    "capacity_density_7x7",
]
GROUPS: OrderedDict[str, list[str]] = OrderedDict(
    [
        ("base", BASE),
        ("mask_static", BASE + MASKS),
        ("hydraulic_static", BASE + HYDRAULICS),
        ("all_static", BASE + MASKS + HYDRAULICS),
        # These two models retain the all-static predictors and isolate the
        # influence of target-aware sampling and sample weighting.
        ("all_static_unweighted", BASE + MASKS + HYDRAULICS),
        ("all_static_uniform", BASE + MASKS + HYDRAULICS),
    ]
)
SUPERSET = list(OrderedDict.fromkeys(BASE + MASKS + HYDRAULICS))
PRIMITIVE_DRAINAGE = [
    "drain_inlet_mask",
    "pipe_mask",
    "drain_inlet_count",
    "pipe_segment_count",
    "network_degree",
    "pipe_diameter",
    "pipe_slope",
    "pipe_capacity",
    "pipe_cover_depth",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", default="region1_20m_drainage_v3_full")
    parser.add_argument("--output-name", default="reviewer_major_revision_v3/drainlite_v3")
    parser.add_argument("--events", nargs="+", default=None)
    parser.add_argument("--pixels-per-step", type=int, default=450)
    parser.add_argument("--max-iter-candidates", nargs="+", type=int, default=[80, 140, 220])
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-leaf-nodes", type=int, default=31)
    parser.add_argument("--l2-regularization", type=float, default=0.01)
    parser.add_argument("--predict-chunk", type=int, default=50000)
    parser.add_argument("--importance-samples", type=int, default=10000)
    parser.add_argument("--importance-repeats", type=int, default=3)
    parser.add_argument("--block-null-repeats", type=int, default=50)
    # Both full-domain dimensions (400 x 560) are divisible by 20, so this
    # block size preserves every drainage value under block permutation.
    parser.add_argument("--block-size", type=int, default=20)
    parser.add_argument("--random-state", type=int, default=866)
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


def csi(pred: np.ndarray, target: np.ndarray, threshold: float) -> float:
    p = pred >= threshold
    t = target >= threshold
    tp = float(np.logical_and(p, t).sum())
    fp = float(np.logical_and(p, ~t).sum())
    fn = float(np.logical_and(~p, t).sum())
    denominator = tp + fp + fn
    return 1.0 if denominator == 0 else tp / denominator


def load_static(geo: Path) -> dict[str, np.ndarray]:
    dem = np.load(geo / "dem.npy").astype(np.float32)
    gy, gx = np.gradient(np.nan_to_num(dem, nan=WALL_HEIGHT_M), CELL_M, CELL_M)
    static: dict[str, np.ndarray] = {
        "dem_m": np.nan_to_num(dem, nan=WALL_HEIGHT_M).astype(np.float32),
        "active_mask": np.load(geo / "active_mask.npy").astype(bool),
        "dem_slope": np.sqrt(gx * gx + gy * gy).astype(np.float32),
    }
    for name in PRIMITIVE_DRAINAGE:
        static[name] = np.nan_to_num(np.load(geo / f"{name}.npy").astype(np.float32), nan=0.0)
    return regenerate_descriptors(static)


def regenerate_descriptors(static: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    out = {name: values.copy() for name, values in static.items()}
    out["pipe_density_3x3"] = uniform_filter(out["pipe_mask"], size=3, mode="constant").astype(np.float32)
    out["pipe_density_7x7"] = uniform_filter(out["pipe_mask"], size=7, mode="constant").astype(np.float32)
    out["inlet_density_7x7"] = uniform_filter(out["drain_inlet_mask"], size=7, mode="constant").astype(np.float32)
    out["capacity_density_7x7"] = uniform_filter(out["pipe_capacity"], size=7, mode="constant").astype(np.float32)
    return out


def shifted(array: np.ndarray, dr: int, dc: int) -> np.ndarray:
    result = np.zeros_like(array)
    src_r0, src_r1 = max(-dr, 0), min(array.shape[0] - dr, array.shape[0])
    src_c0, src_c1 = max(-dc, 0), min(array.shape[1] - dc, array.shape[1])
    dst_r0, dst_r1 = max(dr, 0), min(array.shape[0] + dr, array.shape[0])
    dst_c0, dst_c1 = max(dc, 0), min(array.shape[1] + dc, array.shape[1])
    if src_r1 > src_r0 and src_c1 > src_c0:
        result[dst_r0:dst_r1, dst_c0:dst_c1] = array[src_r0:src_r1, src_c0:src_c1]
    return result


def shift_static(static: dict[str, np.ndarray], dr: int, dc: int) -> dict[str, np.ndarray]:
    result = {name: values.copy() for name, values in static.items() if name not in SUPERSET}
    result["dem_m"] = static["dem_m"]
    result["dem_slope"] = static["dem_slope"]
    result["active_mask"] = static["active_mask"]
    for name in PRIMITIVE_DRAINAGE:
        result[name] = shifted(static[name], dr, dc)
    return regenerate_descriptors(result)


def block_shuffle_static(
    static: dict[str, np.ndarray], block_size: int, random_state: int
) -> dict[str, np.ndarray]:
    height, width = static["active_mask"].shape
    nrows = math.ceil(height / block_size)
    ncols = math.ceil(width / block_size)
    destinations = [(r, c) for r in range(nrows) for c in range(ncols)]
    rng = np.random.default_rng(random_state)
    sources = [destinations[index] for index in rng.permutation(len(destinations))]
    result: dict[str, np.ndarray] = {
        "dem_m": static["dem_m"],
        "dem_slope": static["dem_slope"],
        "active_mask": static["active_mask"],
    }
    for name in PRIMITIVE_DRAINAGE:
        source = static[name]
        target = np.zeros_like(source)
        for (dst_r, dst_c), (src_r, src_c) in zip(destinations, sources):
            dr0, dc0 = dst_r * block_size, dst_c * block_size
            sr0, sc0 = src_r * block_size, src_c * block_size
            block = source[sr0 : min(sr0 + block_size, height), sc0 : min(sc0 + block_size, width)]
            take_h = min(block.shape[0], height - dr0)
            take_w = min(block.shape[1], width - dc0)
            target[dr0 : dr0 + take_h, dc0 : dc0 + take_w] = block[:take_h, :take_w]
        result[name] = target
    return regenerate_descriptors(result)


def load_event(flood: Path, event: str) -> dict[str, np.ndarray]:
    path = flood / event
    arrays = {
        "surface_original": np.load(path / "h_itzi_surface.npy").astype(np.float32),
        "surface": np.load(path / "h_itzi_surface_matched.npy").astype(np.float32),
        "coupled": np.load(path / "h_itzi_swmm.npy").astype(np.float32),
        "mike": np.load(path / "h_mike_ref.npy").astype(np.float32),
        "rainfall": np.load(path / "rainfall.npy").astype(np.float32),
    }
    arrays["cumulative"] = np.cumsum(arrays["rainfall"], axis=0, dtype=np.float32)
    shapes = {tuple(values.shape) for values in arrays.values()}
    if len(shapes) != 1 or next(iter(shapes))[0] != 72:
        raise ValueError(f"{event}: inconsistent or non-72-step arrays: {sorted(shapes)}")
    if any(not np.isfinite(values).all() for values in arrays.values()):
        raise ValueError(f"{event}: NaN or Inf detected")
    return arrays


def choose_pixels(
    rng: np.random.Generator,
    static: dict[str, np.ndarray],
    surface: np.ndarray,
    residual: np.ndarray,
    n: int,
) -> tuple[np.ndarray, np.ndarray]:
    active = static["active_mask"]
    near_network = active & ((static["pipe_density_7x7"] > 0) | (static["inlet_density_7x7"] > 0))
    strata = [
        (active & (np.abs(residual) > 0.0002), int(n * 0.35)),
        (active & (surface > 0.03), int(n * 0.25)),
        (near_network, int(n * 0.25)),
        (active, n),
    ]
    pieces: list[np.ndarray] = []
    for mask, quota in strata:
        candidates = np.flatnonzero(mask)
        if quota > 0 and candidates.size:
            pieces.append(rng.choice(candidates, min(quota, candidates.size), replace=False))
    selected = np.unique(np.concatenate(pieces))
    if selected.size < n:
        background = np.setdiff1d(np.flatnonzero(active), selected, assume_unique=False)
        if background.size:
            selected = np.concatenate(
                [selected, rng.choice(background, min(n - selected.size, background.size), replace=False)]
            )
    if selected.size > n:
        selected = rng.choice(selected, n, replace=False)
    return np.unravel_index(selected, active.shape)


def feature_matrix(
    arrays: dict[str, np.ndarray],
    static: dict[str, np.ndarray],
    t: int,
    rows: np.ndarray,
    cols: np.ndarray,
    features: list[str] = SUPERSET,
) -> np.ndarray:
    surface = arrays["surface"][t]
    surf3 = uniform_filter(surface, size=3, mode="nearest")
    surf7 = uniform_filter(surface, size=7, mode="nearest")
    phase = 2.0 * np.pi * (t + 1) / 72.0
    values: dict[str, np.ndarray] = {
        "surface_depth_mm": surface[rows, cols] * 1000.0,
        "rainfall_mm": arrays["rainfall"][t, rows, cols],
        "cumulative_rainfall_mm": arrays["cumulative"][t, rows, cols],
        "dem_m": static["dem_m"][rows, cols],
        "dem_slope": static["dem_slope"][rows, cols],
        "time_norm": np.full(rows.shape, (t + 1) / 72.0, dtype=np.float32),
        "time_sin": np.full(rows.shape, np.sin(phase), dtype=np.float32),
        "time_cos": np.full(rows.shape, np.cos(phase), dtype=np.float32),
        "surface_depth_3x3_mean_mm": surf3[rows, cols] * 1000.0,
        "surface_depth_7x7_mean_mm": surf7[rows, cols] * 1000.0,
    }
    for name in MASKS + HYDRAULICS:
        values[name] = static[name][rows, cols]
    return np.column_stack([values[name] for name in features]).astype(np.float32, copy=False)


def event_samples(
    event: str,
    arrays: dict[str, np.ndarray],
    static: dict[str, np.ndarray],
    pixels_per_step: int,
    random_state: int,
    sampling: str = "stratified",
    weighted: bool = True,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(random_state)
    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    w_parts: list[np.ndarray] = []
    time_parts: list[np.ndarray] = []
    row_parts: list[np.ndarray] = []
    col_parts: list[np.ndarray] = []
    for t in range(72):
        residual = arrays["coupled"][t] - arrays["surface"][t]
        if sampling == "uniform":
            candidates = np.flatnonzero(static["active_mask"])
            selected = rng.choice(
                candidates, min(pixels_per_step, candidates.size), replace=False
            )
            rows, cols = np.unravel_index(selected, static["active_mask"].shape)
        elif sampling == "stratified":
            rows, cols = choose_pixels(
                rng, static, arrays["surface"][t], residual, pixels_per_step
            )
        else:
            raise ValueError(f"Unknown sampling strategy: {sampling}")
        y = residual[rows, cols] * 1000.0
        weights = np.ones(y.shape, dtype=np.float32)
        if weighted:
            weights += 4.0 * (np.abs(y) > 0.2)
            weights += 2.0 * (arrays["surface"][t, rows, cols] > 0.03)
            weights += 1.0 * (static["pipe_density_7x7"][rows, cols] > 0)
        x_parts.append(feature_matrix(arrays, static, t, rows, cols))
        y_parts.append(y.astype(np.float32))
        w_parts.append(weights.astype(np.float32))
        time_parts.append(np.full(rows.shape, t, dtype=np.int16))
        row_parts.append(np.asarray(rows, dtype=np.int16))
        col_parts.append(np.asarray(cols, dtype=np.int16))
    return {
        "x": np.vstack(x_parts),
        "y": np.concatenate(y_parts),
        "weight": np.concatenate(w_parts),
        "t": np.concatenate(time_parts),
        "row": np.concatenate(row_parts),
        "col": np.concatenate(col_parts),
    }


def fit_model(
    x: np.ndarray,
    y: np.ndarray,
    weight: np.ndarray,
    features: list[str],
    max_iter: int,
    args: argparse.Namespace,
) -> HistGradientBoostingRegressor:
    indexes = {name: index for index, name in enumerate(SUPERSET)}
    cols = [indexes[name] for name in features]
    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=args.learning_rate,
        max_iter=max_iter,
        max_leaf_nodes=args.max_leaf_nodes,
        l2_regularization=args.l2_regularization,
        early_stopping=False,
        random_state=args.random_state,
    )
    model.fit(x[:, cols], y, sample_weight=weight)
    return model


def predict_sample(model: HistGradientBoostingRegressor, features: list[str], x: np.ndarray) -> np.ndarray:
    indexes = {name: index for index, name in enumerate(SUPERSET)}
    return model.predict(x[:, [indexes[name] for name in features]]).astype(np.float32)


def tune_and_refit(
    model_name: str,
    outer_train: list[str],
    validation_event: str,
    samples: dict[str, dict[str, np.ndarray]],
    args: argparse.Namespace,
    use_weights: bool = True,
) -> tuple[HistGradientBoostingRegressor, dict[str, object]]:
    if validation_event not in outer_train:
        raise ValueError(f"Inner validation event {validation_event} is not in outer training events")
    inner_train = [event for event in outer_train if event != validation_event]
    features = GROUPS[model_name]
    x_train = np.vstack([samples[event]["x"] for event in inner_train])
    y_train = np.concatenate([samples[event]["y"] for event in inner_train])
    w_train = np.concatenate([samples[event]["weight"] for event in inner_train])
    if not use_weights:
        w_train = np.ones_like(w_train)
    validation_scores = []
    for max_iter in args.max_iter_candidates:
        candidate = fit_model(x_train, y_train, w_train, features, max_iter, args)
        pred = predict_sample(candidate, features, samples[validation_event]["x"])
        validation_weight = (
            samples[validation_event]["weight"]
            if use_weights
            else np.ones_like(samples[validation_event]["weight"])
        )
        score = float(
            np.average(
                np.abs(pred - samples[validation_event]["y"]),
                weights=validation_weight,
            )
        )
        validation_scores.append({"max_iter": max_iter, "weighted_mae_mm": score})
    selected = min(validation_scores, key=lambda row: (row["weighted_mae_mm"], row["max_iter"]))["max_iter"]
    x_all = np.vstack([samples[event]["x"] for event in outer_train])
    y_all = np.concatenate([samples[event]["y"] for event in outer_train])
    w_all = np.concatenate([samples[event]["weight"] for event in outer_train])
    if not use_weights:
        w_all = np.ones_like(w_all)
    started = time.perf_counter()
    model = fit_model(x_all, y_all, w_all, features, int(selected), args)
    fit_seconds = time.perf_counter() - started
    return model, {
        "model": model_name,
        "inner_validation_event": validation_event,
        "inner_training_events": inner_train,
        "scores": validation_scores,
        "selected_max_iter": int(selected),
        "refit_rows": int(x_all.shape[0]),
        "fit_seconds": fit_seconds,
        "sample_weighting": "target-aware" if use_weights else "uniform weights",
    }


def predict_models_full(
    arrays: dict[str, np.ndarray],
    static: dict[str, np.ndarray],
    models: dict[str, HistGradientBoostingRegressor],
    chunk_size: int,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    height, width = static["active_mask"].shape
    active_rows, active_cols = np.where(static["active_mask"])
    outputs = {name: np.zeros((72, height, width), dtype=np.float32) for name in models}
    indexes = {name: index for index, name in enumerate(SUPERSET)}
    group_cols = {name: [indexes[feature] for feature in GROUPS[name]] for name in models}
    runtimes = {name: 0.0 for name in models}
    for t in range(72):
        for start in range(0, active_rows.size, chunk_size):
            rows = active_rows[start : start + chunk_size]
            cols = active_cols[start : start + chunk_size]
            x = feature_matrix(arrays, static, t, rows, cols)
            for name, model in models.items():
                then = time.perf_counter()
                residual_mm = np.clip(model.predict(x[:, group_cols[name]]), -1200.0, 1200.0)
                runtimes[name] += time.perf_counter() - then
                outputs[name][t, rows, cols] = np.maximum(
                    arrays["surface"][t, rows, cols] + residual_mm / 1000.0, 0.0
                )
    return outputs, runtimes


def climatology_predictions(
    flood: Path,
    train_events: list[str],
    surface: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    spatial_sum = np.zeros(surface.shape[1:], dtype=np.float64)
    temporal_sum = np.zeros(surface.shape, dtype=np.float64)
    for event in train_events:
        path = flood / event
        residual = (
            np.load(path / "h_itzi_swmm.npy", mmap_mode="r")
            - np.load(path / "h_itzi_surface_matched.npy", mmap_mode="r")
        )
        spatial_sum += np.asarray(residual, dtype=np.float64).sum(axis=0)
        temporal_sum += np.asarray(residual, dtype=np.float64)
    spatial_residual = (spatial_sum / (len(train_events) * 72.0)).astype(np.float32)
    temporal_residual = (temporal_sum / len(train_events)).astype(np.float32)
    spatial_pred = np.maximum(surface + spatial_residual[None, :, :], 0.0).astype(np.float32)
    temporal_pred = np.maximum(surface + temporal_residual, 0.0).astype(np.float32)
    return spatial_pred, temporal_pred


def metric_row(
    event: str,
    model: str,
    pred: np.ndarray,
    target: np.ndarray,
    surface: np.ndarray,
    active: np.ndarray,
    reference: str = "coupled_label",
) -> dict[str, object]:
    p = pred[:, active]
    t = target[:, active]
    s = surface[:, active]
    error = p - t
    peak_p = pred.max(axis=0)[active]
    peak_t = target.max(axis=0)[active]
    top_cut = float(np.quantile(peak_t, 0.999))
    top = peak_t >= top_cut
    p_volume = p.sum(axis=1, dtype=np.float64) * CELL_AREA_M2
    t_volume = t.sum(axis=1, dtype=np.float64) * CELL_AREA_M2
    p_global = float(p.max())
    t_global = float(t.max())
    target_residual = t - s
    predicted_residual = p - s
    return {
        "event": event,
        "model": model,
        "reference": reference,
        "active_cells": int(active.sum()),
        "n_cell_times": int(p.size),
        "mae_mm": float(np.mean(np.abs(error)) * 1000.0),
        "rmse_mm": float(np.sqrt(np.mean(error * error)) * 1000.0),
        "bias_mm": float(np.mean(error) * 1000.0),
        "csi_0p03": csi(p, t, 0.03),
        "csi_0p15": csi(p, t, 0.15),
        "peak_map_mae_mm": float(np.mean(np.abs(peak_p - peak_t)) * 1000.0),
        "global_peak_bias_mm": (p_global - t_global) * 1000.0,
        "global_peak_abs_error_mm": abs(p_global - t_global) * 1000.0,
        "top0p1_peak_map_mae_mm": float(np.mean(np.abs(peak_p[top] - peak_t[top])) * 1000.0),
        "target_peak_p99p9_mm": top_cut * 1000.0,
        "final_volume_bias_m3": float(p_volume[-1] - t_volume[-1]),
        "final_volume_abs_error_m3": float(abs(p_volume[-1] - t_volume[-1])),
        "residual_mae_mm": float(np.mean(np.abs(predicted_residual - target_residual)) * 1000.0),
    }


def conditional_rows(
    event: str,
    model: str,
    pred: np.ndarray,
    arrays: dict[str, np.ndarray],
    static: dict[str, np.ndarray],
    proximity_m: np.ndarray,
) -> list[dict[str, object]]:
    target = arrays["coupled"]
    surface = arrays["surface"]
    residual = target - surface
    active3 = np.broadcast_to(static["active_mask"], target.shape)
    conditions = OrderedDict(
        [
            ("all_active", active3),
            ("wet_either_1mm", active3 & ((target > 0.001) | (surface > 0.001))),
            ("target_over_30mm", active3 & (target > 0.03)),
            ("target_over_150mm", active3 & (target > 0.15)),
            ("residual_abs_over_1mm", active3 & (np.abs(residual) > 0.001)),
            ("negative_residual_over_1mm", active3 & (residual < -0.001)),
            ("positive_residual_over_1mm", active3 & (residual > 0.001)),
        ]
    )
    for low, high in [(0, 20), (20, 60), (60, 140), (140, math.inf)]:
        label = f"network_{low}_{'inf' if math.isinf(high) else high}m"
        mask2 = static["active_mask"] & (proximity_m >= low) & (proximity_m < high)
        conditions[label] = np.broadcast_to(mask2, target.shape)
    total = int(active3.sum())
    rows = []
    for condition, mask in conditions.items():
        count = int(mask.sum())
        if not count:
            continue
        p, t = pred[mask], target[mask]
        error = p - t
        rows.append(
            {
                "event": event,
                "model": model,
                "condition": condition,
                "n_cell_times": count,
                "proportion_of_active_pct": 100.0 * count / total,
                "mae_mm": float(np.mean(np.abs(error)) * 1000.0),
                "rmse_mm": float(np.sqrt(np.mean(error * error)) * 1000.0),
                "bias_mm": float(np.mean(error) * 1000.0),
                "csi_0p03": csi(p, t, 0.03),
                "csi_0p15": csi(p, t, 0.15),
            }
        )
    return rows


def transformed_sample_matrix(
    arrays: dict[str, np.ndarray],
    transformed: dict[str, np.ndarray],
    sample: dict[str, np.ndarray],
) -> np.ndarray:
    parts = []
    for t in range(72):
        select = sample["t"] == t
        parts.append(feature_matrix(arrays, transformed, t, sample["row"][select], sample["col"][select]))
    return np.vstack(parts)


def residual_correlations(
    flood: Path, events: list[str], active: np.ndarray
) -> tuple[list[dict[str, object]], np.ndarray]:
    maps = []
    for event in events:
        path = flood / event
        residual = np.load(path / "h_itzi_swmm.npy") - np.load(path / "h_itzi_surface_matched.npy")
        maps.append(np.mean(residual, axis=0)[active])
    matrix = np.corrcoef(np.asarray(maps))
    rows = [
        {"event_a": first, "event_b": second, "pearson_r": float(matrix[i, j])}
        for i, first in enumerate(events)
        for j, second in enumerate(events)
    ]
    return rows, matrix


def summarize(rows: list[dict[str, object]], reference: str) -> list[dict[str, object]]:
    selected = [row for row in rows if row["reference"] == reference]
    models = list(OrderedDict.fromkeys(str(row["model"]) for row in selected))
    result = []
    for model in models:
        group = [row for row in selected if row["model"] == model]
        result.append(
            {
                "model": model,
                "reference": reference,
                "n_events": len(group),
                **{
                    metric: float(np.mean([float(row[metric]) for row in group]))
                    for metric in [
                        "mae_mm",
                        "rmse_mm",
                        "csi_0p03",
                        "csi_0p15",
                        "peak_map_mae_mm",
                        "global_peak_abs_error_mm",
                        "top0p1_peak_map_mae_mm",
                        "final_volume_abs_error_m3",
                    ]
                },
            }
        )
    return sorted(result, key=lambda row: row["mae_mm"])


def plot_summary(out: Path, summary: list[dict[str, object]], event_rows: list[dict[str, object]]) -> None:
    figures = out / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    names = [str(row["model"]) for row in summary]
    values = [float(row["mae_mm"]) for row in summary]
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.2), constrained_layout=True)
    axes[0].bar(np.arange(len(names)), values, color="#2b6cb0", alpha=0.85)
    axes[0].set_xticks(np.arange(len(names)))
    axes[0].set_xticklabels(names, rotation=50, ha="right", fontsize=7)
    axes[0].set_ylabel("MAE to matched coupled label (mm)")
    axes[0].set_title("Event-level cross-validation")
    for position, name in enumerate(names):
        points = [float(row["mae_mm"]) for row in event_rows if row["reference"] == "coupled_label" and row["model"] == name]
        axes[0].scatter(np.full(len(points), position), points, s=10, color="#1a202c", zorder=3)
    display = [name for name in ["surface_matched", "spatial_climatology", "spatiotemporal_climatology", "base", "all_static"] if name in names]
    events = list(OrderedDict.fromkeys(str(row["event"]) for row in event_rows))
    for name, colour in zip(display, ["#6b7280", "#7c3aed", "#db2777", "#d97706", "#2563eb"]):
        lookup = {(str(row["event"]), str(row["model"])): float(row["mae_mm"]) for row in event_rows if row["reference"] == "coupled_label"}
        axes[1].plot(events, [lookup[(event, name)] for event in events], marker="o", ms=3, lw=1.1, label=name, color=colour)
    axes[1].set_ylabel("Held-event MAE (mm)")
    axes[1].set_title("Paired performance by event")
    axes[1].tick_params(axis="x", rotation=35)
    axes[1].legend(fontsize=6.5)
    add_panel_labels(axes, x=-0.13, y=1.03)
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
    for suffix in ["png", "pdf", "svg"]:
        fig.savefig(figures / f"drainlite_v3_cross_validation.{suffix}", dpi=400 if suffix == "png" else None)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    process = psutil.Process(os.getpid())
    flood = ROOT / "LarNO-main" / "benchmark" / "urbanflood" / "flood" / args.dataset_name
    geo = ROOT / "LarNO-main" / "benchmark" / "urbanflood" / "geodata" / args.dataset_name
    out = ROOT / "extended_study" / "output" / args.output_name
    for directory in [out / "metrics", out / "models", out / "predictions", out / "figures"]:
        directory.mkdir(parents=True, exist_ok=True)
    dataset_audit = json.loads((flood / "dataset_audit.json").read_text(encoding="utf-8"))
    events = args.events or dataset_audit["accepted_events"]
    if len(events) < 5:
        raise RuntimeError(f"At least five accepted events are required; found {events}")
    static = load_static(geo)
    arrays_by_event: dict[str, dict[str, np.ndarray]] = {}
    samples: dict[str, dict[str, np.ndarray]] = {}
    uniform_samples: dict[str, dict[str, np.ndarray]] = {}
    for index, event in enumerate(events):
        print(f"loading and sampling {event}", flush=True)
        arrays_by_event[event] = load_event(flood, event)
        samples[event] = event_samples(
            event,
            arrays_by_event[event],
            static,
            args.pixels_per_step,
            args.random_state + 1009 * (index + 1),
        )
        uniform_samples[event] = event_samples(
            event,
            arrays_by_event[event],
            static,
            args.pixels_per_step,
            args.random_state + 200003 + 1009 * (index + 1),
            sampling="uniform",
            weighted=False,
        )
    network_mask = (static["drain_inlet_mask"] > 0) | (static["pipe_mask"] > 0)
    proximity_m = distance_transform_edt(~network_mask) * CELL_M
    event_rows: list[dict[str, object]] = []
    conditional: list[dict[str, object]] = []
    tuning_rows: list[dict[str, object]] = []
    importance_rows: list[dict[str, object]] = []
    alignment_rows: list[dict[str, object]] = []
    runtime_rows: list[dict[str, object]] = []

    for fold, test_event in enumerate(events):
        outer_train = [event for event in events if event != test_event]
        validation_event = events[(fold + 1) % len(events)]
        print(f"outer fold {fold + 1}/{len(events)}: test={test_event}", flush=True)
        models: dict[str, HistGradientBoostingRegressor] = {}
        fold_tuning = []
        for model_name in GROUPS:
            model_samples = (
                uniform_samples if model_name == "all_static_uniform" else samples
            )
            use_weights = model_name not in {
                "all_static_unweighted",
                "all_static_uniform",
            }
            model, tuning = tune_and_refit(
                model_name,
                outer_train,
                validation_event,
                model_samples,
                args,
                use_weights=use_weights,
            )
            tuning["sampling"] = (
                "uniform active cells"
                if model_name == "all_static_uniform"
                else "residual/wet/network-stratified"
            )
            tuning["outer_test_event"] = test_event
            fold_tuning.append(tuning)
            models[model_name] = model
            model_path = out / "models" / test_event / f"{model_name}.joblib"
            model_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump({"model": model, "features": GROUPS[model_name], "tuning": tuning}, model_path)
            runtime_rows.append(
                {
                    "event": test_event,
                    "model": model_name,
                    "stage": "fit",
                    "seconds": tuning["fit_seconds"],
                    "model_bytes": model_path.stat().st_size,
                    "process_peak_working_set_mb": process.memory_info().peak_wset / (1024.0 * 1024.0),
                }
            )
        tuning_rows.extend(fold_tuning)
        arrays = arrays_by_event[test_event]
        predictions, predict_times = predict_models_full(arrays, static, models, args.predict_chunk)
        spatial_clim, temporal_clim = climatology_predictions(flood, outer_train, arrays["surface"])
        predictions = {
            "surface_original": arrays["surface_original"],
            "surface_matched": arrays["surface"],
            "spatial_climatology": spatial_clim,
            "spatiotemporal_climatology": temporal_clim,
            **predictions,
        }
        for model_name, seconds in predict_times.items():
            runtime_rows.append({"event": test_event, "model": model_name, "stage": "full_domain_inference", "seconds": seconds, "model_bytes": 0, "process_peak_working_set_mb": process.memory_info().peak_wset / (1024.0 * 1024.0)})
        for model_name, pred in predictions.items():
            event_rows.append(metric_row(test_event, model_name, pred, arrays["coupled"], arrays["surface"], static["active_mask"]))
            event_rows.append(metric_row(test_event, model_name, pred, arrays["mike"], arrays["surface"], static["active_mask"], reference="mike_external"))
            if model_name in {"surface_matched", "base", "all_static"}:
                conditional.extend(conditional_rows(test_event, model_name, pred, arrays, static, proximity_m))
        event_rows.append(metric_row(test_event, "itzi_swmm_coupled", arrays["coupled"], arrays["mike"], arrays["surface"], static["active_mask"], reference="mike_external"))

        pred_dir = out / "predictions" / test_event
        pred_dir.mkdir(parents=True, exist_ok=True)
        np.save(pred_dir / "h_all_static.npy", predictions["all_static"])
        np.save(pred_dir / "residual_all_static_mm.npy", (predictions["all_static"] - arrays["surface"]) * 1000.0)
        np.save(pred_dir / "h_spatial_climatology.npy", spatial_clim)
        np.save(pred_dir / "h_spatiotemporal_climatology.npy", temporal_clim)

        sample = samples[test_event]
        all_model = models["all_static"]
        all_features = GROUPS["all_static"]
        aligned_pred = predict_sample(all_model, all_features, sample["x"])
        aligned_mae = float(np.mean(np.abs(aligned_pred - sample["y"])))
        alignment_rows.append({"event": test_event, "control": "aligned", "replicate": 0, "mae_mm": aligned_mae, "delta_vs_aligned_mm": 0.0})
        for distance in [1, 2, 4, 8]:
            for direction, (dr, dc) in {"north": (-distance, 0), "south": (distance, 0), "west": (0, -distance), "east": (0, distance)}.items():
                variant = shift_static(static, dr, dc)
                x_variant = transformed_sample_matrix(arrays, variant, sample)
                pred = predict_sample(all_model, all_features, x_variant)
                mae = float(np.mean(np.abs(pred - sample["y"])))
                alignment_rows.append({"event": test_event, "control": f"shift_{distance}_{direction}", "replicate": 0, "mae_mm": mae, "delta_vs_aligned_mm": mae - aligned_mae})
        for repeat in range(args.block_null_repeats):
            variant = block_shuffle_static(static, args.block_size, args.random_state + 100000 * (fold + 1) + repeat)
            x_variant = transformed_sample_matrix(arrays, variant, sample)
            pred = predict_sample(all_model, all_features, x_variant)
            mae = float(np.mean(np.abs(pred - sample["y"])))
            alignment_rows.append({"event": test_event, "control": "block_shuffle", "replicate": repeat + 1, "mae_mm": mae, "delta_vs_aligned_mm": mae - aligned_mae})

        take_rng = np.random.default_rng(args.random_state + fold)
        take = take_rng.choice(sample["x"].shape[0], min(args.importance_samples, sample["x"].shape[0]), replace=False)
        indexes = {name: index for index, name in enumerate(SUPERSET)}
        cols = [indexes[name] for name in all_features]
        importance = permutation_importance(
            all_model,
            sample["x"][take][:, cols],
            sample["y"][take],
            scoring="neg_mean_absolute_error",
            n_repeats=args.importance_repeats,
            random_state=args.random_state + fold,
        )
        for feature, mean, std in zip(all_features, importance.importances_mean, importance.importances_std):
            importance_rows.append({"event": test_event, "feature": feature, "importance_mae_mm": float(mean), "importance_std_mm": float(std), "n_samples": int(take.size)})
        write_csv(out / "metrics" / "event_metrics_partial.csv", event_rows)

    summary_coupled = summarize(event_rows, "coupled_label")
    summary_mike = summarize(event_rows, "mike_external")
    correlation_rows, correlation_matrix = residual_correlations(flood, events, static["active_mask"])
    write_csv(out / "metrics" / "event_metrics.csv", event_rows)
    write_csv(out / "metrics" / "summary_coupled.csv", summary_coupled)
    write_csv(out / "metrics" / "summary_mike_external.csv", summary_mike)
    write_csv(out / "metrics" / "conditional_metrics.csv", conditional)
    write_csv(out / "metrics" / "inner_validation.csv", [
        {
            "outer_test_event": row["outer_test_event"],
            "model": row["model"],
            "inner_validation_event": row["inner_validation_event"],
            "sampling": row["sampling"],
            "sample_weighting": row["sample_weighting"],
            "selected_max_iter": row["selected_max_iter"],
            "refit_rows": row["refit_rows"],
            "fit_seconds": row["fit_seconds"],
            "candidate_scores_json": json.dumps(row["scores"]),
        }
        for row in tuning_rows
    ])
    write_csv(out / "metrics" / "spatial_alignment_controls.csv", alignment_rows)
    write_csv(out / "metrics" / "permutation_importance.csv", importance_rows)
    write_csv(out / "metrics" / "runtime_model_size.csv", runtime_rows)
    write_csv(out / "metrics" / "residual_map_correlations.csv", correlation_rows)
    np.save(out / "metrics" / "residual_map_correlation_matrix.npy", correlation_matrix)
    plot_summary(out, summary_coupled, event_rows)

    manifest = []
    for path in sorted(out.rglob("*")):
        if path.is_file():
            manifest.append({"path": str(path.relative_to(out)), "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    metadata = {
        "dataset": args.dataset_name,
        "events": events,
        "target": "signed residual h_C_itzi_swmm - h_B_surface_matched",
        "upstream": "h_B_surface_matched; same roughness as the coupled run",
        "external_reference": "MIKE; descriptive only and never used for training or tuning",
        "validation": "leave-one-event-out with one event-level inner validation event",
        "feature_groups": GROUPS,
        "excluded_inputs": ["row coordinate", "column coordinate", "distance_to_outfall", "outfall mask", "SWMM target-side dynamic states"],
        "climatology_baselines": ["training-event time-independent spatial residual mean", "training-event time-dependent spatial residual mean"],
        "sampling_sensitivity": {
            "all_static_unweighted": "same stratified rows as all_static with unit sample weights",
            "all_static_uniform": "uniform active-cell rows with unit sample weights",
        },
        "spatial_controls": {"cardinal_shifts_cells": [1, 2, 4, 8], "block_shuffle_repeats_per_fold": args.block_null_repeats, "block_size_cells": args.block_size},
        "arguments": vars(args),
        "runtime_seconds": time.perf_counter() - started,
        "peak_working_set_mb": process.memory_info().peak_wset / (1024.0 * 1024.0),
        "hardware": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "physical_cpu_cores": psutil.cpu_count(logical=False),
            "logical_cpu_cores": psutil.cpu_count(logical=True),
            "total_memory_gb": psutil.virtual_memory().total / (1024.0 ** 3),
        },
        "pid": os.getpid(),
        "manifest": manifest,
    }
    (out / "experiment_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"summary_coupled": summary_coupled, "summary_mike": summary_mike, "runtime_seconds": metadata["runtime_seconds"]}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
