#!/usr/bin/env python3
"""Extended lightweight validation for DrainLite paper gaps.

This script addresses three manuscript gaps without retraining LarNO:

1. Paper-level simple baselines:
   constant reduction rate, distance-decay empirical baseline, ridge,
   random forest, extra trees, and histogram gradient boosting.
2. Peak-enhanced residual learning:
   a high-water / high-reduction weighted HGB variant.
3. Spatial generalization:
   four held-out spatial tiles plus pipe-density stratified evaluation.

Outputs are written to:
    extended_study/output/drainlite_extended_validation/
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "extended_study"))
import train_drainlite_residual as dl  # noqa: E402


OUT = ROOT / "extended_study" / "output" / "drainlite_extended_validation"
MODEL_DIR = OUT / "models"
METRIC_DIR = OUT / "metrics"
FIG_DIR = OUT / "figures"
CELL_AREA_M2 = 20.0 * 20.0
FEATURES = dl.FEATURE_GROUPS["all_static"]


@dataclass
class FittedModel:
    name: str
    kind: str
    payload: Any
    features: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pixels-per-step", type=int, default=300)
    parser.add_argument("--spatial-pixels-per-step", type=int, default=220)
    parser.add_argument("--max-iter", type=int, default=220)
    parser.add_argument("--spatial-max-iter", type=int, default=160)
    parser.add_argument("--rf-trees", type=int, default=50)
    parser.add_argument("--random-state", type=int, default=666)
    parser.add_argument("--predict-chunk", type=int, default=80000)
    parser.add_argument("--eval-pixels-per-step", type=int, default=2200)
    parser.add_argument("--skip-dense", action="store_true")
    parser.add_argument("--skip-spatial", action="store_true")
    return parser.parse_args()


def ensure_dirs() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    METRIC_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def csi(pred: np.ndarray, ref: np.ndarray, threshold: float, mask: np.ndarray | None = None) -> float:
    if mask is not None:
        pred = pred[:, mask]
        ref = ref[:, mask]
    p = pred >= threshold
    r = ref >= threshold
    tp = float(np.logical_and(p, r).sum())
    fp = float(np.logical_and(p, ~r).sum())
    fn = float(np.logical_and(~p, r).sum())
    denom = tp + fp + fn
    return 1.0 if denom == 0 else tp / denom


def subset_metrics(
    event: str,
    model: str,
    pred: np.ndarray,
    target: np.ndarray,
    surface: np.ndarray,
    mask: np.ndarray | None,
    subset: str,
) -> dict[str, Any]:
    if mask is None:
        pred_v = pred
        target_v = target
        surface_v = surface
    else:
        pred_v = pred[:, mask]
        target_v = target[:, mask]
        surface_v = surface[:, mask]
    diff = pred_v - target_v
    target_peak = np.max(target, axis=0)
    pred_peak = np.max(pred, axis=0)
    if mask is not None:
        target_peak_v = target_peak[mask]
        pred_peak_v = pred_peak[mask]
    else:
        target_peak_v = target_peak.ravel()
        pred_peak_v = pred_peak.ravel()
    wet_peak = target_peak_v[target_peak_v > 0]
    top_thr = float(np.nanpercentile(wet_peak, 99.0)) if wet_peak.size else np.inf
    top_mask = target_peak_v >= top_thr

    true_red = np.maximum(surface_v - target_v, 0.0)
    pred_red = np.maximum(surface_v - pred_v, 0.0)
    cell_count = int(target_v.size)
    return {
        "event": event,
        "model": model,
        "subset": subset,
        "cell_time_count": cell_count,
        "mae_mm": float(np.mean(np.abs(diff)) * 1000.0),
        "rmse_mm": float(np.sqrt(np.mean(diff * diff)) * 1000.0),
        "csi_0p03": csi(pred, target, 0.03, mask),
        "csi_0p15": csi(pred, target, 0.15, mask),
        "peak_pred_m": float(np.nanmax(pred_peak_v)) if pred_peak_v.size else math.nan,
        "peak_target_m": float(np.nanmax(target_peak_v)) if target_peak_v.size else math.nan,
        "peak_error_mm": float((np.nanmax(pred_peak_v) - np.nanmax(target_peak_v)) * 1000.0) if pred_peak_v.size else math.nan,
        "top1_peak_mae_mm": float(np.mean(np.abs(pred_peak_v[top_mask] - target_peak_v[top_mask])) * 1000.0) if np.any(top_mask) else math.nan,
        "deep_0p15_mae_mm": float(np.mean(np.abs(pred_peak_v[target_peak_v >= 0.15] - target_peak_v[target_peak_v >= 0.15])) * 1000.0) if np.any(target_peak_v >= 0.15) else math.nan,
        "reduction_mae_mm": float(np.mean(np.abs(pred_red - true_red)) * 1000.0),
        "true_reduction_volume_m3": float(np.sum(true_red) * CELL_AREA_M2),
        "pred_reduction_volume_m3": float(np.sum(pred_red) * CELL_AREA_M2),
    }


def build_samples(
    events: list[str],
    static: dict[str, np.ndarray],
    swmm: dict[str, dict[str, float]],
    pixels_per_step: int,
    random_state: int,
    spatial_mask: np.ndarray | None = None,
    peak_focus: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_state)
    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    w_parts: list[np.ndarray] = []
    for event in events:
        arrays = dl.load_event(event)
        for t in range(arrays["surface"].shape[0]):
            surface_t = arrays["surface"][t]
            reduction_t = np.maximum(arrays["surface"][t] - arrays["sink"][t], 0.0)
            rows, cols = dl.choose_pixels(rng, static, surface_t, reduction_t, pixels_per_step)
            if spatial_mask is not None:
                keep = spatial_mask[rows, cols]
                attempts = 0
                while keep.sum() < max(20, pixels_per_step // 3) and attempts < 5:
                    more_r, more_c = dl.choose_pixels(rng, static, surface_t, reduction_t, pixels_per_step)
                    rows = np.concatenate([rows[keep], more_r])
                    cols = np.concatenate([cols[keep], more_c])
                    keep = spatial_mask[rows, cols]
                    attempts += 1
                rows = rows[keep][:pixels_per_step]
                cols = cols[keep][:pixels_per_step]
                if rows.size == 0:
                    continue
            x = dl.feature_matrix(event, arrays, static, swmm, t, rows, cols, FEATURES)
            y = reduction_t[rows, cols] * 1000.0
            weights = np.ones_like(y, dtype=np.float32)
            weights += (y > 0.5).astype(np.float32) * 4.0
            weights += (surface_t[rows, cols] > 0.03).astype(np.float32) * 2.0
            if peak_focus:
                weights += (surface_t[rows, cols] > 0.10).astype(np.float32) * 5.0
                weights += (surface_t[rows, cols] > 0.20).astype(np.float32) * 10.0
                weights += (y > 10.0).astype(np.float32) * 5.0
            x_parts.append(x)
            y_parts.append(y.astype(np.float32))
            w_parts.append(weights)
        print(f"samples {event}: {sum(len(p) for p in y_parts):,} rows")
    return np.vstack(x_parts), np.concatenate(y_parts), np.concatenate(w_parts)


def fit_constant_rate(x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> FittedModel:
    col = FEATURES.index("surface_depth_mm")
    surface = x[:, col]
    valid = surface > 1.0
    ratio = np.average(np.clip(y[valid] / surface[valid], 0.0, 1.0), weights=weights[valid]) if np.any(valid) else 0.0
    return FittedModel("constant_rate", "constant_rate", {"ratio": float(ratio)}, FEATURES)


def fit_distance_empirical(x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> FittedModel:
    s_col = FEATURES.index("surface_depth_mm")
    d_col = FEATURES.index("distance_to_outfall")
    surface = x[:, s_col]
    distance = x[:, d_col]
    valid = surface > 1.0
    edges = np.array([0.0, 40.0, 80.0, 160.0, 320.0, 640.0, 1280.0, np.inf], dtype=np.float32)
    ratios = []
    global_ratio = np.average(np.clip(y[valid] / surface[valid], 0.0, 1.0), weights=weights[valid]) if np.any(valid) else 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = valid & (distance >= lo) & (distance < hi)
        if np.any(mask):
            ratios.append(float(np.average(np.clip(y[mask] / surface[mask], 0.0, 1.0), weights=weights[mask])))
        else:
            ratios.append(float(global_ratio))
    return FittedModel("distance_decay_empirical", "distance_empirical", {"edges": edges, "ratios": np.array(ratios, dtype=np.float32)}, FEATURES)


def fit_models(x: np.ndarray, y: np.ndarray, weights: np.ndarray, args: argparse.Namespace) -> list[FittedModel]:
    models: list[FittedModel] = [fit_constant_rate(x, y, weights), fit_distance_empirical(x, y, weights)]
    ridge = make_pipeline(StandardScaler(), Ridge(alpha=5.0, random_state=args.random_state))
    ridge.fit(x, y, ridge__sample_weight=weights)
    models.append(FittedModel("ridge_linear", "sklearn", ridge, FEATURES))

    hgb = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_iter=args.max_iter,
        max_leaf_nodes=31,
        l2_regularization=0.01,
        early_stopping=True,
        validation_fraction=0.15,
        random_state=args.random_state,
    )
    hgb.fit(x, y, sample_weight=weights)
    models.append(FittedModel("hist_gbdt_all_static_refit", "sklearn", hgb, FEATURES))

    peak_weights = weights.copy()
    surface = x[:, FEATURES.index("surface_depth_mm")]
    peak_weights += (surface > 100.0).astype(np.float32) * 6.0
    peak_weights += (surface > 200.0).astype(np.float32) * 12.0
    peak_weights += (y > 10.0).astype(np.float32) * 6.0
    peak_hgb = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.04,
        max_iter=max(args.max_iter, 260),
        max_leaf_nodes=31,
        l2_regularization=0.02,
        early_stopping=True,
        validation_fraction=0.15,
        random_state=args.random_state + 17,
    )
    peak_hgb.fit(x, y, sample_weight=peak_weights)
    models.append(FittedModel("hist_gbdt_peak_weighted", "sklearn", peak_hgb, FEATURES))

    rf = RandomForestRegressor(
        n_estimators=args.rf_trees,
        max_depth=18,
        min_samples_leaf=4,
        max_features=0.75,
        n_jobs=-1,
        random_state=args.random_state,
    )
    rf.fit(x, y, sample_weight=weights)
    models.append(FittedModel("random_forest", "sklearn", rf, FEATURES))

    extra = ExtraTreesRegressor(
        n_estimators=args.rf_trees,
        max_depth=20,
        min_samples_leaf=3,
        max_features=0.85,
        n_jobs=-1,
        random_state=args.random_state,
    )
    extra.fit(x, y, sample_weight=weights)
    models.append(FittedModel("extra_trees", "sklearn", extra, FEATURES))

    for fitted in models:
        joblib.dump(fitted, MODEL_DIR / f"{fitted.name}.joblib")
    return models


def predict_reduction_points(
    fitted: FittedModel,
    event: str,
    arrays: dict[str, np.ndarray],
    static: dict[str, np.ndarray],
    swmm: dict[str, dict[str, float]],
    t: int,
    rows: np.ndarray,
    cols: np.ndarray,
) -> np.ndarray:
    x = dl.feature_matrix(event, arrays, static, swmm, t, rows, cols, fitted.features)
    surface_col = fitted.features.index("surface_depth_mm")
    if fitted.kind == "constant_rate":
        pred = x[:, surface_col] * float(fitted.payload["ratio"])
    elif fitted.kind == "distance_empirical":
        distance_col = fitted.features.index("distance_to_outfall")
        edges = fitted.payload["edges"]
        ratios = fitted.payload["ratios"]
        bins = np.searchsorted(edges[1:], x[:, distance_col], side="right")
        bins = np.clip(bins, 0, len(ratios) - 1)
        pred = x[:, surface_col] * ratios[bins]
    else:
        pred = fitted.payload.predict(x).astype(np.float32)
    max_reduction = arrays["surface"][t, rows, cols] * 1000.0
    return np.clip(pred, 0.0, max_reduction).astype(np.float32)


def predict_reduction_for_event(
    fitted: FittedModel,
    event: str,
    arrays: dict[str, np.ndarray],
    static: dict[str, np.ndarray],
    swmm: dict[str, dict[str, float]],
    chunk_size: int,
    eval_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    t_count, h, w = arrays["surface"].shape
    if eval_mask is None:
        eval_mask = static["active_mask"]
    active_rows, active_cols = np.where(eval_mask)
    reduction_mm = np.zeros((t_count, h, w), dtype=np.float32)
    surface_col = FEATURES.index("surface_depth_mm")
    distance_col = FEATURES.index("distance_to_outfall")
    for t in range(t_count):
        pred_t = np.zeros((h, w), dtype=np.float32)
        for start in range(0, active_rows.size, chunk_size):
            rows = active_rows[start:start + chunk_size]
            cols = active_cols[start:start + chunk_size]
            x = dl.feature_matrix(event, arrays, static, swmm, t, rows, cols, fitted.features)
            if fitted.kind == "constant_rate":
                pred = x[:, surface_col] * float(fitted.payload["ratio"])
            elif fitted.kind == "distance_empirical":
                edges = fitted.payload["edges"]
                ratios = fitted.payload["ratios"]
                bins = np.searchsorted(edges[1:], x[:, distance_col], side="right")
                bins = np.clip(bins, 0, len(ratios) - 1)
                pred = x[:, surface_col] * ratios[bins]
            else:
                pred = fitted.payload.predict(x).astype(np.float32)
            max_reduction = arrays["surface"][t, rows, cols] * 1000.0
            pred = np.clip(pred, 0.0, max_reduction)
            pred_t[rows, cols] = pred.astype(np.float32)
        reduction_mm[t] = pred_t
    h_pred = np.maximum(arrays["surface"] - reduction_mm / 1000.0, 0.0).astype(np.float32)
    return h_pred, reduction_mm


def choose_eval_pixels(
    rng: np.random.Generator,
    static: dict[str, np.ndarray],
    surface_t: np.ndarray,
    sink_peak: np.ndarray,
    reduction_t: np.ndarray,
    n: int,
) -> tuple[np.ndarray, np.ndarray]:
    active = static["active_mask"]
    wet = active & (surface_t > 0.001)
    positive = active & (reduction_t > 1e-5)
    deep_peak = active & (sink_peak >= np.nanpercentile(sink_peak[active], 99.0))
    near_pipe = active & ((static["pipe_density_7x7"] > 0) | (static["inlet_density_7x7"] > 0))
    chosen: list[np.ndarray] = []
    for mask, quota in [
        (deep_peak, int(n * 0.25)),
        (positive, int(n * 0.30)),
        (wet, int(n * 0.20)),
        (near_pipe, int(n * 0.15)),
        (active, n),
    ]:
        flat = np.flatnonzero(mask)
        if flat.size == 0 or quota <= 0:
            continue
        chosen.append(rng.choice(flat, size=min(quota, flat.size), replace=False))
    idx = np.unique(np.concatenate(chosen)) if chosen else np.flatnonzero(active)
    if idx.size > n:
        idx = rng.choice(idx, size=n, replace=False)
    return np.unravel_index(idx, active.shape)


def sampled_event_metrics(event: str, model: str, sample_rows: list[dict[str, float | bool]]) -> dict[str, Any]:
    pred = np.array([r["pred_m"] for r in sample_rows], dtype=np.float32)
    target = np.array([r["target_m"] for r in sample_rows], dtype=np.float32)
    surface = np.array([r["surface_m"] for r in sample_rows], dtype=np.float32)
    peak_flag = np.array([bool(r["deep_peak"]) for r in sample_rows], dtype=bool)
    diff = pred - target
    true_red = np.maximum(surface - target, 0.0)
    pred_red = np.maximum(surface - pred, 0.0)
    csi03_den = max(int(np.sum((pred >= 0.03) | (target >= 0.03))), 1)
    csi15_den = max(int(np.sum((pred >= 0.15) | (target >= 0.15))), 1)
    return {
        "event": event,
        "model": model,
        "subset": "stratified_sample",
        "sample_count": int(pred.size),
        "mae_mm": float(np.mean(np.abs(diff)) * 1000.0),
        "rmse_mm": float(np.sqrt(np.mean(diff * diff)) * 1000.0),
        "csi_0p03": float(np.sum((pred >= 0.03) & (target >= 0.03)) / csi03_den),
        "csi_0p15": float(np.sum((pred >= 0.15) & (target >= 0.15)) / csi15_den),
        "sample_peak_pred_m": float(np.max(pred)),
        "sample_peak_target_m": float(np.max(target)),
        "sample_peak_error_mm": float((np.max(pred) - np.max(target)) * 1000.0),
        "deep_peak_sample_mae_mm": float(np.mean(np.abs(diff[peak_flag])) * 1000.0) if np.any(peak_flag) else math.nan,
        "deep_0p15_sample_mae_mm": float(np.mean(np.abs(diff[target >= 0.15])) * 1000.0) if np.any(target >= 0.15) else math.nan,
        "reduction_mae_mm": float(np.mean(np.abs(pred_red - true_red)) * 1000.0),
    }


def evaluate_models_sampled(
    models: list[FittedModel],
    events: list[str],
    static: dict[str, np.ndarray],
    swmm: dict[str, dict[str, float]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(args.random_state + 909)
    for event in events:
        arrays = dl.load_event(event)
        sink_peak = np.max(arrays["sink"], axis=0)
        active_peak = sink_peak[static["active_mask"]]
        deep_thr = float(np.nanpercentile(active_peak, 99.0))
        samples_by_model: dict[str, list[dict[str, float | bool]]] = {"surface_only": []}
        for fitted in models:
            samples_by_model[fitted.name] = []
        for t in range(arrays["surface"].shape[0]):
            reduction_t = np.maximum(arrays["surface"][t] - arrays["sink"][t], 0.0)
            pr, pc = choose_eval_pixels(rng, static, arrays["surface"][t], sink_peak, reduction_t, args.eval_pixels_per_step)
            target_v = arrays["sink"][t, pr, pc]
            surface_v = arrays["surface"][t, pr, pc]
            deep_flag = sink_peak[pr, pc] >= deep_thr
            for i in range(pr.size):
                samples_by_model["surface_only"].append({
                    "pred_m": float(surface_v[i]),
                    "target_m": float(target_v[i]),
                    "surface_m": float(surface_v[i]),
                    "deep_peak": bool(deep_flag[i]),
                })
            for fitted in models:
                red_mm = predict_reduction_points(fitted, event, arrays, static, swmm, t, pr, pc)
                pred_v = np.maximum(surface_v - red_mm / 1000.0, 0.0)
                for i in range(pr.size):
                    samples_by_model[fitted.name].append({
                        "pred_m": float(pred_v[i]),
                        "target_m": float(target_v[i]),
                        "surface_m": float(surface_v[i]),
                        "deep_peak": bool(deep_flag[i]),
                    })
        for model_name, sample_rows in samples_by_model.items():
            rows.append(sampled_event_metrics(event, model_name, sample_rows))
        print(f"sample-evaluated {event}")
    write_csv(METRIC_DIR / "paper_level_baseline_event_metrics.csv", rows)
    summary = aggregate(rows, group_cols=["model", "subset"])
    write_csv(METRIC_DIR / "paper_level_baseline_summary.csv", summary)
    make_sample_model_comparison_figure(summary)
    return rows


def evaluate_dense_peak_models(
    models: list[FittedModel],
    events: list[str],
    static: dict[str, np.ndarray],
    swmm: dict[str, dict[str, float]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    wanted = {"hist_gbdt_all_static_refit", "hist_gbdt_peak_weighted", "distance_decay_empirical"}
    selected = [m for m in models if m.name in wanted]
    rows: list[dict[str, Any]] = []
    pred_root = OUT / "predictions"
    for event in events:
        arrays = dl.load_event(event)
        event_dir = pred_root / event
        event_dir.mkdir(parents=True, exist_ok=True)
        rows.append(subset_metrics(event, "surface_only", arrays["surface"], arrays["sink"], arrays["surface"], None, "full_domain"))
        for fitted in selected:
            print(f"dense peak predict {event} {fitted.name}")
            pred, red = predict_reduction_for_event(fitted, event, arrays, static, swmm, args.predict_chunk)
            np.save(event_dir / f"h_{fitted.name}.npy", pred)
            np.save(event_dir / f"reduction_{fitted.name}_mm.npy", red)
            rows.append(subset_metrics(event, fitted.name, pred, arrays["sink"], arrays["surface"], None, "full_domain"))
    write_csv(METRIC_DIR / "dense_peak_model_event_metrics.csv", rows)
    summary = aggregate(rows, group_cols=["model", "subset"])
    write_csv(METRIC_DIR / "dense_peak_model_summary.csv", summary)
    make_model_comparison_figure(summary, "dense_peak_model_summary.png", "Dense full-domain peak-enhancement check")
    return rows


def aggregate(rows: list[dict[str, Any]], group_cols: list[str]) -> list[dict[str, Any]]:
    df = pd.DataFrame(rows)
    metric_cols = [c for c in df.columns if c not in {"event", "model", "subset"}]
    out = []
    for keys, g in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: key for col, key in zip(group_cols, keys)}
        row["n_events"] = int(g["event"].nunique()) if "event" in g.columns else int(len(g))
        for col in metric_cols:
            if pd.api.types.is_numeric_dtype(g[col]):
                row[col] = float(g[col].mean())
        out.append(row)
    return sorted(out, key=lambda r: (str(r.get("subset", "")), float(r.get("mae_mm", 1e9))))


def make_model_comparison_figure(summary: list[dict[str, Any]], filename: str, title: str) -> None:
    df = pd.DataFrame(summary)
    if "subset" in df.columns:
        df = df[df["subset"] == "full_domain"].copy()
    order = df.sort_values("mae_mm")["model"].tolist()
    df = df.set_index("model").loc[order].reset_index()
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.8))
    axes[0].bar(df["model"], df["mae_mm"], color="#4477aa")
    axes[0].set_title("MAE to ITZI + sink")
    axes[0].set_ylabel("mm")
    axes[1].bar(df["model"], df["top1_peak_mae_mm"], color="#cc6677")
    axes[1].set_title("Top 1% peak MAE")
    axes[1].set_ylabel("mm")
    axes[2].bar(df["model"], df["peak_error_mm"], color="#ddcc77")
    axes[2].axhline(0, color="black", lw=0.8)
    axes[2].set_title("Peak error")
    axes[2].set_ylabel("mm")
    axes[3].bar(df["model"], df["csi_0p15"], color="#228833")
    axes[3].set_ylim(0, 1)
    axes[3].set_title("CSI at 0.15 m")
    for ax in axes:
        ax.tick_params(axis="x", rotation=55, labelsize=8)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(FIG_DIR / filename, dpi=220)
    plt.close(fig)


def make_sample_model_comparison_figure(summary: list[dict[str, Any]]) -> None:
    df = pd.DataFrame(summary)
    df = df[df["subset"] == "stratified_sample"].copy().sort_values("mae_mm")
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.8))
    axes[0].bar(df["model"], df["mae_mm"], color="#4477aa")
    axes[0].set_title("Sample MAE")
    axes[0].set_ylabel("mm")
    axes[1].bar(df["model"], df["deep_peak_sample_mae_mm"], color="#cc6677")
    axes[1].set_title("Deep-peak sample MAE")
    axes[1].set_ylabel("mm")
    axes[2].bar(df["model"], df["sample_peak_error_mm"], color="#ddcc77")
    axes[2].axhline(0, color="black", lw=0.8)
    axes[2].set_title("Sample peak error")
    axes[2].set_ylabel("mm")
    axes[3].bar(df["model"], df["csi_0p15"], color="#228833")
    axes[3].set_ylim(0, 1)
    axes[3].set_title("Sample CSI at 0.15 m")
    for ax in axes:
        ax.tick_params(axis="x", rotation=55, labelsize=8)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Paper-level baseline comparison on stratified evaluation samples", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(FIG_DIR / "paper_level_baseline_summary.png", dpi=220)
    plt.close(fig)


def tile_masks(shape: tuple[int, int]) -> dict[str, np.ndarray]:
    h, w = shape
    r_mid, c_mid = h // 2, w // 2
    masks = {
        "north_west": np.zeros(shape, dtype=bool),
        "north_east": np.zeros(shape, dtype=bool),
        "south_west": np.zeros(shape, dtype=bool),
        "south_east": np.zeros(shape, dtype=bool),
    }
    masks["north_west"][:r_mid, :c_mid] = True
    masks["north_east"][:r_mid, c_mid:] = True
    masks["south_west"][r_mid:, :c_mid] = True
    masks["south_east"][r_mid:, c_mid:] = True
    return masks


def run_spatial_generalization(
    train_events: list[str],
    test_events: list[str],
    static: dict[str, np.ndarray],
    swmm: dict[str, dict[str, float]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    masks = tile_masks(static["active_mask"].shape)
    active = static["active_mask"]
    for fold, holdout in masks.items():
        print(f"spatial fold {fold}")
        train_mask = active & ~holdout
        eval_mask = active & holdout
        x, y, weights = build_samples(
            train_events,
            static,
            swmm,
            args.spatial_pixels_per_step,
            args.random_state + abs(hash(fold)) % 10000,
            spatial_mask=train_mask,
            peak_focus=False,
        )
        model = HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=0.05,
            max_iter=args.spatial_max_iter,
            max_leaf_nodes=31,
            l2_regularization=0.02,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=args.random_state + 31,
        )
        model.fit(x, y, sample_weight=weights)
        fitted = FittedModel(f"spatial_holdout_{fold}", "sklearn", model, FEATURES)
        joblib.dump(fitted, MODEL_DIR / f"spatial_holdout_{fold}.joblib")
        tile_pipe_density = float(np.mean(static["pipe_density_7x7"][eval_mask]))
        tile_inlet_density = float(np.mean(static["inlet_density_7x7"][eval_mask]))
        for event in test_events:
            arrays = dl.load_event(event)
            pred, _red = predict_reduction_for_event(fitted, event, arrays, static, swmm, args.predict_chunk, eval_mask=eval_mask)
            row = subset_metrics(event, fitted.name, pred, arrays["sink"], arrays["surface"], eval_mask, fold)
            surface_row = subset_metrics(event, "surface_only", arrays["surface"], arrays["sink"], arrays["surface"], eval_mask, fold)
            row["surface_only_mae_mm"] = surface_row["mae_mm"]
            row["mae_improvement_vs_surface_pct"] = (
                100.0 * (surface_row["mae_mm"] - row["mae_mm"]) / surface_row["mae_mm"]
                if surface_row["mae_mm"] > 0 else math.nan
            )
            row["tile_pipe_density_7x7_mean"] = tile_pipe_density
            row["tile_inlet_density_7x7_mean"] = tile_inlet_density
            rows.append(row)
    write_csv(METRIC_DIR / "spatial_holdout_event_metrics.csv", rows)
    summary = aggregate(rows, group_cols=["subset", "model"])
    write_csv(METRIC_DIR / "spatial_holdout_summary.csv", summary)
    make_spatial_figure(static, rows, summary)
    return rows


def evaluate_density_strata(
    model: FittedModel,
    test_events: list[str],
    static: dict[str, np.ndarray],
    swmm: dict[str, dict[str, float]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    active = static["active_mask"]
    dens = static["pipe_density_7x7"]
    active_vals = dens[active]
    q25, q75 = np.quantile(active_vals, [0.25, 0.75])
    masks = {
        "low_pipe_density": active & (dens <= q25),
        "middle_pipe_density": active & (dens > q25) & (dens < q75),
        "high_pipe_density": active & (dens >= q75),
    }
    rows = []
    for event in test_events:
        arrays = dl.load_event(event)
        pred, _ = predict_reduction_for_event(model, event, arrays, static, swmm, args.predict_chunk)
        for name, mask in masks.items():
            row = subset_metrics(event, model.name, pred, arrays["sink"], arrays["surface"], mask, name)
            row["pipe_density_7x7_mean"] = float(np.mean(dens[mask]))
            row["inlet_density_7x7_mean"] = float(np.mean(static["inlet_density_7x7"][mask]))
            rows.append(row)
    write_csv(METRIC_DIR / "pipe_density_strata_event_metrics.csv", rows)
    summary = aggregate(rows, group_cols=["subset", "model"])
    write_csv(METRIC_DIR / "pipe_density_strata_summary.csv", summary)
    make_density_figure(summary)
    return rows


def make_spatial_figure(static: dict[str, np.ndarray], rows: list[dict[str, Any]], summary: list[dict[str, Any]]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))
    dem = np.ma.masked_where(~static["active_mask"], static["dem_m"])
    axes[0].imshow(dem, cmap="terrain")
    colors = {
        "north_west": "#1b9e77",
        "north_east": "#d95f02",
        "south_west": "#7570b3",
        "south_east": "#e7298a",
    }
    for name, mask in tile_masks(static["active_mask"].shape).items():
        yy, xx = np.where(mask)
        axes[0].contour(mask.astype(float), levels=[0.5], colors=[colors[name]], linewidths=1.2)
        axes[0].text(xx.mean(), yy.mean(), name.replace("_", "\n"), ha="center", va="center", fontsize=8, color="black")
    axes[0].set_title("Held-out spatial windows")
    axes[0].axis("off")

    df = pd.DataFrame(summary).sort_values("subset")
    axes[1].bar(df["subset"], df["mae_mm"], color=[colors.get(x, "#4477aa") for x in df["subset"]])
    axes[1].set_title("Held-out tile MAE")
    axes[1].set_ylabel("mm")
    axes[1].tick_params(axis="x", rotation=35, labelsize=8)
    axes[1].grid(axis="y", alpha=0.25)

    axes[2].bar(df["subset"], df["mae_improvement_vs_surface_pct"], color=[colors.get(x, "#4477aa") for x in df["subset"]])
    axes[2].axhline(0, color="black", lw=0.8)
    axes[2].set_title("Improvement vs surface-only")
    axes[2].set_ylabel("%")
    axes[2].tick_params(axis="x", rotation=35, labelsize=8)
    axes[2].grid(axis="y", alpha=0.25)

    fig.suptitle("Spatial-window generalization of DrainLite", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(FIG_DIR / "spatial_holdout_generalization.png", dpi=220)
    plt.close(fig)


def make_density_figure(summary: list[dict[str, Any]]) -> None:
    df = pd.DataFrame(summary).sort_values("subset")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    axes[0].bar(df["subset"], df["mae_mm"], color="#4477aa")
    axes[0].set_ylabel("MAE (mm)")
    axes[0].set_title("Error by pipe-density stratum")
    axes[1].bar(df["subset"], df["top1_peak_mae_mm"], color="#cc6677")
    axes[1].set_ylabel("mm")
    axes[1].set_title("Top 1% peak MAE")
    axes[2].bar(df["subset"], df["reduction_mae_mm"], color="#228833")
    axes[2].set_ylabel("mm")
    axes[2].set_title("Reduction MAE")
    for ax in axes:
        ax.tick_params(axis="x", rotation=35, labelsize=8)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("DrainLite behavior under different pipe-network densities", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(FIG_DIR / "pipe_density_strata.png", dpi=220)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    ensure_dirs()
    train_events = dl.read_event_list("region1_drainage_train.txt")
    test_events = dl.read_event_list("region1_drainage_test.txt")
    static = dl.load_static()
    swmm = dl.read_swmm()

    x, y, weights = build_samples(
        train_events,
        static,
        swmm,
        args.pixels_per_step,
        args.random_state,
        peak_focus=False,
    )
    np.save(METRIC_DIR / "extended_training_y_reduction_mm.npy", y.astype(np.float32))
    models = fit_models(x, y, weights, args)
    rows = evaluate_models_sampled(models, test_events, static, swmm, args)
    if not args.skip_dense:
        evaluate_dense_peak_models(models, test_events, static, swmm, args)
    peak_model = next(m for m in models if m.name == "hist_gbdt_peak_weighted")
    evaluate_density_strata(peak_model, test_events, static, swmm, args)
    if not args.skip_spatial:
        run_spatial_generalization(train_events, test_events, static, swmm, args)

    metadata = {
        "train_events": train_events,
        "test_events": test_events,
        "features": FEATURES,
        "pixels_per_step": args.pixels_per_step,
        "eval_pixels_per_step": args.eval_pixels_per_step,
        "spatial_pixels_per_step": args.spatial_pixels_per_step,
        "models": [m.name for m in models],
        "purpose": "paper-level baselines, peak-enhanced DrainLite variant, and spatial/density generalization checks",
    }
    (OUT / "extended_validation_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote extended validation outputs to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
