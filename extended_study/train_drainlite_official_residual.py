#!/usr/bin/env python3
"""Train DrainLite-OfficialResidual on formal ITZI-SWMM coupled labels.

This branch uses the copied calibrated ITZI-SWMM outputs imported as
`h_itzi_swmm_official.npy`.  Unlike the conceptual sink DrainLite model, the
target here is signed:

    residual_mm = 1000 * (h_itzi_swmm_official - h_itzi_swmm_official_surface)

Negative residuals mean net drainage; positive residuals mean local surcharge or
backwater.  Therefore this model must not enforce h_pred <= h_surface.
"""

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
from sklearn.ensemble import HistGradientBoostingRegressor

from train_drainlite_residual import (
    CELL_AREA_M2,
    FEATURE_GROUPS,
    FLOOD_DIR,
    OUT as SINK_OUT,
    SUPERSET_FEATURES,
    choose_pixels,
    csi,
    feature_matrix,
    load_static,
    read_swmm,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "extended_study" / "output" / "drainlite_official_residual"
DEFAULT_TRAIN = ["event1", "event20", "event65", "event66", "event67"]
DEFAULT_TEST = ["event68", "event69", "event70"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-events", nargs="+", default=DEFAULT_TRAIN)
    parser.add_argument("--test-events", nargs="+", default=DEFAULT_TEST)
    parser.add_argument("--pixels-per-step", type=int, default=2500)
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-leaf-nodes", type=int, default=31)
    parser.add_argument("--l2-regularization", type=float, default=0.01)
    parser.add_argument("--random-state", type=int, default=667)
    parser.add_argument("--predict-chunk", type=int, default=60000)
    parser.add_argument(
        "--residual-deadband-mm",
        type=float,
        default=2.0,
        help="Set predicted signed residuals with absolute value below this threshold to zero.",
    )
    return parser.parse_args()


def load_event(event: str) -> dict[str, np.ndarray]:
    event_dir = FLOOD_DIR / event
    surface = np.load(event_dir / "h_itzi_swmm_official_surface.npy").astype(np.float32)
    official = np.load(event_dir / "h_itzi_swmm_official.npy").astype(np.float32)
    mike = np.load(event_dir / "h_mike_ref.npy").astype(np.float32)
    rainfall = np.load(event_dir / "rainfall.npy").astype(np.float32)
    return {
        "surface": surface,
        "official": official,
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
    changed = np.abs(residual_t) > 1e-5
    pseudo_target = np.abs(residual_t)
    rows, cols = choose_pixels(rng, static, surface_t, pseudo_target, n)
    if np.any(changed):
        changed_flat = np.flatnonzero(static["active_mask"] & changed)
        extra = rng.choice(changed_flat, size=min(max(n // 3, 1), changed_flat.size), replace=False)
        idx = np.unique(np.concatenate([np.ravel_multi_index((rows, cols), static["active_mask"].shape), extra]))
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
        print(f"training {model_name}: rows={x_all.shape[0]:,}, features={len(cols)}")
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
    residual_deadband_mm: float = 0.0,
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
        residual_mm[t] = np.clip(pred_t, -1000.0, 1000.0)
    h_pred = np.maximum(arrays["surface"] + residual_mm / 1000.0, 0.0).astype(np.float32)
    return h_pred, residual_mm


def event_metrics(event: str, model: str, pred: np.ndarray, target: np.ndarray, surface: np.ndarray) -> dict[str, object]:
    err = pred - target
    target_residual = target - surface
    pred_residual = pred - surface
    peak_pred = pred.max(axis=0)
    peak_target = target.max(axis=0)
    final_surcharge = np.maximum(pred[-1] - surface[-1], 0.0)
    final_reduction = np.maximum(surface[-1] - pred[-1], 0.0)
    return {
        "event": event,
        "model": model,
        "mae_m": float(np.mean(np.abs(err))),
        "rmse_m": float(np.sqrt(np.mean(err * err))),
        "csi_0p03": csi(pred, target, 0.03),
        "csi_0p15": csi(pred, target, 0.15),
        "peak_pred_m": float(np.max(pred)),
        "peak_target_m": float(np.max(target)),
        "peak_error_m": float(np.max(pred) - np.max(target)),
        "signed_residual_mae_m": float(np.mean(np.abs(pred_residual - target_residual))),
        "drainage_reduction_final_m3": float(np.sum(final_reduction) * CELL_AREA_M2),
        "local_surcharge_final_m3": float(np.sum(final_surcharge) * CELL_AREA_M2),
        "local_surcharge_cells": int(np.sum(final_surcharge > 1e-6)),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_peak_map(event: str, arrays: dict[str, np.ndarray], pred: np.ndarray, residual_mm: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mike_peak = arrays["mike"].max(axis=0)
    surface_peak = arrays["surface"].max(axis=0)
    official_peak = arrays["official"].max(axis=0)
    pred_peak = pred.max(axis=0)
    error_peak = pred_peak - official_peak
    residual_peak = residual_mm[np.argmax(np.abs(residual_mm), axis=0), np.indices(residual_mm.shape[1:])[0], np.indices(residual_mm.shape[1:])[1]] / 1000.0
    vmax = max(float(mike_peak.max()), float(surface_peak.max()), float(official_peak.max()), float(pred_peak.max()))
    lim = max(0.05, float(np.nanmax(np.abs(error_peak))), float(np.nanmax(np.abs(residual_peak))))
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    panels = [
        ("MIKE reference peak", mike_peak, "viridis", 0, vmax),
        ("ITZI surface-only peak", surface_peak, "viridis", 0, vmax),
        ("Official ITZI-SWMM peak", official_peak, "viridis", 0, vmax),
        ("DrainLite official residual peak", pred_peak, "viridis", 0, vmax),
        ("Prediction - official error", error_peak, "coolwarm", -lim, lim),
        ("Predicted signed residual", residual_peak, "coolwarm", -lim, lim),
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
                "mae_improvement_vs_surface_pct": 100.0 * (surface_mae - mae) / surface_mae if surface_mae > 0 else math.nan,
                "mae_improvement_vs_base_pct": 100.0 * (base_mae - mae) / base_mae if base_mae > 0 else math.nan,
            }
        )
    return sorted(out, key=lambda r: float(r["mae_mm"]))


def main() -> int:
    args = parse_args()
    (OUT / "metrics").mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    static = load_static()
    swmm = read_swmm()
    x_all, y, weights = build_training_samples(args.train_events, static, swmm, args.pixels_per_step, args.random_state)
    np.save(OUT / "metrics" / "training_target_signed_residual_mm_sample.npy", y)
    models = train_models(x_all, y, weights, args)
    rows: list[dict[str, object]] = []
    pred_root = OUT / "predictions"
    for event in args.test_events:
        arrays = load_event(event)
        event_dir = pred_root / event
        event_dir.mkdir(parents=True, exist_ok=True)
        rows.append(event_metrics(event, "surface_only", arrays["surface"], arrays["official"], arrays["surface"]))
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
            np.save(event_dir / f"h_drainlite_official_{model_name}.npy", pred)
            np.save(event_dir / f"signed_residual_pred_{model_name}_mm.npy", residual_mm)
            rows.append(event_metrics(event, model_name, pred, arrays["official"], arrays["surface"]))
            if model_name == "all_static":
                make_peak_map(event, arrays, pred, residual_mm, OUT / "figures" / f"{event}_official_peak_maps.png")
        print(f"predicted {event}")

    write_csv(OUT / "metrics" / "official_residual_event_metrics.csv", rows)
    write_csv(OUT / "metrics" / "official_residual_ablation_summary.csv", summarize(rows))
    metadata = {
        "model": "DrainLite-OfficialResidual",
        "target": "1000 * (h_itzi_swmm_official - h_itzi_swmm_official_surface), signed mm",
        "prediction": "h_pred = max(h_itzi_swmm_official_surface + signed_residual_mm / 1000, 0)",
        "train_events": args.train_events,
        "test_events": args.test_events,
        "feature_groups": FEATURE_GROUPS,
        "source_labels": "h_itzi_swmm_official.npy imported from copied calibrated ITZI-SWMM project",
        "supersedes": str(SINK_OUT / "drainlite_run_metadata.json"),
        "note": "This formal-label model allows local surcharge and is not constrained to monotone drainage reduction.",
        "residual_deadband_mm": args.residual_deadband_mm,
    }
    (OUT / "official_residual_run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote official residual outputs to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
