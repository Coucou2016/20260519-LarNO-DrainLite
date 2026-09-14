#!/usr/bin/env python3
"""Leave-one-event-out validation for DrainLite-ConnectedResidual."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from train_drainlite_connected_residual import (
    FEATURE_GROUPS,
    SUPERSET_FEATURES,
    build_training_samples,
    configure_paths,
    event_metrics,
    load_event,
    load_static,
    predict_event,
    read_connected_swmm,
    write_csv,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "extended_study" / "output" / "drainlite_connected_cross_validation"
EVENTS = ["event1", "event20", "event65", "event66", "event67", "event68", "event69", "event70"]
DEFAULT_MODELS = ["base", "all_static", "swmm_assisted"]
DEFAULT_DATASET_NAME = "region1_20m_connected_swmm_v1"
DEFAULT_OUTPUT_NAME = "drainlite_connected_cross_validation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--output-name", default=DEFAULT_OUTPUT_NAME)
    parser.add_argument("--connected-metrics", default=None)
    parser.add_argument("--events", nargs="+", default=EVENTS)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--pixels-per-step", type=int, default=700)
    parser.add_argument("--max-iter", type=int, default=220)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-leaf-nodes", type=int, default=31)
    parser.add_argument("--l2-regularization", type=float, default=0.01)
    parser.add_argument("--random-state", type=int, default=671)
    parser.add_argument("--predict-chunk", type=int, default=60000)
    parser.add_argument("--residual-deadband-mm", type=float, default=0.0)
    return parser.parse_args()


def train_selected_models(
    x_all: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    model_names: list[str],
    args: argparse.Namespace,
    fold_dir: Path,
) -> dict[str, HistGradientBoostingRegressor]:
    fold_dir.mkdir(parents=True, exist_ok=True)
    col_index = {name: i for i, name in enumerate(SUPERSET_FEATURES)}
    models: dict[str, HistGradientBoostingRegressor] = {}
    for model_name in model_names:
        features = FEATURE_GROUPS[model_name]
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
        print(f"  training {model_name}: rows={x_all.shape[0]:,}, features={len(cols)}", flush=True)
        model.fit(x_all[:, cols], y, sample_weight=weights)
        models[model_name] = model
        joblib.dump({"model": model, "features": features}, fold_dir / f"{model_name}.joblib")
    return models


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
                "abs_peak_error_mm": np.mean([float(r.get("abs_peak_error_m", abs(float(r["peak_error_m"])))) for r in model_rows]) * 1000.0,
                "peak_time_error_h": np.mean([float(r.get("peak_time_error_h", math.nan)) for r in model_rows]),
                "abs_peak_time_error_h": np.mean([float(r.get("abs_peak_time_error_h", math.nan)) for r in model_rows]),
                "mean_abs_peak_volume_error_m3": np.mean([abs(float(r.get("peak_volume_error_m3", math.nan))) for r in model_rows]),
                "mean_abs_final_volume_error_m3": np.mean([abs(float(r.get("final_volume_error_m3", math.nan))) for r in model_rows]),
                "final_reduction_capture_pct": np.mean([float(r.get("final_reduction_capture_pct", math.nan)) for r in model_rows]),
                "signed_residual_mae_mm": np.mean([float(r["signed_residual_mae_m"]) for r in model_rows]) * 1000.0,
                "mae_to_mike_mm": np.mean([float(r["mae_to_mike_m"]) for r in model_rows]) * 1000.0,
                "mae_improvement_vs_surface_pct": 100.0 * (surface_mae - mae) / surface_mae if surface_mae > 0 else math.nan,
                "mae_improvement_vs_base_pct": (
                    math.nan
                    if model == "surface_only"
                    else 100.0 * (base_mae - mae) / base_mae if surface_mae > 0 and base_mae > 0 else math.nan
                ),
            }
        )
    return sorted(out, key=lambda r: float(r["mae_mm"]))


def main() -> int:
    args = parse_args()
    global OUT
    OUT = ROOT / "extended_study" / "output" / args.output_name
    configure_paths(args.dataset_name, args.output_name.replace("_cross_validation", "_residual"), args.connected_metrics)
    for model_name in args.models:
        if model_name not in FEATURE_GROUPS:
            raise ValueError(f"Unknown model: {model_name}")

    for d in [OUT / "metrics", OUT / "models", OUT / "predictions"]:
        d.mkdir(parents=True, exist_ok=True)

    static = load_static()
    swmm = read_connected_swmm()
    rows: list[dict[str, object]] = []

    for fold_idx, test_event in enumerate(args.events, start=1):
        train_events = [event for event in args.events if event != test_event]
        print(f"\n=== fold {fold_idx}/{len(args.events)}: test={test_event}; train={train_events} ===", flush=True)
        x_all, y, weights = build_training_samples(
            train_events,
            static,
            swmm,
            pixels_per_step=args.pixels_per_step,
            random_state=args.random_state + fold_idx,
        )
        fold_model_dir = OUT / "models" / test_event
        models = train_selected_models(x_all, y, weights, args.models, args, fold_model_dir)

        arrays = load_event(test_event)
        rows.append(
            event_metrics(
                test_event,
                "surface_only",
                arrays["surface"],
                arrays["official"],
                arrays["surface"],
                arrays["mike"],
                static["active_mask"],
            )
        )
        pred_dir = OUT / "predictions" / test_event
        pred_dir.mkdir(parents=True, exist_ok=True)
        for model_name, model in models.items():
            pred, residual_mm = predict_event(
                test_event,
                model,
                FEATURE_GROUPS[model_name],
                arrays,
                static,
                swmm,
                args.predict_chunk,
                args.residual_deadband_mm,
            )
            np.save(pred_dir / f"h_drainlite_connected_cv_{model_name}.npy", pred)
            np.save(pred_dir / f"signed_residual_cv_{model_name}_mm.npy", residual_mm)
            rows.append(event_metrics(test_event, model_name, pred, arrays["official"], arrays["surface"], arrays["mike"], static["active_mask"]))
        write_csv(OUT / "metrics" / "connected_residual_cv_event_metrics_partial.csv", rows)

    summary = summarize(rows)
    write_csv(OUT / "metrics" / "connected_residual_cv_event_metrics.csv", rows)
    write_csv(OUT / "metrics" / "connected_residual_cv_summary.csv", summary)
    metadata = {
        "model": "LarNO-DrainLite-ConnectedResidual leave-one-event-out",
        "dataset": args.dataset_name,
        "events": args.events,
        "models": args.models,
        "pixels_per_step": args.pixels_per_step,
        "max_iter": args.max_iter,
        "learning_rate": args.learning_rate,
        "max_leaf_nodes": args.max_leaf_nodes,
        "l2_regularization": args.l2_regularization,
        "random_state": args.random_state,
        "connected_metrics": args.connected_metrics,
    }
    (OUT / "connected_residual_cv_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
