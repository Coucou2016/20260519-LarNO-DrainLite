#!/usr/bin/env python3
"""Leave-one-event-out validation for the formal ITZI-SWMM residual model."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from train_drainlite_official_residual import (
    OUT as OFFICIAL_OUT,
    build_training_samples,
    event_metrics,
    load_event,
    predict_event,
)
from train_drainlite_residual import FEATURE_GROUPS, SUPERSET_FEATURES, load_static, read_swmm


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "extended_study" / "output" / "official_paper_package" / "metrics"
EVENTS = ["event1", "event20", "event65", "event66", "event67", "event68", "event69", "event70"]
MODELS = ["base", "all_static"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pixels-per-step", type=int, default=1200)
    parser.add_argument("--max-iter", type=int, default=160)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-leaf-nodes", type=int, default=31)
    parser.add_argument("--l2-regularization", type=float, default=0.01)
    parser.add_argument("--random-state", type=int, default=668)
    parser.add_argument("--predict-chunk", type=int, default=60000)
    parser.add_argument("--residual-deadband-mm", type=float, default=2.0)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def train_one(
    model_name: str,
    x_all: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    args: argparse.Namespace,
) -> HistGradientBoostingRegressor:
    col_index = {name: i for i, name in enumerate(SUPERSET_FEATURES)}
    cols = [col_index[name] for name in FEATURE_GROUPS[model_name]]
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
    model.fit(x_all[:, cols], y, sample_weight=weights)
    return model


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_model: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_model.setdefault(str(row["model"]), []).append(row)
    surface_mae = np.mean([float(r["mae_m"]) for r in by_model["surface_only"]])
    base_mae = np.mean([float(r["mae_m"]) for r in by_model["base"]])
    out = []
    for model, model_rows in by_model.items():
        mae = np.array([float(r["mae_m"]) for r in model_rows])
        rmse = np.array([float(r["rmse_m"]) for r in model_rows])
        out.append(
            {
                "model": model,
                "n_events": len(model_rows),
                "mae_mean_mm": float(mae.mean() * 1000.0),
                "mae_std_mm": float(mae.std(ddof=0) * 1000.0),
                "rmse_mean_mm": float(rmse.mean() * 1000.0),
                "csi_0p03_mean": float(np.mean([float(r["csi_0p03"]) for r in model_rows])),
                "csi_0p15_mean": float(np.mean([float(r["csi_0p15"]) for r in model_rows])),
                "peak_error_mean_mm": float(np.mean([float(r["peak_error_m"]) for r in model_rows]) * 1000.0),
                "improvement_vs_surface_pct": float(100.0 * (surface_mae - mae.mean()) / surface_mae) if surface_mae else math.nan,
                "improvement_vs_base_pct": float(100.0 * (base_mae - mae.mean()) / base_mae) if base_mae else math.nan,
            }
        )
    return sorted(out, key=lambda r: float(r["mae_mean_mm"]))


def main() -> int:
    args = parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    static = load_static()
    swmm = read_swmm()
    rows: list[dict[str, object]] = []

    for holdout in EVENTS:
        train_events = [e for e in EVENTS if e != holdout]
        arrays = load_event(holdout)
        rows.append(
            {
                "fold": holdout,
                **event_metrics(holdout, "surface_only", arrays["surface"], arrays["official"], arrays["surface"]),
            }
        )
        x_all, y, weights = build_training_samples(
            train_events,
            static,
            swmm,
            pixels_per_step=args.pixels_per_step,
            random_state=args.random_state + EVENTS.index(holdout),
        )
        for model_name in MODELS:
            print(f"fold={holdout} model={model_name} train_rows={x_all.shape[0]:,}")
            model = train_one(model_name, x_all, y, weights, args)
            pred, _ = predict_event(
                holdout,
                model,
                FEATURE_GROUPS[model_name],
                arrays,
                static,
                swmm,
                args.predict_chunk,
                args.residual_deadband_mm,
            )
            rows.append(
                {
                    "fold": holdout,
                    **event_metrics(holdout, model_name, pred, arrays["official"], arrays["surface"]),
                }
            )

    write_csv(OUT / "official_residual_leave_one_event_out_metrics.csv", rows)
    summary = summarize(rows)
    write_csv(OUT / "official_residual_leave_one_event_out_summary.csv", summary)
    metadata = {
        "events": EVENTS,
        "models": ["surface_only"] + MODELS,
        "pixels_per_step": args.pixels_per_step,
        "max_iter": args.max_iter,
        "source_single_split": str(OFFICIAL_OUT),
        "residual_deadband_mm": args.residual_deadband_mm,
    }
    (OUT / "official_residual_leave_one_event_out_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote cross-validation metrics to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
