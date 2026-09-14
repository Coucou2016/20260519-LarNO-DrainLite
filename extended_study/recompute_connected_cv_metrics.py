#!/usr/bin/env python3
"""Recompute connected DrainLite CV metrics on valid DEM cells from saved predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_drainlite_connected_cross_validation import summarize
from train_drainlite_connected_residual import (
    configure_paths,
    event_metrics,
    load_event,
    load_static,
    write_csv,
)


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--output-name", required=True)
    parser.add_argument("--connected-metrics", default=None)
    parser.add_argument("--events", nargs="+", required=True)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["base", "mask_only", "hydraulic_only", "all_static", "swmm_assisted"],
    )
    parser.add_argument("--prediction-template", default="h_drainlite_connected_cv_{model}.npy")
    parser.add_argument("--metrics-kind", choices=["cv", "main"], default="cv")
    args = parser.parse_args()

    output = ROOT / "extended_study" / "output" / args.output_name
    configure_paths(args.dataset_name, args.output_name.replace("_cross_validation", "_residual"), args.connected_metrics)
    static = load_static()
    active = static["active_mask"]
    rows: list[dict[str, object]] = []
    for event in args.events:
        arrays = load_event(event)
        rows.append(
            event_metrics(
                event,
                "surface_only",
                arrays["surface"],
                arrays["official"],
                arrays["surface"],
                arrays["mike"],
                active,
            )
        )
        prediction_dir = output / "predictions" / event
        for model in args.models:
            prediction_path = prediction_dir / args.prediction_template.format(model=model)
            if not prediction_path.exists():
                raise FileNotFoundError(prediction_path)
            import numpy as np

            pred = np.load(prediction_path).astype(np.float32)
            rows.append(
                event_metrics(
                    event,
                    model,
                    pred,
                    arrays["official"],
                    arrays["surface"],
                    arrays["mike"],
                    active,
                )
            )

    summary = summarize(rows)
    metrics_dir = output / "metrics"
    if args.metrics_kind == "cv":
        event_name = "connected_residual_cv_event_metrics.csv"
        partial_name = "connected_residual_cv_event_metrics_partial.csv"
        summary_name = "connected_residual_cv_summary.csv"
    else:
        event_name = "connected_residual_event_metrics.csv"
        partial_name = "connected_residual_event_metrics_partial.csv"
        summary_name = "connected_residual_ablation_summary.csv"
    write_csv(metrics_dir / event_name, rows)
    write_csv(metrics_dir / partial_name, rows)
    write_csv(metrics_dir / summary_name, summary)
    audit = {
        "dataset": args.dataset_name,
        "events": args.events,
        "models": args.models,
        "metric_scope": "valid DEM cells only",
        "active_cells": int(active.sum()),
        "total_cells": int(active.size),
        "active_fraction": float(active.mean()),
        "saved_predictions_reused": True,
        "prediction_template": args.prediction_template,
        "metrics_kind": args.metrics_kind,
    }
    (output / "connected_residual_metric_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
