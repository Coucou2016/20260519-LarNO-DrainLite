#!/usr/bin/env python3
"""Evaluate leakage-controlled climatology-plus-dynamics DrainLite models.

Each outer fold holds out one complete rainfall event.  The spatiotemporal
drainage prior for that event is computed only from the remaining events.  For
model fitting, every training row receives a leave-one-event-out prior so that
its own coupled target never contributes to its predictor.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import time
from collections import OrderedDict
from pathlib import Path

import joblib
import numpy as np
import psutil
from sklearn.ensemble import HistGradientBoostingRegressor

import run_reviewer_v3_drainlite as dl


ROOT = Path(__file__).resolve().parents[1]
GROUPS = OrderedDict(
    [
        ("clim_dynamic", ["climatology_residual_mm"] + dl.BASE),
        ("clim_all_static", ["climatology_residual_mm"] + dl.SUPERSET),
    ]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", default="region1_20m_drainage_v3_full")
    parser.add_argument(
        "--output-name", default="reviewer_major_revision_v3/drainlite_hybrid_v3"
    )
    parser.add_argument("--events", nargs="+", default=None)
    parser.add_argument("--pixels-per-step", type=int, default=450)
    parser.add_argument("--max-iter-candidates", nargs="+", type=int, default=[80, 140, 220])
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-leaf-nodes", type=int, default=31)
    parser.add_argument("--l2-regularization", type=float, default=0.01)
    parser.add_argument("--predict-chunk", type=int, default=50000)
    parser.add_argument("--random-state", type=int, default=1907)
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
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def residual(arrays: dict[str, np.ndarray]) -> np.ndarray:
    return arrays["coupled"] - arrays["surface"]


def mean_residual(
    arrays_by_event: dict[str, dict[str, np.ndarray]], events: list[str]
) -> np.ndarray:
    if not events:
        raise ValueError("A drainage prior requires at least one source event")
    total = np.zeros_like(arrays_by_event[events[0]]["surface"], dtype=np.float64)
    for event in events:
        total += residual(arrays_by_event[event])
    return (total / len(events)).astype(np.float32)


def encoded_sample(
    sample: dict[str, np.ndarray], prior: np.ndarray, group: str
) -> tuple[np.ndarray, np.ndarray]:
    base_indexes = {name: index for index, name in enumerate(dl.SUPERSET)}
    base_names = GROUPS[group][1:]
    selected = sample["x"][:, [base_indexes[name] for name in base_names]]
    prior_values = prior[sample["t"], sample["row"], sample["col"]] * 1000.0
    x = np.column_stack([prior_values, selected]).astype(np.float32, copy=False)
    return x, sample["y"]


def fit(
    x: np.ndarray, y: np.ndarray, max_iter: int, args: argparse.Namespace
) -> HistGradientBoostingRegressor:
    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=args.learning_rate,
        max_iter=max_iter,
        max_leaf_nodes=args.max_leaf_nodes,
        l2_regularization=args.l2_regularization,
        early_stopping=False,
        random_state=args.random_state,
    )
    model.fit(x, y)
    return model


def tune_model(
    group: str,
    outer_train: list[str],
    validation_event: str,
    samples: dict[str, dict[str, np.ndarray]],
    arrays_by_event: dict[str, dict[str, np.ndarray]],
    args: argparse.Namespace,
) -> tuple[HistGradientBoostingRegressor, dict[str, object], np.ndarray]:
    inner_train = [event for event in outer_train if event != validation_event]
    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    for event in inner_train:
        source = [other for other in inner_train if other != event]
        prior = mean_residual(arrays_by_event, source)
        x, y = encoded_sample(samples[event], prior, group)
        x_parts.append(x)
        y_parts.append(y)
    x_train, y_train = np.vstack(x_parts), np.concatenate(y_parts)
    validation_prior = mean_residual(arrays_by_event, inner_train)
    x_valid, y_valid = encoded_sample(samples[validation_event], validation_prior, group)
    scores = []
    for max_iter in args.max_iter_candidates:
        candidate = fit(x_train, y_train, max_iter, args)
        prediction = candidate.predict(x_valid)
        scores.append(
            {
                "max_iter": int(max_iter),
                "validation_mae_mm": float(np.mean(np.abs(prediction - y_valid))),
            }
        )
    selected = int(min(scores, key=lambda row: (row["validation_mae_mm"], row["max_iter"]))["max_iter"])

    # Re-encode each of the seven outer-training events without using its own
    # target when constructing the prior, then refit at the selected complexity.
    x_parts.clear()
    y_parts.clear()
    for event in outer_train:
        prior = mean_residual(arrays_by_event, [other for other in outer_train if other != event])
        x, y = encoded_sample(samples[event], prior, group)
        x_parts.append(x)
        y_parts.append(y)
    x_refit, y_refit = np.vstack(x_parts), np.concatenate(y_parts)
    started = time.perf_counter()
    model = fit(x_refit, y_refit, selected, args)
    fit_seconds = time.perf_counter() - started
    test_prior = mean_residual(arrays_by_event, outer_train)
    tuning = {
        "model": group,
        "inner_training_events": inner_train,
        "inner_validation_event": validation_event,
        "scores": scores,
        "selected_max_iter": selected,
        "refit_rows": int(x_refit.shape[0]),
        "fit_seconds": fit_seconds,
        "prior_encoding": "leave-one-event-out for every fitted event",
    }
    return model, tuning, test_prior


def dense_prediction(
    model: HistGradientBoostingRegressor,
    group: str,
    arrays: dict[str, np.ndarray],
    static: dict[str, np.ndarray],
    prior: np.ndarray,
    chunk_size: int,
) -> tuple[np.ndarray, float]:
    active_rows, active_cols = np.where(static["active_mask"])
    prediction = np.zeros_like(arrays["surface"], dtype=np.float32)
    base_indexes = {name: index for index, name in enumerate(dl.SUPERSET)}
    base_names = GROUPS[group][1:]
    base_cols = [base_indexes[name] for name in base_names]
    elapsed = 0.0
    for t in range(72):
        for start in range(0, active_rows.size, chunk_size):
            rows = active_rows[start : start + chunk_size]
            cols = active_cols[start : start + chunk_size]
            base = dl.feature_matrix(arrays, static, t, rows, cols)
            x = np.column_stack([prior[t, rows, cols] * 1000.0, base[:, base_cols]])
            then = time.perf_counter()
            estimated = np.clip(model.predict(x), -1200.0, 1200.0)
            elapsed += time.perf_counter() - then
            prediction[t, rows, cols] = np.maximum(
                arrays["surface"][t, rows, cols] + estimated / 1000.0, 0.0
            )
    return prediction, elapsed


def summarize(rows: list[dict[str, object]], reference: str) -> list[dict[str, object]]:
    models = list(OrderedDict.fromkeys(str(row["model"]) for row in rows if row["reference"] == reference))
    result = []
    metrics = [
        "mae_mm", "rmse_mm", "csi_0p03", "csi_0p15", "peak_map_mae_mm",
        "global_peak_abs_error_mm", "top0p1_peak_map_mae_mm", "final_volume_abs_error_m3",
    ]
    for model in models:
        group = [row for row in rows if row["reference"] == reference and row["model"] == model]
        result.append(
            {
                "model": model,
                "reference": reference,
                "n_events": len(group),
                **{metric: float(np.mean([float(row[metric]) for row in group])) for metric in metrics},
            }
        )
    return sorted(result, key=lambda row: row["mae_mm"])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    process = psutil.Process(os.getpid())
    flood = ROOT / "LarNO-main" / "benchmark" / "urbanflood" / "flood" / args.dataset_name
    geo = ROOT / "LarNO-main" / "benchmark" / "urbanflood" / "geodata" / args.dataset_name
    out = ROOT / "extended_study" / "output" / args.output_name
    for directory in [out / "metrics", out / "models", out / "predictions"]:
        directory.mkdir(parents=True, exist_ok=True)
    audit = json.loads((flood / "dataset_audit.json").read_text(encoding="utf-8"))
    events = args.events or audit["accepted_events"]
    if len(events) < 5:
        raise RuntimeError("At least five events are required")
    static = dl.load_static(geo)
    arrays_by_event: dict[str, dict[str, np.ndarray]] = {}
    samples: dict[str, dict[str, np.ndarray]] = {}
    for index, event in enumerate(events):
        print(f"load and uniformly sample {event}", flush=True)
        arrays_by_event[event] = dl.load_event(flood, event)
        samples[event] = dl.event_samples(
            event,
            arrays_by_event[event],
            static,
            args.pixels_per_step,
            args.random_state + 1009 * (index + 1),
            sampling="uniform",
            weighted=False,
        )

    rows: list[dict[str, object]] = []
    tuning_rows: list[dict[str, object]] = []
    runtime_rows: list[dict[str, object]] = []
    for fold, test_event in enumerate(events):
        outer_train = [event for event in events if event != test_event]
        validation_event = events[(fold + 1) % len(events)]
        arrays = arrays_by_event[test_event]
        print(f"fold {fold + 1}/{len(events)} test={test_event} validation={validation_event}", flush=True)
        test_prior = mean_residual(arrays_by_event, outer_train)
        baseline = np.maximum(arrays["surface"] + test_prior, 0.0).astype(np.float32)
        predictions: dict[str, np.ndarray] = {"spatiotemporal_climatology": baseline}
        for group in GROUPS:
            model, tuning, group_prior = tune_model(
                group, outer_train, validation_event, samples, arrays_by_event, args
            )
            prediction, seconds = dense_prediction(
                model, group, arrays, static, group_prior, args.predict_chunk
            )
            predictions[group] = prediction
            model_path = out / "models" / f"fold_{test_event}_{group}.joblib"
            joblib.dump(model, model_path, compress=3)
            tuning_rows.append({"fold": fold, "test_event": test_event, **tuning})
            runtime_rows.append(
                {
                    "fold": fold,
                    "test_event": test_event,
                    "model": group,
                    "dense_inference_seconds": seconds,
                    "active_cell_steps": int(static["active_mask"].sum() * 72),
                }
            )
        pred_dir = out / "predictions" / test_event
        pred_dir.mkdir(parents=True, exist_ok=True)
        for name, prediction in predictions.items():
            np.save(pred_dir / f"h_{name}.npy", prediction)
            rows.append(
                dl.metric_row(
                    test_event, name, prediction, arrays["coupled"], arrays["surface"],
                    static["active_mask"], "coupled_label",
                )
            )
            rows.append(
                dl.metric_row(
                    test_event, name, prediction, arrays["mike"], arrays["surface"],
                    static["active_mask"], "mike_external",
                )
            )

    coupled_summary = summarize(rows, "coupled_label")
    mike_summary = summarize(rows, "mike_external")
    write_csv(out / "metrics" / "event_metrics.csv", rows)
    write_csv(out / "metrics" / "summary_coupled.csv", coupled_summary)
    write_csv(out / "metrics" / "summary_mike_external.csv", mike_summary)
    write_csv(out / "metrics" / "runtime.csv", runtime_rows)
    (out / "metrics" / "tuning.json").write_text(
        json.dumps(tuning_rows, indent=2), encoding="utf-8"
    )
    metadata = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "dataset": args.dataset_name,
        "events": events,
        "outer_validation": "leave one complete event out",
        "inner_validation": "one complete event selected without test-event access",
        "prior_leakage_control": (
            "Held-event prior uses only seven outer-training events; every fitted row uses a "
            "prior averaged from other training events and excludes its own target."
        ),
        "groups": GROUPS,
        "pixels_per_step": args.pixels_per_step,
        "total_seconds": time.perf_counter() - started,
        "peak_rss_mb": process.memory_info().rss / (1024**2),
        "platform": platform.platform(),
        "logical_cpu_count": os.cpu_count(),
    }
    (out / "experiment_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    files = [path for path in out.rglob("*") if path.is_file() and path.name != "sha256_manifest.csv"]
    write_csv(
        out / "sha256_manifest.csv",
        [{"path": str(path.relative_to(out)), "bytes": path.stat().st_size, "sha256": sha256(path)} for path in files],
    )
    print(json.dumps({"coupled": coupled_summary, "mike": mike_summary}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
