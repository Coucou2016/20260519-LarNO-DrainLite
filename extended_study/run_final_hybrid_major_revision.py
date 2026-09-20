#!/usr/bin/env python3
"""Run the reviewer-requested final-hybrid controls on the audited V3 dataset.

The script keeps the complete rainfall event as the outer validation unit.  It
uses a fixed model complexity, repeats the experiment over independent spatial
sampling seeds, and tests the final prior-plus-dynamics-plus-network model with
network subsets, spatial displacement, block shuffling, group permutation,
residual-clipping sensitivity, conditional metrics, and end-to-end correction
timing.  MIKE remains a descriptive external reference and is never used for
model fitting or selection.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import time
from collections import OrderedDict
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import psutil
from scipy.ndimage import distance_transform_edt
from sklearn.ensemble import HistGradientBoostingRegressor

import run_reviewer_v3_drainlite as dl


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = "region1_20m_drainage_v3_full"
DEFAULT_OUTPUT = "reviewer_major_revision_v4/final_hybrid_controls"
DEFAULT_SEEDS = [1907, 2718, 3141, 5772, 8119]
MODEL_GROUPS: OrderedDict[str, tuple[list[str], bool]] = OrderedDict(
    [
        ("dynamic_no_prior", (dl.BASE, False)),
        ("hybrid_dynamic", (dl.BASE, True)),
        ("hybrid_mask", (dl.BASE + dl.MASKS, True)),
        ("hybrid_hydraulic", (dl.BASE + dl.HYDRAULICS, True)),
        ("hybrid_all", (dl.SUPERSET, True)),
    ]
)
PRIMARY_MODELS = ["surface_matched", "prior_only", *MODEL_GROUPS]
REPEATED_SEED_MODELS = {"hybrid_dynamic", "hybrid_all"}
SHIFT_DIRECTIONS = OrderedDict(
    [("north", (-1, 0)), ("south", (1, 0)), ("west", (0, -1)), ("east", (0, 1))]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET)
    parser.add_argument("--output-name", default=DEFAULT_OUTPUT)
    parser.add_argument("--events", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--pixels-per-step", type=int, default=450)
    parser.add_argument("--max-iter", type=int, default=140)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-leaf-nodes", type=int, default=31)
    parser.add_argument("--l2-regularization", type=float, default=0.01)
    parser.add_argument("--predict-chunk", type=int, default=50000)
    parser.add_argument("--block-size", type=int, default=20)
    parser.add_argument("--block-repeats", type=int, default=10)
    parser.add_argument("--permutation-repeats", type=int, default=5)
    parser.add_argument("--canonical-seed", type=int, default=1907)
    parser.add_argument("--skip-controls", action="store_true")
    parser.add_argument(
        "--all-models-every-seed",
        action="store_true",
        help="Run every subgroup model for every seed instead of reserving subgroup ablations for the canonical seed.",
    )
    parser.add_argument("--resume", action="store_true")
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


def append_fold_payload(
    payload: dict[str, list[dict[str, object]]],
    event_rows: list[dict[str, object]],
    runtime_rows: list[dict[str, object]],
    conditional_rows: list[dict[str, object]],
    control_rows: list[dict[str, object]],
    clipping_rows: list[dict[str, object]],
    training_rows: list[dict[str, object]],
) -> None:
    event_rows.extend(payload["event_rows"])
    runtime_rows.extend(payload["runtime_rows"])
    conditional_rows.extend(payload["conditional_rows"])
    control_rows.extend(payload["control_rows"])
    clipping_rows.extend(payload["clipping_rows"])
    training_rows.extend(payload["training_rows"])


def save_fold_payload(path: Path, payload: dict[str, list[dict[str, object]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def mean_residual(
    arrays_by_event: dict[str, dict[str, np.ndarray]], events: Iterable[str]
) -> np.ndarray:
    selected = list(events)
    if not selected:
        raise ValueError("A leakage-controlled prior needs at least one source event")
    result = np.zeros_like(arrays_by_event[selected[0]]["surface"], dtype=np.float64)
    for event in selected:
        result += arrays_by_event[event]["coupled"] - arrays_by_event[event]["surface"]
    return (result / len(selected)).astype(np.float32)


def encode_sample(
    sample: dict[str, np.ndarray],
    prior: np.ndarray,
    features: list[str],
    use_prior: bool,
) -> tuple[np.ndarray, np.ndarray]:
    index = {name: position for position, name in enumerate(dl.SUPERSET)}
    selected = sample["x"][:, [index[name] for name in features]]
    prior_values = prior[sample["t"], sample["row"], sample["col"]] * 1000.0
    x = np.column_stack([prior_values, selected]) if use_prior else selected
    return x.astype(np.float32, copy=False), sample["y"]


def fit_model(
    x: np.ndarray, y: np.ndarray, seed: int, args: argparse.Namespace
) -> HistGradientBoostingRegressor:
    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=args.learning_rate,
        max_iter=args.max_iter,
        max_leaf_nodes=args.max_leaf_nodes,
        l2_regularization=args.l2_regularization,
        early_stopping=False,
        random_state=seed,
    )
    model.fit(x, y)
    return model


def train_fold_model(
    features: list[str],
    outer_train: list[str],
    samples: dict[str, dict[str, np.ndarray]],
    arrays_by_event: dict[str, dict[str, np.ndarray]],
    seed: int,
    args: argparse.Namespace,
    use_prior: bool,
) -> tuple[HistGradientBoostingRegressor, float, int]:
    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    for event in outer_train:
        own_event_excluded = [other for other in outer_train if other != event]
        prior = mean_residual(arrays_by_event, own_event_excluded)
        x, y = encode_sample(samples[event], prior, features, use_prior)
        x_parts.append(x)
        y_parts.append(y)
    x_train = np.vstack(x_parts)
    y_train = np.concatenate(y_parts)
    started = time.perf_counter()
    model = fit_model(x_train, y_train, seed, args)
    elapsed = time.perf_counter() - started
    return model, elapsed, int(x_train.shape[0])


def dense_prediction(
    model: HistGradientBoostingRegressor,
    features: list[str],
    arrays: dict[str, np.ndarray],
    static: dict[str, np.ndarray],
    prior: np.ndarray,
    chunk_size: int,
    clip_mm: float | None = None,
    use_prior: bool = True,
    neighbourhoods: dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Predict a full event and time feature construction separately."""
    started_total = time.perf_counter()
    neighbourhood_seconds = 0.0
    if neighbourhoods is None:
        started_neighbourhoods = time.perf_counter()
        neighbourhoods = dl.precompute_surface_neighbourhoods(arrays, static)
        neighbourhood_seconds = time.perf_counter() - started_neighbourhoods
    active_rows, active_cols = np.where(static["active_mask"])
    prediction = np.zeros_like(arrays["surface"], dtype=np.float32)
    index = {name: position for position, name in enumerate(dl.SUPERSET)}
    columns = [index[name] for name in features]
    feature_seconds = neighbourhood_seconds
    assembly_seconds = 0.0
    model_seconds = 0.0
    postprocess_seconds = 0.0
    raw_below_clip = 0
    raw_above_clip = 0
    depth_clamped = 0
    active_values = 0
    for t in range(arrays["surface"].shape[0]):
        for start in range(0, active_rows.size, chunk_size):
            rows = active_rows[start : start + chunk_size]
            cols = active_cols[start : start + chunk_size]
            then = time.perf_counter()
            base = dl.feature_matrix(
                arrays,
                static,
                t,
                rows,
                cols,
                neighbourhoods=neighbourhoods,
            )
            feature_seconds += time.perf_counter() - then
            then = time.perf_counter()
            x = (
                np.column_stack([prior[t, rows, cols] * 1000.0, base[:, columns]])
                if use_prior
                else base[:, columns]
            )
            assembly_seconds += time.perf_counter() - then
            then = time.perf_counter()
            raw = model.predict(x)
            model_seconds += time.perf_counter() - then
            then = time.perf_counter()
            if clip_mm is not None:
                raw_below_clip += int(np.count_nonzero(raw < -clip_mm))
                raw_above_clip += int(np.count_nonzero(raw > clip_mm))
                residual_mm = np.clip(raw, -clip_mm, clip_mm)
            else:
                residual_mm = raw
            unbounded_depth = arrays["surface"][t, rows, cols] + residual_mm / 1000.0
            depth_clamped += int(np.count_nonzero(unbounded_depth < 0.0))
            active_values += int(unbounded_depth.size)
            prediction[t, rows, cols] = np.maximum(unbounded_depth, 0.0)
            postprocess_seconds += time.perf_counter() - then
    total_seconds = time.perf_counter() - started_total
    return prediction, {
        "feature_seconds": feature_seconds,
        "neighbourhood_seconds": neighbourhood_seconds,
        "uses_cached_neighbourhoods": neighbourhood_seconds == 0.0,
        "assembly_seconds": assembly_seconds,
        "model_seconds": model_seconds,
        "postprocess_seconds": postprocess_seconds,
        "total_correction_seconds": total_seconds,
        "raw_below_clip_count": raw_below_clip,
        "raw_above_clip_count": raw_above_clip,
        "depth_clamped_count": depth_clamped,
        "active_value_count": active_values,
        "depth_clamped_pct": 100.0 * depth_clamped / max(active_values, 1),
    }


def drop_network_static(static: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    result = {
        "dem_m": static["dem_m"],
        "dem_slope": static["dem_slope"],
        "active_mask": static["active_mask"],
    }
    for name in dl.PRIMITIVE_DRAINAGE:
        result[name] = np.zeros_like(static[name])
    return dl.regenerate_descriptors(result)


def permuted_network_static(
    static: dict[str, np.ndarray], random_state: int
) -> dict[str, np.ndarray]:
    result = {
        "dem_m": static["dem_m"],
        "dem_slope": static["dem_slope"],
        "active_mask": static["active_mask"],
    }
    active_flat = np.flatnonzero(static["active_mask"])
    permutation = np.random.default_rng(random_state).permutation(active_flat.size)
    for name in dl.PRIMITIVE_DRAINAGE:
        source = static[name].ravel()
        target = np.zeros_like(source)
        target[active_flat] = source[active_flat[permutation]]
        result[name] = target.reshape(static[name].shape)
    return dl.regenerate_descriptors(result)


def metric_row(
    event: str,
    seed: int,
    model: str,
    pred: np.ndarray,
    reference: np.ndarray,
    surface: np.ndarray,
    active: np.ndarray,
    reference_name: str,
) -> dict[str, object]:
    row = dl.metric_row(event, model, pred, reference, surface, active, reference_name)
    p = pred[:, active]
    t = reference[:, active]
    p_volume = p.sum(axis=1, dtype=np.float64) * dl.CELL_AREA_M2
    t_volume = t.sum(axis=1, dtype=np.float64) * dl.CELL_AREA_M2
    p_area03 = (p >= 0.03).sum(axis=1) * dl.CELL_AREA_M2
    t_area03 = (t >= 0.03).sum(axis=1) * dl.CELL_AREA_M2
    p_area15 = (p >= 0.15).sum(axis=1) * dl.CELL_AREA_M2
    t_area15 = (t >= 0.15).sum(axis=1) * dl.CELL_AREA_M2
    row.update(
        {
            "seed": seed,
            "mean_abs_volume_error_m3": float(np.mean(np.abs(p_volume - t_volume))),
            "max_abs_volume_error_m3": float(np.max(np.abs(p_volume - t_volume))),
            "mean_abs_area_error_0p03_m2": float(np.mean(np.abs(p_area03 - t_area03))),
            "mean_abs_area_error_0p15_m2": float(np.mean(np.abs(p_area15 - t_area15))),
            "global_peak_time_error_min": float(
                abs(np.unravel_index(np.argmax(pred), pred.shape)[0]
                    - np.unravel_index(np.argmax(reference), reference.shape)[0])
                * 5.0
            ),
        }
    )
    return row


def csi_subset(pred: np.ndarray, target: np.ndarray, threshold: float) -> float:
    p = pred >= threshold
    t = target >= threshold
    tp = int(np.logical_and(p, t).sum())
    fp = int(np.logical_and(p, ~t).sum())
    fn = int(np.logical_and(~p, t).sum())
    return 1.0 if tp + fp + fn == 0 else tp / (tp + fp + fn)


def conditional_metrics(
    event: str,
    model: str,
    pred: np.ndarray,
    arrays: dict[str, np.ndarray],
    static: dict[str, np.ndarray],
) -> list[dict[str, object]]:
    target = arrays["coupled"]
    surface = arrays["surface"]
    residual = target - surface
    active = static["active_mask"]
    active3 = np.broadcast_to(active, target.shape)
    inlet_distance = distance_transform_edt(static["drain_inlet_mask"] <= 0) * dl.CELL_M
    pipe_distance = distance_transform_edt(static["pipe_mask"] <= 0) * dl.CELL_M
    conditions: OrderedDict[str, np.ndarray] = OrderedDict(
        [
            ("all_active", active3),
            ("target_wet_0p03m", active3 & (target >= 0.03)),
            ("target_wet_0p15m", active3 & (target >= 0.15)),
            ("wet_either_0p03m", active3 & ((target >= 0.03) | (surface >= 0.03))),
        ]
    )
    for threshold in [1, 5, 10]:
        metres = threshold / 1000.0
        conditions[f"effect_abs_over_{threshold}mm"] = active3 & (np.abs(residual) > metres)
        conditions[f"net_drainage_over_{threshold}mm"] = active3 & (residual < -metres)
        conditions[f"positive_residual_over_{threshold}mm"] = active3 & (residual > metres)
    for feature, distance in [("inlet", inlet_distance), ("pipe", pipe_distance)]:
        for low, high in [(0, 20), (20, 60), (60, 100), (100, math.inf)]:
            upper = "inf" if math.isinf(high) else str(high)
            mask2 = active & (distance >= low) & (distance < high)
            conditions[f"near_{feature}_{low}_{upper}m"] = np.broadcast_to(mask2, target.shape)
    rows = []
    for condition, mask in conditions.items():
        count = int(mask.sum())
        if count == 0:
            continue
        p = pred[mask]
        t = target[mask]
        error = p - t
        predicted_residual = p - surface[mask]
        target_residual = t - surface[mask]
        rows.append(
            {
                "event": event,
                "model": model,
                "condition": condition,
                "n_cell_times": count,
                "proportion_active_pct": 100.0 * count / int(active3.sum()),
                "mae_mm": float(np.mean(np.abs(error)) * 1000.0),
                "rmse_mm": float(np.sqrt(np.mean(error * error)) * 1000.0),
                "bias_mm": float(np.mean(error) * 1000.0),
                "residual_mae_mm": float(
                    np.mean(np.abs(predicted_residual - target_residual)) * 1000.0
                ),
                "csi_0p03": csi_subset(p, t, 0.03),
                "csi_0p15": csi_subset(p, t, 0.15),
            }
        )
    return rows


def summarize_event_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    metrics = [
        "mae_mm",
        "rmse_mm",
        "csi_0p03",
        "csi_0p15",
        "peak_map_mae_mm",
        "global_peak_bias_mm",
        "global_peak_abs_error_mm",
        "top0p1_peak_map_mae_mm",
        "final_volume_bias_m3",
        "final_volume_abs_error_m3",
        "mean_abs_volume_error_m3",
        "max_abs_volume_error_m3",
        "mean_abs_area_error_0p03_m2",
        "mean_abs_area_error_0p15_m2",
        "global_peak_time_error_min",
        "residual_mae_mm",
    ]
    groups: OrderedDict[tuple[object, str, str], list[dict[str, object]]] = OrderedDict()
    for row in rows:
        key = (row["seed"], str(row["reference"]), str(row["model"]))
        groups.setdefault(key, []).append(row)
    result = []
    for (seed, reference, model), group in groups.items():
        result.append(
            {
                "seed": seed,
                "reference": reference,
                "model": model,
                "n_events": len(group),
                **{
                    metric: float(np.mean([float(row[metric]) for row in group]))
                    for metric in metrics
                },
            }
        )
    return result


def summarize_seeds(seed_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    metrics = [key for key in seed_rows[0] if key not in {"seed", "reference", "model", "n_events"}]
    groups: OrderedDict[tuple[str, str], list[dict[str, object]]] = OrderedDict()
    for row in seed_rows:
        groups.setdefault((str(row["reference"]), str(row["model"])), []).append(row)
    result = []
    for (reference, model), group in groups.items():
        record: dict[str, object] = {
            "reference": reference,
            "model": model,
            "n_seeds": len(group),
            "n_events": group[0]["n_events"],
        }
        for metric in metrics:
            values = np.asarray([float(row[metric]) for row in group])
            record[f"{metric}_mean"] = float(values.mean())
            record[f"{metric}_sd"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        result.append(record)
    return result


def event_inventory(
    public_flood: Path, used_events: list[str], active: np.ndarray
) -> list[dict[str, object]]:
    rows = []
    event_paths = sorted(public_flood.glob("event*"), key=lambda path: int(path.name[5:]))
    for path in event_paths:
        try:
            rainfall = np.load(path / "rainfall.npy", mmap_mode="r")
            mike = np.load(path / "h.npy", mmap_mode="r")
            if rainfall.shape != (72, 400, 560) or mike.shape != rainfall.shape:
                raise ValueError(f"unexpected shapes {rainfall.shape}, {mike.shape}")
            rainfall_total = np.asarray(rainfall).sum(axis=0)
            mike_array = np.asarray(mike)
            row = {
                "event": path.name,
                "public_arrays_valid": True,
                "used_in_paired_v3": path.name in used_events,
                "selection_reason": (
                    "complete locally generated matched A/B/C calculation"
                    if path.name in used_events
                    else "public forcing/reference available; matched A/B/C calculation not generated"
                ),
                "mean_6h_rainfall_mm_active": float(rainfall_total[active].mean()),
                "max_local_6h_rainfall_mm_active": float(rainfall_total[active].max()),
                "max_5min_rainfall_mm_active": float(np.asarray(rainfall)[:, active].max()),
                "mike_mean_depth_mm_active": float(mike_array[:, active].mean() * 1000.0),
                "mike_global_peak_m_active": float(mike_array[:, active].max()),
            }
        except Exception as exc:
            row = {
                "event": path.name,
                "public_arrays_valid": False,
                "used_in_paired_v3": False,
                "selection_reason": f"unreadable public arrays: {type(exc).__name__}",
            }
        rows.append(row)
    return rows


def load_physics_runtime(root: Path) -> dict[str, dict[str, float]]:
    path = root / "extended_study" / "output" / "reviewer_major_revision_v3" / "formal_matched_full" / "physics_quality.csv"
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {
            row["event"]: {
                "surface_runtime_s": float(row["matched_control_runtime_s"]),
                "coupled_runtime_s": float(row["coupled_runtime_s"]),
            }
            for row in csv.DictReader(handle)
        }


def evaluate_prediction(
    rows: list[dict[str, object]],
    event: str,
    seed: int,
    model: str,
    prediction: np.ndarray,
    arrays: dict[str, np.ndarray],
    active: np.ndarray,
) -> None:
    rows.append(metric_row(event, seed, model, prediction, arrays["coupled"], arrays["surface"], active, "coupled_label"))
    rows.append(metric_row(event, seed, model, prediction, arrays["mike"], arrays["surface"], active, "mike_external"))


def main() -> int:
    args = parse_args()
    if args.canonical_seed not in args.seeds:
        raise ValueError("--canonical-seed must be included in --seeds")
    started = time.perf_counter()
    flood = ROOT / "LarNO-main" / "benchmark" / "urbanflood" / "flood" / args.dataset_name
    geo = ROOT / "LarNO-main" / "benchmark" / "urbanflood" / "geodata" / args.dataset_name
    out = ROOT / "extended_study" / "output" / args.output_name
    for directory in [out / "metrics", out / "models", out / "predictions"]:
        directory.mkdir(parents=True, exist_ok=True)
    audit = json.loads((flood / "dataset_audit.json").read_text(encoding="utf-8"))
    events = args.events or audit["accepted_events"]
    if len(events) < 5:
        raise ValueError("At least five complete events are required")
    static = dl.load_static(geo)
    active = static["active_mask"]
    arrays_by_event = {event: dl.load_event(flood, event) for event in events}
    physics_runtime = load_physics_runtime(ROOT)

    event_rows: list[dict[str, object]] = []
    runtime_rows: list[dict[str, object]] = []
    conditional_rows: list[dict[str, object]] = []
    control_rows: list[dict[str, object]] = []
    clipping_rows: list[dict[str, object]] = []
    training_rows: list[dict[str, object]] = []

    for seed_index, seed in enumerate(args.seeds):
        print(f"sampling seed {seed} ({seed_index + 1}/{len(args.seeds)})", flush=True)
        samples = {
            event: dl.event_samples(
                event,
                arrays_by_event[event],
                static,
                args.pixels_per_step,
                seed + 1009 * (index + 1),
                sampling="uniform",
                weighted=False,
            )
            for index, event in enumerate(events)
        }
        for fold, test_event in enumerate(events):
            print(f"  fold {fold + 1}/{len(events)}: {test_event}", flush=True)
            checkpoint = out / "checkpoints" / f"seed_{seed}_{test_event}.json"
            if args.resume and checkpoint.exists():
                payload = json.loads(checkpoint.read_text(encoding="utf-8"))
                append_fold_payload(
                    payload,
                    event_rows,
                    runtime_rows,
                    conditional_rows,
                    control_rows,
                    clipping_rows,
                    training_rows,
                )
                print("    restored completed fold from checkpoint", flush=True)
                continue
            fold_event_rows: list[dict[str, object]] = []
            fold_runtime_rows: list[dict[str, object]] = []
            fold_conditional_rows: list[dict[str, object]] = []
            fold_control_rows: list[dict[str, object]] = []
            fold_clipping_rows: list[dict[str, object]] = []
            fold_training_rows: list[dict[str, object]] = []
            outer_train = [event for event in events if event != test_event]
            test_prior = mean_residual(arrays_by_event, outer_train)
            arrays = arrays_by_event[test_event]
            neighbourhoods = dl.precompute_surface_neighbourhoods(arrays, static)
            prior_prediction = np.maximum(arrays["surface"] + test_prior, 0.0).astype(np.float32)
            evaluate_prediction(fold_event_rows, test_event, seed, "surface_matched", arrays["surface"], arrays, active)
            evaluate_prediction(fold_event_rows, test_event, seed, "prior_only", prior_prediction, arrays, active)
            if seed == args.canonical_seed:
                fold_conditional_rows.extend(conditional_metrics(test_event, "surface_matched", arrays["surface"], arrays, static))
                fold_conditional_rows.extend(conditional_metrics(test_event, "prior_only", prior_prediction, arrays, static))

            fitted: dict[str, HistGradientBoostingRegressor] = {}
            active_model_groups = MODEL_GROUPS if (
                seed == args.canonical_seed or args.all_models_every_seed
            ) else OrderedDict(
                (name, specification)
                for name, specification in MODEL_GROUPS.items()
                if name in REPEATED_SEED_MODELS
            )
            for offset, (model_name, (features, use_prior)) in enumerate(active_model_groups.items()):
                model, fit_seconds, fitted_rows = train_fold_model(
                    features,
                    outer_train,
                    samples,
                    arrays_by_event,
                    seed + offset,
                    args,
                    use_prior,
                )
                fitted[model_name] = model
                fold_training_rows.append(
                    {
                        "seed": seed,
                        "fold": fold,
                        "test_event": test_event,
                        "model": model_name,
                        "fit_seconds": fit_seconds,
                        "fitted_rows": fitted_rows,
                        "max_iter_fixed": args.max_iter,
                    }
                )
                prediction, timing = dense_prediction(
                    model,
                    features,
                    arrays,
                    static,
                    test_prior,
                    args.predict_chunk,
                    clip_mm=None,
                    use_prior=use_prior,
                    neighbourhoods=neighbourhoods,
                )
                evaluate_prediction(fold_event_rows, test_event, seed, model_name, prediction, arrays, active)
                run = physics_runtime[test_event]
                fold_runtime_rows.append(
                    {
                        "seed": seed,
                        "fold": fold,
                        "test_event": test_event,
                        "model": model_name,
                        **timing,
                        **run,
                        "surface_plus_correction_seconds": run["surface_runtime_s"] + float(timing["total_correction_seconds"]),
                        "speedup_vs_coupled": run["coupled_runtime_s"] / (run["surface_runtime_s"] + float(timing["total_correction_seconds"])),
                    }
                )
                if seed == args.canonical_seed:
                    fold_conditional_rows.extend(conditional_metrics(test_event, model_name, prediction, arrays, static))
                    model_dir = out / "models" / f"seed_{seed}"
                    model_dir.mkdir(parents=True, exist_ok=True)
                    joblib.dump(model, model_dir / f"fold_{test_event}_{model_name}.joblib", compress=3)
                    pred_dir = out / "predictions" / test_event
                    pred_dir.mkdir(parents=True, exist_ok=True)
                    np.save(pred_dir / f"h_{model_name}.npy", prediction)

            if seed != args.canonical_seed or args.skip_controls:
                payload = {
                    "event_rows": fold_event_rows,
                    "runtime_rows": fold_runtime_rows,
                    "conditional_rows": fold_conditional_rows,
                    "control_rows": fold_control_rows,
                    "clipping_rows": fold_clipping_rows,
                    "training_rows": fold_training_rows,
                }
                save_fold_payload(checkpoint, payload)
                append_fold_payload(
                    payload,
                    event_rows,
                    runtime_rows,
                    conditional_rows,
                    control_rows,
                    clipping_rows,
                    training_rows,
                )
                continue

            all_model = fitted["hybrid_all"]
            all_features, _ = MODEL_GROUPS["hybrid_all"]
            for clip_mm in [500.0, 1200.0]:
                prediction, diagnostics = dense_prediction(
                    all_model,
                    all_features,
                    arrays,
                    static,
                    test_prior,
                    args.predict_chunk,
                    clip_mm=clip_mm,
                    neighbourhoods=neighbourhoods,
                )
                row = metric_row(test_event, seed, f"hybrid_all_clip_{int(clip_mm)}mm", prediction, arrays["coupled"], arrays["surface"], active, "coupled_label")
                fold_clipping_rows.append({**row, "clip_mm": clip_mm, **diagnostics})
            primary_runtime = next(
                row for row in fold_runtime_rows
                if row["seed"] == seed and row["test_event"] == test_event and row["model"] == "hybrid_all"
            )
            primary_metric = next(
                row for row in fold_event_rows
                if row["seed"] == seed and row["event"] == test_event
                and row["model"] == "hybrid_all" and row["reference"] == "coupled_label"
            )
            fold_clipping_rows.append(
                {
                    **primary_metric,
                    "clip_mm": "none",
                    "feature_seconds": primary_runtime["feature_seconds"],
                    "assembly_seconds": primary_runtime["assembly_seconds"],
                    "model_seconds": primary_runtime["model_seconds"],
                    "postprocess_seconds": primary_runtime["postprocess_seconds"],
                    "total_correction_seconds": primary_runtime["total_correction_seconds"],
                    "raw_below_clip_count": 0,
                    "raw_above_clip_count": 0,
                    "depth_clamped_count": primary_runtime["depth_clamped_count"],
                    "active_value_count": primary_runtime["active_value_count"],
                    "depth_clamped_pct": primary_runtime["depth_clamped_pct"],
                }
            )

            controls: list[tuple[str, dict[str, np.ndarray], int]] = [
                ("zero_static_network_fields", drop_network_static(static), 0)
            ]
            for distance_m in [20, 40, 80, 160]:
                cells = distance_m // int(dl.CELL_M)
                for direction, (dr, dc) in SHIFT_DIRECTIONS.items():
                    controls.append(
                        (f"shift_{distance_m}m_{direction}", dl.shift_static(static, dr * cells, dc * cells), 0)
                    )
            for repeat in range(args.block_repeats):
                controls.append(
                    (
                        "block_shuffle",
                        dl.block_shuffle_static(static, args.block_size, seed + 3001 + repeat),
                        repeat + 1,
                    )
                )
            for repeat in range(args.permutation_repeats):
                controls.append(
                    (
                        "group_permutation",
                        permuted_network_static(static, seed + 5003 + repeat),
                        repeat + 1,
                    )
                )
            aligned_mae = float(primary_metric["mae_mm"])
            for control, transformed, replicate in controls:
                prediction, timing = dense_prediction(
                    all_model,
                    all_features,
                    arrays,
                    transformed,
                    test_prior,
                    args.predict_chunk,
                    clip_mm=None,
                    neighbourhoods=neighbourhoods,
                )
                row = metric_row(test_event, seed, control, prediction, arrays["coupled"], arrays["surface"], active, "coupled_label")
                fold_control_rows.append(
                    {
                        **row,
                        "control": control,
                        "replicate": replicate,
                        "aligned_hybrid_all_mae_mm": aligned_mae,
                        "delta_mae_vs_aligned_mm": float(row["mae_mm"]) - aligned_mae,
                        **timing,
                    }
                )

            payload = {
                "event_rows": fold_event_rows,
                "runtime_rows": fold_runtime_rows,
                "conditional_rows": fold_conditional_rows,
                "control_rows": fold_control_rows,
                "clipping_rows": fold_clipping_rows,
                "training_rows": fold_training_rows,
            }
            save_fold_payload(checkpoint, payload)
            append_fold_payload(
                payload,
                event_rows,
                runtime_rows,
                conditional_rows,
                control_rows,
                clipping_rows,
                training_rows,
            )

    seed_summary = summarize_event_rows(event_rows)
    model_summary = summarize_seeds(seed_summary)
    write_csv(out / "metrics" / "event_metrics_all_seeds.csv", event_rows)
    write_csv(out / "metrics" / "seed_macro_metrics.csv", seed_summary)
    write_csv(out / "metrics" / "model_summary_mean_sd.csv", model_summary)
    write_csv(out / "metrics" / "runtime_end_to_end.csv", runtime_rows)
    write_csv(out / "metrics" / "training_runtime.csv", training_rows)
    write_csv(out / "metrics" / "conditional_metrics_canonical_seed.csv", conditional_rows)
    write_csv(out / "metrics" / "final_hybrid_network_controls.csv", control_rows)
    write_csv(out / "metrics" / "clipping_sensitivity.csv", clipping_rows)
    write_csv(
        out / "metrics" / "event_selection_inventory.csv",
        event_inventory(
            ROOT / "LarNO-main" / "benchmark" / "urbanflood" / "flood" / "region1_20m",
            events,
            active,
        ),
    )
    metadata = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "dataset": args.dataset_name,
        "events": events,
        "seeds": args.seeds,
        "canonical_seed": args.canonical_seed,
        "target": "signed residual h_C_itzi_swmm - h_B_surface_matched",
        "outer_validation": "leave one complete rainfall event out",
        "complexity_selection": f"pre-specified fixed max_iter={args.max_iter}; no held-event or row-random early stopping",
        "prior_control": "held event excluded; each fitted event receives a prior excluding its own target",
        "network_controls": {
            "shift": "20, 40, 80 and 160 m in four cardinal directions at inference",
            "block_shuffle": f"{args.block_repeats} jointly permuted {args.block_size}x{args.block_size} cell network-block replicates",
            "group_permutation": f"{args.permutation_repeats} joint active-cell permutations of all primitive network fields",
            "zero_static_network_fields": (
                "all primitive and derived network fields set to zero at inference; the "
                "leakage-controlled fixed-network prior remains present"
            ),
        },
        "runtime_scope": "feature construction + input assembly + estimator prediction + physical non-negativity post-processing",
        "primary_residual_clipping": "none",
        "clipping_sensitivity_mm": [500, 1200],
        "surface_depth_neighbourhood": "mask-aware mean excluding inactive/building cells",
        "mike_role": "descriptive external reference; never used for fitting or model selection",
        "arguments": vars(args),
        "runtime_seconds": time.perf_counter() - started,
        "peak_rss_mb": psutil.Process(os.getpid()).memory_info().rss / (1024**2),
        "platform": platform.platform(),
        "python": platform.python_version(),
    }
    (out / "experiment_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(out), "models": len(model_summary), "runtime_seconds": metadata["runtime_seconds"]}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
