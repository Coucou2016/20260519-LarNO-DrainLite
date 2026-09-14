#!/usr/bin/env python3
"""Generate LarNO-DrainLite v2 quality, MIKE-reference, and manuscript assets."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from publication_plot_style import MODEL_COLORS, configure_publication_style


configure_publication_style()


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "LarNO-main" / "benchmark" / "urbanflood" / "flood" / "region1_20m_connected_swmm_v1"
GEO = ROOT / "LarNO-main" / "benchmark" / "urbanflood" / "geodata" / "region1_20m_connected_swmm_v1"
CONNECTED_METRICS = ROOT / "extended_study" / "output" / "connected_itzi_swmm" / "connected_itzi_swmm_metrics.csv"
CV_OUT = ROOT / "extended_study" / "output" / "drainlite_connected_cross_validation"
MAIN_OUT = ROOT / "extended_study" / "output" / "drainlite_connected_residual"
OUT = ROOT / "extended_study" / "output" / "larno_drainlite_v2_package"
METRICS = OUT / "metrics"
FIG = OUT / "figures"
EVENTS = ["event1", "event20", "event65", "event66", "event67", "event68", "event69", "event70"]
CELL_AREA_M2 = 20.0 * 20.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", default="region1_20m_connected_swmm_v2_inf1mmh")
    parser.add_argument("--connected-metrics", default=None)
    parser.add_argument("--main-output-name", default="drainlite_connected_residual_v2_inf1mmh")
    parser.add_argument("--cv-output-name", default="drainlite_connected_cross_validation_v2_inf1mmh")
    parser.add_argument("--package-output-name", default="larno_drainlite_v2_inf1mmh_package")
    parser.add_argument("--run-label", default="LarNO-DrainLite v2, effective loss 1 mm/h")
    parser.add_argument("--infiltration-mmh", type=float, default=1.0)
    return parser.parse_args()


def configure_paths(args: argparse.Namespace) -> None:
    global DATA, GEO, CONNECTED_METRICS, CV_OUT, MAIN_OUT, OUT, METRICS, FIG
    DATA = ROOT / "LarNO-main" / "benchmark" / "urbanflood" / "flood" / args.dataset_name
    GEO = ROOT / "LarNO-main" / "benchmark" / "urbanflood" / "geodata" / args.dataset_name
    if args.connected_metrics:
        CONNECTED_METRICS = Path(args.connected_metrics)
    else:
        rebuilt = DATA / "connected_swmm_metrics_rebuilt.csv"
        CONNECTED_METRICS = rebuilt if rebuilt.exists() else CONNECTED_METRICS
    CV_OUT = ROOT / "extended_study" / "output" / args.cv_output_name
    MAIN_OUT = ROOT / "extended_study" / "output" / args.main_output_name
    OUT = ROOT / "extended_study" / "output" / args.package_output_name
    METRICS = OUT / "metrics"
    FIG = OUT / "figures"


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def safe_float(value: object, default: float = math.nan) -> float:
    try:
        return float(value)
    except Exception:
        return default


def csi(pred: np.ndarray, target: np.ndarray, threshold: float) -> float:
    p = pred >= threshold
    t = target >= threshold
    tp = float(np.logical_and(p, t).sum())
    fp = float(np.logical_and(p, ~t).sum())
    fn = float(np.logical_and(~p, t).sum())
    den = tp + fp + fn
    return tp / den if den else math.nan


def load_cv_pred(event: str, model: str = "all_static") -> np.ndarray:
    return np.load(CV_OUT / "predictions" / event / f"h_drainlite_connected_cv_{model}.npy").astype(np.float32)


def quality_status(error_pct: float, report_exists: bool = True) -> str:
    if not report_exists or math.isnan(error_pct):
        return "excluded"
    err = abs(error_pct)
    if err <= 2.0:
        return "accepted"
    if err <= 8.0:
        return "warning"
    return "excluded"


def build_quality_tables() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows = []
    for row in read_csv(CONNECTED_METRICS):
        event = row["event"]
        continuity = safe_float(row.get("continuity_error_pct"))
        report_exists = True if "report_exists" not in row else str(row.get("report_exists", "")).lower() == "true"
        status = quality_status(continuity, report_exists)
        nonconverging = safe_float(row.get("nonconverging_steps_pct"))
        rows.append(
            {
                "event": event,
                "quality_status": status,
                "continuity_error_pct": continuity,
                "abs_continuity_error_pct": abs(continuity) if not math.isnan(continuity) else math.nan,
                "nonconverging_steps_pct": nonconverging,
                "external_inflow_m3": safe_float(row.get("external_inflow_m3")),
                "external_outflow_m3": safe_float(row.get("external_outflow_m3")),
                "final_stored_m3": safe_float(row.get("final_stored_m3")),
                "flooding_loss_m3": safe_float(row.get("flooding_loss_m3")),
                "peak_reduction_mm": safe_float(row.get("peak_reduction_mm")),
                "final_volume_reduction_m3": safe_float(row.get("final_volume_reduction_m3")),
                "use_as_training_label": status in ["accepted", "warning"],
                "must_disclose_stability_warning": status != "accepted" or (not math.isnan(nonconverging) and nonconverging > 1.0),
            }
        )
    counts: dict[str, int] = {"accepted": 0, "warning": 0, "excluded": 0}
    for row in rows:
        counts[str(row["quality_status"])] += 1
    summary = [
        {
            "quality_status": key,
            "n_events": value,
            "rule": {
                "accepted": "abs(continuity_error_pct) <= 2",
                "warning": "2 < abs(continuity_error_pct) <= 8",
                "excluded": "abs(continuity_error_pct) > 8 or missing/failed report",
            }[key],
        }
        for key, value in counts.items()
    ]
    return rows, summary


def model_vs_mike(
    event: str,
    model_name: str,
    pred: np.ndarray,
    mike: np.ndarray,
    active_mask: np.ndarray,
) -> dict[str, object]:
    pred_v = pred[:, active_mask]
    mike_v = mike[:, active_mask]
    err = pred_v - mike_v
    peak_pred = pred.max(axis=0)[active_mask]
    peak_mike = mike.max(axis=0)[active_mask]
    peak_time_pred = pred_v.max(axis=1)
    peak_time_mike = mike_v.max(axis=1)
    volume_pred = pred_v.sum(axis=1) * CELL_AREA_M2
    volume_mike = mike_v.sum(axis=1) * CELL_AREA_M2
    return {
        "event": event,
        "model": model_name,
        "mae_m": float(np.mean(np.abs(err))),
        "rmse_m": float(np.sqrt(np.mean(err * err))),
        "evaluation_active_cells": int(active_mask.sum()),
        "csi_0p03": csi(pred_v, mike_v, 0.03),
        "csi_0p15": csi(pred_v, mike_v, 0.15),
        "peak_depth_error_m": float(np.max(pred_v) - np.max(mike_v)),
        "abs_peak_depth_error_m": float(abs(np.max(pred_v) - np.max(mike_v))),
        "peak_map_mae_m": float(np.mean(np.abs(peak_pred - peak_mike))),
        "peak_time_error_step": int(np.argmax(peak_time_pred) - np.argmax(peak_time_mike)),
        "final_volume_error_m3": float(volume_pred[-1] - volume_mike[-1]),
        "peak_volume_error_m3": float(volume_pred[np.argmax(volume_mike)] - volume_mike[np.argmax(volume_mike)]),
    }


def build_mike_reference_tables() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows = []
    dem = np.load(GEO / "dem.npy").astype(np.float32)
    active_mask = np.isfinite(dem) & (dem < 49.9)
    for event in EVENTS:
        d = DATA / event
        mike = np.load(d / "h_mike_ref.npy").astype(np.float32)
        surface = np.load(d / "h_itzi_surface.npy").astype(np.float32)
        connected = np.load(d / "h_itzi_swmm_connected.npy").astype(np.float32)
        drainlite = load_cv_pred(event, "all_static")
        rows.append(model_vs_mike(event, "ITZI surface-only", surface, mike, active_mask))
        rows.append(model_vs_mike(event, "ITZI-SWMM connected", connected, mike, active_mask))
        rows.append(model_vs_mike(event, "DrainLite all_static CV", drainlite, mike, active_mask))

    by_model: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_model.setdefault(str(row["model"]), []).append(row)
    summary = []
    for model, model_rows in by_model.items():
        summary.append(
            {
                "model": model,
                "n_events": len(model_rows),
                "mae_mm": float(np.mean([safe_float(r["mae_m"]) for r in model_rows]) * 1000.0),
                "rmse_mm": float(np.mean([safe_float(r["rmse_m"]) for r in model_rows]) * 1000.0),
                "csi_0p03": float(np.mean([safe_float(r["csi_0p03"]) for r in model_rows])),
                "csi_0p15": float(np.mean([safe_float(r["csi_0p15"]) for r in model_rows])),
                "peak_depth_error_mm": float(np.mean([safe_float(r["peak_depth_error_m"]) for r in model_rows]) * 1000.0),
                "abs_peak_depth_error_mm": float(np.mean([safe_float(r["abs_peak_depth_error_m"]) for r in model_rows]) * 1000.0),
                "peak_map_mae_mm": float(np.mean([safe_float(r["peak_map_mae_m"]) for r in model_rows]) * 1000.0),
                "final_volume_error_m3": float(np.mean([safe_float(r["final_volume_error_m3"]) for r in model_rows])),
            }
        )
    return rows, sorted(summary, key=lambda r: safe_float(r["mae_mm"]))


def plot_quality(quality: list[dict[str, object]]) -> None:
    events = [str(r["event"]) for r in quality]
    continuity = [safe_float(r["continuity_error_pct"]) for r in quality]
    nonconv = [safe_float(r["nonconverging_steps_pct"]) for r in quality]
    colors = [
        {"accepted": "#009E73", "warning": "#E69F00", "excluded": "#D55E00"}[str(r["quality_status"])]
        for r in quality
    ]
    x = np.arange(len(events))
    fig, ax1 = plt.subplots(figsize=(7.1, 3.6))
    ax1.bar(x, continuity, color=colors, label="Continuity error")
    ax1.axhline(2.0, color="#009E73", linestyle="--", linewidth=1)
    ax1.axhline(-2.0, color="#009E73", linestyle="--", linewidth=1)
    ax1.axhline(8.0, color="#D55E00", linestyle=":", linewidth=1)
    ax1.axhline(-8.0, color="#D55E00", linestyle=":", linewidth=1)
    ax1.set_ylabel("SWMM continuity error (%)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(events, rotation=20)
    ax2 = ax1.twinx()
    ax2.plot(x, nonconv, color="#111111", marker="o", label="Nonconverging steps")
    ax2.set_ylabel("Nonconverging steps (%)")
    ax1.grid(axis="y", alpha=0.25)
    handles = [
        Patch(facecolor="#009E73", label="Accepted"),
        Patch(facecolor="#E69F00", label="Warning"),
        Patch(facecolor="#D55E00", label="Excluded"),
        Line2D([0], [0], color="#009E73", linestyle="--", label="+/-2% threshold"),
        Line2D([0], [0], color="#D55E00", linestyle=":", label="+/-8% threshold"),
        Line2D([0], [0], color="#111111", marker="o", label="Nonconverging steps"),
    ]
    fig.suptitle("ITZI-SWMM physical-label quality grading", y=0.985)
    fig.legend(
        handles=handles,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.90),
        fontsize=7.0,
        frameon=False,
        handlelength=1.8,
        columnspacing=1.2,
    )
    fig.subplots_adjust(left=0.09, right=0.90, bottom=0.20, top=0.68)
    fig.savefig(FIG / "v2_label_quality_grading.png")
    plt.close(fig)


def plot_mike_summary(summary: list[dict[str, object]]) -> None:
    order = ["ITZI surface-only", "ITZI-SWMM connected", "DrainLite all_static CV"]
    rows = sorted(summary, key=lambda r: order.index(str(r["model"])))
    labels = [str(r["model"]) for r in rows]
    mae = [safe_float(r["mae_mm"]) for r in rows]
    peak_mae = [safe_float(r["peak_map_mae_mm"]) for r in rows]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7.1, 3.4))
    ax.bar(x - 0.18, mae, width=0.36, label="Full time-series MAE", color="#0072B2")
    ax.bar(x + 0.18, peak_mae, width=0.36, label="Peak-map MAE", color="#E69F00")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=12, ha="right")
    ax.set_ylabel("Error to MIKE reference (mm)")
    ax.set_title("External MIKE-reference comparison")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "v2_mike_reference_summary.png")
    plt.close(fig)


def plot_mike_event_mae(rows: list[dict[str, object]]) -> None:
    models = ["ITZI surface-only", "ITZI-SWMM connected", "DrainLite all_static CV"]
    by = {(str(r["event"]), str(r["model"])): safe_float(r["mae_m"]) * 1000.0 for r in rows}
    x = np.arange(len(EVENTS))
    width = 0.24
    colors = MODEL_COLORS
    fig, ax = plt.subplots(figsize=(7.1, 3.5))
    for i, model in enumerate(models):
        ax.bar(x + (i - 1) * width, [by.get((event, model), math.nan) for event in EVENTS], width=width, label=model, color=colors[model])
    ax.set_xticks(x)
    ax.set_xticklabels(EVENTS, rotation=20)
    ax.set_ylabel("MAE to MIKE reference (mm)")
    ax.set_title("Event-wise MIKE-reference comparison")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "v2_mike_event_mae.png")
    plt.close(fig)


def write_reproducibility_manifest(run_label: str, args: argparse.Namespace) -> None:
    metadata = {}
    meta_path = DATA / "metadata.json"
    if meta_path.exists():
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    infiltration = args.infiltration_mmh
    if infiltration is None:
        infiltration = metadata.get("infiltration_mmh", 0.0)
    audit_name = "connected_swmm_dataset_v2_inf1mmh" if str(args.dataset_name).endswith("inf1mmh") else "connected_swmm_dataset_v2_infiltration"
    run_dir_name = f"connected_itzi_swmm_inf_{infiltration:g}mmh" if infiltration else "connected_itzi_swmm"
    metrics_arg = f"LarNO-main\\benchmark\\urbanflood\\flood\\{args.dataset_name}\\connected_swmm_metrics_rebuilt.csv"
    manifest = {
        "title": run_label,
        "run_order": [
            "python extended_study\\build_component_outfall_swmm_network.py",
            "python extended_study\\run_connected_swmm_validation.py",
            "python extended_study\\diagnose_hydrograph_timing.py",
            "python extended_study\\run_connected_itzi_swmm_comparison.py --events event68 --infiltration-mmh 1",
            "python extended_study\\run_connected_itzi_swmm_comparison.py --events event68 --infiltration-mmh 2",
            "python extended_study\\run_connected_itzi_swmm_comparison.py --events event68 --infiltration-mmh 3",
            "python extended_study\\run_connected_itzi_swmm_comparison.py --events event68 --infiltration-mmh 4",
            "python extended_study\\run_connected_itzi_swmm_comparison.py --events event68 --infiltration-mmh 5",
            "python extended_study\\summarize_infiltration_pilot.py",
            "python extended_study\\summarize_infiltration_sensitivity.py",
            f"python extended_study\\run_connected_itzi_swmm_comparison.py --events event1 --infiltration-mmh {infiltration:g}",
            f"python extended_study\\run_connected_itzi_swmm_comparison.py --events event20 --infiltration-mmh {infiltration:g}",
            f"python extended_study\\run_connected_itzi_swmm_comparison.py --events event65 --infiltration-mmh {infiltration:g}",
            f"python extended_study\\run_connected_itzi_swmm_comparison.py --events event66 --infiltration-mmh {infiltration:g}",
            f"python extended_study\\run_connected_itzi_swmm_comparison.py --events event67 --infiltration-mmh {infiltration:g}",
            f"python extended_study\\run_connected_itzi_swmm_comparison.py --events event68 --infiltration-mmh {infiltration:g}",
            f"python extended_study\\run_connected_itzi_swmm_comparison.py --events event69 --infiltration-mmh {infiltration:g}",
            f"python extended_study\\run_connected_itzi_swmm_comparison.py --events event70 --infiltration-mmh {infiltration:g}",
            f"python extended_study\\build_connected_swmm_drainage_dataset_v2_infiltration.py --run-dir extended_study\\output\\{run_dir_name} --dataset-name {args.dataset_name} --audit-output-name {audit_name} --infiltration-mmh {infiltration:g}",
            f"python extended_study\\train_drainlite_connected_residual.py --dataset-name {args.dataset_name} --output-name {args.main_output_name} --connected-metrics {metrics_arg}",
            f"python extended_study\\recompute_connected_cv_metrics.py --dataset-name {args.dataset_name} --output-name {args.main_output_name} --connected-metrics {metrics_arg} --events event68 event69 event70 --models base mask_only hydraulic_only all_static swmm_assisted --prediction-template h_drainlite_connected_{{model}}.npy --metrics-kind main",
            f"python extended_study\\run_drainlite_connected_cross_validation.py --dataset-name {args.dataset_name} --output-name {args.cv_output_name} --connected-metrics {metrics_arg}",
            f"python extended_study\\recompute_connected_cv_metrics.py --dataset-name {args.dataset_name} --output-name {args.cv_output_name} --connected-metrics {metrics_arg} --events event1 event20 event65 event66 event67 event68 event69 event70 --models base all_static swmm_assisted",
            f"python extended_study\\run_drainlite_connected_cross_validation.py --dataset-name {args.dataset_name} --output-name drainlite_connected_cross_validation_v2_inf1mmh_quality_filtered --connected-metrics {metrics_arg} --events event1 event67 event68 event69 event70 --models base mask_only hydraulic_only all_static swmm_assisted --pixels-per-step 700 --max-iter 220",
            f"python extended_study\\recompute_connected_cv_metrics.py --dataset-name {args.dataset_name} --output-name drainlite_connected_cross_validation_v2_inf1mmh_quality_filtered --connected-metrics {metrics_arg} --events event1 event67 event68 event69 event70 --models base mask_only hydraulic_only all_static swmm_assisted",
            "python extended_study\\summarize_quality_filtered_validation.py",
            "python extended_study\\summarize_quality_filtered_mike.py",
            f"python extended_study\\diagnose_hydrograph_timing.py --dataset-name {args.dataset_name} --output-name hydrograph_timing_diagnostics_v2_inf1mmh",
            f"python extended_study\\generate_drainlite_connected_diagnostics.py --dataset-name {args.dataset_name} --main-output-name {args.main_output_name} --cv-output-name {args.cv_output_name}",
            "python extended_study\\summarize_infiltration_model_selection.py",
            f"python extended_study\\generate_larno_drainlite_v2_package.py --dataset-name {args.dataset_name} --connected-metrics {metrics_arg} --main-output-name {args.main_output_name} --cv-output-name {args.cv_output_name} --package-output-name {args.package_output_name} --infiltration-mmh {infiltration:g}",
            "python extended_study\\generate_larno_drainlite_v2_inf1mmh_manuscript.py",
            f"python extended_study\\generate_drainlite_connected_report.py --dataset-name {args.dataset_name} --main-output-name {args.main_output_name} --cv-output-name {args.cv_output_name} --package-output-name {args.package_output_name} --connected-output-name {run_dir_name} --dataset-audit-name {audit_name} --hydro-output-name hydrograph_timing_diagnostics_v2_inf1mmh",
        ],
        "fixed_workflow": str(ROOT / "extended_study" / "ITZI_SWMM_FIXED_WORKFLOW.md"),
        "dataset": str(DATA),
        "main_model_output": str(MAIN_OUT),
        "cross_validation_output": str(CV_OUT),
        "quality_filtered_cross_validation_output": str(ROOT / "extended_study" / "output" / "drainlite_connected_cross_validation_v2_inf1mmh_quality_filtered"),
        "quality_stratified_output": str(ROOT / "extended_study" / "output" / "drainlite_quality_stratified_v2_inf1mmh"),
        "v2_package_output": str(OUT),
        "hydrograph_timing_diagnostics": str(ROOT / "extended_study" / "output" / "hydrograph_timing_diagnostics_v2_inf1mmh"),
        "infiltration_pilot": str(ROOT / "extended_study" / "output" / "infiltration_pilot"),
        "manuscript": str(OUT / "manuscript_larno_drainlite_v2.md"),
        "boundary_statement": (
            "This package supports a road-aligned conceptual drainage prior and "
            "lightweight residual correction. It does not claim a surveyed sewer "
            "network or a fully retrained LarNO backbone."
        ),
        "connected_metrics": str(CONNECTED_METRICS),
        "infiltration_mmh": infiltration,
        "metric_scope": "43,606 valid DEM cells; building-wall and invalid cells excluded",
        "primary_events": ["event1", "event67", "event68", "event69", "event70"],
        "diagnostic_excluded_events": ["event20", "event65", "event66"],
        "primary_model": "all_static",
        "oracle_only_model": "swmm_assisted",
    }
    (OUT / "reproducibility_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# LarNO-DrainLite v2 Reproducibility Manifest",
        "",
        "## Scope",
        "",
        manifest["boundary_statement"],
        "",
        "## Run Order",
        "",
    ]
    lines.extend(f"{idx}. `{cmd}`" for idx, cmd in enumerate(manifest["run_order"], start=1))
    lines.extend(
        [
            "",
            "## Key Outputs",
            "",
            f"- Quality-filtered cross-validation: {manifest['quality_filtered_cross_validation_output']}",
            f"- Quality-stratified analysis: {manifest['quality_stratified_output']}",
            f"- Fixed workflow: `{manifest['fixed_workflow']}`",
            f"- Dataset: `{manifest['dataset']}`",
            f"- Main model output: `{manifest['main_model_output']}`",
            f"- Cross-validation output: `{manifest['cross_validation_output']}`",
            f"- v2 package output: `{manifest['v2_package_output']}`",
            f"- Hydrograph timing diagnostics: `{manifest['hydrograph_timing_diagnostics']}`",
            f"- Infiltration pilot: `{manifest['infiltration_pilot']}`",
            f"- Manuscript draft: `{manifest['manuscript']}`",
            "",
            "## Metric Scope",
            "",
            f"- {manifest['metric_scope']}",
            f"- Primary events: {', '.join(manifest['primary_events'])}",
            f"- Diagnostic-only excluded events: {', '.join(manifest['diagnostic_excluded_events'])}",
            "- all_static is the deployable primary model; swmm_assisted is an oracle-assisted diagnostic.",
        ]
    )
    (OUT / "REPRODUCIBILITY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    configure_paths(args)
    METRICS.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    quality, quality_summary = build_quality_tables()
    mike_rows, mike_summary = build_mike_reference_tables()
    write_csv(METRICS / "v2_label_quality_grading.csv", quality)
    write_csv(METRICS / "v2_label_quality_summary.csv", quality_summary)
    write_csv(METRICS / "v2_mike_reference_event_metrics.csv", mike_rows)
    write_csv(METRICS / "v2_mike_reference_summary.csv", mike_summary)
    plot_quality(quality)
    plot_mike_summary(mike_summary)
    plot_mike_event_mae(mike_rows)
    write_reproducibility_manifest(args.run_label, args)
    summary = {
        "quality_counts": quality_summary,
        "mike_reference_summary": mike_summary,
        "outputs": {
            "metrics": str(METRICS),
            "figures": str(FIG),
            "manifest": str(OUT / "REPRODUCIBILITY.md"),
        },
    }
    (OUT / "v2_package_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
