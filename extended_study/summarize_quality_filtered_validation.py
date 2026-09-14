#!/usr/bin/env python3
"""Summarize DrainLite validation by ITZI-SWMM label-quality tier."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from publication_plot_style import MODEL_COLORS, add_panel_labels, configure_publication_style, model_label


configure_publication_style()


ROOT = Path(__file__).resolve().parents[1]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def mean(rows: list[dict[str, str]], key: str, scale: float = 1.0) -> float:
    values = [float(row[key]) for row in rows if row.get(key, "") not in {"", "nan", "NaN"}]
    return float(np.mean(values) * scale) if values else float("nan")


def mean_abs(rows: list[dict[str, str]], key: str, scale: float = 1.0) -> float:
    values = [abs(float(row[key])) for row in rows if row.get(key, "") not in {"", "nan", "NaN"}]
    return float(np.mean(values) * scale) if values else float("nan")


def summarize(
    rows: list[dict[str, str]],
    analysis_set: str,
    validation_design: str,
    events: set[str],
) -> list[dict[str, object]]:
    selected = [row for row in rows if row["event"] in events]
    by_model: dict[str, list[dict[str, str]]] = {}
    for row in selected:
        by_model.setdefault(row["model"], []).append(row)
    surface_mae = mean(by_model.get("surface_only", []), "mae_m", 1000.0)
    base_mae = mean(by_model.get("base", []), "mae_m", 1000.0)
    output = []
    for model, model_rows in by_model.items():
        mae_mm = mean(model_rows, "mae_m", 1000.0)
        output.append(
            {
                "analysis_set": analysis_set,
                "validation_design": validation_design,
                "model": model,
                "n_events": len(model_rows),
                "mae_mm": mae_mm,
                "rmse_mm": mean(model_rows, "rmse_m", 1000.0),
                "csi_0p03": mean(model_rows, "csi_0p03"),
                "csi_0p15": mean(model_rows, "csi_0p15"),
                "peak_error_mm": mean(model_rows, "peak_error_m", 1000.0),
                "abs_peak_error_mm": (
                    mean(model_rows, "abs_peak_error_m", 1000.0)
                    if model_rows and "abs_peak_error_m" in model_rows[0]
                    else mean_abs(model_rows, "peak_error_m", 1000.0)
                ),
                "peak_time_error_h": mean(model_rows, "peak_time_error_h"),
                "abs_peak_time_error_h": mean(model_rows, "abs_peak_time_error_h"),
                "mean_abs_peak_volume_error_m3": mean_abs(model_rows, "peak_volume_error_m3"),
                "mean_abs_final_volume_error_m3": mean_abs(model_rows, "final_volume_error_m3"),
                "final_reduction_capture_pct": mean(model_rows, "final_reduction_capture_pct"),
                "signed_residual_mae_mm": mean(model_rows, "signed_residual_mae_m", 1000.0),
                "mae_to_mike_mm": mean(model_rows, "mae_to_mike_m", 1000.0),
                "peak_error_to_mike_mm": mean(model_rows, "peak_error_to_mike_m", 1000.0),
                "improvement_vs_surface_pct": (
                    100.0 * (surface_mae - mae_mm) / surface_mae if surface_mae > 0 else float("nan")
                ),
                "improvement_vs_base_pct": (
                    float("nan")
                    if model == "surface_only"
                    else 100.0 * (base_mae - mae_mm) / base_mae if base_mae > 0 else float("nan")
                ),
            }
        )
    return sorted(output, key=lambda row: (str(row["analysis_set"]), float(row["mae_mm"])))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--quality-csv",
        default="extended_study/output/larno_drainlite_v2_inf1mmh_package/metrics/v2_label_quality_grading.csv",
    )
    parser.add_argument(
        "--all8-metrics",
        default="extended_study/output/drainlite_connected_cross_validation_v2_inf1mmh/metrics/connected_residual_cv_event_metrics.csv",
    )
    parser.add_argument(
        "--filtered-metrics",
        default="extended_study/output/drainlite_connected_cross_validation_v2_inf1mmh_quality_filtered/metrics/connected_residual_cv_event_metrics.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="extended_study/output/drainlite_quality_stratified_v2_inf1mmh",
    )
    args = parser.parse_args()

    quality_path = ROOT / args.quality_csv
    all8_path = ROOT / args.all8_metrics
    filtered_path = ROOT / args.filtered_metrics
    output_dir = ROOT / args.output_dir
    metrics_dir = output_dir / "metrics"
    figures_dir = output_dir / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    quality_rows = read_csv(quality_path)
    all8_rows = read_csv(all8_path)
    filtered_rows = read_csv(filtered_path)
    status = {row["event"]: row["quality_status"] for row in quality_rows}
    all_events = set(status)
    usable_events = {event for event, tier in status.items() if tier in {"accepted", "warning"}}
    accepted_events = {event for event, tier in status.items() if tier == "accepted"}

    membership = []
    for row in quality_rows:
        membership.append(
            {
                "event": row["event"],
                "quality_status": row["quality_status"],
                "continuity_error_pct": row["continuity_error_pct"],
                "nonconverging_steps_pct": row["nonconverging_steps_pct"],
                "in_all8_diagnostic": True,
                "in_quality_filtered_formal": row["event"] in usable_events,
                "in_accepted_sensitivity": row["event"] in accepted_events,
            }
        )
    write_csv(metrics_dir / "quality_stratified_event_membership.csv", membership)

    summary = []
    summary.extend(summarize(all8_rows, "all8_diagnostic", "8-event LOEO; excluded labels may enter training", all_events))
    summary.extend(
        summarize(
            all8_rows,
            "quality_usable_subset_from_all8",
            "subset of 8-event LOEO predictions; excluded labels may enter training",
            usable_events,
        )
    )
    summary.extend(
        summarize(
            filtered_rows,
            "quality_filtered_formal",
            "strict 5-event LOEO; excluded labels absent from training and testing",
            usable_events,
        )
    )
    summary.extend(
        summarize(
            filtered_rows,
            "accepted_only_sensitivity",
            "accepted test events from strict 5-event LOEO",
            accepted_events,
        )
    )
    write_csv(metrics_dir / "quality_stratified_drainlite_summary.csv", summary)

    enriched_events = []
    for source, rows in [("all8_loeo", all8_rows), ("quality_filtered_loeo", filtered_rows)]:
        for row in rows:
            enriched_events.append({"validation_source": source, "quality_status": status.get(row["event"], "unknown"), **row})
    write_csv(metrics_dir / "quality_stratified_event_metrics.csv", enriched_events)

    formal = [row for row in summary if row["analysis_set"] == "quality_filtered_formal"]
    formal_by_model = {str(row["model"]): row for row in formal}
    all_static = formal_by_model.get("all_static", {})
    surface = formal_by_model.get("surface_only", {})
    base = formal_by_model.get("base", {})
    conclusions = {
        "formal_event_set": sorted(usable_events),
        "accepted_only_events": sorted(accepted_events),
        "excluded_events": sorted(all_events - usable_events),
        "formal_validation_design": "Five-event leave-one-event-out; excluded labels are absent from both training and testing.",
        "formal_all_static_mae_mm": all_static.get("mae_mm"),
        "formal_surface_only_mae_mm": surface.get("mae_mm"),
        "formal_base_mae_mm": base.get("mae_mm"),
        "formal_improvement_vs_surface_pct": all_static.get("improvement_vs_surface_pct"),
        "formal_improvement_vs_base_pct": all_static.get("improvement_vs_base_pct"),
        "interpretation": (
            "Use the strict five-event result as the primary model claim. Retain the eight-event result as a diagnostic "
            "sensitivity analysis, because three excluded SWMM labels otherwise enter training folds."
        ),
    }
    (metrics_dir / "quality_stratified_conclusions.json").write_text(
        json.dumps(conclusions, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    models = [
        model
        for model in ["surface_only", "base", "mask_only", "hydraulic_only", "all_static", "swmm_assisted"]
        if model in formal_by_model
    ]
    colors = MODEL_COLORS
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.8))
    x = np.arange(len(models))
    axes[0].bar(x, [float(formal_by_model[m]["mae_mm"]) for m in models], color=[colors[m] for m in models])
    axes[0].set_xticks(x, [model_label(model) for model in models], rotation=28, ha="right")
    axes[0].set_ylabel("MAE to ITZI-SWMM label (mm)")
    axes[0].set_title("Strict quality-filtered LOEO")

    axes[1].bar(x, [float(formal_by_model[m]["mae_to_mike_mm"]) for m in models], color=[colors[m] for m in models])
    axes[1].set_xticks(x, [model_label(model) for model in models], rotation=28, ha="right")
    axes[1].set_ylabel("MAE to MIKE reference (mm)")
    axes[1].set_title("External plausibility check")

    accepted_rows = [row for row in filtered_rows if row["event"] in accepted_events]
    warning_rows = [row for row in filtered_rows if status.get(row["event"]) == "warning"]
    tiers = ["accepted", "warning"]
    tier_rows = [accepted_rows, warning_rows]
    width = 0.38
    tx = np.arange(len(tiers))
    for offset, model in [(-width / 2, "surface_only"), (width / 2, "all_static")]:
        values = [mean([row for row in rows if row["model"] == model], "mae_m", 1000.0) for rows in tier_rows]
        axes[2].bar(tx + offset, values, width=width, label=model_label(model), color=colors[model])
    axes[2].set_xticks(tx, tiers)
    axes[2].set_ylabel("MAE to ITZI-SWMM label (mm)")
    axes[2].set_title("Performance by label tier")
    axes[2].legend(frameon=False)
    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    add_panel_labels(axes, x=-0.18, y=1.04)
    fig.suptitle("Quality-filtered DrainLite validation")
    fig.tight_layout()
    fig.savefig(figures_dir / "quality_filtered_validation_summary.png")
    plt.close(fig)

    print(json.dumps(conclusions, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
