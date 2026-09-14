#!/usr/bin/env python3
"""MIKE external plausibility metrics for the quality-filtered DrainLite analysis."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from publication_plot_style import MODEL_COLORS, add_panel_labels, configure_publication_style


configure_publication_style()


ROOT = Path(__file__).resolve().parents[1]
CELL_AREA_M2 = 400.0


def csi(pred: np.ndarray, target: np.ndarray, threshold: float) -> float:
    p = pred >= threshold
    t = target >= threshold
    denominator = np.logical_or(p, t).sum()
    return float(np.logical_and(p, t).sum() / denominator) if denominator else 1.0


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", default="region1_20m_connected_swmm_v2_inf1mmh")
    parser.add_argument("--cv-output-name", default="drainlite_connected_cross_validation_v2_inf1mmh_quality_filtered")
    parser.add_argument("--output-name", default="drainlite_quality_stratified_v2_inf1mmh")
    parser.add_argument("--events", nargs="+", default=["event1", "event67", "event68", "event69", "event70"])
    args = parser.parse_args()

    flood = ROOT / "LarNO-main" / "benchmark" / "urbanflood" / "flood" / args.dataset_name
    geo = ROOT / "LarNO-main" / "benchmark" / "urbanflood" / "geodata" / args.dataset_name
    cv = ROOT / "extended_study" / "output" / args.cv_output_name
    output = ROOT / "extended_study" / "output" / args.output_name
    metrics_dir = output / "metrics"
    figures_dir = output / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    dem = np.load(geo / "dem.npy").astype(np.float32)
    active = np.isfinite(dem) & (dem < 49.9)
    event_rows: list[dict[str, object]] = []
    for event in args.events:
        event_dir = flood / event
        mike = np.load(event_dir / "h_mike_ref.npy").astype(np.float32)
        arrays = {
            "ITZI surface-only": np.load(event_dir / "h_itzi_surface.npy").astype(np.float32),
            "ITZI-SWMM connected": np.load(event_dir / "h_itzi_swmm_connected.npy").astype(np.float32),
            "DrainLite all_static": np.load(
                cv / "predictions" / event / "h_drainlite_connected_cv_all_static.npy"
            ).astype(np.float32),
        }
        mike_v = mike[:, active]
        mike_peak_map = mike.max(axis=0)[active]
        mike_hmax = mike_v.max(axis=1)
        mike_peak_t = int(np.argmax(mike_hmax))
        mike_final_volume = float(mike_v[-1].sum() * CELL_AREA_M2)
        for model, pred in arrays.items():
            pred_v = pred[:, active]
            error = pred_v - mike_v
            pred_peak_map = pred.max(axis=0)[active]
            pred_hmax = pred_v.max(axis=1)
            pred_peak_t = int(np.argmax(pred_hmax))
            event_rows.append(
                {
                    "event": event,
                    "model": model,
                    "evaluation_active_cells": int(active.sum()),
                    "mae_mm": float(np.mean(np.abs(error)) * 1000.0),
                    "rmse_mm": float(np.sqrt(np.mean(error * error)) * 1000.0),
                    "csi_0p03": csi(pred_v, mike_v, 0.03),
                    "csi_0p15": csi(pred_v, mike_v, 0.15),
                    "peak_depth_error_mm": float((pred_hmax.max() - mike_hmax.max()) * 1000.0),
                    "abs_peak_depth_error_mm": float(abs(pred_hmax.max() - mike_hmax.max()) * 1000.0),
                    "peak_time_error_h": float((pred_peak_t - mike_peak_t) * 5.0 / 60.0),
                    "abs_peak_time_error_h": float(abs(pred_peak_t - mike_peak_t) * 5.0 / 60.0),
                    "peak_map_mae_mm": float(np.mean(np.abs(pred_peak_map - mike_peak_map)) * 1000.0),
                    "final_volume_error_m3": float(pred_v[-1].sum() * CELL_AREA_M2 - mike_final_volume),
                    "abs_final_volume_error_m3": float(abs(pred_v[-1].sum() * CELL_AREA_M2 - mike_final_volume)),
                }
            )

    summary: list[dict[str, object]] = []
    for model in ["ITZI surface-only", "ITZI-SWMM connected", "DrainLite all_static"]:
        rows = [row for row in event_rows if row["model"] == model]
        summary.append(
            {
                "model": model,
                "n_events": len(rows),
                "mae_mm": float(np.mean([row["mae_mm"] for row in rows])),
                "rmse_mm": float(np.mean([row["rmse_mm"] for row in rows])),
                "csi_0p03": float(np.mean([row["csi_0p03"] for row in rows])),
                "csi_0p15": float(np.mean([row["csi_0p15"] for row in rows])),
                "peak_depth_error_mm": float(np.mean([row["peak_depth_error_mm"] for row in rows])),
                "abs_peak_depth_error_mm": float(np.mean([row["abs_peak_depth_error_mm"] for row in rows])),
                "abs_peak_time_error_h": float(np.mean([row["abs_peak_time_error_h"] for row in rows])),
                "peak_map_mae_mm": float(np.mean([row["peak_map_mae_mm"] for row in rows])),
                "final_volume_error_m3": float(np.mean([row["final_volume_error_m3"] for row in rows])),
                "abs_final_volume_error_m3": float(np.mean([row["abs_final_volume_error_m3"] for row in rows])),
            }
        )
    write_csv(metrics_dir / "quality_filtered_mike_event_metrics.csv", event_rows)
    write_csv(metrics_dir / "quality_filtered_mike_summary.csv", summary)

    labels = [row["model"] for row in summary]
    colors = [MODEL_COLORS[label] for label in labels]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.05))
    axes[0].bar(x, [row["mae_mm"] for row in summary], color=colors)
    axes[0].set_ylabel("MAE to MIKE (mm)")
    axes[0].set_title("Full time-space error")
    axes[1].bar(x, [row["peak_depth_error_mm"] for row in summary], color=colors)
    axes[1].axhline(0, color="#111827", linewidth=0.8)
    axes[1].set_ylabel("Peak-depth bias (mm)")
    axes[1].set_title("Signed peak bias")
    axes[2].bar(x, [row["final_volume_error_m3"] for row in summary], color=colors)
    axes[2].axhline(0, color="#111827", linewidth=0.8)
    axes[2].set_ylabel(r"Final-volume bias (m$^3$)")
    axes[2].set_title("Signed final-volume bias")
    value_groups = [
        [row["mae_mm"] for row in summary],
        [row["peak_depth_error_mm"] for row in summary],
        [row["final_volume_error_m3"] for row in summary],
    ]
    for axis_index, (ax, values) in enumerate(zip(axes, value_groups)):
        maximum = max(abs(float(value)) for value in values) or 1.0
        for index, value in enumerate(values):
            label = f"{value:.1f}" if axis_index < 2 else f"{value / 1000:.1f}k"
            if value >= 0:
                ax.annotate(
                    label,
                    (index, value),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=6.6,
                )
            elif abs(value) >= 0.25 * maximum:
                ax.text(
                    index,
                    value * 0.5,
                    label,
                    ha="center",
                    va="center",
                    fontsize=6.6,
                    color="white",
                )
            else:
                ax.text(
                    index,
                    value - 0.05 * maximum,
                    label,
                    ha="center",
                    va="top",
                    fontsize=6.6,
                )
    for ax in axes:
        ax.set_xticks(x, labels, rotation=18, ha="right")
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    add_panel_labels(axes, x=-0.18, y=1.04)
    fig.suptitle("External comparison with MIKE on five quality-usable events")
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    fig.savefig(figures_dir / "quality_filtered_mike_summary.png")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
