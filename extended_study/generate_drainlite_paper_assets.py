#!/usr/bin/env python3
"""Generate manuscript-grade figures and tables for the DrainLite paper draft."""

from __future__ import annotations

import csv
import json
import math
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DL = ROOT / "extended_study" / "output" / "drainlite_residual"
FIG_SRC = DL / "figures"
METRICS = DL / "metrics"
PRED = DL / "predictions"
FLOOD = ROOT / "LarNO-main" / "benchmark" / "urbanflood" / "flood" / "region1_20m_drainage_v1"
GEO = ROOT / "LarNO-main" / "benchmark" / "urbanflood" / "geodata" / "region1_20m_drainage_v1"
SWMM_CSV = ROOT / "extended_study" / "output" / "swmm_network" / "swmm_summary_metrics.csv"
COUPLED_PROTO_FIG = ROOT / "extended_study" / "output" / "itzi_swmm_coupled_prototype" / "event75" / "prototype_comparison.png"
COUPLED_PROTO_SUMMARY_FIG = ROOT / "extended_study" / "output" / "itzi_swmm_coupled_prototype" / "prototype_summary_all_events.png"
COUPLED_PROTO_SUMMARY_CSV = ROOT / "extended_study" / "output" / "itzi_swmm_coupled_prototype" / "prototype_summary_all_events.csv"
OUT = ROOT / "paper_draft" / "drainlite"
FIG = OUT / "figures"
TAB = OUT / "tables"
CELL_AREA = 400.0


def ensure_dirs() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)


def load_metadata() -> dict:
    return json.loads((DL / "drainlite_run_metadata.json").read_text(encoding="utf-8"))


def box(ax, xy, wh, text, fc="#f6f9fc", ec="#305f8f", fontsize=10):
    x, y = xy
    w, h = wh
    patch = plt.Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, lw=1.4, joinstyle="round")
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, wrap=True)
    return patch


def arrow(ax, start, end, color="#305f8f"):
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(arrowstyle="->", lw=1.5, color=color, shrinkA=4, shrinkB=4),
    )


def make_framework_figure() -> None:
    fig, ax = plt.subplots(figsize=(13, 6.8))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 7)
    ax.axis("off")
    ax.set_title("DrainLite: lightweight drainage-aware residual correction", fontsize=15, weight="bold")

    box(ax, (0.4, 4.9), (2.3, 1.0), "Rainfall field\n5-min sequence", "#eaf3ff")
    box(ax, (0.4, 3.55), (2.3, 1.0), "DEM and local\nsurface predictors", "#eaf3ff")
    box(ax, (0.4, 2.2), (2.3, 1.0), "Road-aligned pipe\nnetwork rasters", "#eaf3ff")
    box(ax, (0.4, 0.85), (2.3, 1.0), "SWMM standalone\nevent metrics", "#fff4e6", "#b56b22")

    box(ax, (3.5, 4.2), (2.4, 1.15), "ITZI surface-only\n2D dynamic runoff", "#edf7ed", "#2d7a3e")
    box(ax, (3.5, 2.2), (2.4, 1.15), "Feature stack\nsurface + rainfall + DEM\n+ drainage priors", "#f6f9fc")
    box(ax, (6.8, 2.2), (2.5, 1.15), "DrainLite residual learner\nHistGradientBoosting", "#f6f9fc")
    box(ax, (10.1, 2.2), (2.4, 1.15), "Predicted reduction\nnon-negative drainage effect", "#f6f9fc")
    box(ax, (10.1, 4.45), (2.4, 1.15), "Drainage-aware forecast\nh = max(surface - reduction, 0)", "#edf7ed", "#2d7a3e")
    box(ax, (6.8, 0.55), (2.5, 1.0), "Training target\nmax(surface - sink, 0)", "#fff0f0", "#b23b3b")

    arrow(ax, (2.7, 5.4), (3.5, 4.85))
    arrow(ax, (2.7, 4.05), (3.5, 4.55))
    arrow(ax, (2.7, 2.7), (3.5, 2.75))
    arrow(ax, (2.7, 1.35), (3.5, 2.45))
    arrow(ax, (5.9, 4.75), (10.1, 5.0))
    arrow(ax, (5.9, 2.75), (6.8, 2.75))
    arrow(ax, (9.3, 2.75), (10.1, 2.75))
    arrow(ax, (11.3, 3.35), (11.3, 4.45))
    arrow(ax, (8.05, 1.55), (8.05, 2.2), "#b23b3b")

    ax.text(
        0.45,
        6.35,
        "a",
        fontsize=18,
        weight="bold",
    )
    ax.text(
        6.8,
        6.35,
        "b",
        fontsize=18,
        weight="bold",
    )
    ax.text(
        6.8,
        5.85,
        "Constraint: 0 <= h_DrainLite <= h_surface-only; no LarNO backbone retraining is required.",
        fontsize=10,
        color="#425466",
    )
    fig.tight_layout()
    fig.savefig(FIG / "fig01_framework.png", dpi=220)
    plt.close(fig)


def make_data_flow_figure() -> None:
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 7)
    ax.axis("off")
    ax.set_title("Drainage-augmented benchmark construction and roles of physical references", fontsize=15, weight="bold")

    box(ax, (0.35, 5.2), (2.2, 0.9), "LarNO public\n20 m benchmark", "#eaf3ff")
    box(ax, (0.35, 3.75), (2.2, 0.9), "OSM / road-aligned\nconceptual network", "#eaf3ff")
    box(ax, (0.35, 2.3), (2.2, 0.9), "ITZI surface solver\nno drainage", "#edf7ed", "#2d7a3e")
    box(ax, (0.35, 0.85), (2.2, 0.9), "ITZI + inlet sink\nconceptual drainage label", "#edf7ed", "#2d7a3e")

    box(ax, (3.35, 5.2), (2.3, 0.9), "MIKE reference\nexternal benchmark", "#fff4e6", "#b56b22")
    box(ax, (3.35, 3.75), (2.3, 0.9), "8 static drainage rasters\n20 m aligned", "#f6f9fc")
    box(ax, (3.35, 2.3), (2.3, 0.9), "SWMM standalone\n1D dynamic-wave metrics", "#fff4e6", "#b56b22")
    box(ax, (3.35, 0.85), (2.3, 0.9), "Reduction target\nsurface - sink", "#fff0f0", "#b23b3b")

    box(ax, (6.55, 3.75), (2.45, 1.1), "DrainLite dataset\n17 events, 72 steps,\n200 x 280 grid", "#f6f9fc")
    box(ax, (9.75, 4.65), (2.55, 0.95), "Train split\n12 events", "#eaf3ff")
    box(ax, (9.75, 3.05), (2.55, 0.95), "Test split\n5 events", "#eaf3ff")
    box(ax, (9.75, 1.45), (2.55, 0.95), "Outputs\nmodels, predictions,\nmetrics and figures", "#edf7ed", "#2d7a3e")

    for start_y, end_y in [(5.65, 4.3), (4.2, 4.3), (2.75, 4.3), (1.3, 4.3)]:
        arrow(ax, (2.55, start_y), (6.55, end_y))
    arrow(ax, (5.65, 5.65), (6.55, 4.55))
    arrow(ax, (5.65, 4.2), (6.55, 4.25))
    arrow(ax, (5.65, 2.75), (6.55, 4.0))
    arrow(ax, (5.65, 1.3), (6.55, 3.95))
    arrow(ax, (9.0, 4.3), (9.75, 5.1))
    arrow(ax, (9.0, 4.25), (9.75, 3.5))
    arrow(ax, (9.0, 4.0), (9.75, 1.95))

    ax.text(0.4, 6.45, "a", fontsize=18, weight="bold")
    ax.text(6.55, 6.45, "b", fontsize=18, weight="bold")
    ax.text(
        3.35,
        6.35,
        "MIKE is used as an external reference, whereas ITZI+sink provides the conceptual drainage target.\n"
        "SWMM remains standalone and is not interpreted as two-way ITZI-SWMM coupling.",
        fontsize=10,
        color="#425466",
    )
    fig.tight_layout()
    fig.savefig(FIG / "fig03_data_flow.png", dpi=220)
    plt.close(fig)


def make_ablation_figure() -> None:
    df = pd.read_csv(METRICS / "drainlite_ablation_summary.csv")
    df = df[~df["model"].isin(["itzi_sink_label"])]
    order = ["surface_only", "base", "mask_only", "hydraulic_only", "all_static", "swmm_assisted"]
    df["model"] = pd.Categorical(df["model"], order, ordered=True)
    df = df.sort_values("model")
    x = np.arange(len(df))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    axes[0].bar(x, df["mae_mm"], color="#4477aa")
    axes[0].set_title("MAE to ITZI+sink")
    axes[0].set_ylabel("MAE (mm)")
    axes[1].bar(x, df["csi_0p03"], color="#228833")
    axes[1].set_title("CSI at 0.03 m")
    axes[1].set_ylim(0, 1)
    axes[2].bar(x, df["reduction_mae_mm"], color="#cc6677")
    axes[2].set_title("Reduction MAE")
    axes[2].set_ylabel("MAE (mm)")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(df["model"], rotation=40, ha="right")
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Ablation of drainage-aware feature groups")
    fig.tight_layout()
    fig.savefig(FIG / "fig05_ablation.png", dpi=220)
    plt.close(fig)


def make_time_series_figure(events: list[str] = ["event75", "event80"]) -> None:
    fig, axes = plt.subplots(len(events), 3, figsize=(14, 7), sharex=True)
    if len(events) == 1:
        axes = axes[None, :]
    for row, event in enumerate(events):
        surface = np.load(FLOOD / event / "h_itzi_surface.npy")
        sink = np.load(FLOOD / event / "h_itzi_sink.npy")
        pred = np.load(PRED / event / "h_drainlite_all_static.npy")
        mike = np.load(FLOOD / event / "h_mike_ref.npy")
        t = np.arange(surface.shape[0]) * 5
        series = {
            "MIKE ref": mike.max(axis=(1, 2)),
            "ITZI surface": surface.max(axis=(1, 2)),
            "ITZI+sink": sink.max(axis=(1, 2)),
            "DrainLite": pred.max(axis=(1, 2)),
        }
        for label, values in series.items():
            axes[row, 0].plot(t, values, label=label, lw=1.5)
        axes[row, 0].set_title(f"{event}: peak depth")
        axes[row, 0].set_ylabel("Depth (m)")
        vol = {
            "ITZI surface": surface.sum(axis=(1, 2)) * CELL_AREA,
            "ITZI+sink": sink.sum(axis=(1, 2)) * CELL_AREA,
            "DrainLite": pred.sum(axis=(1, 2)) * CELL_AREA,
        }
        for label, values in vol.items():
            axes[row, 1].plot(t, values / 1000.0, label=label, lw=1.5)
        axes[row, 1].set_title(f"{event}: water volume")
        axes[row, 1].set_ylabel(r"Volume ($10^3$ m$^3$)")
        axes[row, 2].plot(t, np.mean(np.abs(surface - sink), axis=(1, 2)) * 1000, label="surface-only", lw=1.5)
        axes[row, 2].plot(t, np.mean(np.abs(pred - sink), axis=(1, 2)) * 1000, label="DrainLite", lw=1.5)
        axes[row, 2].set_title(f"{event}: MAE to ITZI+sink")
        axes[row, 2].set_ylabel("MAE (mm)")
        for ax in axes[row]:
            ax.grid(alpha=0.25)
            ax.set_xlabel("Time (min)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(FIG / "extended_fig_time_series.png", dpi=220)
    plt.close(fig)


def copy_existing_figures() -> None:
    copies = {
        "static_dem_network.png": "fig02_dem_network.png",
        "event75_peak_maps.png": "fig04_event75_peak_maps.png",
        "event76_peak_maps.png": "extended_fig_event76_peak_maps.png",
        "event77_peak_maps.png": "extended_fig_event77_peak_maps.png",
        "event78_peak_maps.png": "extended_fig_event78_peak_maps.png",
        "event80_peak_maps.png": "extended_fig_event80_peak_maps.png",
        "feature_importance_all_static.png": "fig06_feature_importance.png",
        "swmm_summary.png": "fig07_swmm_summary.png",
    }
    for src, dst in copies.items():
        shutil.copy2(FIG_SRC / src, FIG / dst)
    if COUPLED_PROTO_FIG.exists():
        shutil.copy2(COUPLED_PROTO_FIG, FIG / "extended_fig_itzi_swmm_prototype.png")
    if COUPLED_PROTO_SUMMARY_FIG.exists():
        shutil.copy2(COUPLED_PROTO_SUMMARY_FIG, FIG / "extended_fig_itzi_swmm_prototype_summary.png")


def feature_definitions() -> list[dict[str, str]]:
    return [
        {"feature": "surface_depth_mm", "unit": "mm", "definition": "ITZI surface-only water depth at the current time step."},
        {"feature": "rainfall_mm", "unit": "mm/5 min", "definition": "Spatial rainfall field at the current 5-min time step."},
        {"feature": "cumsum_rainfall_mm", "unit": "mm", "definition": "Cumulative rainfall from the beginning of the event."},
        {"feature": "dem_m", "unit": "m", "definition": "Digital elevation model aligned to the 20 m grid."},
        {"feature": "dem_slope", "unit": "m/cell", "definition": "Local DEM gradient magnitude."},
        {"feature": "drain_inlet_mask", "unit": "-", "definition": "Binary raster indicating conceptual drainage inlet locations."},
        {"feature": "drain_outfall_mask", "unit": "-", "definition": "Binary raster indicating conceptual outfall locations."},
        {"feature": "pipe_mask", "unit": "-", "definition": "Binary raster of road-aligned conceptual pipe cells."},
        {"feature": "pipe_diameter", "unit": "m", "definition": "Rasterized conceptual pipe diameter assigned to pipe cells."},
        {"feature": "pipe_slope", "unit": "-", "definition": "Rasterized pipe slope derived from local terrain slope."},
        {"feature": "pipe_capacity", "unit": "m3/s", "definition": "Conceptual full-flow pipe capacity from Manning-type approximation."},
        {"feature": "pipe_cover_depth", "unit": "m", "definition": "Ground elevation minus conceptual pipe invert elevation."},
        {"feature": "distance_to_outfall", "unit": "m", "definition": "Euclidean distance to the nearest conceptual outfall."},
        {"feature": "pipe_density_3x3/7x7", "unit": "-", "definition": "Local moving-average density of pipe cells."},
        {"feature": "inlet_density_7x7", "unit": "-", "definition": "Local moving-average density of inlet cells."},
        {"feature": "capacity_density_7x7", "unit": "m3/s", "definition": "Local moving-average pipe capacity."},
        {"feature": "SWMM event metrics", "unit": "various", "definition": "Standalone 1D dynamic-wave summary metrics; not two-way coupling."},
    ]


def write_tables(metadata: dict) -> None:
    ablation = pd.read_csv(METRICS / "drainlite_ablation_summary.csv")
    event_metrics = pd.read_csv(METRICS / "drainlite_event_metrics.csv")
    quality = pd.read_csv(ROOT / "extended_study" / "output" / "drainage_dataset_v1" / "dataset_quality.csv")
    swmm = pd.read_csv(SWMM_CSV)

    table1 = pd.DataFrame([
        {"item": "Study area window", "value": "200 x 280 cells at 20 m resolution"},
        {"item": "Temporal length", "value": "72 steps at 5 min intervals (6 h)"},
        {"item": "Valid events", "value": "17 events; event79 excluded due to corrupt local files"},
        {"item": "Train/test split", "value": f"{len(metadata['train_events'])}/{len(metadata['test_events'])} events"},
        {"item": "Base physical state", "value": "ITZI surface-only dynamic 2D runoff"},
        {"item": "Drainage target", "value": "ITZI + conceptual road-aligned inlet sink"},
        {"item": "External reference", "value": "MIKE reference from LarNO benchmark"},
        {"item": "Drainage features", "value": "8 static pipe-network rasters plus derived local densities"},
        {"item": "Residual learner", "value": "HistGradientBoostingRegressor"},
        {"item": "Training samples", "value": f"{metadata['pixels_per_step']} pixels per event-step in this local run"},
    ])
    table1.to_csv(TAB / "table1_dataset_model_configuration.csv", index=False)

    ablation.to_csv(TAB / "table2_ablation_metrics.csv", index=False)

    mike = event_metrics[event_metrics["target"] == "mike_ref"].copy()
    table3 = mike.groupby("model", as_index=False).agg(
        n_events=("event", "count"),
        mae_m=("mae_m", "mean"),
        rmse_m=("rmse_m", "mean"),
        csi_0p03=("csi_0p03", "mean"),
        csi_0p15=("csi_0p15", "mean"),
        peak_error_m=("peak_error_m", "mean"),
    )
    table3.to_csv(TAB / "table3_mike_external_reference.csv", index=False)

    sink_metrics = event_metrics[
        (event_metrics["target"] == "itzi_sink")
        & (event_metrics["model"] != "itzi_sink_label")
    ].copy()
    table4 = sink_metrics.groupby("model", as_index=False).agg(
        n_events=("event", "count"),
        peak_volume_error_m3=("peak_volume_error_m3", "mean"),
        peak_area_error_m2_0p03=("peak_area_error_m2_0p03", "mean"),
        reduction_mae_mm=("reduction_mae_m", lambda s: float(np.mean(s) * 1000.0)),
        reduction_capture_error_pct=("reduction_capture_error_pct", "mean"),
    )
    order = ["surface_only", "base", "mask_only", "hydraulic_only", "all_static", "swmm_assisted"]
    table4["model"] = pd.Categorical(table4["model"], order, ordered=True)
    table4.sort_values("model").to_csv(TAB / "table4_secondary_itzi_sink_metrics.csv", index=False)

    quality.merge(swmm[["event", "routing_continuity_error_pct", "outfall_volume_m3", "node_flooding_volume_m3"]], on="event", how="left").to_csv(
        TAB / "extended_table1_event_inventory.csv",
        index=False,
    )
    pd.DataFrame(feature_definitions()).to_csv(TAB / "extended_table2_feature_definitions.csv", index=False)
    if COUPLED_PROTO_SUMMARY_CSV.exists():
        pd.read_csv(COUPLED_PROTO_SUMMARY_CSV).to_csv(
            TAB / "extended_table3_coupled_prototype_summary.csv",
            index=False,
        )


def main() -> int:
    ensure_dirs()
    metadata = load_metadata()
    make_framework_figure()
    make_data_flow_figure()
    make_ablation_figure()
    make_time_series_figure()
    copy_existing_figures()
    write_tables(metadata)
    print(f"Wrote paper assets to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
