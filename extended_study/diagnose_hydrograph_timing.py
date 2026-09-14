#!/usr/bin/env python3
"""Diagnose timing mismatch between MIKE and ITZI hydrographs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from publication_plot_style import MODEL_COLORS, add_panel_labels, configure_publication_style


configure_publication_style()


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "LarNO-main" / "benchmark" / "urbanflood" / "flood" / "region1_20m_connected_swmm_v1"
GEO = ROOT / "LarNO-main" / "benchmark" / "urbanflood" / "geodata" / "region1_20m_connected_swmm_v1"
OUT = ROOT / "extended_study" / "output" / "hydrograph_timing_diagnostics"
FIG = OUT / "figures"
EVENTS = ["event1", "event20", "event65", "event66", "event67", "event68", "event69", "event70"]
CELL_AREA_M2 = 20.0 * 20.0
DT_HOURS = 5.0 / 60.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", default="region1_20m_connected_swmm_v1")
    parser.add_argument("--output-name", default="hydrograph_timing_diagnostics")
    parser.add_argument("--events", nargs="+", default=EVENTS)
    return parser.parse_args()


def configure_paths(args: argparse.Namespace) -> None:
    global DATA, GEO, OUT, FIG, EVENTS
    DATA = ROOT / "LarNO-main" / "benchmark" / "urbanflood" / "flood" / args.dataset_name
    GEO = ROOT / "LarNO-main" / "benchmark" / "urbanflood" / "geodata" / args.dataset_name
    OUT = ROOT / "extended_study" / "output" / args.output_name
    FIG = OUT / "figures"
    EVENTS = list(args.events)


def active_mask() -> np.ndarray:
    dem = np.load(GEO / "dem.npy").astype(np.float32)
    return dem < 49.9


def volume(h: np.ndarray, active: np.ndarray | None = None) -> np.ndarray:
    if active is None:
        return h.sum(axis=(1, 2)) * CELL_AREA_M2
    return h[:, active].sum(axis=1) * CELL_AREA_M2


def max_depth(h: np.ndarray) -> np.ndarray:
    return h.max(axis=(1, 2))


def flooded_area(h: np.ndarray, threshold: float = 0.03) -> np.ndarray:
    return (h >= threshold).sum(axis=(1, 2)) * CELL_AREA_M2 / 1_000_000.0


def peak_summary(series: np.ndarray) -> tuple[int, float, float, float]:
    idx = int(np.nanargmax(series))
    return idx + 1, (idx + 1) * DT_HOURS, float(series[idx]), float(series[-1])


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def event_summary(event: str) -> list[dict[str, object]]:
    d = DATA / event
    active = active_mask()
    rainfall = np.load(d / "rainfall.npy").astype(np.float32)
    arrays = {
        "MIKE reference": np.load(d / "h_mike_ref.npy").astype(np.float32),
        "ITZI surface-only": np.load(d / "h_itzi_surface.npy").astype(np.float32),
        "ITZI-SWMM connected": np.load(d / "h_itzi_swmm_connected.npy").astype(np.float32),
    }
    rain_mean_mm_step = rainfall.mean(axis=(1, 2))
    rain_max_mm_step = rainfall.max(axis=(1, 2))
    rain_peak_step, rain_peak_hour, rain_peak_mean, rain_final_mean = peak_summary(rain_mean_mm_step)
    rows = []
    for model, h in arrays.items():
        hmax = max_depth(h)
        vol = volume(h, active)
        area = flooded_area(h)
        h_step, h_hour, h_peak, h_final = peak_summary(hmax)
        v_step, v_hour, v_peak, v_final = peak_summary(vol)
        a_step, a_hour, a_peak, a_final = peak_summary(area)
        rows.append(
            {
                "event": event,
                "model": model,
                "rain_peak_step": rain_peak_step,
                "rain_peak_hour": rain_peak_hour,
                "rain_peak_mean_mm_per_5min": rain_peak_mean,
                "rain_final_mean_mm_per_5min": rain_final_mean,
                "rain_total_mean_mm_if_depth_per_step": float(rain_mean_mm_step.sum()),
                "rain_total_maxcell_mm_if_depth_per_step": float(rainfall.sum(axis=0).max()),
                "rain_peak_mean_mmh_if_depth_per_step": float(rain_peak_mean * 12.0),
                "rain_max_cell_mmh_if_depth_per_step": float(rain_max_mm_step.max() * 12.0),
                "max_depth_peak_step": h_step,
                "max_depth_peak_hour": h_hour,
                "max_depth_peak_m": h_peak,
                "max_depth_final_m": h_final,
                "max_depth_final_to_peak_ratio": h_final / h_peak if h_peak else np.nan,
                "volume_peak_step": v_step,
                "volume_peak_hour": v_hour,
                "volume_peak_m3": v_peak,
                "volume_final_m3": v_final,
                "volume_final_to_peak_ratio": v_final / v_peak if v_peak else np.nan,
                "volume_mask": "active_nonbuilding_cells",
                "flooded_area_peak_step": a_step,
                "flooded_area_peak_hour": a_hour,
                "flooded_area_peak_km2": a_peak,
                "flooded_area_final_km2": a_final,
                "flooded_area_final_to_peak_ratio": a_final / a_peak if a_peak else np.nan,
                "peaks_at_final_frame": h_step == h.shape[0] or v_step == h.shape[0],
            }
        )
    return rows


def plot_event(event: str) -> None:
    d = DATA / event
    active = active_mask()
    rainfall = np.load(d / "rainfall.npy").astype(np.float32)
    mike = np.load(d / "h_mike_ref.npy").astype(np.float32)
    surface = np.load(d / "h_itzi_surface.npy").astype(np.float32)
    connected = np.load(d / "h_itzi_swmm_connected.npy").astype(np.float32)
    t = np.arange(1, rainfall.shape[0] + 1) * DT_HOURS
    rain_mean = rainfall.mean(axis=(1, 2))

    fig, axes = plt.subplots(3, 1, figsize=(7.1, 6.4), sharex=True, constrained_layout=True)
    ax = axes[0]
    ax.bar(t, rain_mean, width=DT_HOURS * 0.85, color="#6b7280", alpha=0.5, label="Mean rainfall depth per 5 min")
    ax2 = ax.twinx()
    ax2.plot(t, rain_mean * 12.0, color="#111827", lw=1.4, label="Equivalent mean rainfall intensity")
    ax.set_ylabel("mm / 5 min")
    ax2.set_ylabel("mm/h")
    ax.set_title(f"{event}: rainfall forcing")
    ax.grid(alpha=0.25)

    ax = axes[1]
    ax.plot(t, max_depth(mike), color=MODEL_COLORS["MIKE reference"], lw=1.8, label="MIKE reference")
    ax.plot(t, max_depth(surface), color=MODEL_COLORS["ITZI surface-only"], lw=1.6, label="ITZI surface-only")
    ax.plot(t, max_depth(connected), color=MODEL_COLORS["ITZI-SWMM connected"], lw=1.5, label="ITZI-SWMM connected")
    ax.set_ylabel("Maximum depth (m)")
    ax.set_title("Maximum water depth hydrograph")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    ax = axes[2]
    ax.plot(t, volume(mike, active) / 1000.0, color=MODEL_COLORS["MIKE reference"], lw=1.8, label="MIKE reference")
    ax.plot(t, volume(surface, active) / 1000.0, color=MODEL_COLORS["ITZI surface-only"], lw=1.6, label="ITZI surface-only")
    ax.plot(t, volume(connected, active) / 1000.0, color=MODEL_COLORS["ITZI-SWMM connected"], lw=1.5, label="ITZI-SWMM connected")
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Surface water volume (10^3 m3)")
    ax.set_title("Surface water volume hydrograph on active cells")
    ax.grid(alpha=0.25)
    add_panel_labels(axes, x=-0.09, y=1.02)
    fig.savefig(FIG / f"{event}_hydrograph_timing_diagnostic.png")
    plt.close(fig)


def plot_summary(rows: list[dict[str, object]]) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    models = ["MIKE reference", "ITZI surface-only", "ITZI-SWMM connected"]
    events = EVENTS
    by = {(r["event"], r["model"]): r for r in rows}
    x = np.arange(len(events))
    width = 0.25
    colors = MODEL_COLORS
    fig, axes = plt.subplots(2, 1, figsize=(7.1, 5.1), sharex=True, constrained_layout=True)
    for i, model in enumerate(models):
        axes[0].bar(
            x + (i - 1) * width,
            [by[(e, model)]["max_depth_peak_hour"] for e in events],
            width=width,
            label=model,
            color=colors[model],
        )
        axes[1].bar(
            x + (i - 1) * width,
            [by[(e, model)]["volume_final_to_peak_ratio"] for e in events],
            width=width,
            label=model,
            color=colors[model],
        )
    axes[0].set_ylabel("Peak time of maximum depth (h)")
    axes[0].set_title("Peak timing mismatch", pad=34)
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].axhline(1.0, color="#6b7280", lw=1, ls="--")
    axes[1].set_ylabel("Final / peak surface volume")
    axes[1].set_title("Recession diagnostic: values near 1 indicate no falling limb")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(events, rotation=20)
    axes[1].grid(axis="y", alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(
        handles,
        labels,
        fontsize=7.2,
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
    )
    add_panel_labels(axes, x=-0.09, y=1.02)
    fig.savefig(FIG / "hydrograph_timing_summary.png")
    plt.close(fig)


def build_rainfall_accounting() -> list[dict[str, object]]:
    dem = np.load(GEO / "dem.npy").astype(np.float32)
    bldg = dem >= 49.9
    active = ~bldg
    total_cells = int(dem.size)
    active_cells = int(active.sum())
    building_cells = int(bldg.sum())
    redistribution_factor = total_cells / active_cells if active_cells else np.nan
    rows = []
    for event in EVENTS:
        rainfall = np.load(DATA / event / "rainfall.npy").astype(np.float32)
        mean_all = float(rainfall.sum(axis=0).mean())
        mean_active_original = float(rainfall[:, active].sum(axis=0).mean())
        mean_building_original = float(rainfall[:, bldg].sum(axis=0).mean()) if building_cells else 0.0
        mean_active_after = float(rainfall.sum() / active_cells) if active_cells else np.nan
        rows.append(
            {
                "event": event,
                "total_cells": total_cells,
                "building_cells": building_cells,
                "active_cells": active_cells,
                "building_fraction_pct": float(building_cells / total_cells * 100.0),
                "redistribution_factor_on_active_cells": float(redistribution_factor),
                "mean_total_rainfall_all_cells_mm": mean_all,
                "mean_total_rainfall_active_original_mm": mean_active_original,
                "mean_total_rainfall_building_original_mm": mean_building_original,
                "mean_total_rainfall_active_after_redistribution_mm": mean_active_after,
                "active_rainfall_increase_pct": float((mean_active_after / mean_active_original - 1.0) * 100.0)
                if mean_active_original
                else np.nan,
            }
        )
    return rows


def main() -> int:
    args = parse_args()
    configure_paths(args)
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for event in EVENTS:
        rows.extend(event_summary(event))
        plot_event(event)
    write_csv(OUT / "hydrograph_timing_summary.csv", rows)
    rainfall_accounting = build_rainfall_accounting()
    write_csv(OUT / "rainfall_accounting_summary.csv", rainfall_accounting)
    plot_summary(rows)
    metadata_path = DATA / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    infiltration_mmh = metadata.get("infiltration_mmh")
    note = {
        "dataset": args.dataset_name,
        "infiltration_mmh": infiltration_mmh,
        "diagnosis": [
            f"ITZI surface-only uses no drainage and no explicit open 2D boundary; this dataset applies an effective infiltration rate of {infiltration_mmh} mm/h when metadata provides it.",
            "ITZI-SWMM connected adds pipe drainage, but current conceptual network capacity and coupling do not reproduce the MIKE falling limb for all events.",
            "The copied runner now applies rainfall frame 0 at t=0 and integrates the full 6 h event. Earlier archived runs may still contain the former one-frame timing delay and must not be mixed with this dataset.",
            "The current runner redistributes rainfall falling on building cells to active cells. Buildings account for about 22.13% of the current 20 m window, increasing active-cell rainfall by about 28.4%. This assumption should be calibrated or tested against a no-redistribution scenario.",
            "MIKE reference likely includes calibrated losses and a stronger/real drainage system; therefore hydrograph timing should be a separate calibration target before using ITZI-SWMM labels as strict physical truth.",
        ],
        "outputs": {
            "summary_csv": str(OUT / "hydrograph_timing_summary.csv"),
            "rainfall_accounting_csv": str(OUT / "rainfall_accounting_summary.csv"),
            "summary_figure": str(FIG / "hydrograph_timing_summary.png"),
            "event_figures": str(FIG),
        },
    }
    (OUT / "hydrograph_timing_diagnosis.json").write_text(json.dumps(note, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(note, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
