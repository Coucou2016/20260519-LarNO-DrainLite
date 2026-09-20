"""Audit which drainage label source produces the visible drainage effect.

This script compares:
1. The original project ITZI + Pipes/orifice results in
   E:/Projects/20260518-itzi-flood/test_cases/shenzhen_region1/output/event*_itzi.npz
2. The current Drainage V1 sink labels in LarNO-main/benchmark/urbanflood/flood/region1_20m_drainage_v1
3. The current official ITZI-SWMM coupled labels, where available.

The purpose is to prevent mixing the conceptual pipe/sink label with the
formal SWMM-coupled label in reports and manuscript text.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OLD_OUTPUT = Path(
    os.environ.get(
        "ITZI_LEGACY_OUTPUT",
        ROOT / "external_models" / "20260518-itzi-flood" / "test_cases" / "shenzhen_region1" / "output",
    )
)
DATASET = ROOT / "LarNO-main" / "benchmark" / "urbanflood" / "flood" / "region1_20m_drainage_v1"
OUT = ROOT / "extended_study" / "output" / "drainage_effect_source_audit"
FIG = OUT / "figures"

EVENTS = ["event1", "event20", "event65", "event66", "event67", "event68", "event69", "event70"]
CELL_AREA_M2 = 20.0 * 20.0


def load_last(path: Path) -> np.ndarray:
    arr = np.load(path).astype(np.float32)
    if arr.ndim == 3:
        return arr[-1]
    return arr


def flooded_cells(arr: np.ndarray, threshold: float = 0.03) -> int:
    return int(np.count_nonzero(arr > threshold))


def pct(value: float) -> float:
    return 100.0 * value


def audit_rows() -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for event in EVENTS:
        old = np.load(OLD_OUTPUT / f"{event}_itzi.npz", allow_pickle=True)
        old_surface = old["h_itzi_surf"].astype(np.float32)
        old_pipe = old["h_itzi_pipe"].astype(np.float32)

        current_surface = load_last(DATASET / event / "h_itzi_surface.npy")
        current_sink = load_last(DATASET / event / "h_itzi_sink.npy")

        sink_reduction = np.maximum(current_surface - current_sink, 0.0)
        surface_flooded = flooded_cells(current_surface)
        sink_flooded = flooded_cells(current_sink)
        area_reduction_pct = pct((surface_flooded - sink_flooded) / surface_flooded) if surface_flooded else 0.0

        row: dict[str, float | str] = {
            "event": event,
            "old_pipe_peak_m": float(np.nanmax(old_pipe)),
            "current_sink_peak_m": float(np.nanmax(current_sink)),
            "old_current_pipe_sink_max_abs_diff_m": float(np.nanmax(np.abs(old_pipe - current_sink))),
            "old_surface_peak_m": float(np.nanmax(old_surface)),
            "current_surface_peak_m": float(np.nanmax(current_surface)),
            "old_current_surface_max_abs_diff_m": float(np.nanmax(np.abs(old_surface - current_surface))),
            "sink_peak_reduction_m": float(np.nanmax(current_surface) - np.nanmax(current_sink)),
            "sink_mean_reduction_mm": float(np.nanmean(sink_reduction) * 1000.0),
            "sink_max_local_reduction_m": float(np.nanmax(sink_reduction)),
            "surface_flooded_cells_3cm": surface_flooded,
            "sink_flooded_cells_3cm": sink_flooded,
            "sink_area_reduction_pct_3cm": area_reduction_pct,
            "official_swmm_available": "no",
            "official_swmm_peak_reduction_m": np.nan,
            "official_swmm_mean_abs_diff_mm": np.nan,
            "official_swmm_fraction_gt_1mm_pct": np.nan,
        }

        official_surface_path = DATASET / event / "h_itzi_swmm_official_surface.npy"
        official_coupled_path = DATASET / event / "h_itzi_swmm_official.npy"
        if official_surface_path.exists() and official_coupled_path.exists():
            official_surface = np.load(official_surface_path).astype(np.float32)
            official_coupled = np.load(official_coupled_path).astype(np.float32)
            diff = official_surface - official_coupled
            row.update(
                {
                    "official_swmm_available": "yes",
                    "official_swmm_peak_reduction_m": float(np.nanmax(official_surface) - np.nanmax(official_coupled)),
                    "official_swmm_mean_abs_diff_mm": float(np.nanmean(np.abs(diff)) * 1000.0),
                    "official_swmm_fraction_gt_1mm_pct": float(np.nanmean(np.abs(diff) > 0.001) * 100.0),
                }
            )

        rows.append(row)
    return rows


def write_csv(rows: list[dict[str, float | str]]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    csv_path = OUT / "drainage_effect_source_audit.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_summary(rows: list[dict[str, float | str]]) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    events = [str(r["event"]) for r in rows]
    x = np.arange(len(events))
    width = 0.38

    sink_peak = np.array([float(r["sink_peak_reduction_m"]) for r in rows]) * 1000.0
    swmm_peak = np.array(
        [float(r["official_swmm_peak_reduction_m"]) if r["official_swmm_available"] == "yes" else np.nan for r in rows]
    ) * 1000.0

    sink_area = np.array([float(r["sink_area_reduction_pct_3cm"]) for r in rows])
    sink_mean = np.array([float(r["sink_mean_reduction_mm"]) for r in rows])
    swmm_mean = np.array(
        [float(r["official_swmm_mean_abs_diff_mm"]) if r["official_swmm_available"] == "yes" else np.nan for r in rows]
    )

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.unicode_minus": False,
            "figure.dpi": 160,
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.0), constrained_layout=True)

    ax = axes[0, 0]
    ax.bar(x - width / 2, sink_peak, width, label="ITZI + Pipes/sink")
    ax.bar(x + width / 2, swmm_peak, width, label="Official ITZI-SWMM")
    ax.set_title("Peak-depth reduction by drainage label source")
    ax.set_ylabel("Peak reduction (mm)")
    ax.set_xticks(x)
    ax.set_xticklabels(events, rotation=35, ha="right")
    ax.legend()

    ax = axes[0, 1]
    ax.bar(x, sink_area, color="#3b82f6")
    ax.set_title("Flooded-area reduction in ITZI + Pipes/sink")
    ax.set_ylabel("Area reduction at h > 0.03 m (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(events, rotation=35, ha="right")

    ax = axes[1, 0]
    ax.bar(x - width / 2, sink_mean, width, label="ITZI + Pipes/sink")
    ax.bar(x + width / 2, swmm_mean, width, label="Official ITZI-SWMM")
    ax.set_title("Mean absolute drainage signal")
    ax.set_ylabel("Mean signal (mm)")
    ax.set_xticks(x)
    ax.set_xticklabels(events, rotation=35, ha="right")
    ax.legend()

    ax = axes[1, 1]
    pipe_sink_diff = np.array([float(r["old_current_pipe_sink_max_abs_diff_m"]) for r in rows]) * 1000.0
    surf_diff = np.array([float(r["old_current_surface_max_abs_diff_m"]) for r in rows]) * 1000.0
    ax.plot(events, pipe_sink_diff, marker="o", label="Old pipe vs current sink")
    ax.plot(events, surf_diff, marker="s", label="Old surface vs current surface")
    ax.set_title("Original-project output copied into current dataset")
    ax.set_ylabel("Maximum absolute final-frame difference (mm)")
    ax.tick_params(axis="x", rotation=35)
    ax.legend()

    fig.suptitle("Drainage-effect source audit: large sink effect vs weak official SWMM effect", fontsize=15, weight="bold")
    fig.savefig(FIG / "drainage_effect_source_audit_summary.png", bbox_inches="tight")
    plt.close(fig)


def plot_event_maps(event: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    surface = load_last(DATASET / event / "h_itzi_surface.npy")
    sink = load_last(DATASET / event / "h_itzi_sink.npy")
    reduction = surface - sink

    maps = [
        ("Surface-only final depth", surface, "viridis"),
        ("ITZI + Pipes/sink final depth", sink, "viridis"),
        ("Pipes/sink reduction", reduction, "coolwarm"),
    ]

    official_surface_path = DATASET / event / "h_itzi_swmm_official_surface.npy"
    official_coupled_path = DATASET / event / "h_itzi_swmm_official.npy"
    if official_surface_path.exists() and official_coupled_path.exists():
        official_surface = np.load(official_surface_path).astype(np.float32)[-1]
        official_coupled = np.load(official_coupled_path).astype(np.float32)[-1]
        maps.extend(
            [
                ("Official SWMM coupled final depth", official_coupled, "viridis"),
                ("Official SWMM reduction", official_surface - official_coupled, "coolwarm"),
            ]
        )

    n = len(maps)
    fig, axes = plt.subplots(1, n, figsize=(4.0 * n, 4.2), constrained_layout=True)
    if n == 1:
        axes = [axes]
    vmax = float(max(np.nanmax(surface), np.nanmax(sink)))
    for ax, (title, arr, cmap) in zip(axes, maps):
        if "reduction" in title.lower():
            lim = max(0.05, float(np.nanmax(np.abs(arr))))
            im = ax.imshow(arr, cmap=cmap, vmin=-lim, vmax=lim)
        else:
            im = ax.imshow(arr, cmap=cmap, vmin=0, vmax=vmax)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.suptitle(f"{event}: drainage label source spatial comparison", fontsize=14, weight="bold")
    fig.savefig(FIG / f"{event}_drainage_label_source_maps.png", bbox_inches="tight")
    plt.close(fig)


def write_markdown(rows: list[dict[str, float | str]]) -> None:
    sink_peak_mean = np.nanmean([float(r["sink_peak_reduction_m"]) for r in rows]) * 1000.0
    sink_area_mean = np.nanmean([float(r["sink_area_reduction_pct_3cm"]) for r in rows])
    swmm_rows = [r for r in rows if r["official_swmm_available"] == "yes"]
    swmm_peak_mean = np.nanmean([float(r["official_swmm_peak_reduction_m"]) for r in swmm_rows]) * 1000.0 if swmm_rows else np.nan
    swmm_abs_mean = np.nanmean([float(r["official_swmm_mean_abs_diff_mm"]) for r in swmm_rows]) if swmm_rows else np.nan

    text = f"""# Drainage Effect Source Audit

This audit separates two drainage labels that should not be mixed in the report:

- `ITZI + Pipes/sink`: the original visible drainage-effect branch from `run_itzi_shenzhen.py`.
- `Official ITZI-SWMM`: the formal coupled branch stored as `h_itzi_swmm_official.npy`.

Main finding:

- The visible drainage effect remembered from the original project is real, but it comes from `ITZI + Pipes/sink`.
- Across the eight checked events, `ITZI + Pipes/sink` reduces the final-frame peak by {sink_peak_mean:.1f} mm on average and reduces the flooded area above 0.03 m by {sink_area_mean:.1f}% on average.
- The current official ITZI-SWMM label is much weaker in available events, with average peak reduction {swmm_peak_mean:.1f} mm and mean absolute signal {swmm_abs_mean:.3f} mm.
- Therefore, a report figure based on `h_itzi_swmm_official.npy` can show nearly overlapping time-series curves even though the original `ITZI + Pipes/sink` branch has a strong drainage effect.

Outputs:

- `drainage_effect_source_audit.csv`
- `figures/drainage_effect_source_audit_summary.png`
- `figures/event1_drainage_label_source_maps.png`
- `figures/event65_drainage_label_source_maps.png`
"""
    (OUT / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    rows = audit_rows()
    write_csv(rows)
    plot_summary(rows)
    plot_event_maps("event1")
    plot_event_maps("event65")
    write_markdown(rows)
    print(f"Wrote audit to {OUT}")


if __name__ == "__main__":
    main()
