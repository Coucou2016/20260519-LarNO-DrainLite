"""Plot drainage/no-drainage process curves and spatial overlays.

The figures separate two drainage branches:

1. ITZI + Pipes/sink:
   `h_itzi_surface.npy` versus `h_itzi_sink.npy`.
   This is the branch with the visible drainage effect.

2. Official ITZI-SWMM coupling:
   `h_itzi_swmm_official_surface.npy` versus `h_itzi_swmm_official.npy`.
   This uses ITZI drainage coupling with PySWMM where available.

The script writes figures and a metrics CSV under
`extended_study/output/drainage_process_spatial_audit`.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FLOOD = ROOT / "LarNO-main" / "benchmark" / "urbanflood" / "flood" / "region1_20m_drainage_v1"
GEO = ROOT / "LarNO-main" / "benchmark" / "urbanflood" / "geodata" / "region1_20m_drainage_v1"
OUT = ROOT / "extended_study" / "output" / "drainage_process_spatial_audit"
FIG = OUT / "figures"

EVENTS = ["event1", "event20", "event65", "event66", "event67", "event68", "event69", "event70"]
CELL_M = 20.0
CELL_AREA = CELL_M * CELL_M
FLOOD_THRESHOLD_M = 0.03


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.unicode_minus": False,
            "figure.dpi": 160,
            "savefig.dpi": 180,
        }
    )


def load_event(event: str) -> dict[str, np.ndarray]:
    event_dir = FLOOD / event
    data = {
        "mike": np.load(event_dir / "h_mike_ref.npy").astype(np.float32),
        "surface": np.load(event_dir / "h_itzi_surface.npy").astype(np.float32),
        "sink": np.load(event_dir / "h_itzi_sink.npy").astype(np.float32),
        "rainfall": np.load(event_dir / "rainfall.npy").astype(np.float32),
    }
    official_surface = event_dir / "h_itzi_swmm_official_surface.npy"
    official_coupled = event_dir / "h_itzi_swmm_official.npy"
    if official_surface.exists() and official_coupled.exists():
        data["official_surface"] = np.load(official_surface).astype(np.float32)
        data["official_swmm"] = np.load(official_coupled).astype(np.float32)
    return data


def load_static() -> dict[str, np.ndarray]:
    return {
        "dem": np.load(GEO / "dem.npy").astype(np.float32),
        "pipe": np.load(GEO / "pipe_mask.npy").astype(np.float32),
        "inlet": np.load(GEO / "drain_inlet_mask.npy").astype(np.float32),
        "outfall": np.load(GEO / "drain_outfall_mask.npy").astype(np.float32),
        "diameter": np.load(GEO / "pipe_diameter.npy").astype(np.float32),
        "slope": np.load(GEO / "pipe_slope.npy").astype(np.float32),
        "capacity": np.load(GEO / "pipe_capacity.npy").astype(np.float32),
        "cover": np.load(GEO / "pipe_cover_depth.npy").astype(np.float32),
        "distance": np.load(GEO / "distance_to_outfall.npy").astype(np.float32),
    }


def extent_for(arr: np.ndarray) -> list[float]:
    h, w = arr.shape[-2:]
    return [0, w * CELL_M / 1000.0, h * CELL_M / 1000.0, 0]


def volume_series(h: np.ndarray) -> np.ndarray:
    return h.sum(axis=(1, 2)) * CELL_AREA


def peak_series(h: np.ndarray) -> np.ndarray:
    return h.max(axis=(1, 2))


def flooded_area_series(h: np.ndarray) -> np.ndarray:
    return (h > FLOOD_THRESHOLD_M).sum(axis=(1, 2)) * CELL_AREA / 1_000_000.0


def mean_abs_diff_mm(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)) * 1000.0)


def overlay_network(ax: plt.Axes, static: dict[str, np.ndarray], alpha: float = 0.75) -> None:
    extent = extent_for(static["dem"])
    pipe = static["pipe"] > 0
    inlet = static["inlet"] > 0
    outfall = static["outfall"] > 0

    if pipe.any():
        ax.contour(pipe.astype(float), levels=[0.5], colors="#111827", linewidths=0.45, alpha=alpha, extent=extent)
    if inlet.any():
        rr, cc = np.where(inlet)
        ax.scatter((cc + 0.5) * CELL_M / 1000.0, (rr + 0.5) * CELL_M / 1000.0, s=8, c="#06b6d4", marker="o", edgecolors="none", alpha=0.9)
    if outfall.any():
        rr, cc = np.where(outfall)
        ax.scatter((cc + 0.5) * CELL_M / 1000.0, (rr + 0.5) * CELL_M / 1000.0, s=36, c="#e11d48", marker="*", edgecolors="white", linewidths=0.4, alpha=1.0)


def plot_static_overlays(static: dict[str, np.ndarray]) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    dem = static["dem"]
    extent = extent_for(dem)
    dem_plot = np.where(dem >= 49.9, np.nan, dem)

    fig, ax = plt.subplots(figsize=(8.5, 7.2), constrained_layout=True)
    im = ax.imshow(dem_plot, cmap="terrain", extent=extent, origin="upper")
    overlay_network(ax, static)
    ax.set_title("DEM with drainage network, inlets and outfall")
    ax.set_xlabel("x distance from upper-left corner (km)")
    ax.set_ylabel("y distance from upper-left corner (km)")
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("DEM elevation excluding building wall cells (m)")
    ax.text(
        0.02,
        0.98,
        "black: pipe grid / cyan: inlet / red star: outfall",
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#cbd5e1", "alpha": 0.85},
        fontsize=9,
    )
    fig.savefig(FIG / "static_dem_network_overlay.png", bbox_inches="tight")
    plt.close(fig)

    maps = [
        ("Pipe mask", static["pipe"], "gray", 0, 1),
        ("Inlet mask", static["inlet"], "Blues", 0, 1),
        ("Outfall mask", static["outfall"], "Reds", 0, 1),
        ("Pipe diameter", np.where(static["diameter"] > 0, static["diameter"], np.nan), "viridis", None, None),
        ("Pipe slope", np.where(static["slope"] > 0, static["slope"], np.nan), "magma", None, None),
        ("Pipe capacity", np.where(static["capacity"] > 0, static["capacity"], np.nan), "plasma", None, None),
        ("Pipe cover depth", np.where(static["cover"] > 0, static["cover"], np.nan), "cividis", None, None),
        ("Distance to outfall", static["distance"], "turbo", None, None),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(16, 7.8), constrained_layout=True)
    for ax, (title, arr, cmap, vmin, vmax) in zip(axes.ravel(), maps):
        im = ax.imshow(arr, cmap=cmap, extent=extent, origin="upper", vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.suptitle("Rasterized drainage features aligned to 20 m DEM grid", fontsize=14, weight="bold")
    fig.savefig(FIG / "static_drainage_feature_maps.png", bbox_inches="tight")
    plt.close(fig)


def plot_event_timeseries(event: str, data: dict[str, np.ndarray]) -> None:
    t = np.arange(1, data["surface"].shape[0] + 1) * 5.0 / 60.0

    surface = data["surface"]
    sink = data["sink"]
    mike = data["mike"]
    has_swmm = "official_swmm" in data

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.2), constrained_layout=True)
    ax = axes[0, 0]
    ax.plot(t, peak_series(mike), color="#111827", lw=1.7, label="MIKE reference")
    ax.plot(t, peak_series(surface), color="#2563eb", lw=1.6, label="ITZI surface-only")
    ax.plot(t, peak_series(sink), color="#16a34a", lw=1.6, label="ITZI + Pipes/sink")
    if has_swmm:
        ax.plot(t, peak_series(data["official_swmm"]), color="#dc2626", lw=1.4, ls="--", label="Official ITZI-SWMM")
    ax.set_title("Maximum water depth")
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Depth (m)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.plot(t, volume_series(mike) / 1000.0, color="#111827", lw=1.7, label="MIKE reference")
    ax.plot(t, volume_series(surface) / 1000.0, color="#2563eb", lw=1.6, label="ITZI surface-only")
    ax.plot(t, volume_series(sink) / 1000.0, color="#16a34a", lw=1.6, label="ITZI + Pipes/sink")
    if has_swmm:
        ax.plot(t, volume_series(data["official_swmm"]) / 1000.0, color="#dc2626", lw=1.4, ls="--", label="Official ITZI-SWMM")
    ax.set_title("Surface water volume")
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Volume (10^3 m3)")
    ax.grid(alpha=0.25)

    ax = axes[1, 0]
    ax.plot(t, flooded_area_series(mike), color="#111827", lw=1.7, label="MIKE reference")
    ax.plot(t, flooded_area_series(surface), color="#2563eb", lw=1.6, label="ITZI surface-only")
    ax.plot(t, flooded_area_series(sink), color="#16a34a", lw=1.6, label="ITZI + Pipes/sink")
    if has_swmm:
        ax.plot(t, flooded_area_series(data["official_swmm"]), color="#dc2626", lw=1.4, ls="--", label="Official ITZI-SWMM")
    ax.set_title("Flooded area above 0.03 m")
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Area (km2)")
    ax.grid(alpha=0.25)

    ax = axes[1, 1]
    sink_peak_reduction = (peak_series(surface) - peak_series(sink)) * 1000.0
    sink_volume_reduction = (volume_series(surface) - volume_series(sink)) / 1000.0
    ax.plot(t, sink_peak_reduction, color="#16a34a", lw=1.8, label="Pipes/sink peak reduction (mm)")
    ax2 = ax.twinx()
    ax2.plot(t, sink_volume_reduction, color="#059669", lw=1.5, ls=":", label="Pipes/sink volume reduction (10^3 m3)")
    if has_swmm:
        official_peak_reduction = (peak_series(data["official_surface"]) - peak_series(data["official_swmm"])) * 1000.0
        ax.plot(t, official_peak_reduction, color="#dc2626", lw=1.5, ls="--", label="Official SWMM peak reduction (mm)")
    ax.set_title("Drainage effect difference curves")
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Peak reduction (mm)")
    ax2.set_ylabel("Volume reduction (10^3 m3)")
    ax.grid(alpha=0.25)
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, fontsize=8, loc="best")

    fig.suptitle(f"{event}: time process comparison of no-drainage and drainage scenarios", fontsize=14, weight="bold")
    fig.savefig(FIG / f"{event}_drainage_timeseries.png", bbox_inches="tight")
    plt.close(fig)


def plot_event_spatial(event: str, data: dict[str, np.ndarray], static: dict[str, np.ndarray]) -> None:
    extent = extent_for(static["dem"])
    surface_peak = data["surface"].max(axis=0)
    sink_peak = data["sink"].max(axis=0)
    mike_peak = data["mike"].max(axis=0)
    sink_red_peak = surface_peak - sink_peak
    vmax = float(max(np.nanmax(surface_peak), np.nanmax(sink_peak), np.nanmax(mike_peak)))

    maps: list[tuple[str, np.ndarray, str, float | None, float | None, bool]] = [
        ("MIKE reference peak", mike_peak, "viridis", 0, vmax, False),
        ("ITZI surface-only peak", surface_peak, "viridis", 0, vmax, True),
        ("ITZI + Pipes/sink peak", sink_peak, "viridis", 0, vmax, True),
        ("Surface - Pipes/sink peak", sink_red_peak, "coolwarm", -max(0.15, float(np.nanmax(np.abs(sink_red_peak)))), max(0.15, float(np.nanmax(np.abs(sink_red_peak)))), True),
    ]
    if "official_swmm" in data:
        official_surface_peak = data["official_surface"].max(axis=0)
        official_peak = data["official_swmm"].max(axis=0)
        official_red_peak = official_surface_peak - official_peak
        lim = max(0.01, float(np.nanmax(np.abs(official_red_peak))))
        maps.extend(
            [
                ("Official ITZI-SWMM peak", official_peak, "viridis", 0, vmax, True),
                ("Official surface - SWMM peak", official_red_peak, "coolwarm", -lim, lim, True),
            ]
        )

    fig, axes = plt.subplots(2, 3, figsize=(15.8, 9.6), constrained_layout=True)
    axes_flat = axes.ravel()
    for ax, spec in zip(axes_flat, maps):
        title, arr, cmap, vmin, vmax_map, add_network = spec
        im = ax.imshow(arr, cmap=cmap, extent=extent, origin="upper", vmin=vmin, vmax=vmax_map)
        if add_network:
            overlay_network(ax, static, alpha=0.65)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("x (km)")
        ax.set_ylabel("y (km)")
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cb.set_label("m")
    for ax in axes_flat[len(maps) :]:
        ax.axis("off")
    fig.suptitle(f"{event}: peak-depth spatial distribution with drainage network overlay", fontsize=14, weight="bold")
    fig.savefig(FIG / f"{event}_drainage_spatial_maps.png", bbox_inches="tight")
    plt.close(fig)


def write_metrics(rows: list[dict[str, float | str]]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "drainage_process_metrics.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def collect_metrics(event: str, data: dict[str, np.ndarray]) -> dict[str, float | str]:
    surface = data["surface"]
    sink = data["sink"]
    row: dict[str, float | str] = {
        "event": event,
        "surface_peak_m": float(peak_series(surface).max()),
        "sink_peak_m": float(peak_series(sink).max()),
        "sink_peak_reduction_mm": float((peak_series(surface).max() - peak_series(sink).max()) * 1000.0),
        "sink_mean_abs_diff_mm": mean_abs_diff_mm(surface, sink),
        "sink_final_volume_reduction_m3": float(volume_series(surface)[-1] - volume_series(sink)[-1]),
        "sink_max_volume_reduction_m3": float(np.max(volume_series(surface) - volume_series(sink))),
        "sink_final_flooded_area_reduction_km2": float(flooded_area_series(surface)[-1] - flooded_area_series(sink)[-1]),
        "official_available": "no",
        "official_surface_peak_m": np.nan,
        "official_swmm_peak_m": np.nan,
        "official_peak_reduction_mm": np.nan,
        "official_mean_abs_diff_mm": np.nan,
        "official_final_volume_reduction_m3": np.nan,
        "official_max_volume_reduction_m3": np.nan,
        "official_fraction_abs_diff_gt_1mm_pct": np.nan,
    }
    if "official_swmm" in data:
        official_surface = data["official_surface"]
        official = data["official_swmm"]
        row.update(
            {
                "official_available": "yes",
                "official_surface_peak_m": float(peak_series(official_surface).max()),
                "official_swmm_peak_m": float(peak_series(official).max()),
                "official_peak_reduction_mm": float((peak_series(official_surface).max() - peak_series(official).max()) * 1000.0),
                "official_mean_abs_diff_mm": mean_abs_diff_mm(official_surface, official),
                "official_final_volume_reduction_m3": float(volume_series(official_surface)[-1] - volume_series(official)[-1]),
                "official_max_volume_reduction_m3": float(np.max(volume_series(official_surface) - volume_series(official))),
                "official_fraction_abs_diff_gt_1mm_pct": float(np.mean(np.abs(official_surface - official) > 0.001) * 100.0),
            }
        )
    return row


def plot_summary(rows: list[dict[str, float | str]]) -> None:
    events = [str(r["event"]) for r in rows]
    x = np.arange(len(events))
    width = 0.38
    sink_peak = np.array([float(r["sink_peak_reduction_mm"]) for r in rows])
    official_peak = np.array([float(r["official_peak_reduction_mm"]) for r in rows])
    sink_mean = np.array([float(r["sink_mean_abs_diff_mm"]) for r in rows])
    official_mean = np.array([float(r["official_mean_abs_diff_mm"]) for r in rows])
    sink_vol = np.array([float(r["sink_max_volume_reduction_m3"]) for r in rows]) / 1000.0
    official_vol = np.array([float(r["official_max_volume_reduction_m3"]) for r in rows]) / 1000.0

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    ax = axes[0]
    ax.bar(x - width / 2, sink_peak, width, label="ITZI + Pipes/sink")
    ax.bar(x + width / 2, official_peak, width, label="Official ITZI-SWMM")
    ax.set_title("Maximum peak-depth reduction")
    ax.set_ylabel("mm")
    ax.set_xticks(x)
    ax.set_xticklabels(events, rotation=35, ha="right")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.bar(x - width / 2, sink_mean, width, label="ITZI + Pipes/sink")
    ax.bar(x + width / 2, official_mean, width, label="Official ITZI-SWMM")
    ax.set_title("Mean absolute time-space difference")
    ax.set_ylabel("mm")
    ax.set_xticks(x)
    ax.set_xticklabels(events, rotation=35, ha="right")
    ax.legend(fontsize=8)

    ax = axes[2]
    ax.bar(x - width / 2, sink_vol, width, label="ITZI + Pipes/sink")
    ax.bar(x + width / 2, official_vol, width, label="Official ITZI-SWMM")
    ax.set_title("Maximum surface-volume reduction")
    ax.set_ylabel("10^3 m3")
    ax.set_xticks(x)
    ax.set_xticklabels(events, rotation=35, ha="right")
    ax.legend(fontsize=8)

    fig.suptitle("Drainage/no-drainage effect summary across events", fontsize=14, weight="bold")
    fig.savefig(FIG / "all_events_drainage_effect_summary.png", bbox_inches="tight")
    plt.close(fig)


def write_readme(rows: list[dict[str, float | str]]) -> None:
    sink_peak = np.mean([float(r["sink_peak_reduction_mm"]) for r in rows])
    sink_mean = np.mean([float(r["sink_mean_abs_diff_mm"]) for r in rows])
    official_peak = np.mean([float(r["official_peak_reduction_mm"]) for r in rows])
    official_mean = np.mean([float(r["official_mean_abs_diff_mm"]) for r in rows])
    text = f"""# Drainage Process And Spatial Audit

This package replots ITZI drainage/no-drainage process curves and spatial maps.

Branches:

- `ITZI surface-only`: no drainage.
- `ITZI + Pipes/sink`: conceptual pipe/inlet drainage sink. This branch has the visible drainage effect.
- `Official ITZI-SWMM`: ITZI drainage coupling with PySWMM/Dynamic Wave where available.

Across the eight checked events:

- `ITZI + Pipes/sink` average maximum peak-depth reduction: {sink_peak:.1f} mm.
- `ITZI + Pipes/sink` average mean time-space difference: {sink_mean:.3f} mm.
- `Official ITZI-SWMM` average maximum peak-depth reduction: {official_peak:.1f} mm.
- `Official ITZI-SWMM` average mean time-space difference: {official_mean:.3f} mm.

Key figures:

- `figures/static_dem_network_overlay.png`
- `figures/static_drainage_feature_maps.png`
- `figures/all_events_drainage_effect_summary.png`
- `figures/event*_drainage_timeseries.png`
- `figures/event*_drainage_spatial_maps.png`
"""
    (OUT / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    setup_style()
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    static = load_static()
    plot_static_overlays(static)

    rows: list[dict[str, float | str]] = []
    for event in EVENTS:
        data = load_event(event)
        rows.append(collect_metrics(event, data))
        plot_event_timeseries(event, data)
        plot_event_spatial(event, data, static)

    write_metrics(rows)
    plot_summary(rows)
    write_readme(rows)
    print(f"Wrote drainage process and spatial audit to {OUT}")


if __name__ == "__main__":
    main()
