#!/usr/bin/env python3
"""Create publication-quality diagnostics for matched hydraulic scenarios."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import TwoSlopeNorm

from audit_reviewer_v3_network import sections
from publication_plot_style import add_panel_labels, configure_publication_style


configure_publication_style()
ROOT = Path(__file__).resolve().parents[1]
DEM_PATH = ROOT / "external_models" / "20260518-itzi-flood" / "test_cases" / "shenzhen_region1" / "input_data" / "geodata" / "region1_20m" / "dem.npy"
CELL_M = 20.0
CELL_AREA = 400.0


def network_geometry(path: Path, domain: str):
    data = sections(path)
    coordinates = {row[0]: (float(row[1]) / 1000.0, float(row[2]) / 1000.0) for row in data["COORDINATES"]}
    lines = []
    for row in data["CONDUITS"]:
        if row[1] in coordinates and row[2] in coordinates:
            lines.append([coordinates[row[1]], coordinates[row[2]]])
    if domain == "sub":
        lines = [line for line in lines if all(2.4 <= x <= 8.0 and 2.4 <= y <= 6.4 for x, y in line)]
    return lines


def extent(domain: str):
    return [0.0, 11.2, 0.0, 8.0] if domain == "full" else [2.4, 8.0, 2.4, 6.4]


def add_network(ax, lines, color="white", linewidth=0.18, alpha=0.65):
    if lines:
        ax.add_collection(LineCollection(lines, colors=color, linewidths=linewidth, alpha=alpha, zorder=3))


def load_event(event_dir: Path):
    metadata = json.loads((event_dir / "metadata.json").read_text(encoding="utf-8"))
    arrays = {
        "MIKE reference": np.load(event_dir / "h_mike_ref.npy"),
        "A: surface n=0.015": np.load(event_dir / "h_A_surface_n015.npy"),
        "B: matched roughness": np.load(event_dir / "h_B_surface_inlet_n012.npy"),
        "C: Itzi-SWMM": np.load(event_dir / "h_C_itzi_swmm.npy"),
    }
    dem = np.load(DEM_PATH)
    if metadata["domain"] == "sub":
        dem = dem[80:280, 120:400]
    return metadata, arrays, dem


def hydrographs(event_dir: Path, output: Path):
    metadata, arrays, dem = load_event(event_dir)
    active = dem < 49.9
    times = (np.arange(next(iter(arrays.values())).shape[0]) + 1) * 5.0 / 60.0
    colors = ["#333333", "#0072B2", "#E69F00", "#009E73"]
    linestyles = ["-", "-", "--", "-"]
    linewidths = [1.8, 2.1, 1.5, 1.9]
    zorders = [4, 2, 3, 5]
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.4), sharex=True)
    for (label, values), color, linestyle, linewidth, zorder in zip(
        arrays.items(), colors, linestyles, linewidths, zorders
    ):
        axes[0, 0].plot(
            times,
            values[:, active].sum(axis=1, dtype=np.float64) * CELL_AREA / 1e6,
            label=label,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            zorder=zorder,
        )
        axes[0, 1].plot(
            times, values[:, active].max(axis=1), label=label, color=color,
            linestyle=linestyle, linewidth=linewidth, zorder=zorder,
        )
        axes[1, 0].plot(
            times, np.count_nonzero(values[:, active] > 0.03, axis=1) * CELL_AREA / 1e6,
            label=label, color=color, linestyle=linestyle, linewidth=linewidth,
            zorder=zorder,
        )
    records = metadata["records"]
    record_time = np.asarray(records["C"]["time_h"])
    axes[1, 1].plot(record_time, np.asarray(records["C"]["surface_held_exchange_m3"]) / 1000.0, color="#D55E00", label="2D exchange integral")
    axes[1, 1].plot(record_time, np.asarray(records["C"]["swmm_net_exchange_m3"]) / 1000.0, color="#56B4E9", linestyle="--", label="SWMM-step integral")
    axes[0, 0].set_ylabel("Surface water volume ($10^6$ m$^3$)")
    axes[0, 1].set_ylabel("Maximum cell depth (m)")
    axes[1, 0].set_ylabel("Area deeper than 0.03 m (km$^2$)")
    axes[1, 1].set_ylabel("Cumulative net exchange ($10^3$ m$^3$)")
    for ax in axes[1]:
        ax.set_xlabel("Elapsed time (h)")
    axes[0, 0].legend(frameon=False, fontsize=7, ncol=2)
    axes[1, 1].legend(frameon=False, fontsize=7)
    add_panel_labels(axes.ravel())
    fig.tight_layout()
    for suffix in ["png", "pdf", "svg"]:
        fig.savefig(output / f"fig_physics_hydrographs.{suffix}")
    plt.close(fig)


def peak_maps(event_dir: Path, network: Path, output: Path):
    metadata, arrays, dem = load_event(event_dir)
    active = dem < 49.9
    map_extent = extent(metadata["domain"])
    lines = network_geometry(network, metadata["domain"])
    peaks = {label: values.max(axis=0) for label, values in arrays.items()}
    vmax = float(np.quantile(np.concatenate([values[active] for values in peaks.values()]), 0.995))
    b = arrays["B: matched roughness"]
    c = arrays["C: Itzi-SWMM"]
    difference = c.max(axis=0) - b.max(axis=0)
    roughness = arrays["B: matched roughness"].max(axis=0) - arrays["A: surface n=0.015"].max(axis=0)
    limit = max(0.01, float(np.quantile(np.abs(difference[active]), 0.995)))
    fig = plt.figure(figsize=(7.5, 5.0), constrained_layout=True)
    grid = fig.add_gridspec(2, 4, height_ratios=[1.0, 1.15])
    depth_axes = [fig.add_subplot(grid[0, index]) for index in range(4)]
    difference_axis = fig.add_subplot(grid[1, 0:2])
    roughness_axis = fig.add_subplot(grid[1, 2:4])
    all_axes = depth_axes + [difference_axis, roughness_axis]
    image = None
    for ax, (label, values) in zip(depth_axes, peaks.items()):
        image = ax.imshow(np.ma.masked_where(~active, values), cmap="Blues", vmin=0, vmax=vmax, extent=map_extent, origin="upper")
        add_network(ax, lines)
        ax.set_title(label, fontsize=8)
    diff_image = difference_axis.imshow(np.ma.masked_where(~active, difference), cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit), extent=map_extent, origin="upper")
    add_network(difference_axis, lines, color="black", alpha=0.35)
    difference_axis.set_title("Peak-map drainage effect, C - B", fontsize=8)
    rough_limit = max(0.001, float(np.quantile(np.abs(roughness[active]), 0.995)))
    rough_image = roughness_axis.imshow(np.ma.masked_where(~active, roughness), cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-rough_limit, vcenter=0, vmax=rough_limit), extent=map_extent, origin="upper")
    roughness_axis.set_title("Peak-map roughness effect, B - A", fontsize=8)
    for ax in all_axes:
        ax.set_xlabel("Easting (km)")
        ax.set_ylabel("Northing (km)")
    fig.colorbar(image, ax=depth_axes, label="Peak water depth (m)", orientation="horizontal", shrink=0.72, pad=0.08)
    fig.colorbar(diff_image, ax=difference_axis, label="Depth difference (m)", orientation="horizontal", shrink=0.78, pad=0.10)
    fig.colorbar(rough_image, ax=roughness_axis, label="Depth difference (m)", orientation="horizontal", shrink=0.78, pad=0.10)
    add_panel_labels(all_axes)
    for suffix in ["png", "pdf", "svg"]:
        fig.savefig(output / f"fig_physics_peak_maps.{suffix}")
    plt.close(fig)


def temporal_difference_maps(event_dir: Path, network: Path, output: Path):
    metadata, arrays, dem = load_event(event_dir)
    active = dem < 49.9
    lines = network_geometry(network, metadata["domain"])
    map_extent = extent(metadata["domain"])
    residual = arrays["C: Itzi-SWMM"] - arrays["B: matched roughness"]
    indices = [23, 35, 47, 71]
    limit = max(0.01, float(np.quantile(np.abs(residual[:, active]), 0.995)))
    fig, axes = plt.subplots(1, 4, figsize=(7.5, 2.35), constrained_layout=True)
    image = None
    for ax, index in zip(axes, indices):
        image = ax.imshow(np.ma.masked_where(~active, residual[index]), cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit), extent=map_extent, origin="upper")
        add_network(ax, lines, color="black", alpha=0.35)
        ax.set_title(f"t = {(index + 1) / 12:.1f} h")
        ax.set_xlabel("Easting (km)")
    axes[0].set_ylabel("Northing (km)")
    fig.colorbar(image, ax=axes, label="Drainage effect C - B (m)", shrink=0.82)
    add_panel_labels(axes)
    for suffix in ["png", "pdf", "svg"]:
        fig.savefig(output / f"fig_physics_difference_times.{suffix}")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("event_dir", type=Path)
    parser.add_argument("--network", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.event_dir / "figures"
    output.mkdir(parents=True, exist_ok=True)
    hydrographs(args.event_dir, output)
    peak_maps(args.event_dir, args.network, output)
    temporal_difference_maps(args.event_dir, args.network, output)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
