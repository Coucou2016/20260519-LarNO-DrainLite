#!/usr/bin/env python3
"""Plot the audited road-derived drainage network in publication style."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from audit_reviewer_v3_network import sections
from publication_plot_style import add_panel_labels, configure_publication_style


configure_publication_style()
ROOT = Path(__file__).resolve().parents[1]
DEM_PATH = (
    ROOT / "external_models" / "20260518-itzi-flood" / "test_cases"
    / "shenzhen_region1" / "input_data" / "geodata" / "region1_20m" / "dem.npy"
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("network", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    data = sections(args.network)
    dem = np.load(DEM_PATH).astype(np.float32)
    active = dem < 49.9
    coordinates = {
        row[0]: (float(row[1]) / 1000.0, float(row[2]) / 1000.0)
        for row in data["COORDINATES"]
    }
    junctions = {
        row[0]: {"invert": float(row[1]), "cover": float(row[2])}
        for row in data["JUNCTIONS"]
    }
    outfalls = {row[0]: float(row[1]) for row in data["OUTFALLS"]}
    invert = {node: values["invert"] for node, values in junctions.items()}
    invert.update(outfalls)
    diameter = {row[0]: float(row[2]) for row in data["XSECTIONS"]}
    lines = []
    diameters = []
    slopes = []
    lengths = []
    link_covers = []
    degree = {node: 0 for node in junctions}
    for row in data["CONDUITS"]:
        link, first, second = row[0], row[1], row[2]
        length = float(row[3])
        slope = (
            invert[first] + float(row[5]) - invert[second] - float(row[6])
        ) / length
        if first in coordinates and second in coordinates:
            lines.append([coordinates[first], coordinates[second]])
            diameters.append(diameter[link])
        slopes.append(slope)
        lengths.append(length)
        endpoint_covers = [
            junctions[node]["cover"] for node in [first, second] if node in junctions
        ]
        link_covers.append(float(np.mean(endpoint_covers)) if endpoint_covers else np.nan)
        if first in degree:
            degree[first] += 1
        if second in degree:
            degree[second] += 1
    covers = np.asarray([value["cover"] for value in junctions.values()])
    slopes = np.asarray(slopes)
    lengths = np.asarray(lengths)
    link_covers = np.asarray(link_covers)
    diameters = np.asarray(diameters)
    node_xy = np.asarray([coordinates[node] for node in junctions if node in coordinates])
    node_degree = np.asarray([degree[node] for node in junctions if node in coordinates])

    fig = plt.figure(figsize=(7.5, 6.4), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.25, 1.0])
    ax_dem = fig.add_subplot(grid[0, 0])
    ax_pipe = fig.add_subplot(grid[0, 1])
    ax_hist = fig.add_subplot(grid[1, 0])
    ax_box = fig.add_subplot(grid[1, 1])
    axes = [ax_dem, ax_pipe, ax_hist, ax_box]

    terrain = np.ma.masked_where(~active, dem)
    image = ax_dem.imshow(
        terrain,
        cmap="terrain",
        origin="upper",
        extent=[0, 11.2, 0, 8.0],
        vmin=float(np.quantile(dem[active], 0.01)),
        vmax=float(np.quantile(dem[active], 0.99)),
    )
    ax_dem.add_collection(
        LineCollection(lines, colors="#202020", linewidths=0.22, alpha=0.72)
    )
    ax_dem.scatter(node_xy[:, 0], node_xy[:, 1], s=0.45, c="#c43c39", alpha=0.65)
    ax_dem.set_title("Terrain and gravity-consistent network")
    fig.colorbar(image, ax=ax_dem, label="Ground elevation (m)", shrink=0.82)

    collection = LineCollection(
        lines,
        array=diameters,
        cmap="viridis",
        linewidths=0.55,
        clim=(float(diameters.min()), float(diameters.max())),
    )
    ax_pipe.add_collection(collection)
    point = ax_pipe.scatter(
        node_xy[:, 0], node_xy[:, 1], c=node_degree, cmap="magma", s=1.1,
        vmin=1, vmax=float(np.quantile(node_degree, 0.99)), zorder=3,
    )
    ax_pipe.set_xlim(0, 11.2)
    ax_pipe.set_ylim(0, 8.0)
    ax_pipe.set_aspect("equal")
    ax_pipe.set_title("Pipe diameter and junction degree")
    fig.colorbar(collection, ax=ax_pipe, label="Pipe diameter (m)", shrink=0.82)
    fig.colorbar(point, ax=ax_pipe, label="Junction degree", shrink=0.82)

    bins = np.geomspace(max(slopes.min(), 1e-5), slopes.max(), 30)
    ax_hist.hist(slopes, bins=bins, color="#0072B2", alpha=0.78, edgecolor="white", linewidth=0.25)
    ax_hist.axvline(0.0005, color="#D55E00", linestyle="--", label="Design minimum")
    ax_hist.set_xscale("log")
    ax_hist.set_xlabel("Conduit slope (m m$^{-1}$)")
    ax_hist.set_ylabel("Number of conduits")
    ax_hist.set_title("Positive-slope constraint")
    ax_hist.legend()

    scatter = ax_box.scatter(
        lengths,
        link_covers,
        c=np.asarray([diameter[row[0]] for row in data["CONDUITS"]]),
        cmap="viridis",
        s=5,
        alpha=0.45,
        edgecolors="none",
        vmin=float(diameters.min()),
        vmax=float(diameters.max()),
    )
    ax_box.set_xscale("log")
    ax_box.set_xlabel("Conduit length (m)")
    ax_box.set_ylabel("Mean endpoint cover depth (m)")
    ax_box.set_title("Geometry of retained conduits")
    fig.colorbar(scatter, ax=ax_box, label="Pipe diameter (m)", shrink=0.82)
    ax_box.text(
        0.03,
        0.97,
        f"n = {len(junctions):,} junctions\n"
        f"n = {len(data['CONDUITS']):,} conduits\n"
        f"n = {len(outfalls):,} receiving interfaces",
        transform=ax_box.transAxes,
        va="top",
        ha="left",
        fontsize=7.2,
    )

    for ax in [ax_dem, ax_pipe]:
        ax.set_xlabel("Easting (km)")
        ax.set_ylabel("Northing (km)")
    add_panel_labels(axes)
    for suffix in ["png", "pdf", "svg"]:
        fig.savefig(
            args.output / f"fig_network_audit.{suffix}",
            dpi=400 if suffix == "png" else None,
        )
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
