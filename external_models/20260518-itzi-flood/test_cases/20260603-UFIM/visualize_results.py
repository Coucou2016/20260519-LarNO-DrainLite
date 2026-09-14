#!/usr/bin/env python3
"""Visualization for UFIM coupled simulation — flood maps with drainage network overlay."""
from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from sample_utils import CASE_DIR, load_network_for_plot, sample_dir

plt.rcParams.update({"font.size": 10, "figure.dpi": 120})


def load_results(sample: str) -> dict:
    path = os.path.join(sample_dir(sample), "output", "coupled_results.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Run simulation first: {path}")
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def overlay_network(ax, nodes: list, links: list) -> None:
    for link in links:
        pts = link["points"]
        if len(pts) < 2:
            continue
        rs = [p[0] for p in pts]
        cs = [p[1] for p in pts]
        ax.plot(cs, rs, color="#2d3748", linewidth=0.8, alpha=0.85, zorder=5)

    j_rows, j_cols, o_rows, o_cols = [], [], [], []
    for n in nodes:
        if str(n.get("type", "J")).upper() == "O":
            o_rows.append(n["row"])
            o_cols.append(n["col"])
        else:
            j_rows.append(n["row"])
            j_cols.append(n["col"])

    if j_rows:
        ax.scatter(j_cols, j_rows, s=12, c="#e53e3e", edgecolors="white",
                   linewidths=0.3, zorder=6, label="_junction")
    if o_rows:
        ax.scatter(o_cols, o_rows, s=18, c="#38a169", marker="s", edgecolors="white",
                   linewidths=0.3, zorder=6, label="_outfall")


def network_legend():
    return [
        Line2D([0], [0], color="#2d3748", linewidth=1.5, label="Pipe links"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#e53e3e",
               markersize=7, label="Junction"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#38a169",
               markersize=7, label="Outfall"),
        Patch(facecolor="white", edgecolor="#3182ce", label="Flood depth"),
    ]


def plot_depth_panel(ax, data, bldg, nodes, links, title, cmap, vmin, vmax):
    show = np.ma.masked_where(bldg, data)
    im = ax.imshow(show, cmap=cmap, vmin=vmin, vmax=vmax)
    overlay_network(ax, nodes, links)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    return im


def plot_sample(sample: str, out_dir: str) -> None:
    d = load_results(sample)
    h_s = d["h_final_surface"]
    h_w = d["h_final_swmm"]
    dem = d["dem"]
    bldg = d["building_mask"]
    rec_s = d["rec_surface"].item()
    rec_w = d["rec_swmm"].item()
    rain = d["rain_mm_h"]
    if rain.ndim == 2:
        rain_plot = rain.max(axis=1)
    else:
        rain_plot = rain

    nodes, links, _ = load_network_for_plot(sample)
    os.makedirs(out_dir, exist_ok=True)

    vmax = max(float(np.nanmax(h_s)), float(np.nanmax(h_w)), 0.05)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    im0 = axes[0].imshow(np.where(bldg, np.nan, dem), cmap="terrain")
    overlay_network(axes[0], nodes, links)
    axes[0].set_title(f"{sample} — DEM + network")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, label="m")

    im1 = plot_depth_panel(
        axes[1], h_s, bldg, nodes, links, "Surface only — final depth", "Blues", 0, vmax
    )
    im2 = plot_depth_panel(
        axes[2], h_w, bldg, nodes, links, "SWMM coupled — final depth", "Blues", 0, vmax
    )
    plt.colorbar(im1, ax=axes[1], fraction=0.046, label="depth (m)")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, label="depth (m)")

    handles = network_legend()
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=9, frameon=True)
    fig.suptitle(f"UFIM {sample}: flood depth with drainage network overlay", fontsize=12)
    fig.tight_layout(rect=[0, 0.06, 1, 0.95])
    fig.savefig(os.path.join(out_dir, f"{sample}_depth_maps.png"), bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    diff = h_s - h_w
    lim = max(abs(np.nanmin(diff)), abs(np.nanmax(diff)), 0.01)
    im = ax.imshow(np.ma.masked_where(bldg, diff), cmap="RdBu_r", vmin=-lim, vmax=lim)
    overlay_network(ax, nodes, links)
    ax.set_title(f"{sample}: depth difference (surface − SWMM) + network")
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, label="m")
    ax.legend(handles=handles[:3], loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{sample}_depth_diff.png"), bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    t = rec_s["time_h"]
    axes[0, 0].plot(t, rec_s["hmax_m"], "b-o", ms=3, label="Surface")
    axes[0, 0].plot(rec_w["time_h"], rec_w["hmax_m"], "r-s", ms=3, label="SWMM")
    axes[0, 0].set_ylabel("Max depth (m)")
    axes[0, 0].legend()
    axes[0, 0].set_title("Peak depth")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(t, rec_s["vol_m3"], "b-o", ms=3, label="Surface")
    axes[0, 1].plot(rec_w["time_h"], rec_w["vol_m3"], "r-s", ms=3, label="SWMM")
    axes[0, 1].set_ylabel("Volume (m³)")
    axes[0, 1].set_title("Stored volume")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(t, rec_s["flooded_cells"], "b-o", ms=3, label="Surface")
    axes[1, 0].plot(rec_w["time_h"], rec_w["flooded_cells"], "r-s", ms=3, label="SWMM")
    axes[1, 0].set_xlabel("Time (h)")
    axes[1, 0].set_ylabel("Flooded cells (>3 cm)")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    rain_t = np.arange(len(rain_plot)) * 5 / 60.0
    axes[1, 1].bar(rain_t, rain_plot, width=0.07, color="steelblue", alpha=0.7)
    axes[1, 1].set_xlabel("Time (h)")
    axes[1, 1].set_ylabel("Rain intensity (mm/h)")
    axes[1, 1].set_title("Rainfall input (max across stations if multi)")
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle(f"UFIM {sample}: simulation time series", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{sample}_timeseries.png"), bbox_inches="tight")
    plt.close(fig)

    print(f"  Saved figures for {sample} -> {out_dir}")


def plot_summary(all_summaries: list, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    names = [s["sample"] for s in all_summaries]
    x = np.arange(len(names))
    width = 0.35

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    surf_peak = [s["surface_peak_m"] for s in all_summaries]
    swmm_peak = [s["swmm_peak_m"] for s in all_summaries]
    axes[0].bar(x - width / 2, surf_peak, width, label="Surface", color="steelblue")
    axes[0].bar(x + width / 2, swmm_peak, width, label="SWMM", color="indianred")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names)
    axes[0].set_ylabel("Peak depth (m)")
    axes[0].set_title("Peak flood depth")
    axes[0].legend()
    axes[0].grid(True, axis="y", alpha=0.3)

    surf_f = [s["surface_flooded"] for s in all_summaries]
    swmm_f = [s["swmm_flooded"] for s in all_summaries]
    axes[1].bar(x - width / 2, surf_f, width, label="Surface", color="steelblue")
    axes[1].bar(x + width / 2, swmm_f, width, label="SWMM", color="indianred")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names)
    axes[1].set_ylabel("Cells")
    axes[1].set_title("Flooded cells (>3 cm)")
    axes[1].legend()
    axes[1].grid(True, axis="y", alpha=0.3)

    drained = [s["swmm_drained_m3"] for s in all_summaries]
    axes[2].bar(names, drained, color="seagreen")
    axes[2].set_ylabel("Volume (m³)")
    axes[2].set_title("Cumulative drainage (SWMM)")
    axes[2].grid(True, axis="y", alpha=0.3)

    fig.suptitle("UFIM samples — cross-sample summary", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "all_samples_summary.png"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved summary chart -> {out_dir}/all_samples_summary.png")


def main():
    parser = argparse.ArgumentParser(description="Visualize UFIM simulation results")
    parser.add_argument("--sample", choices=["sample1", "sample2", "sample3"])
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    viz_dir = os.path.join(CASE_DIR, "visualization_output")
    samples = ["sample1", "sample2", "sample3"] if args.all else [args.sample]
    if not samples or (not args.all and not args.sample):
        parser.error("Specify --sample or --all")

    summaries = []
    for s in samples:
        plot_sample(s, viz_dir)
        d = load_results(s)
        summaries.append(d["summary"].item())

    if len(summaries) > 1:
        plot_summary(summaries, viz_dir)


if __name__ == "__main__":
    main()
