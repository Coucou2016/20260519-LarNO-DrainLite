#!/usr/bin/env python3
"""Plot numerical-stability diagnostics for candidate coupling settings."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from publication_plot_style import add_panel_labels, configure_publication_style


configure_publication_style()


def compact(name: str) -> str:
    replacements = {
        "stability_fixed_step05_full_3h": "Fixed, r=0.8, 3 h",
        "stability_hydraulic_sized_full_3h": "Oversized, r=0.8, 3 h",
        "stability_original_extran_r08_full_1p5h": "EXTRAN, r=0.8, 1.5 h",
        "stability_original_slot_r02_full_1p5h": "Fixed, r=0.2, 1.5 h",
        "stability_original_slot_r02_full_3h": "Fixed, r=0.2, 3 h",
        "stability_original_free_r02_full_3h": "Free, r=0.2, 3 h",
        "stability_original_normal_r02_full_3h": "Normal, r=0.2, 3 h",
        "stability_outlet200_free_r08_full_3h": "Free, Lout=200 m, r=0.8",
        "stability_outlet200_free_r02_full_3h": "Free, Lout=200 m, r=0.2",
        "stability_outlet200_normal_r08_full_3h": "Normal, Lout=200 m, r=0.8",
    }
    return replacements.get(name, name.replace("stability_", "").replace("_", " "))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frame = pd.read_csv(args.input)
    frame = frame[np.isfinite(frame["steps_not_converging_pct"])].copy()
    labels = [compact(value) for value in frame["run"]]
    accepted = (
        frame["steps_not_converging_pct"].abs().to_numpy() <= 2.0
    ) & (frame["continuity_error_pct"].abs().to_numpy() <= 2.0) & (
        frame["warning_count"].to_numpy() == 0
    )
    colours = np.where(accepted, "#009E73", "#D55E00")
    positions = np.arange(len(frame))
    fig, axes = plt.subplots(2, 2, figsize=(7.5, 5.8), constrained_layout=True)
    axes = axes.ravel()
    axes[0].bar(positions, frame["steps_not_converging_pct"], color=colours)
    axes[0].axhline(2.0, color="#111111", linestyle="--", label="Acceptance limit")
    axes[0].set_ylabel("Steps not converging (%)")
    axes[0].set_title("Dynamic Wave convergence")
    axes[0].legend()
    axes[1].bar(positions, np.abs(frame["continuity_error_pct"]), color=colours)
    axes[1].axhline(2.0, color="#111111", linestyle="--")
    axes[1].set_ylabel("Absolute continuity error (%)")
    axes[1].set_title("Flow-routing continuity")
    axes[2].bar(positions, frame["average_iterations"], color="#0072B2", alpha=0.82)
    axes[2].set_ylabel("Iterations per routing step")
    axes[2].set_title("Solver effort")
    axes[3].scatter(
        frame["net_exchange_m3"] / 1000.0,
        frame["final_surface_volume_m3"] / 1e6,
        c=colours,
        s=28,
        edgecolor="white",
        linewidth=0.4,
    )
    for position, (_, row) in enumerate(frame.iterrows()):
        axes[3].annotate(
            str(position + 1),
            (row["net_exchange_m3"] / 1000.0, row["final_surface_volume_m3"] / 1e6),
            xytext=(3, 2),
            textcoords="offset points",
            fontsize=6.5,
        )
    axes[3].set_xlabel("Cumulative net drainage exchange ($10^3$ m$^3$)")
    axes[3].set_ylabel("Final surface volume ($10^6$ m$^3$)")
    axes[3].set_title("Hydraulic-response sensitivity")
    for axis in axes[:3]:
        axis.set_xticks(positions)
        axis.set_xticklabels([f"{index + 1}. {label}" for index, label in enumerate(labels)], rotation=55, ha="right", fontsize=6.3)
        axis.grid(axis="y", alpha=0.22)
    add_panel_labels(axes)
    args.output.mkdir(parents=True, exist_ok=True)
    for suffix in ["png", "pdf", "svg"]:
        fig.savefig(args.output / f"fig_stability_sensitivity.{suffix}", dpi=400 if suffix == "png" else None)
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
