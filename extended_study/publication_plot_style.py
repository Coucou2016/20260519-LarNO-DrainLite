#!/usr/bin/env python3
"""Shared SciencePlots configuration for publication figures."""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401 - importing registers the SciencePlots styles


MODEL_COLORS = {
    "MIKE reference": "#111111",
    "ITZI surface-only": "#7A7A7A",
    "ITZI-SWMM connected": "#D55E00",
    "DrainLite all_static": "#0072B2",
    "DrainLite all_static CV": "#0072B2",
    "surface_only": "#7A7A7A",
    "base": "#E69F00",
    "mask_only": "#56B4E9",
    "hydraulic_only": "#CC79A7",
    "all_static": "#0072B2",
    "swmm_assisted": "#009E73",
}

MODEL_LABELS = {
    "surface_only": "Surface only",
    "base": "Base residual",
    "mask_only": "Network masks",
    "hydraulic_only": "Hydraulic fields",
    "all_static": "All static",
    "swmm_assisted": "SWMM-assisted",
}


def configure_publication_style() -> None:
    """Apply a consistent journal-ready style without requiring LaTeX."""

    plt.style.use(["science", "no-latex"])
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "SimSun", "Microsoft YaHei", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "figure.titlesize": 10.5,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.35,
            "lines.markersize": 4.0,
            "grid.linewidth": 0.45,
            "grid.alpha": 0.25,
            "legend.frameon": False,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.unicode_minus": False,
        }
    )


def model_label(name: str) -> str:
    """Return a compact, reader-facing label for a model identifier."""

    return MODEL_LABELS.get(name, name)


def add_panel_labels(axes, *, x: float = -0.12, y: float = 1.04) -> None:
    """Add stable (a), (b), ... labels to a sequence of axes."""

    for index, axis in enumerate(list(axes)):
        axis.text(
            x,
            y,
            f"({chr(97 + index)})",
            transform=axis.transAxes,
            fontsize=9.5,
            fontweight="bold",
            va="bottom",
            ha="left",
        )
