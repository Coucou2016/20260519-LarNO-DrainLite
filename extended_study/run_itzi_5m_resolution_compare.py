"""
Build a 5 m resolution comparison package for the LarNO/MIKE+/ITZI study.

The public LarNO dataset commonly available for this case is `region1_20m`.
The paper discusses 5 m high-resolution forecasting, but the true 5 m MIKE+
reference files may not be present in a local checkout. This script therefore
uses two explicit modes:

1. true_5m
   If `region1_5m` files exist locally, compare the real 5 m MIKE+ reference
   against the 20 m reference over the same physical window.

2. proxy_5m
   If true 5 m files are missing, create a labelled proxy comparison by
   refining the public 20 m reference with nearest-neighbour expansion. This is
   only a grid-sensitivity/visualization aid. It must not be described as a
   true MIKE+ 5 m reference reproduction.

Outputs:
  extended_study/output/itzi_5m_compare/
    resolution_summary.csv
    mike_reference_20m_vs_5m_peak_stats.csv
    mike_reference_20m_vs_5m_peaks.png
    README_5m_mode.txt

Run from the repository root:
    python extended_study/run_itzi_5m_resolution_compare.py
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "LarNO-main" / "benchmark" / "urbanflood"
OUT = ROOT / "extended_study" / "output" / "itzi_5m_compare"

EVENTS = ["event1", "event20", "event65", "event66", "event67", "event68", "event69", "event70"]

# Existing 20 m ITZI dynamic comparison window: rows 80:280, cols 120:400.
# Same physical box at 5 m uses index multiplier 4.
SUB_20M = (slice(80, 280), slice(120, 400))
SUB_5M = (slice(80 * 4, 280 * 4), slice(120 * 4, 400 * 4))


@dataclass(frozen=True)
class DatasetPaths:
    geodata_dir: Path
    flood_dir: Path

    @property
    def dem(self) -> Path:
        return self.geodata_dir / "dem.npy"

    def h(self, event: str) -> Path:
        return self.flood_dir / event / "h.npy"

    def rainfall(self, event: str) -> Path:
        return self.flood_dir / event / "rainfall.npy"


P20 = DatasetPaths(
    BENCH / "geodata" / "region1_20m",
    BENCH / "flood" / "region1_20m",
)
P5 = DatasetPaths(
    BENCH / "geodata" / "region1_5m",
    BENCH / "flood" / "region1_5m",
)


def true_5m_available() -> bool:
    paths = [P5.dem]
    for event in EVENTS:
        paths.extend([P5.h(event), P5.rainfall(event)])
    return all(path.exists() for path in paths)


def load_peak(path: Path, sub: tuple[slice, slice] | None = None) -> np.ndarray:
    arr = np.load(path)
    if arr.ndim == 3:
        peak = np.nanmax(arr, axis=0)
    elif arr.ndim == 2:
        peak = arr
    else:
        raise ValueError(f"Unsupported array shape for {path}: {arr.shape}")
    if sub is not None:
        peak = peak[sub]
    return peak.astype(np.float32, copy=False)


def upscale_20m_to_proxy_5m(arr: np.ndarray) -> np.ndarray:
    return np.repeat(np.repeat(arr, 4, axis=0), 4, axis=1)


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize_resolution(mode: str) -> list[dict[str, object]]:
    dem20 = np.load(P20.dem)
    rows: list[dict[str, object]] = [{
        "dataset": "MIKE/LarNO reference",
        "mode": "public_20m",
        "resolution_m": 20,
        "dem_shape": f"{dem20.shape[0]}x{dem20.shape[1]}",
        "domain_km": f"{dem20.shape[0] * 20 / 1000:.1f}x{dem20.shape[1] * 20 / 1000:.1f}",
        "compare_window_cells": "200x280",
        "compare_window_km": "4.0x5.6",
    }]

    if mode == "true_5m":
        dem5 = np.load(P5.dem)
        dem_shape = f"{dem5.shape[0]}x{dem5.shape[1]}"
        domain_km = f"{dem5.shape[0] * 5 / 1000:.1f}x{dem5.shape[1] * 5 / 1000:.1f}"
    else:
        dem_shape = f"{dem20.shape[0] * 4}x{dem20.shape[1] * 4}"
        domain_km = f"{dem20.shape[0] * 20 / 1000:.1f}x{dem20.shape[1] * 20 / 1000:.1f}"

    rows.append({
        "dataset": "MIKE/LarNO reference",
        "mode": mode,
        "resolution_m": 5,
        "dem_shape": dem_shape,
        "domain_km": domain_km,
        "compare_window_cells": "800x1120",
        "compare_window_km": "4.0x5.6",
    })
    return rows


def get_5m_peak(event: str, mode: str) -> np.ndarray:
    if mode == "true_5m":
        return load_peak(P5.h(event), SUB_5M)
    peak20 = load_peak(P20.h(event), SUB_20M)
    return upscale_20m_to_proxy_5m(peak20)


def write_event_peak_csv(mode: str) -> None:
    rows: list[dict[str, object]] = []
    for event in EVENTS:
        peak20 = load_peak(P20.h(event), SUB_20M)
        peak5 = get_5m_peak(event, mode)
        rows.append({
            "event": event,
            "comparison_mode": mode,
            "mike20_peak_m": float(np.nanmax(peak20)),
            "mike5_peak_m": float(np.nanmax(peak5)),
            "mike5_minus_mike20_peak_m": float(np.nanmax(peak5) - np.nanmax(peak20)),
            "mike20_mean_m": float(np.nanmean(peak20)),
            "mike5_mean_m": float(np.nanmean(peak5)),
            "mike20_p95_m": float(np.nanpercentile(peak20, 95)),
            "mike5_p95_m": float(np.nanpercentile(peak5, 95)),
        })
    write_csv(rows, OUT / "mike_reference_20m_vs_5m_peak_stats.csv")


def make_reference_figures(mode: str) -> None:
    import matplotlib.pyplot as plt

    OUT.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, len(EVENTS), figsize=(3.2 * len(EVENTS), 7.0), constrained_layout=True)
    vmax = 2.6
    for col, event in enumerate(EVENTS):
        peak20 = load_peak(P20.h(event), SUB_20M)
        peak5 = get_5m_peak(event, mode)
        axes[0, col].imshow(peak20, cmap="Blues", vmin=0, vmax=vmax, origin="lower")
        axes[0, col].set_title(f"{event}\nMIKE 20m max={np.nanmax(peak20):.3f}m", fontsize=9)
        axes[1, col].imshow(peak5, cmap="Blues", vmin=0, vmax=vmax, origin="lower")
        label = "MIKE 5m" if mode == "true_5m" else "Proxy 5m"
        axes[1, col].set_title(f"{label} max={np.nanmax(peak5):.3f}m", fontsize=9)
        for ax in axes[:, col]:
            ax.set_xticks([])
            ax.set_yticks([])
    title_suffix = "true region1_5m" if mode == "true_5m" else "proxy from public 20m fields"
    fig.suptitle(f"Resolution Check: 20 m vs 5 m ({title_suffix})", fontsize=15, fontweight="bold")
    fig.savefig(OUT / "mike_reference_20m_vs_5m_peaks.png", dpi=220)
    plt.close(fig)


def write_readme(mode: str) -> None:
    if mode == "true_5m":
        text = """5 m comparison mode: true_5m

The required region1_5m files were found locally. The figures and CSV in this
folder compare real MIKE+ 5 m reference data against the public 20 m reference
over the same 4.0 km x 5.6 km physical window.
"""
    else:
        text = """5 m comparison mode: proxy_5m

The true region1_5m MIKE+ reference files were not found locally. This folder
therefore contains a labelled proxy 5 m comparison created by nearest-neighbour
refinement of the public 20 m reference fields.

This is useful for checking plotting, physical window consistency, and the
expected grid-size change from 200 x 280 cells to 800 x 1120 cells. It is not a
true 5 m MIKE+ reference result and must not be reported as one.

To run a true 5 m comparison, place these files under LarNO-main:
  benchmark/urbanflood/geodata/region1_5m/dem.npy
  benchmark/urbanflood/flood/region1_5m/event*/h.npy
  benchmark/urbanflood/flood/region1_5m/event*/rainfall.npy
Then rerun this script.
"""
    (OUT / "README_5m_mode.txt").write_text(text, encoding="utf-8")


def main() -> int:
    mode = "true_5m" if true_5m_available() else "proxy_5m"
    OUT.mkdir(parents=True, exist_ok=True)
    write_csv(summarize_resolution(mode), OUT / "resolution_summary.csv")
    write_event_peak_csv(mode)
    make_reference_figures(mode)
    write_readme(mode)

    print(f"Wrote {mode} resolution comparison outputs to {OUT}")
    if mode == "proxy_5m":
        print("True region1_5m files were not found; generated a labelled proxy_5m comparison instead.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
