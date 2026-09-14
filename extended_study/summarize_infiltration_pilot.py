#!/usr/bin/env python3
"""Summarize the first ITZI infiltration pilot run."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from publication_plot_style import add_panel_labels, configure_publication_style


configure_publication_style()


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "LarNO-main" / "benchmark" / "urbanflood" / "flood" / "region1_20m_connected_swmm_v1"
GEO = ROOT / "LarNO-main" / "benchmark" / "urbanflood" / "geodata" / "region1_20m_connected_swmm_v1"
INF_RUN = ROOT / "extended_study" / "output" / "connected_itzi_swmm_inf_5mmh"
OUT = ROOT / "extended_study" / "output" / "infiltration_pilot"
FIG = OUT / "figures"
EVENT = "event68"
CELL_AREA_M2 = 20.0 * 20.0
DT_H = 5.0 / 60.0


def max_depth(h: np.ndarray) -> np.ndarray:
    return h.max(axis=(1, 2))


def active_mask() -> np.ndarray:
    dem = np.load(GEO / "dem.npy").astype(np.float32)
    return dem < 49.9


def volume(h: np.ndarray, active: np.ndarray | None = None) -> np.ndarray:
    if active is None:
        return h.sum(axis=(1, 2)) * CELL_AREA_M2
    return h[:, active].sum(axis=1) * CELL_AREA_M2


def metrics(model: str, h: np.ndarray, mike: np.ndarray, active: np.ndarray) -> dict[str, object]:
    hmax = max_depth(h)
    vol = volume(h, active)
    mike_peak_map = mike.max(axis=0)
    return {
        "event": EVENT,
        "model": model,
        "max_depth_peak_hour": float((np.argmax(hmax) + 1) * DT_H),
        "max_depth_peak_m": float(hmax.max()),
        "max_depth_final_m": float(hmax[-1]),
        "max_depth_final_to_peak_ratio": float(hmax[-1] / hmax.max()) if hmax.max() else np.nan,
        "volume_peak_hour": float((np.argmax(vol) + 1) * DT_H),
        "volume_peak_m3": float(vol.max()),
        "volume_final_m3": float(vol[-1]),
        "volume_final_to_peak_ratio": float(vol[-1] / vol.max()) if vol.max() else np.nan,
        "volume_mask": "active_nonbuilding_cells",
        "mae_to_mike_mm": float(np.mean(np.abs(h - mike)) * 1000.0) if model != "MIKE reference" else 0.0,
        "peak_map_mae_to_mike_mm": float(np.mean(np.abs(h.max(axis=0) - mike_peak_map)) * 1000.0)
        if model != "MIKE reference"
        else 0.0,
    }


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


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    base_dir = DATA / EVENT
    inf_npz = INF_RUN / EVENT / f"{EVENT}_connected_itzi_swmm.npz"
    if not inf_npz.exists():
        raise FileNotFoundError(f"Missing pilot result: {inf_npz}")

    mike = np.load(base_dir / "h_mike_ref.npy").astype(np.float32)
    old_surface = np.load(base_dir / "h_itzi_surface.npy").astype(np.float32)
    old_connected = np.load(base_dir / "h_itzi_swmm_connected.npy").astype(np.float32)
    active = active_mask()
    z = np.load(inf_npz)
    inf_surface = z["h_surf"].astype(np.float32)
    inf_connected = z["h_swmm_connected"].astype(np.float32)

    series = {
        "MIKE reference": mike,
        "No infiltration surface": old_surface,
        "No infiltration ITZI-SWMM": old_connected,
        "5 mm/h infiltration surface": inf_surface,
        "5 mm/h infiltration ITZI-SWMM": inf_connected,
    }
    rows = [metrics(name, h, mike, active) for name, h in series.items()]
    write_csv(OUT / "event68_infiltration_pilot_summary.csv", rows)

    t = np.arange(1, mike.shape[0] + 1) * DT_H
    fig, axes = plt.subplots(2, 1, figsize=(7.1, 5.1), sharex=True, constrained_layout=True)
    colors = {
        "MIKE reference": "#111827",
        "No infiltration surface": "#60a5fa",
        "No infiltration ITZI-SWMM": "#ef4444",
        "5 mm/h infiltration surface": "#2563eb",
        "5 mm/h infiltration ITZI-SWMM": "#b91c1c",
    }
    for name, h in series.items():
        axes[0].plot(t, max_depth(h), label=name, color=colors[name], lw=1.6)
        axes[1].plot(t, volume(h, active) / 1000.0, label=name, color=colors[name], lw=1.6)
    axes[0].set_ylabel("Maximum depth (m)")
    axes[0].set_title("Event68 maximum-depth hydrograph: infiltration pilot")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8, ncol=2)
    axes[1].set_xlabel("Time (h)")
    axes[1].set_ylabel("Surface water volume (10^3 m3)")
    axes[1].set_title("Event68 active-cell surface-water volume: infiltration pilot")
    axes[1].grid(alpha=0.25)
    add_panel_labels(axes, x=-0.09, y=1.02)
    fig.savefig(FIG / "event68_infiltration_pilot_hydrograph.png")
    plt.close(fig)

    note = {
        "event": EVENT,
        "infiltration_mmh": 5.0,
        "summary_csv": str(OUT / "event68_infiltration_pilot_summary.csv"),
        "figure": str(FIG / "event68_infiltration_pilot_hydrograph.png"),
        "interpretation": (
            "Adding 5 mm/h active-cell infiltration improves hydrograph timing and creates a falling limb, "
            "but it over-reduces peak depth and surface-water volume for event68. The rate should be calibrated "
            "instead of used as the final setting."
        ),
    }
    (OUT / "event68_infiltration_pilot_summary.json").write_text(
        json.dumps(note, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(note, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
