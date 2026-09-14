#!/usr/bin/env python3
"""
Run a real ITZI 5 m refined-grid sensitivity experiment.

Important scope:
- The public local dataset is 20 m. This script derives 5 m DEM/rainfall by
  nearest-neighbour refinement of the public 20 m fields.
- ITZI itself is run on a 5 m grid, so this is a genuine 5 m ITZI numerical
  run, but it is not a true 5 m MIKE-reference reproduction.
- Results are aggregated back to 20 m for comparison with the available MIKE+
  20 m reference over the same physical window.

The window is intentionally 1.0 km x 1.4 km to keep the 5 m ITZI runs tractable:
  20 m source window: 50 x 70 cells
  5 m ITZI window:    200 x 280 cells
"""

from __future__ import annotations

import csv
import os
import sys
import time
from datetime import timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, r"E:\Miniconda3\Lib\site-packages")
import itzi.rasterdomain as rasterdomain
import itzi.surfaceflow as surfaceflow
from itzi.hydrology import Hydrology
from itzi.infiltration import InfNull


ROOT = Path(__file__).resolve().parents[1]
DEM_PATH = ROOT / "LarNO-main" / "benchmark" / "urbanflood" / "geodata" / "region1_20m" / "dem.npy"
FLOOD_DIR = ROOT / "LarNO-main" / "benchmark" / "urbanflood" / "flood" / "region1_20m"
NET_PATH = ROOT / "extended_study" / "output" / "osm_merged_network.npz"
OUT_DIR = ROOT / "extended_study" / "output" / "itzi_5m_refined"

EVENTS = ["event1", "event20", "event65", "event66", "event67", "event68", "event69", "event70"]

# 20 m source window. Located inside the previous 4.0 km x 5.6 km comparison
# region and includes road/pipe-dense urban cells.
Y0_20, Y1_20 = 120, 170
X0_20, X1_20 = 160, 230

CELL20 = 20.0
CELL5 = 5.0
SCALE = 4
G = 9.81


def upscale_2d(a: np.ndarray) -> np.ndarray:
    return np.repeat(np.repeat(a, SCALE, axis=0), SCALE, axis=1)


def upscale_3d(a: np.ndarray) -> np.ndarray:
    return np.repeat(np.repeat(a, SCALE, axis=1), SCALE, axis=2)


def block_mean_5_to_20(a5: np.ndarray) -> np.ndarray:
    h, w = a5.shape
    return a5.reshape(h // SCALE, SCALE, w // SCALE, SCALE).mean(axis=(1, 3))


def build_5m_pipe_nodes(bldg5: np.ndarray) -> list[dict]:
    if not NET_PATH.exists():
        return []
    net = np.load(NET_PATH, allow_pickle=True)
    nodes = list(net["nodes"])
    y0_m = Y0_20 * CELL20
    x0_m = X0_20 * CELL20
    h5, w5 = bldg5.shape
    out = []
    for n in nodes:
        x_m = float(n.get("x_m", float(n["col"]) * CELL20))
        y_m = float(n.get("y_m", float(n["row"]) * CELL20))
        r5 = int(round((y_m - y0_m) / CELL5))
        c5 = int(round((x_m - x0_m) / CELL5))
        if 0 <= r5 < h5 and 0 <= c5 < w5 and not bldg5[r5, c5]:
            out.append({
                "row": r5,
                "col": c5,
                "invert": float(n.get("invert", 0.0)),
                "type": n.get("type", "junction"),
            })
    return out


def run_itzi_5m(dem5: np.ndarray, bldg5: np.ndarray, rainfall5: np.ndarray,
                pipe_nodes: list[dict] | None = None) -> tuple[dict, np.ndarray]:
    h, w = dem5.shape
    active = ~bldg5
    cell_area = CELL5 * CELL5
    duration = rainfall5.shape[0] * 300
    dt_rain = 300

    dem_itzi = dem5.astype(np.float32).copy()
    dem_itzi[bldg5] = 50.0

    domain = rasterdomain.RasterDomain(
        dtype=np.float32,
        arr_mask=np.zeros((h, w), dtype=bool),
        cell_shape=(CELL5, CELL5),
    )
    sim_param = {
        "hmin": 0.001,
        "cfl": 0.7,
        "theta": 0.9,
        "g": 9.80665,
        "vrouting": 0.1,
        "dtmax": 1.0,
        "slmax": 0.1,
    }
    sf_sim = surfaceflow.SurfaceFlowSimulation(domain, sim_param)
    hydrology = Hydrology(domain, 60.0, InfNull(domain, 60.0))

    friction = np.full((h, w), 0.015, dtype=np.float32)
    friction[bldg5] = 100.0

    domain.update_array(k="dem", arr=dem_itzi)
    domain.update_array(k="friction", arr=friction)
    domain.update_array(k="h", arr=np.zeros((h, w), dtype=np.float32))
    domain.update_array(k="rain", arr=np.zeros((h, w), dtype=np.float32))
    sf_sim.update_flow_dir()

    inlets = []
    if pipe_nodes:
        for n in pipe_nodes:
            r, c = int(n["row"]), int(n["col"])
            if 0 <= r < h and 0 <= c < w and active[r, c]:
                inlets.append((r, c, float(n["invert"])))

    t = 0.0
    dt = 0.001
    rain_idx = 0
    next_rain = 0.0
    next_record = 1800.0
    n_active = max(int(active.sum()), 1)
    drained = 0.0
    records = {"time_h": [], "vol_m3": [], "hmax_m": [], "flooded_cells": [], "drained_m3": []}

    next_drain = 30.0

    while t < duration:
        if t >= next_rain and rain_idx < rainfall5.shape[0]:
            rain_2d = rainfall5[rain_idx]
            bldg_rain = float(np.sum(rain_2d[bldg5]))
            bldg_extra = bldg_rain / n_active
            rain_field = np.zeros((h, w), dtype=np.float32)
            rain_field[active] = (rain_2d[active] + bldg_extra) / 1000.0 / dt_rain
            domain.update_array(k="rain", arr=rain_field)
            rain_idx += 1
            next_rain += dt_rain

        domain.update_ext_array()
        hydrology.solve_dt()
        hydrology.step()

        if inlets and t >= next_drain:
            h_arr = domain.get_array("h")
            for r, c, invert in inlets:
                if h_arr[r, c] > 0.01:
                    surf_wl = dem_itzi[r, c] + h_arr[r, c]
                    head = max(float(surf_wl - invert), 0.01)
                    q = 0.65 * 2.0 * np.sqrt(2 * G * head)
                    q = min(q, float(h_arr[r, c]) * cell_area / 30.0 * 0.3)
                    removal = q * 30.0 / cell_area
                    h_arr[r, c] = max(0.0, h_arr[r, c] - removal)
                    drained += q * 30.0
            domain.update_array(k="h", arr=h_arr)
            next_drain += 30.0

        sf_sim.dt = timedelta(seconds=dt)
        sf_sim.step()
        sf_sim.solve_dt()
        t += dt
        dt = sf_sim.dt.total_seconds()

        if t >= next_record:
            h_arr = domain.get_array("h")
            records["time_h"].append(t / 3600.0)
            records["vol_m3"].append(float(np.sum(h_arr[active]) * cell_area))
            records["hmax_m"].append(float(np.max(h_arr)))
            records["flooded_cells"].append(int(np.sum(h_arr > 0.03)))
            records["drained_m3"].append(float(drained))
            next_record += 1800.0

    return records, domain.get_array("h").copy()


def write_csv(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_figures(rows: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    events = [r["event"] for r in rows]
    mike = [r["mike20_peak_m"] for r in rows]
    surf = [r["itzi5_surface_peak_m"] for r in rows]
    pipe = [r["itzi5_pipe_peak_m"] for r in rows]
    red = [r["pipe_peak_reduction_pct"] for r in rows]

    x = np.arange(len(events))
    w = 0.25
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.bar(x - w, mike, w, label="MIKE+ 20m reference", color="#2ecc71", edgecolor="black")
    ax.bar(x, surf, w, label="ITZI 5m refined surface", color="#e74c3c", edgecolor="black")
    ax.bar(x + w, pipe, w, label="ITZI 5m refined + pipes", color="#2980b9", edgecolor="black")
    for i, val in enumerate(red):
        ax.annotate(f"{val:.1f}%", (x[i] + w, pipe[i]), xytext=(0, 6),
                    textcoords="offset points", ha="center", fontsize=8, color="#2980b9")
    ax.set_xticks(x)
    ax.set_xticklabels([e.replace("event", "E") for e in events])
    ax.set_ylabel("Peak depth (m)")
    ax.set_title("5 m Refined-Grid ITZI Sensitivity: Peaks vs 20 m MIKE Reference")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "itzi_5m_refined_peak_bars.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, [r["pipe_volume_reduction_pct"] for r in rows], marker="o", label="Volume reduction")
    ax.plot(x, [r["pipe_flooded_reduction_pct"] for r in rows], marker="s", label="Flooded-cell reduction")
    ax.plot(x, red, marker="^", label="Global peak reduction")
    ax.set_xticks(x)
    ax.set_xticklabels([e.replace("event", "E") for e in events])
    ax.set_ylabel("Reduction (%)")
    ax.set_title("5 m Refined-Grid Pipe Effect Metrics")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "itzi_5m_refined_reduction_metrics.png", dpi=180)
    plt.close(fig)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dem20_full = np.load(DEM_PATH).astype(np.float32)
    bldg20_full = dem20_full >= 49.9
    dem20 = dem20_full[Y0_20:Y1_20, X0_20:X1_20]
    bldg20 = bldg20_full[Y0_20:Y1_20, X0_20:X1_20]
    dem5 = upscale_2d(dem20)
    bldg5 = upscale_2d(bldg20)
    pipe_nodes5 = build_5m_pipe_nodes(bldg5)

    rows = []
    for event in EVENTS:
        print(f"\n=== {event}: 5 m refined ITZI window ===")
        rainfall20 = np.load(FLOOD_DIR / event / "rainfall.npy")[:, Y0_20:Y1_20, X0_20:X1_20]
        h20_ref = np.load(FLOOD_DIR / event / "h.npy")[:, Y0_20:Y1_20, X0_20:X1_20]
        rainfall5 = upscale_3d(rainfall20)
        peak20_ref = np.max(h20_ref, axis=0)

        t0 = time.time()
        rec_s, h5_s = run_itzi_5m(dem5, bldg5, rainfall5, pipe_nodes=None)
        print(f"  surface done in {time.time() - t0:.1f}s peak={h5_s.max():.3f}")
        t0 = time.time()
        rec_p, h5_p = run_itzi_5m(dem5, bldg5, rainfall5, pipe_nodes=pipe_nodes5)
        print(f"  pipe done in {time.time() - t0:.1f}s peak={h5_p.max():.3f}")

        h20_s = block_mean_5_to_20(h5_s)
        h20_p = block_mean_5_to_20(h5_p)
        surface_vol = float(rec_s["vol_m3"][-1])
        pipe_vol = float(rec_p["vol_m3"][-1])
        surface_flooded = int(rec_s["flooded_cells"][-1])
        pipe_flooded = int(rec_p["flooded_cells"][-1])
        row = {
            "event": event,
            "mode": "itzi_5m_refined_from_public_20m_inputs",
            "window_km": "1.0x1.4",
            "grid_5m_cells": f"{dem5.shape[0]}x{dem5.shape[1]}",
            "pipe_nodes_5m_window": len(pipe_nodes5),
            "mike20_peak_m": float(np.max(peak20_ref)),
            "itzi5_surface_peak_m": float(np.max(h5_s)),
            "itzi5_pipe_peak_m": float(np.max(h5_p)),
            "itzi5_surface_agg20_peak_m": float(np.max(h20_s)),
            "itzi5_pipe_agg20_peak_m": float(np.max(h20_p)),
            "pipe_peak_reduction_pct": float((np.max(h5_s) - np.max(h5_p)) / np.max(h5_s) * 100.0),
            "surface_volume_m3": surface_vol,
            "pipe_volume_m3": pipe_vol,
            "pipe_volume_reduction_pct": float((surface_vol - pipe_vol) / surface_vol * 100.0) if surface_vol else 0.0,
            "drained_volume_m3": float(rec_p["drained_m3"][-1]),
            "surface_flooded_cells": surface_flooded,
            "pipe_flooded_cells": pipe_flooded,
            "pipe_flooded_reduction_pct": float((surface_flooded - pipe_flooded) / surface_flooded * 100.0) if surface_flooded else 0.0,
        }
        rows.append(row)
        np.savez(
            OUT_DIR / f"{event}_itzi_5m_refined.npz",
            peak20_ref=peak20_ref,
            h5_surface=h5_s,
            h5_pipe=h5_p,
            h20_surface_agg=h20_s,
            h20_pipe_agg=h20_p,
            rec_surface=rec_s,
            rec_pipe=rec_p,
            dem5=dem5,
            bldg5=bldg5,
        )

    write_csv(rows, OUT_DIR / "itzi_5m_refined_metrics.csv")
    make_figures(rows)
    (OUT_DIR / "README_5m_refined.txt").write_text(
        "This folder contains real ITZI runs on a 5 m refined grid derived from public 20 m DEM/rainfall fields.\n"
        "It is a 5 m ITZI numerical sensitivity experiment, not a true 5 m MIKE+ reference validation.\n"
        "Results are also aggregated back to 20 m for comparison with the available MIKE+ 20 m reference.\n",
        encoding="utf-8",
    )
    print(f"\nSaved 5 m refined ITZI outputs to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
