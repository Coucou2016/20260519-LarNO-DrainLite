#!/usr/bin/env python3
"""Generate 72-step ITZI drainage labels for LarNO-D dataset v1.

This script runs the corrected ITZI dynamic surface solver on the formal
4.0 km x 5.6 km comparison window and records water depth every 5 minutes.
It writes both surface-only and conceptual road-aligned inlet-sink scenarios.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import timedelta
from pathlib import Path

import numpy as np

import itzi.rasterdomain as rasterdomain
import itzi.surfaceflow as surfaceflow
from itzi.hydrology import Hydrology
from itzi.infiltration import InfNull


ROOT = Path(__file__).resolve().parents[1]
DEM_PATH = ROOT / "LarNO-main" / "benchmark" / "urbanflood" / "geodata" / "region1_20m" / "dem.npy"
FLOOD_DIR = ROOT / "LarNO-main" / "benchmark" / "urbanflood" / "flood" / "region1_20m"
NET_PATH = ROOT / "extended_study" / "output" / "osm_merged_network.npz"
OUT_DIR = ROOT / "extended_study" / "output" / "itzi_drainage_timeseries"

DEFAULT_EVENTS = ["event1", "event20", "event65", "event66", "event67", "event68", "event69", "event70"]
CELL = 20.0
CELL_AREA = CELL * CELL
G = 9.81
Y0, Y1 = 80, 280
X0, X1 = 120, 400


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", nargs="*", default=DEFAULT_EVENTS)
    parser.add_argument("--surface-only", action="store_true", help="Only run surface-only labels.")
    parser.add_argument("--sink-only", action="store_true", help="Only run inlet-sink labels.")
    return parser.parse_args()


def load_pipe_nodes(bldg: np.ndarray) -> list[dict]:
    net = np.load(NET_PATH, allow_pickle=True)
    nodes = [dict(n) for n in net["nodes"].tolist()]
    pipe_nodes = []
    h, w = bldg.shape
    for n in nodes:
        row = int(n["row"]) - Y0
        col = int(n["col"]) - X0
        if 0 <= row < h and 0 <= col < w and not bldg[row, col]:
            pipe_nodes.append({
                "row": row,
                "col": col,
                "invert": float(n.get("invert", n.get("elevation", 0.0))),
                "type": n.get("type", "junction"),
            })
    return pipe_nodes


def run_itzi_event(
    dem: np.ndarray,
    bldg: np.ndarray,
    rainfall_3d: np.ndarray,
    pipe_nodes: list[dict] | None,
) -> tuple[np.ndarray, dict]:
    h, w = dem.shape
    active = ~bldg
    duration = rainfall_3d.shape[0] * 300.0
    dt_rain = 300.0

    dem_itzi = dem.astype(np.float32).copy()
    dem_itzi[bldg] = 50.0

    domain = rasterdomain.RasterDomain(
        dtype=np.float32,
        arr_mask=np.zeros((h, w), dtype=bool),
        cell_shape=(CELL, CELL),
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
    friction[bldg] = 100.0
    domain.update_array(k="dem", arr=dem_itzi)
    domain.update_array(k="friction", arr=friction)
    domain.update_array(k="h", arr=np.zeros((h, w), dtype=np.float32))
    domain.update_array(k="rain", arr=np.zeros((h, w), dtype=np.float32))
    sf_sim.update_flow_dir()

    inlets = []
    if pipe_nodes:
        inlets = [
            (int(n["row"]), int(n["col"]), float(n["invert"]))
            for n in pipe_nodes
            if 0 <= int(n["row"]) < h and 0 <= int(n["col"]) < w and active[int(n["row"]), int(n["col"])]
        ]

    t = 0.0
    dt = 0.001
    rain_idx = 0
    next_rain = 0.0
    next_drain = 30.0
    next_record = 300.0
    n_active = max(int(active.sum()), 1)
    drained = 0.0
    frames: list[np.ndarray] = []
    records = {"time_s": [], "vol_m3": [], "hmax_m": [], "flooded_cells": [], "drained_m3": []}

    while t < duration:
        if t >= next_rain and rain_idx < rainfall_3d.shape[0]:
            rain_2d = rainfall_3d[rain_idx]
            building_rain = float(np.sum(rain_2d[bldg]))
            building_extra = building_rain / n_active
            rain_field = np.zeros((h, w), dtype=np.float32)
            rain_field[active] = (rain_2d[active] + building_extra) / 1000.0 / dt_rain
            domain.update_array(k="rain", arr=rain_field)
            rain_idx += 1
            next_rain += dt_rain

        domain.update_ext_array()
        hydrology.solve_dt()
        hydrology.step()

        if inlets and t >= next_drain:
            h_arr = domain.get_array("h")
            for row, col, invert in inlets:
                if h_arr[row, col] > 0.01:
                    surf_wl = dem_itzi[row, col] + h_arr[row, col]
                    head = max(float(surf_wl - invert), 0.01)
                    q = 0.65 * 2.0 * np.sqrt(2.0 * G * head)
                    q = min(q, float(h_arr[row, col]) * CELL_AREA / 30.0 * 0.3)
                    removal = q * 30.0 / CELL_AREA
                    h_arr[row, col] = max(0.0, float(h_arr[row, col]) - removal)
                    drained += q * 30.0
            domain.update_array(k="h", arr=h_arr)
            next_drain += 30.0

        sf_sim.dt = timedelta(seconds=dt)
        sf_sim.step()
        sf_sim.solve_dt()
        t += dt
        dt = sf_sim.dt.total_seconds()

        while t >= next_record and len(frames) < rainfall_3d.shape[0]:
            h_arr = domain.get_array("h").astype(np.float32).copy()
            frames.append(h_arr)
            records["time_s"].append(float(next_record))
            records["vol_m3"].append(float(np.sum(h_arr[active]) * CELL_AREA))
            records["hmax_m"].append(float(np.max(h_arr)))
            records["flooded_cells"].append(int(np.sum(h_arr > 0.03)))
            records["drained_m3"].append(float(drained))
            next_record += 300.0

    if len(frames) != rainfall_3d.shape[0]:
        raise RuntimeError(f"Expected {rainfall_3d.shape[0]} frames, got {len(frames)}")
    return np.stack(frames, axis=0), records


def main() -> int:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dem_full = np.load(DEM_PATH, allow_pickle=True).astype(np.float32)
    bldg_full = dem_full >= 49.9
    dem = dem_full[Y0:Y1, X0:X1].copy()
    bldg = bldg_full[Y0:Y1, X0:X1].copy()
    pipe_nodes = load_pipe_nodes(bldg)

    metadata = {
        "source": "ITZI SurfaceFlowSimulation with InfNull infiltration",
        "cell_size_m": CELL,
        "window": {"row_start": Y0, "row_stop": Y1, "col_start": X0, "col_stop": X1},
        "rainfall_units": "rainfall.npy interpreted as mm per 5 min",
        "building_rainfall": "uniformly redistributed from building cells to active non-building cells",
        "sink_model": "conceptual road-aligned inlet sink; not full 1D pipe routing",
        "pipe_node_count": len(pipe_nodes),
    }
    (OUT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    run_surface = not args.sink_only
    run_sink = not args.surface_only
    for event in args.events:
        event_dir = OUT_DIR / event
        event_dir.mkdir(parents=True, exist_ok=True)
        rainfall = np.load(FLOOD_DIR / event / "rainfall.npy")[:, Y0:Y1, X0:X1].astype(np.float32)
        print(f"\n=== {event}: rainfall={rainfall.shape}, pipe_nodes={len(pipe_nodes)} ===")

        if run_surface:
            t0 = time.time()
            h_surface, rec_surface = run_itzi_event(dem, bldg, rainfall, pipe_nodes=None)
            np.save(event_dir / "h_itzi_surface.npy", h_surface)
            np.savez(event_dir / "records_surface.npz", records=rec_surface)
            print(f"surface: frames={h_surface.shape}, peak={h_surface.max():.3f}, time={time.time()-t0:.1f}s")

        if run_sink:
            t0 = time.time()
            h_sink, rec_sink = run_itzi_event(dem, bldg, rainfall, pipe_nodes=pipe_nodes)
            np.save(event_dir / "h_itzi_sink.npy", h_sink)
            np.savez(event_dir / "records_sink.npz", records=rec_sink)
            print(f"sink: frames={h_sink.shape}, peak={h_sink.max():.3f}, time={time.time()-t0:.1f}s")

    print(f"\nSaved ITZI drainage time-series labels to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
