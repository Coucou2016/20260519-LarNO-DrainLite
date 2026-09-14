#!/usr/bin/env python3
"""Build a clean LarNO-style dataset from fixed native ITZI-SWMM outputs."""

from __future__ import annotations

import csv
import json
import math
import shutil
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC_FLOOD = ROOT / "LarNO-main" / "benchmark" / "urbanflood" / "flood" / "region1_20m_drainage_v1"
SRC_GEO = ROOT / "LarNO-main" / "benchmark" / "urbanflood" / "geodata" / "region1_20m_drainage_v1"
CONNECTED = ROOT / "extended_study" / "output" / "connected_itzi_swmm"
CONNECTED_METRICS = CONNECTED / "connected_itzi_swmm_metrics.csv"
INP = (
    ROOT
    / "external_models"
    / "20260518-itzi-flood"
    / "test_cases"
    / "shenzhen_region1"
    / "input_data"
    / "networks"
    / "swmm_connected_sub.inp"
)
OUT_GEO = ROOT / "LarNO-main" / "benchmark" / "urbanflood" / "geodata" / "region1_20m_connected_swmm_v1"
OUT_FLOOD = ROOT / "LarNO-main" / "benchmark" / "urbanflood" / "flood" / "region1_20m_connected_swmm_v1"
OUT_AUDIT = ROOT / "extended_study" / "output" / "connected_swmm_dataset"

CELL = 20.0
H_FULL = 400
W_FULL = 560
ROW_OFF = 80
COL_OFF = 120
CELL_AREA_M2 = CELL * CELL


def read_sections(path: Path) -> dict[str, list[list[str]]]:
    sections: dict[str, list[list[str]]] = {}
    cur = None
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith(";"):
            continue
        if line.startswith("[") and line.endswith("]"):
            cur = line.upper()
            sections.setdefault(cur, [])
            continue
        if cur:
            sections[cur].append(line.split())
    return sections


def local_rc(x_m: float, y_m: float) -> tuple[int, int]:
    """Use the exact coordinate-to-cell rule used by the ITZI coupling runner."""
    col = int(x_m / CELL) - COL_OFF
    row = int((H_FULL * CELL - y_m) / CELL) - ROW_OFF
    return row, col


def inside(row: int, col: int, shape: tuple[int, int]) -> bool:
    return 0 <= row < shape[0] and 0 <= col < shape[1]


def line_cells(r0: int, c0: int, r1: int, c1: int, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    n = int(max(abs(r1 - r0), abs(c1 - c0))) + 1
    n = max(n, 2)
    # Truncation toward zero matches int() in the coupled runner.
    rr = np.trunc(np.linspace(r0, r1, n)).astype(np.int32)
    cc = np.trunc(np.linspace(c0, c1, n)).astype(np.int32)
    mask = (rr >= 0) & (rr < shape[0]) & (cc >= 0) & (cc < shape[1])
    if not np.any(mask):
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)
    flat = np.unique(np.ravel_multi_index((rr[mask], cc[mask]), shape))
    return np.unravel_index(flat, shape)


def full_pipe_capacity(diameter_m: float, slope: float, mannings_n: float) -> float:
    diameter_m = max(float(diameter_m), 0.05)
    slope = max(float(abs(slope)), 1.0e-5)
    mannings_n = max(float(mannings_n), 0.005)
    area = math.pi * diameter_m * diameter_m / 4.0
    hydraulic_radius = diameter_m / 4.0
    return (1.0 / mannings_n) * area * hydraulic_radius ** (2.0 / 3.0) * math.sqrt(slope)


def parse_swmm_network(shape: tuple[int, int]) -> dict[str, np.ndarray | dict]:
    sections = read_sections(INP)
    junctions: dict[str, dict[str, float]] = {}
    outfalls: dict[str, dict[str, float]] = {}
    coords: dict[str, tuple[float, float, float, float]] = {}
    diameters: dict[str, float] = {}

    for row in sections.get("[JUNCTIONS]", []):
        junctions[row[0]] = {"invert": float(row[1])}
    for row in sections.get("[OUTFALLS]", []):
        outfalls[row[0]] = {"invert": float(row[1])}
    for row in sections.get("[COORDINATES]", []):
        node_id = row[0]
        x_m, y_m = float(row[1]), float(row[2])
        rr, cc = local_rc(x_m, y_m)
        coords[node_id] = (x_m, y_m, rr, cc)
    for row in sections.get("[XSECTIONS]", []):
        diameters[row[0]] = float(row[2])

    dem = np.load(SRC_GEO / "dem.npy").astype(np.float32)
    active = np.isfinite(dem) & (dem < 49.9)
    inlet_mask = np.zeros(shape, dtype=np.float32)
    inlet_count = np.zeros(shape, dtype=np.float32)
    outfall_mask = np.zeros(shape, dtype=np.float32)
    pipe_mask = np.zeros(shape, dtype=np.float32)
    pipe_segment_count = np.zeros(shape, dtype=np.float32)
    pipe_diameter = np.zeros(shape, dtype=np.float32)
    pipe_slope = np.zeros(shape, dtype=np.float32)
    pipe_capacity = np.zeros(shape, dtype=np.float32)
    pipe_cover_depth_sum = np.zeros(shape, dtype=np.float32)
    pipe_cover_depth_count = np.zeros(shape, dtype=np.float32)

    coupled_junction_count = 0
    for node_id, data in junctions.items():
        if node_id not in coords:
            continue
        _, _, rr, cc = coords[node_id]
        if inside(rr, cc, shape):
            if active[rr, cc]:
                inlet_count[rr, cc] += 1.0
                inlet_mask[rr, cc] = 1.0
                coupled_junction_count += 1

    outfall_points: list[tuple[float, float]] = []
    for node_id in outfalls:
        if node_id not in coords:
            continue
        _, _, rr, cc = coords[node_id]
        outfall_points.append((rr, cc))
        if inside(rr, cc, shape):
            outfall_mask[rr, cc] = 1.0

    link_count = 0
    for row in sections.get("[CONDUITS]", []):
        link_id, from_id, to_id = row[0], row[1], row[2]
        if from_id not in coords or to_id not in coords:
            continue
        r0, c0 = coords[from_id][2], coords[from_id][3]
        r1, c1 = coords[to_id][2], coords[to_id][3]
        rr, cc = line_cells(r0, c0, r1, c1, shape)
        if rr.size == 0:
            continue
        from_invert = junctions.get(from_id, outfalls.get(from_id, {})).get("invert")
        to_invert = junctions.get(to_id, outfalls.get(to_id, {})).get("invert")
        length = max(float(row[3]), CELL)
        mannings_n = float(row[4])
        diameter = diameters.get(link_id, 1.2)
        slope = 1.0e-5
        if from_invert is not None and to_invert is not None:
            slope = max((float(from_invert) - float(to_invert)) / length, 1.0e-5)
        capacity = full_pipe_capacity(diameter, slope, mannings_n)

        pipe_mask[rr, cc] = 1.0
        pipe_segment_count[rr, cc] += 1.0
        pipe_diameter[rr, cc] = np.maximum(pipe_diameter[rr, cc], diameter)
        pipe_slope[rr, cc] = np.maximum(pipe_slope[rr, cc], slope)
        pipe_capacity[rr, cc] = np.maximum(pipe_capacity[rr, cc], capacity)
        if from_invert is not None and to_invert is not None:
            cover = np.maximum(dem[rr, cc] - ((float(from_invert) + float(to_invert)) / 2.0), 0.0)
            pipe_cover_depth_sum[rr, cc] += np.nan_to_num(cover, nan=0.0)
            pipe_cover_depth_count[rr, cc] += 1.0
        link_count += 1

    pipe_cover_depth = np.zeros(shape, dtype=np.float32)
    covered = pipe_cover_depth_count > 0
    pipe_cover_depth[covered] = pipe_cover_depth_sum[covered] / pipe_cover_depth_count[covered]

    rr, cc = np.indices(shape, dtype=np.float32)
    distance = np.full(shape, np.inf, dtype=np.float32)
    for r_out, c_out in outfall_points:
        distance = np.minimum(distance, np.sqrt((rr - r_out) ** 2 + (cc - c_out) ** 2) * CELL)
    distance[~np.isfinite(distance)] = 0.0

    return {
        "dem": dem,
        "drain_inlet_mask": inlet_mask,
        "drain_inlet_count": inlet_count,
        "drain_outfall_mask": outfall_mask,
        "pipe_mask": pipe_mask,
        "pipe_segment_count": pipe_segment_count,
        "pipe_diameter": pipe_diameter,
        "pipe_slope": pipe_slope,
        "pipe_capacity": pipe_capacity,
        "pipe_cover_depth": pipe_cover_depth,
        "distance_to_outfall": distance.astype(np.float32),
        "audit": {
            "inp": str(INP),
            "junction_count": len(junctions),
            "outfall_count": len(outfalls),
            "conduit_count": len(sections.get("[CONDUITS]", [])),
            "coupled_junction_cells": int(inlet_mask.sum()),
            "coupled_junction_count": int(inlet_count.sum()),
            "multi_junction_cells": int(np.count_nonzero(inlet_count > 1)),
            "rasterized_conduits": link_count,
            "pipe_cells": int(pipe_mask.sum()),
            "overlapping_pipe_cells": int(np.count_nonzero(pipe_segment_count > 1)),
            "outfall_cells": int(outfall_mask.sum()),
            "coordinate_mapping": "int(x / cell), int((H * cell - y) / cell), identical to ITZI runner",
            "outfall_raster_rule": "window-exterior outfalls remain outside; no clipping to boundary cells",
        },
    }


def read_metrics() -> dict[str, dict[str, str]]:
    out = {}
    with CONNECTED_METRICS.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            out[row["event"]] = row
    return out


def write_event_metrics(path: Path, row: dict[str, str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def build_flood_dataset(events: list[str], metrics: dict[str, dict[str, str]]) -> list[dict[str, object]]:
    rows = []
    OUT_FLOOD.mkdir(parents=True, exist_ok=True)
    for event in events:
        src_npz = CONNECTED / event / f"{event}_connected_itzi_swmm.npz"
        if not src_npz.exists():
            continue
        event_out = OUT_FLOOD / event
        event_out.mkdir(parents=True, exist_ok=True)
        z = np.load(src_npz, allow_pickle=True)
        surface = z["h_surf"].astype(np.float32)
        connected = z["h_swmm_connected"].astype(np.float32)
        residual = (connected - surface).astype(np.float32)
        rainfall = np.load(SRC_FLOOD / event / "rainfall.npy").astype(np.float32)
        mike = np.load(SRC_FLOOD / event / "h_mike_ref.npy").astype(np.float32)

        np.save(event_out / "rainfall.npy", rainfall)
        np.save(event_out / "h_mike_ref.npy", mike)
        np.save(event_out / "h_itzi_surface.npy", surface)
        np.save(event_out / "h_itzi_swmm_connected.npy", connected)
        np.save(event_out / "h_connected_residual.npy", residual)
        np.save(event_out / "h.npy", connected)
        if event in metrics:
            write_event_metrics(event_out / "swmm_connected_metrics.csv", metrics[event])

        row = {
            "event": event,
            "shape": list(connected.shape),
            "surface_peak_m": float(surface.max()),
            "connected_peak_m": float(connected.max()),
            "peak_reduction_mm": float((surface.max() - connected.max()) * 1000.0),
            "mean_abs_residual_mm": float(np.mean(np.abs(residual)) * 1000.0),
            "final_volume_reduction_m3": float((surface[-1].sum() - connected[-1].sum()) * CELL_AREA_M2),
            "continuity_error_pct": float(metrics.get(event, {}).get("continuity_error_pct", "nan")),
        }
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    OUT_GEO.mkdir(parents=True, exist_ok=True)
    OUT_AUDIT.mkdir(parents=True, exist_ok=True)
    metrics = read_metrics()
    events = [event for event in ["event1", "event20", "event65", "event66", "event67", "event68", "event69", "event70"] if event in metrics]

    geo = parse_swmm_network((200, 280))
    for key in [
        "dem",
        "drain_inlet_mask",
        "drain_inlet_count",
        "drain_outfall_mask",
        "pipe_mask",
        "pipe_segment_count",
        "pipe_diameter",
        "pipe_slope",
        "pipe_capacity",
        "pipe_cover_depth",
        "distance_to_outfall",
    ]:
        np.save(OUT_GEO / f"{key}.npy", geo[key])

    event_rows = build_flood_dataset(events, metrics)
    metadata = {
        "location": "region1_20m_connected_swmm_v1",
        "cell_size_m": 20.0,
        "shape": [200, 280],
        "time_steps": 72,
        "events": events,
        "target_h": "h.npy is native ITZI-SWMM coupled water depth from fixed connected conceptual SWMM network",
        "surface_baseline": "h_itzi_surface.npy is ITZI 2D dynamic surface-only water depth",
        "connected_label": "h_itzi_swmm_connected.npy is ITZI native DrainageSimulation + SWMM DYNWAVE coupled water depth",
        "residual": "h_connected_residual.npy = h_itzi_swmm_connected - h_itzi_surface, in m",
        "static_features": [
            "drain_inlet_mask",
            "drain_outfall_mask",
            "pipe_mask",
            "pipe_diameter",
            "pipe_slope",
            "pipe_capacity",
            "pipe_cover_depth",
            "distance_to_outfall",
        ],
        "source_workflow": str(ROOT / "extended_study" / "ITZI_SWMM_FIXED_WORKFLOW.md"),
        "network_audit": geo["audit"],
        "quality_note": "Events with abs(SWMM continuity error) > 2% must be reported with a stability warning.",
    }
    (OUT_GEO / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_FLOOD / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_AUDIT / "connected_swmm_dataset_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(OUT_AUDIT / "connected_swmm_dataset_event_summary.csv", event_rows)
    if INP.exists():
        shutil.copy2(INP, OUT_AUDIT / INP.name)
    print(json.dumps({"events": events, "geodata": str(OUT_GEO), "flood": str(OUT_FLOOD), "audit": geo["audit"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
