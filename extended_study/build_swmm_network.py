#!/usr/bin/env python3
"""
Build a SWMM dynamic-wave pipe-network model from the current OSM-derived
conceptual pipe network.

This is the "full 1D pipe routing" branch of the extended study. It is kept
separate from the existing ITZI inlet-sink branch so both drainage
representations can be compared:

1. ITZI surface-only
2. ITZI + road-aligned conceptual inlet-sink drainage
3. SWMM 1D dynamic-wave pipe-network routing

The generated SWMM model uses the OSM-derived nodes/links as junctions/conduits
and assigns subcatchments in the formal 4.0 km x 5.6 km ITZI comparison window
to the nearest in-window pipe node. Rainfall is the available public 20 m
spatial field averaged over that window and converted to intensity.

This creates a real SWMM hydraulic network input file. Running uses either an
EPA SWMM command-line executable (`swmm5.exe` or equivalent) or the PySWMM
runtime when available. If neither runtime is available, the script still
writes all .inp files and a status table.
"""

from __future__ import annotations

import csv
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEM_PATH = ROOT / "LarNO-main" / "benchmark" / "urbanflood" / "geodata" / "region1_20m" / "dem.npy"
FLOOD_DIR = ROOT / "LarNO-main" / "benchmark" / "urbanflood" / "flood" / "region1_20m"
NET_PATH = ROOT / "extended_study" / "output" / "osm_merged_network.npz"
OUT_DIR = ROOT / "extended_study" / "output" / "swmm_network"

EVENTS = [
    "event1", "event20",
    "event65", "event66", "event67", "event68", "event69", "event70",
    "event71", "event72", "event73", "event74", "event75", "event76",
    "event77", "event78", "event80",
]
CELL = 20.0
Y0, Y1 = 80, 280
X0, X1 = 120, 400


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in str(value))


def find_swmm_exe() -> str | None:
    candidates = []
    env = os.environ.get("SWMM5_EXE")
    if env:
        candidates.append(env)
    for name in ["swmm5.exe", "runswmm.exe", "swmm.exe"]:
        found = shutil.which(name)
        if found:
            candidates.append(found)
    candidates.extend([
        r"C:\Program Files\EPA SWMM 5.2.4 (64-bit)\runswmm.exe",
        r"C:\Program Files\EPA SWMM 5.2.4 (64-bit)\swmm5.exe",
        r"C:\Program Files\EPA SWMM 5.2.3 (64-bit)\runswmm.exe",
        r"C:\Program Files (x86)\EPA SWMM 5.2\runswmm.exe",
    ])
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return str(candidate)
    return None


def load_network() -> tuple[list[dict], list[dict]]:
    net = np.load(NET_PATH, allow_pickle=True)
    nodes = [dict(n) for n in net["nodes"].tolist()]
    links = [dict(l) for l in net["links"].tolist()]
    return nodes, links


def estimate_node_areas(nodes: list[dict]) -> dict[str, float]:
    """Assign active cells in the ITZI comparison window to nearest pipe node."""
    dem = np.load(DEM_PATH).astype(np.float32)
    bldg = dem >= 49.9
    active = ~bldg[Y0:Y1, X0:X1]
    region_nodes = []
    for n in nodes:
        row = int(n.get("row", -9999))
        col = int(n.get("col", -9999))
        if Y0 <= row < Y1 and X0 <= col < X1:
            region_nodes.append(n)
    if not region_nodes:
        return {}

    node_xy = np.array([
        [float(n.get("x_m", float(n["col"]) * CELL)), float(n.get("y_m", float(n["row"]) * CELL))]
        for n in region_nodes
    ], dtype=np.float64)
    node_ids = [safe_name(n["id"]) for n in region_nodes]
    rr, cc = np.where(active)
    cell_xy = np.column_stack(((cc + X0 + 0.5) * CELL, (rr + Y0 + 0.5) * CELL))

    counts = np.zeros(len(region_nodes), dtype=np.int64)
    chunk = 2048
    for start in range(0, len(cell_xy), chunk):
        part = cell_xy[start:start + chunk]
        dist2 = ((part[:, None, :] - node_xy[None, :, :]) ** 2).sum(axis=2)
        nearest = np.argmin(dist2, axis=1)
        counts += np.bincount(nearest, minlength=len(region_nodes))

    return {node_ids[i]: float(counts[i]) * CELL * CELL for i in range(len(node_ids)) if counts[i] > 0}


def event_mean_rain_intensity(event: str) -> list[float]:
    rain = np.load(FLOOD_DIR / event / "rainfall.npy")[:, Y0:Y1, X0:X1]
    dem = np.load(DEM_PATH)
    active = dem[Y0:Y1, X0:X1] < 49.9
    # rainfall.npy is treated as mm per 5 min. SWMM intensity is mm/hr.
    return [float(np.nanmean(frame[active]) * 12.0) for frame in rain]


def write_inp(event: str, nodes: list[dict], links: list[dict], areas_m2: dict[str, float]) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    inp = OUT_DIR / f"{event}_osm_dynamic_wave.inp"
    rain_ts = event_mean_rain_intensity(event)

    node_lookup = {safe_name(n["id"]): n for n in nodes}
    outfall_ids = {
        safe_name(n["id"]) for n in nodes
        if str(n.get("type", "")).lower() == "outfall"
    }
    if not outfall_ids:
        # Fallback: use the lowest-invert node as an artificial free outfall.
        lowest = min(nodes, key=lambda n: float(n.get("invert", n.get("elevation", 0.0))))
        outfall_ids.add(safe_name(lowest["id"]))

    lines: list[str] = []
    lines += ["[TITLE]", f"; OSM-derived dynamic-wave SWMM network for {event}", ""]
    lines += [
        "[OPTIONS]",
        "FLOW_UNITS           CMS",
        "INFILTRATION         HORTON",
        "FLOW_ROUTING         DYNWAVE",
        "LINK_OFFSETS         DEPTH",
        "MIN_SLOPE            0.0005",
        "ALLOW_PONDING        YES",
        "SKIP_STEADY_STATE    NO",
        "START_DATE           01/01/2020",
        "START_TIME           00:00:00",
        "REPORT_START_DATE    01/01/2020",
        "REPORT_START_TIME    00:00:00",
        "END_DATE             01/01/2020",
        "END_TIME             06:00:00",
        "REPORT_STEP          00:05:00",
        "WET_STEP             00:05:00",
        "DRY_STEP             01:00:00",
        "ROUTING_STEP         00:00:30",
        "INERTIAL_DAMPING     PARTIAL",
        "NORMAL_FLOW_LIMITED  BOTH",
        "FORCE_MAIN_EQUATION  H-W",
        "VARIABLE_STEP        0.75",
        "LENGTHENING_STEP     0",
        "MIN_SURFAREA         1.14",
        "MAX_TRIALS           8",
        "HEAD_TOLERANCE       0.0015",
        "SYS_FLOW_TOL         5",
        "LAT_FLOW_TOL         5",
        "",
    ]
    lines += ["[EVAPORATION]", "CONSTANT 0.0", ""]
    lines += ["[RAINGAGES]", ";;Name Format Interval SCF Source", f"RG_{event} INTENSITY 0:05 1.0 TIMESERIES TS_{event}", ""]
    lines += ["[SUBCATCHMENTS]", ";;Name RainGage Outlet Area %Imperv Width Slope CurbLen"]
    for node_id, area_m2 in sorted(areas_m2.items()):
        area_ha = max(area_m2 / 10000.0, 0.0001)
        width_m = max(area_m2 ** 0.5, 5.0)
        lines.append(f"S_{node_id} RG_{event} {node_id} {area_ha:.6f} 85 {width_m:.3f} 0.010 0")
    lines += ["", "[SUBAREAS]", ";;Subcatch N-Imperv N-Perv S-Imperv S-Perv PctZero RouteTo PctRouted"]
    for node_id in sorted(areas_m2):
        lines.append(f"S_{node_id} 0.015 0.15 1.0 3.0 25 OUTLET 100")
    lines += ["", "[INFILTRATION]", ";;Subcatch MaxRate MinRate Decay DryTime MaxInfil"]
    for node_id in sorted(areas_m2):
        lines.append(f"S_{node_id} 0.0 0.0 0.0 0.0 0.0")

    lines += ["", "[JUNCTIONS]", ";;Name Elev MaxDepth InitDepth SurDepth Aponded"]
    for node_id, n in sorted(node_lookup.items()):
        if node_id in outfall_ids:
            continue
        elev = float(n.get("invert", n.get("elevation", 0.0)))
        max_depth = max(float(n.get("elevation", elev)) - elev + 3.0, 2.0)
        lines.append(f"{node_id} {elev:.3f} {max_depth:.3f} 0.0 1.0 50.0")

    lines += ["", "[OUTFALLS]", ";;Name Elev Type StageData Gated RouteTo"]
    for node_id in sorted(outfall_ids):
        n = node_lookup[node_id]
        elev = float(n.get("invert", n.get("elevation", 0.0)))
        lines.append(f"{node_id} {elev:.3f} FREE NO")

    lines += ["", "[CONDUITS]", ";;Name FromNode ToNode Length Roughness InOffset OutOffset InitFlow MaxFlow"]
    valid_nodes = set(node_lookup)
    for l in links:
        lid = safe_name(l["id"])
        a = safe_name(l["from_node"])
        b = safe_name(l["to_node"])
        if a not in valid_nodes or b not in valid_nodes or a == b:
            continue
        length = max(float(l.get("length", CELL)), 1.0)
        rough = float(l.get("mannings_n", 0.013))
        lines.append(f"{lid} {a} {b} {length:.3f} {rough:.4f} 0 0 0 0")

    lines += ["", "[XSECTIONS]", ";;Link Shape Geom1 Geom2 Geom3 Geom4 Barrels Culvert"]
    for l in links:
        lid = safe_name(l["id"])
        dia = max(float(l.get("diameter", 0.6)), 0.2)
        lines.append(f"{lid} CIRCULAR {dia:.3f} 0 0 0 1")

    lines += ["", "[TIMESERIES]", ";;Name Time Value"]
    lines.append(f"TS_{event} 00:00 0.000")
    for i, intensity in enumerate(rain_ts):
        total_min = (i + 1) * 5
        hh, mm = divmod(total_min, 60)
        lines.append(f"TS_{event} {hh:02d}:{mm:02d} {intensity:.6f}")

    lines += ["", "[REPORT]", "INPUT NO", "CONTROLS NO", "SUBCATCHMENTS ALL", "NODES ALL", "LINKS ALL", ""]
    lines += ["[COORDINATES]", ";;Node X Y"]
    for node_id, n in sorted(node_lookup.items()):
        lines.append(f"{node_id} {float(n.get('x_m', 0.0)):.3f} {float(n.get('y_m', 0.0)):.3f}")
    lines += [""]

    inp.write_text("\n".join(lines), encoding="utf-8")
    return inp


def run_swmm(inp: Path, swmm_exe: str | None) -> tuple[str, Path, Path]:
    rpt = inp.with_suffix(".rpt")
    out = inp.with_suffix(".out")
    err = inp.with_suffix(".error.txt")
    if not swmm_exe:
        try:
            from pyswmm import Simulation
            with Simulation(str(inp), reportfile=str(rpt), outputfile=str(out)) as sim:
                for _ in sim:
                    pass
            err.unlink(missing_ok=True)
            return "run_ok_pyswmm", rpt, out
        except Exception as exc:
            err.write_text(str(exc), encoding="utf-8")
            return "run_failed_pyswmm", rpt, out
    try:
        subprocess.run([swmm_exe, str(inp), str(rpt), str(out)], check=True, cwd=str(OUT_DIR))
        err.unlink(missing_ok=True)
        return "run_ok", rpt, out
    except Exception as exc:
        err.write_text(str(exc), encoding="utf-8")
        return "run_failed", rpt, out


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    nodes, links = load_network()
    areas = estimate_node_areas(nodes)
    swmm_exe = find_swmm_exe()
    rows = []
    for event in EVENTS:
        inp = write_inp(event, nodes, links, areas)
        status, rpt, out = run_swmm(inp, swmm_exe)
        rows.append({
            "event": event,
            "status": status,
            "swmm_exe": swmm_exe or "",
            "inp": str(inp),
            "rpt_exists": rpt.exists(),
            "out_exists": out.exists(),
            "nodes": len(nodes),
            "links": len(links),
            "subcatchments": len(areas),
            "drainage_area_m2": round(sum(areas.values()), 3),
        })

    with (OUT_DIR / "swmm_build_status.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    readme = OUT_DIR / "README_SWMM_NETWORK.txt"
    readme.write_text(
        "SWMM dynamic-wave pipe-network branch.\n\n"
        "This folder contains SWMM .inp files generated from the OSM-derived pipe network.\n"
        "It is the full 1D pipe-routing comparison branch, separate from the ITZI inlet-sink branch.\n"
        "If no SWMM executable is available, files are generated but not run. Set SWMM5_EXE to runswmm.exe/swmm5.exe and rerun.\n\n"
        "Current coupling level: standalone 1D SWMM dynamic-wave routing driven by event rainfall-runoff subcatchments.\n"
        "It is not yet a fully two-way ITZI-SWMM surface/sewer co-simulation.\n",
        encoding="utf-8",
    )
    print(f"SWMM executable: {swmm_exe or 'not found'}")
    print(f"Wrote SWMM files to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
