#!/usr/bin/env python3
"""Prototype one-event ITZI + PySWMM two-way coupling on the Shenzhen window.

This script is intentionally separate from the DrainLite training dataset.
It tests whether the existing 20 m ITZI window can exchange water online with
a SWMM dynamic-wave pipe network through ITZI's DrainageSimulation API.

Default run:
    python extended_study/run_itzi_swmm_coupled_prototype.py --event event75 --hours 1

Outputs are written to:
    extended_study/output/itzi_swmm_coupled_prototype/<event>/
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, r"E:\Miniconda3\Lib\site-packages")
import pyswmm
import itzi.rasterdomain as rasterdomain
import itzi.surfaceflow as surfaceflow
from itzi.drainage import CouplingTypes, DrainageLink, DrainageNode, DrainageSimulation
from itzi.hydrology import Hydrology
from itzi.infiltration import InfNull
from itzi.swmm_input_parser import SwmmInputParser


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "LarNO-main" / "benchmark" / "urbanflood"
GEO = BASE / "geodata" / "region1_20m_drainage_v1"
FLOOD = BASE / "flood" / "region1_20m_drainage_v1"
NET_PATH = ROOT / "extended_study" / "output" / "osm_merged_network.npz"
OUT_ROOT = ROOT / "extended_study" / "output" / "itzi_swmm_coupled_prototype"

CELL = 20.0
CELL_AREA = CELL * CELL
FULL_ROWS = 400
Y0, Y1 = 80, 280
X0, X1 = 120, 400
G = 9.80665


class _ClosedSwmmModel:
    """No-op replacement to avoid double-close errors from DrainageSimulation.__del__."""

    def swmm_report(self):
        return None

    def swmm_close(self):
        return None


def close_drainage(drainage: DrainageSimulation | None) -> None:
    if drainage is None:
        return
    try:
        drainage.swmm_model.swmm_report()
        drainage.swmm_model.swmm_close()
    except Exception:
        pass
    drainage.swmm_model = _ClosedSwmmModel()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--event", default="event75")
    parser.add_argument("--hours", type=float, default=1.0)
    parser.add_argument("--node-limit", type=int, default=0, help="Optional cap for coupled nodes; 0 means all in-window nodes.")
    parser.add_argument("--dtmax", type=float, default=1.0)
    return parser.parse_args()


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in str(value))


def load_network() -> tuple[list[dict], list[dict]]:
    net = np.load(NET_PATH, allow_pickle=True)
    nodes = [dict(n) for n in net["nodes"].tolist()]
    links = [dict(l) for l in net["links"].tolist()]
    return nodes, links


def write_network_only_swmm(nodes: list[dict], links: list[dict], out_dir: Path, event: str, hours: float) -> Path:
    """Write a SWMM network with no subcatchment rainfall to avoid double-counting rainfall."""
    out_dir.mkdir(parents=True, exist_ok=True)
    inp = out_dir / f"{event}_network_only_coupled.inp"
    node_lookup = {safe_name(n["id"]): n for n in nodes}
    outfall_ids = {
        safe_name(n["id"]) for n in nodes
        if str(n.get("type", "")).lower() == "outfall"
    }
    if not outfall_ids:
        lowest = min(nodes, key=lambda n: float(n.get("invert", n.get("elevation", 0.0))))
        outfall_ids.add(safe_name(lowest["id"]))

    end_hours = int(hours)
    end_minutes = int(round((hours - end_hours) * 60))
    lines: list[str] = []
    lines += ["[TITLE]", f"; Network-only SWMM dynamic-wave branch for ITZI coupling, {event}", ""]
    lines += [
        "[OPTIONS]",
        "FLOW_UNITS           CMS",
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
        f"END_TIME             {end_hours:02d}:{end_minutes:02d}:00",
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

    lines += ["[JUNCTIONS]", ";;Name Elev MaxDepth InitDepth SurDepth Aponded"]
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

    valid_nodes = set(node_lookup)
    valid_link_ids: list[str] = []
    lines += ["", "[CONDUITS]", ";;Name FromNode ToNode Length Roughness InOffset OutOffset InitFlow MaxFlow"]
    for link in links:
        lid = safe_name(link["id"])
        a = safe_name(link["from_node"])
        b = safe_name(link["to_node"])
        if a not in valid_nodes or b not in valid_nodes or a == b:
            continue
        length = max(float(link.get("length", CELL)), 1.0)
        rough = float(link.get("mannings_n", 0.013))
        lines.append(f"{lid} {a} {b} {length:.3f} {rough:.4f} 0 0 0 0")
        valid_link_ids.append(lid)

    lines += ["", "[XSECTIONS]", ";;Link Shape Geom1 Geom2 Geom3 Geom4 Barrels Culvert"]
    for link in links:
        lid = safe_name(link["id"])
        if lid not in valid_link_ids:
            continue
        dia = max(float(link.get("diameter", 0.6)), 0.2)
        lines.append(f"{lid} CIRCULAR {dia:.3f} 0 0 0 1")

    lines += ["", "[COORDINATES]", ";;Node X Y"]
    for node_id, n in sorted(node_lookup.items()):
        lines.append(f"{node_id} {float(n.get('x_m', 0.0)):.3f} {float(n.get('y_m', 0.0)):.3f}")

    lines += ["", "[REPORT]", "INPUT NO", "CONTROLS NO", "NODES ALL", "LINKS ALL", ""]
    inp.write_text("\n".join(lines), encoding="utf-8")
    return inp


def node_coord_to_window_cell(x: float, y: float) -> tuple[int, int]:
    full_col = int(np.floor(x / CELL))
    full_row = int(np.floor(FULL_ROWS - y / CELL))
    return full_row - Y0, full_col - X0


def setup_drainage(inp: Path, active: np.ndarray, node_limit: int = 0) -> tuple[DrainageSimulation, dict[str, tuple[int, int]], dict[str, object]]:
    swmm_sim = pyswmm.Simulation(str(inp))
    parser = SwmmInputParser(str(inp))
    nodes_coors = parser.get_nodes_id_as_dict()
    links_vertices = parser.get_links_id_as_dict()

    all_nodes = pyswmm.Nodes(swmm_sim)
    node_wrappers: list[DrainageNode] = []
    node_id_to_loc: dict[str, tuple[int, int]] = {}
    candidate_count = 0
    coupled_count = 0
    h, w = active.shape

    for pyswmm_node in all_nodes:
        coors = nodes_coors.get(pyswmm_node.nodeid)
        coupling = CouplingTypes.NOT_COUPLED
        if coors is not None and pyswmm_node.is_junction():
            row, col = node_coord_to_window_cell(coors.x, coors.y)
            if 0 <= row < h and 0 <= col < w and active[row, col]:
                candidate_count += 1
                if node_limit <= 0 or coupled_count < node_limit:
                    coupling = CouplingTypes.COUPLED_NO_FLOW
                    node_id_to_loc[pyswmm_node.nodeid] = (row, col)
                    coupled_count += 1
        node_wrappers.append(
            DrainageNode(
                node_object=pyswmm_node,
                coordinates=coors,
                coupling_type=coupling,
                g=G,
            )
        )

    pyswmm_links = pyswmm.Links(swmm_sim)
    link_wrappers: list[DrainageLink] = []
    for pyswmm_link in pyswmm_links:
        in_coor = nodes_coors.get(pyswmm_link.inlet_node)
        out_coor = nodes_coors.get(pyswmm_link.outlet_node)
        vertices = [in_coor] if in_coor else []
        vdata = links_vertices.get(pyswmm_link.linkid)
        if vdata and vdata.vertices:
            vertices.extend(vdata.vertices)
        if out_coor:
            vertices.append(out_coor)
        link_wrappers.append(DrainageLink(link_object=pyswmm_link, vertices=vertices))

    drainage = DrainageSimulation(swmm_sim, node_wrappers, link_wrappers)
    meta = {
        "swmm_nodes": len(node_wrappers),
        "swmm_links": len(link_wrappers),
        "candidate_in_window_active_junctions": candidate_count,
        "coupled_nodes": coupled_count,
        "node_limit": node_limit,
    }
    return drainage, node_id_to_loc, meta


def run_coupled_event(event: str, hours: float, node_limit: int, dtmax: float) -> dict[str, object]:
    out_dir = OUT_ROOT / event
    out_dir.mkdir(parents=True, exist_ok=True)
    nodes, links = load_network()
    inp = write_network_only_swmm(nodes, links, out_dir, event, hours)

    dem = np.load(GEO / "dem.npy").astype(np.float32)
    rainfall = np.load(FLOOD / event / "rainfall.npy").astype(np.float32)
    n_frames = min(rainfall.shape[0], max(1, int(round(hours * 12))))
    rainfall = rainfall[:n_frames]
    duration = n_frames * 300.0
    bldg = dem >= 49.9
    active = ~bldg
    h, w = dem.shape

    dem_itzi = dem.copy()
    dem_itzi[bldg] = 50.0
    friction = np.full((h, w), 0.015, dtype=np.float32)
    friction[bldg] = 100.0

    domain = rasterdomain.RasterDomain(
        dtype=np.float32,
        arr_mask=np.zeros((h, w), dtype=bool),
        cell_shape=(CELL, CELL),
    )
    sim_param = {
        "hmin": 0.001,
        "cfl": 0.7,
        "theta": 0.9,
        "g": G,
        "vrouting": 0.1,
        "dtmax": float(dtmax),
        "slmax": 0.1,
    }
    sf_sim = surfaceflow.SurfaceFlowSimulation(domain, sim_param)
    hydrology = Hydrology(domain, 60.0, InfNull(domain, 60.0))
    domain.update_array(k="dem", arr=dem_itzi)
    domain.update_array(k="friction", arr=friction)
    domain.update_array(k="h", arr=np.zeros((h, w), dtype=np.float32))
    domain.update_array(k="rain", arr=np.zeros((h, w), dtype=np.float32))
    sf_sim.update_flow_dir()

    drainage = None
    t0 = time.time()
    exchange_rows: list[dict[str, object]] = []
    frames: list[np.ndarray] = []
    records: list[dict[str, object]] = []
    surface_to_pipe_m3 = 0.0
    pipe_to_surface_m3 = 0.0
    meta: dict[str, object] = {}
    try:
        drainage, node_id_to_loc, meta = setup_drainage(inp, active, node_limit=node_limit)
        t = 0.0
        dt = 0.001
        rain_idx = 0
        next_rain = 0.0
        next_record = 300.0
        n_active = max(int(active.sum()), 1)
        drainage_done = False

        while t < duration:
            if t >= next_rain and rain_idx < rainfall.shape[0]:
                rain_2d = rainfall[rain_idx]
                building_rain = float(np.sum(rain_2d[bldg]))
                building_extra = building_rain / n_active
                rain_field = np.zeros((h, w), dtype=np.float32)
                rain_field[active] = (rain_2d[active] + building_extra) / 1000.0 / 300.0
                domain.update_array(k="rain", arr=rain_field)
                rain_idx += 1
                next_rain += 300.0

            hydrology.solve_dt()
            hydrology.step()

            if drainage and not drainage_done and t >= drainage.elapsed_time and drainage.elapsed_time < duration - 1.0:
                try:
                    drainage.step()
                except AssertionError:
                    # PySWMM returns a zero-length step after its configured
                    # end time. The ITZI wrapper asserts positive dt, so stop
                    # calling SWMM and keep the last drainage field unchanged.
                    drainage_done = True
                    close_drainage(drainage)
                    drainage = None
                    continue
                dt_drain = max(drainage.dt.total_seconds(), 1e-6)
                arr_z = domain.get_array("dem")
                arr_h = domain.get_array("h")
                surface_states = {
                    node_id: {"z": float(arr_z[row, col]), "h": float(arr_h[row, col])}
                    for node_id, (row, col) in node_id_to_loc.items()
                }
                coupling_flows = drainage.apply_coupling_to_nodes(surface_states, CELL_AREA)
                arr_qd = domain.get_array("n_drain")
                arr_qd.fill(0.0)
                q_surface_to_pipe = 0.0
                q_pipe_to_surface = 0.0
                for node_id, flow in coupling_flows.items():
                    row, col = node_id_to_loc[node_id]
                    arr_qd[row, col] = flow / CELL_AREA
                    if flow < 0:
                        q_surface_to_pipe += -float(flow)
                    elif flow > 0:
                        q_pipe_to_surface += float(flow)
                surface_to_pipe_m3 += q_surface_to_pipe * dt_drain
                pipe_to_surface_m3 += q_pipe_to_surface * dt_drain
                exchange_rows.append({
                    "time_s": round(t, 6),
                    "swmm_elapsed_s": round(drainage.elapsed_time, 6),
                    "dt_drain_s": round(dt_drain, 6),
                    "surface_to_pipe_cms": q_surface_to_pipe,
                    "pipe_to_surface_cms": q_pipe_to_surface,
                    "coupled_nodes": len(coupling_flows),
                })

            domain.update_ext_array()
            sf_sim.dt = timedelta(seconds=dt)
            sf_sim.step()
            sf_sim.solve_dt()
            t += dt
            dt = sf_sim.dt.total_seconds()

            while t >= next_record and len(frames) < rainfall.shape[0]:
                h_arr = domain.get_array("h").astype(np.float32).copy()
                frames.append(h_arr)
                records.append({
                    "time_s": float(next_record),
                    "volume_m3": float(np.sum(h_arr[active]) * CELL_AREA),
                    "hmax_m": float(np.max(h_arr)),
                    "flooded_cells_0p03": int(np.sum(h_arr > 0.03)),
                    "surface_to_pipe_m3": float(surface_to_pipe_m3),
                    "pipe_to_surface_m3": float(pipe_to_surface_m3),
                })
                print(
                    f"t={next_record/3600:.2f}h frames={len(frames)}/{n_frames} "
                    f"hmax={records[-1]['hmax_m']:.3f}m vol={records[-1]['volume_m3']:.0f}m3 "
                    f"in={surface_to_pipe_m3:.0f}m3 out={pipe_to_surface_m3:.0f}m3"
                )
                next_record += 300.0

        if len(frames) != n_frames:
            raise RuntimeError(f"Expected {n_frames} frames, got {len(frames)}")

        h_coupled = np.stack(frames, axis=0).astype(np.float32)
        np.save(out_dir / "h_coupled_prototype.npy", h_coupled)
        with (out_dir / "records.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)
        with (out_dir / "exchange_log.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(exchange_rows[0].keys()))
            writer.writeheader()
            writer.writerows(exchange_rows)

        meta.update({
            "event": event,
            "hours": hours,
            "frames": n_frames,
            "shape": list(h_coupled.shape),
            "cell_size_m": CELL,
            "network_only_inp": str(inp),
            "elapsed_wall_s": round(time.time() - t0, 3),
            "surface_to_pipe_m3": float(surface_to_pipe_m3),
            "pipe_to_surface_m3": float(pipe_to_surface_m3),
            "final_volume_m3": float(records[-1]["volume_m3"]),
            "final_hmax_m": float(records[-1]["hmax_m"]),
            "status": "ok",
            "interpretation": "one-event prototype only; not a 17-event coupled training label",
        })
        (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return meta
    finally:
        close_drainage(drainage)


def main() -> int:
    args = parse_args()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    meta = run_coupled_event(args.event, args.hours, args.node_limit, args.dtmax)
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
