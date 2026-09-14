#!/usr/bin/env python3
"""
ITZI 2D partial-inertia surface flow + SWMM DYNWAVE drainage coupling for UFIM samples.

Surface: itzi.surfaceflow.SurfaceFlowSimulation (NOT depression filling / instant routing)
Drainage: pyswmm + DrainageSimulation.apply_coupling_to_nodes (bi-directional weir/orifice)

Usage:
  python run_coupled_simulation.py --sample sample1
  python run_coupled_simulation.py --all --rebuild-swmm
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from datetime import timedelta

import numpy as np

sys.path.insert(0, r"E:\Miniconda3\Lib\site-packages")
import itzi.rasterdomain as rasterdomain
import itzi.surfaceflow as surfaceflow
from itzi.const import DefaultValues
from itzi.drainage import CouplingTypes, DrainageLink, DrainageNode, DrainageSimulation
from itzi.hydrology import Hydrology
from itzi.infiltration import InfNull
from itzi.swmm_input_parser import SwmmInputParser
import pyswmm

from build_swmm_from_shapefiles import build_swmm_inp
from sample_utils import (
    CASE_DIR,
    GridMeta,
    dem_source_stats,
    interpolate_rainfall_idw,
    load_rainfall_spatial,
    local_to_rowcol,
    prepare_surface_fields,
    redistribute_building_rain,
    sample_dir,
    save_integration_log,
    setup_boundary_arrays,
)

G = 9.80665
RECORD_STEP = 600


def setup_swmm_drainage(swmm_inp: str, dem: np.ndarray, bldg: np.ndarray, meta: GridMeta):
    print("  Setting up SWMM drainage network (DYNWAVE + ITZI coupling)...")
    swmm_sim = pyswmm.Simulation(swmm_inp)
    parser = SwmmInputParser(swmm_inp)
    nodes_coors = parser.get_nodes_id_as_dict()
    links_verts = parser.get_links_id_as_dict()

    params = {
        "orifice_coeff": DefaultValues.ORIFICE_COEFF,
        "free_weir_coeff": DefaultValues.FREE_WEIR_COEFF,
        "submerged_weir_coeff": DefaultValues.SUBMERGED_WEIR_COEFF,
    }

    H, W = dem.shape
    node_id_to_loc = {}
    node_objects = []

    for pnode in pyswmm.Nodes(swmm_sim):
        coors = nodes_coors.get(pnode.nodeid)
        node = DrainageNode(
            node_object=pnode,
            coordinates=coors,
            coupling_type=CouplingTypes.NOT_COUPLED,
            orifice_coeff=params["orifice_coeff"],
            free_weir_coeff=params["free_weir_coeff"],
            submerged_weir_coeff=params["submerged_weir_coeff"],
            g=G,
        )
        if coors is not None:
            x, y = coors.x, coors.y
            if 0 <= x <= meta.width_m and 0 <= y <= meta.height_m:
                row, col = local_to_rowcol(x, y, meta)
                row = max(0, min(H - 1, row))
                col = max(0, min(W - 1, col))
                if 0 <= row < H and 0 <= col < W and not bldg[row, col]:
                    node.coupling_type = CouplingTypes.COUPLED_NO_FLOW
                    node_id_to_loc[pnode.nodeid] = (row, col)
        node_objects.append(node)

    links_list = []
    for plink in pyswmm.Links(swmm_sim):
        in_c = nodes_coors.get(plink.inlet_node)
        out_c = nodes_coors.get(plink.outlet_node)
        verts = [in_c] if in_c else []
        vdata = links_verts.get(plink.linkid)
        if vdata and vdata.vertices:
            verts.extend(vdata.vertices)
        if out_c:
            verts.append(out_c)
        links_list.append(DrainageLink(link_object=plink, vertices=verts))

    drainage = DrainageSimulation(swmm_sim, node_objects, links_list)
    print(
        f"    SWMM nodes: {len(node_objects)}, coupled: {len(node_id_to_loc)}, "
        f"links: {len(links_list)}"
    )
    return drainage, node_id_to_loc


def run_simulation(
    label: str,
    dem: np.ndarray,
    mannings: np.ndarray,
    bldg: np.ndarray,
    meta: GridMeta,
    rain_mm_h: np.ndarray,
    rain_stations: list,
    dt_rain: float,
    bctype: np.ndarray,
    bcval: np.ndarray,
    swmm_inp: str | None = None,
):
    H, W = dem.shape
    cell = meta.cellsize
    cell_area = cell * cell
    duration_s = rain_mm_h.shape[0] * dt_rain
    active = ~bldg

    mask = np.zeros((H, W), dtype=bool)
    domain = rasterdomain.RasterDomain(
        dtype=np.float32, arr_mask=mask, cell_shape=(cell, cell)
    )
    sim_param = {
        "hmin": 0.001, "cfl": 0.7, "theta": 0.9, "g": G,
        "vrouting": 0.1, "dtmax": 1.0, "slmax": 0.1,
    }
    sf_sim = surfaceflow.SurfaceFlowSimulation(domain, sim_param)
    hydrology = Hydrology(domain, 60.0, InfNull(domain, 60.0))

    domain.update_array(k="dem", arr=dem.astype(np.float32))
    domain.update_array(k="friction", arr=mannings.astype(np.float32))
    domain.update_array(k="h", arr=np.zeros((H, W), dtype=np.float32))
    domain.update_array(k="rain", arr=np.zeros((H, W), dtype=np.float32))
    domain.update_array(k="bctype", arr=bctype.astype(np.int32))
    domain.update_array(k="bcval", arr=bcval.astype(np.float32))
    sf_sim.update_flow_dir()

    drainage = None
    node_id_to_loc = {}
    if swmm_inp and os.path.exists(swmm_inp):
        drainage, node_id_to_loc = setup_swmm_drainage(swmm_inp, dem, bldg, meta)
        for row, col in node_id_to_loc.values():
            r0, r1 = max(0, row - 1), min(H, row + 2)
            c0, c1 = max(0, col - 1), min(W, col + 2)
            mannings[r0:r1, c0:c1] = np.minimum(mannings[r0:r1, c0:c1], 0.012)
        domain.update_array(k="friction", arr=mannings.astype(np.float32))

    t = 0.0
    dt = 0.001
    rain_idx = 0
    next_rain = dt_rain
    next_record = RECORD_STEP
    cumulative_drain = 0.0

    records = {
        "time_h": [], "vol_m3": [], "hmax_m": [],
        "flooded_cells": [], "drained_m3": [],
    }
    t0 = time.time()

    multi_station = rain_mm_h.ndim == 2 and rain_mm_h.shape[1] > 1
    print(
        f"\n  [{label}] duration={duration_s/3600:.2f}h grid={H}x{W} cell={cell}m "
        f"rain={'IDW multi-station' if multi_station else 'uniform'}"
    )

    while t < duration_s and rain_idx < rain_mm_h.shape[0]:
        if t >= next_rain and rain_idx < rain_mm_h.shape[0]:
            if multi_station:
                step_vals = rain_mm_h[rain_idx]
                rain_2d = interpolate_rainfall_idw(
                    step_vals, rain_stations, meta, active
                )
            else:
                val = float(rain_mm_h[rain_idx]) if rain_mm_h.ndim == 1 else float(rain_mm_h[rain_idx, 0])
                rain_2d = np.zeros((H, W), dtype=np.float32)
                rain_2d[active] = val
            rain_field = redistribute_building_rain(rain_2d, bldg, dt_rain)
            domain.update_array(k="rain", arr=rain_field)
            rain_idx += 1
            next_rain += dt_rain

        domain.update_ext_array()
        hydrology.solve_dt()
        hydrology.step()

        if drainage is not None:
            if t >= drainage.elapsed_time and drainage.elapsed_time < duration_s - 5:
                try:
                    drainage.step()
                    surface_states = {}
                    arr_z = domain.get_array("dem")
                    arr_h = domain.get_array("h")
                    for nid, (row, col) in node_id_to_loc.items():
                        surface_states[nid] = {
                            "z": float(arr_z[row, col]),
                            "h": float(arr_h[row, col]),
                        }
                    flows = drainage.apply_coupling_to_nodes(surface_states, cell_area)
                    arr_qd = domain.get_array("n_drain")
                    arr_qd[:] = 0.0
                    step_drain = 0.0
                    for nid, q in flows.items():
                        if nid in node_id_to_loc:
                            row, col = node_id_to_loc[nid]
                            arr_qd[row, col] = q / cell_area
                            step_drain += max(q, 0) * drainage._dt
                    cumulative_drain += step_drain
                except Exception as exc:
                    if rain_idx <= 1:
                        print(f"    SWMM warning: {exc}")

        sf_sim.dt = timedelta(seconds=dt)
        sf_sim.step()
        sf_sim.solve_dt()
        t += dt
        dt = sf_sim.dt.total_seconds()

        if t >= next_record:
            h_arr = domain.get_array("h")
            vol = float(np.sum(h_arr[active]) * cell_area)
            hmax = float(np.max(h_arr[active])) if active.any() else 0.0
            flooded = int(np.sum(h_arr > 0.03))
            records["time_h"].append(t / 3600)
            records["vol_m3"].append(vol)
            records["hmax_m"].append(hmax)
            records["flooded_cells"].append(flooded)
            records["drained_m3"].append(cumulative_drain)
            print(
                f"  [{label}] t={t/3600:.2f}h vol={vol:.0f}m3 hmax={hmax:.3f}m "
                f"flooded={flooded} drained={cumulative_drain:.0f}m3"
            )
            next_record += RECORD_STEP

    h_final = domain.get_array("h").copy()

    if drainage is not None:
        try:
            drainage.swmm_model.swmm_end()
        except Exception:
            pass
        try:
            drainage.swmm_model.swmm_close()
        except Exception:
            pass
        drainage.swmm_model = None
        del drainage
        gc.collect()

    elapsed = time.time() - t0
    print(f"  [{label}] Done in {elapsed:.0f}s")
    return records, h_final


def run_sample(sample: str, rebuild_swmm: bool = False) -> dict:
    print("=" * 70)
    print(f"  UFIM Coupled Simulation — {sample}")
    print("=" * 70)

    meta, dem, mannings, bldg, active, integration = prepare_surface_fields(sample)
    src_stats = dem_source_stats(sample)

    rain_2d, dt_rain, rain_paths, rain_stations = load_rainfall_spatial(sample, prefer_multi=True)
    sim_hours = rain_2d.shape[0] * dt_rain / 3600.0

    bctype, bcval, bc_info = setup_boundary_arrays(
        sample, meta, dem, active, bldg, rain_2d.shape[0] * dt_rain
    )

    print(
        f"  DEM source: {meta.nrows}x{meta.ncols} @ {meta.cellsize}m "
        f"elev [{src_stats['min']:.2f}, {src_stats['max']:.2f}] m (unchanged on valid cells)"
    )
    print(f"  Active cells: {active.sum():,}  Blocked: {bldg.sum():,}")
    print(
        f"  Rainfall: {len(rain_paths)} station(s), {rain_2d.shape[0]} steps, "
        f"peak {float(rain_2d.max()):.1f} mm/h"
    )
    print(f"  River/tidal BC cells: {bc_info.get('river_cells', 0)}")

    swmm_inp = os.path.join(sample_dir(sample), "network", "drainage.inp")
    if rebuild_swmm or not os.path.exists(swmm_inp):
        build_swmm_inp(sample, swmm_inp, sim_hours=sim_hours)

    out_dir = os.path.join(sample_dir(sample), "output")
    os.makedirs(out_dir, exist_ok=True)

    rec_s, h_s = run_simulation(
        f"{sample} surface", dem.copy(), mannings.copy(), bldg, meta,
        rain_2d, rain_stations, dt_rain, bctype, bcval, swmm_inp=None,
    )
    rec_w, h_w = run_simulation(
        f"{sample} swmm", dem.copy(), mannings.copy(), bldg, meta,
        rain_2d, rain_stations, dt_rain, bctype, bcval, swmm_inp=swmm_inp,
    )

    summary = {
        "sample": sample,
        "dem_shape": dem.shape,
        "cellsize": meta.cellsize,
        "dem_source_min": src_stats["min"],
        "dem_source_max": src_stats["max"],
        "rainfall_sources": rain_paths,
        "rain_stations_inferred": len(rain_stations),
        "rain_peak_mm_h": float(rain_2d.max()),
        "sim_hours": sim_hours,
        "river_bc_cells": bc_info.get("river_cells", 0),
        "surface_peak_m": float(h_s.max()),
        "surface_flooded": int(np.sum(h_s > 0.03)),
        "surface_vol_m3": float(rec_s["vol_m3"][-1]) if rec_s["vol_m3"] else 0.0,
        "swmm_peak_m": float(h_w.max()),
        "swmm_flooded": int(np.sum(h_w > 0.03)),
        "swmm_vol_m3": float(rec_w["vol_m3"][-1]) if rec_w["vol_m3"] else 0.0,
        "swmm_drained_m3": float(rec_w["drained_m3"][-1]) if rec_w["drained_m3"] else 0.0,
    }

    log_path = save_integration_log(
        sample, integration,
        {"simulation": summary, "boundary": bc_info, "physics": {
            "surface_solver": "itzi.surfaceflow.SurfaceFlowSimulation",
            "drainage_solver": "SWMM DYNWAVE via pyswmm + DrainageSimulation",
        }},
    )

    npz_path = os.path.join(out_dir, "coupled_results.npz")
    np.savez(
        npz_path,
        dem=dem, mannings=mannings, building_mask=bldg,
        h_final_surface=h_s, h_final_swmm=h_w,
        rec_surface=rec_s, rec_swmm=rec_w,
        rain_mm_h=rain_2d, rain_stations=rain_stations,
        meta_xll=meta.xll, meta_yll=meta.yll,
        meta_cellsize=meta.cellsize, summary=summary,
    )

    np.savetxt(os.path.join(out_dir, "depth_surface_final.asc"), h_s, fmt="%.4f")
    np.savetxt(os.path.join(out_dir, "depth_swmm_final.asc"), h_w, fmt="%.4f")

    print(f"\n  Summary {sample}:")
    print(f"    Surface peak={summary['surface_peak_m']:.3f}m  flooded={summary['surface_flooded']}")
    print(f"    SWMM peak={summary['swmm_peak_m']:.3f}m  flooded={summary['swmm_flooded']}  "
          f"drained={summary['swmm_drained_m3']:.0f}m3")
    print(f"  Saved: {npz_path}")
    print(f"  Integration log: {log_path}")
    return summary


def main():
    import subprocess

    parser = argparse.ArgumentParser(description="UFIM ITZI+SWMM coupled simulation")
    parser.add_argument("--sample", choices=["sample1", "sample2", "sample3"])
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--rebuild-swmm", action="store_true")
    args = parser.parse_args()

    if not args.all and not args.sample:
        parser.error("Specify --sample or --all")

    samples = ["sample1", "sample2", "sample3"] if args.all else [args.sample]

    # EPA-SWMM allows only one Simulation per Python process (pyswmm MultiSimulationError).
    if args.all and len(samples) > 1:
        script = os.path.abspath(__file__)
        for s in samples:
            cmd = [sys.executable, script, "--sample", s]
            if args.rebuild_swmm:
                cmd.append("--rebuild-swmm")
            print(f"\n>>> Subprocess: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
        summaries = []
        for s in samples:
            npz = os.path.join(sample_dir(s), "output", "coupled_results.npz")
            d = np.load(npz, allow_pickle=True)
            summaries.append(d["summary"].item())
    else:
        summaries = []
        for s in samples:
            summaries.append(run_sample(s, rebuild_swmm=args.rebuild_swmm))

    summary_path = os.path.join(CASE_DIR, "output", "all_samples_summary.npz")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    np.savez(summary_path, summaries=summaries)
    print(f"\nAll summaries: {summary_path}")


if __name__ == "__main__":
    main()
