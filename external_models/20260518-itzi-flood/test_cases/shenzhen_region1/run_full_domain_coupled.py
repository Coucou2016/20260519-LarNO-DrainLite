#!/usr/bin/env python3
"""
ITZI 2D Surface Hydrodynamics + SWMM Full Coupling — Shenzhen Region1
=====================================================================
Implements bi-directional ITZI-SWMM coupling following:
  - test_cases/urban_drainage/run_coupled_simulation.py (official ITZI pattern)
  - itzi/simulation.py (drainage.step + apply_coupling_to_nodes)

Scenarios per event:
  A) Surface-only (no drainage)
  B) Surface + SWMM coupled (full pipe network from OSM)

Comparison reference: MIKE+ h.npy only (NOT LarNO neural network).

Usage:
  python build_swmm_from_osm.py          # first time
  python run_full_domain_coupled.py      # full domain, all events
  python run_full_domain_coupled.py --domain sub --events event65
"""
import os
import sys
import time
import gc
import argparse
import numpy as np
from datetime import timedelta
from importlib.metadata import PackageNotFoundError, version
from scipy.ndimage import distance_transform_edt

import itzi.rasterdomain as rasterdomain
import itzi.surfaceflow as surfaceflow
from itzi.hydrology import Hydrology
from itzi.infiltration import InfConstantRate, InfNull
from itzi.drainage import DrainageSimulation, DrainageNode, DrainageLink, CouplingTypes
from itzi.swmm_input_parser import SwmmInputParser
from itzi.const import DefaultValues
import pyswmm
from pyswmm import toolkitapi as tka
from swmm.toolkit import solver as swmm_solver

EXPECTED_RUNTIME = {
    "itzi": "25.4",
    "pyswmm": "2.1.0",
    "swmm-toolkit": "0.17.0",
}


def assert_supported_runtime():
    """Reject unreviewed native-coupling versions before opening SWMM."""
    observed = {}
    for package, expected in EXPECTED_RUNTIME.items():
        try:
            observed[package] = version(package)
        except PackageNotFoundError as exc:
            raise RuntimeError(f"Required package is not installed: {package}") from exc
        if observed[package] != expected:
            raise RuntimeError(
                f"Unsupported {package} version {observed[package]!r}; "
                f"the audited coupling requires {expected!r}."
            )
    return observed


RUNTIME_VERSIONS = assert_supported_runtime()

# Itzi 25.4 may call swmm_report() from DrainageSimulation.__del__ after this
# batch runner has already closed the native engine explicitly.  The guarded
# compatibility shim prevents a second native close/report call.  It is kept in
# this version-asserted adapter rather than relying on an unversioned private API.
DrainageSimulation.__del__ = lambda self: None

# ============================================================
# Paths & constants
# ============================================================
CASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(CASE_DIR, "input_data")
DEM_PATH = os.path.join(INPUT_DIR, "geodata", "region1_20m", "dem.npy")
FLOOD_DIR = os.path.join(INPUT_DIR, "flood", "region1_20m")
SWMM_INP_FULL = os.path.join(INPUT_DIR, "networks", "swmm_coupled_full.inp")
SWMM_INP_SUB = os.path.join(INPUT_DIR, "networks", "swmm_coupled_sub.inp")
OUT_DIR = os.path.join(CASE_DIR, "output", "full_domain_swmm")
os.makedirs(OUT_DIR, exist_ok=True)

CELL = 20.0
CELL_AREA = CELL ** 2
G = 9.80665
DT_RAIN = 300       # 5 min per rainfall step
RECORD_STEP = 1800  # 30 min

# Sub-region (LarNO extended study area) vs full domain
SUB_SLICE = (slice(80, 280), slice(120, 400))
ALL_EVENTS = ['event1', 'event20', 'event65', 'event66',
              'event67', 'event68', 'event69', 'event70']


def coor_to_pixel(x_m, y_m, H_full, W_full, row_off=0, col_off=0):
    """Map SWMM coordinates to local raster row/col (with sub-region offset)."""
    col = int(x_m / CELL) - col_off
    row = int((H_full * CELL - y_m) / CELL) - row_off
    return row, col


def get_swmm_outfall_ids(swmm_inp):
    """Return outfall IDs so receiving boundaries are never surface inlets."""
    outfalls = set()
    section = None
    with open(swmm_inp, "r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.split(";", 1)[0].strip()
            if not line:
                continue
            if line.startswith("[") and line.endswith("]"):
                section = line[1:-1].upper()
            elif section == "OUTFALLS":
                outfalls.add(line.split()[0])
    return outfalls


def setup_swmm_drainage(swmm_inp, dem, bldg, H, W, H_full=400, W_full=560,
                        row_off=0, col_off=0,
                        coupling_relaxation=DefaultValues.RELAXATION_FACTOR,
                        coupling_damping=DefaultValues.DAMPING_FACTOR):
    """
    Initialize SWMM + ITZI DrainageSimulation with surface coupling.
    Reference: urban_drainage/run_coupled_simulation.py
    """
    print("  Setting up SWMM drainage network...")
    swmm_sim = pyswmm.Simulation(swmm_inp)
    parser = SwmmInputParser(swmm_inp)
    nodes_coors = parser.get_nodes_id_as_dict()
    links_verts = parser.get_links_id_as_dict()
    outfall_ids = get_swmm_outfall_ids(swmm_inp)

    params = {
        'orifice_coeff': DefaultValues.ORIFICE_COEFF,
        'free_weir_coeff': DefaultValues.FREE_WEIR_COEFF,
        'submerged_weir_coeff': DefaultValues.SUBMERGED_WEIR_COEFF,
    }

    node_id_to_loc = {}
    node_objects = []

    for pnode in pyswmm.Nodes(swmm_sim):
        coors = nodes_coors.get(pnode.nodeid)
        node = DrainageNode(
            node_object=pnode,
            coordinates=coors,
            coupling_type=CouplingTypes.NOT_COUPLED,
            orifice_coeff=params['orifice_coeff'],
            free_weir_coeff=params['free_weir_coeff'],
            submerged_weir_coeff=params['submerged_weir_coeff'],
            g=G,
            relaxation_factor=coupling_relaxation,
            damping_factor=coupling_damping,
        )
        if coors is not None and pnode.nodeid not in outfall_ids:
            x, y = coors.x, coors.y
            if 0 <= x <= W_full * CELL and 0 <= y <= H_full * CELL:
                row, col = coor_to_pixel(x, y, H_full, W_full, row_off, col_off)
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
    n_coupled = len(node_id_to_loc)
    print(f"    SWMM nodes: {len(node_objects)}, coupled to surface: {n_coupled}", flush=True)
    print(f"    SWMM links: {len(links_list)}", flush=True)
    return drainage, node_id_to_loc


def get_surface_node_locations(swmm_inp, bldg, H, W, H_full=400, W_full=560,
                               row_off=0, col_off=0):
    """Read only the SWMM node coordinates used by a matched surface control."""
    parser = SwmmInputParser(swmm_inp)
    outfall_ids = get_swmm_outfall_ids(swmm_inp)
    locations = {}
    for node_id, coors in parser.get_nodes_id_as_dict().items():
        if node_id in outfall_ids:
            continue
        if coors is None:
            continue
        x, y = coors.x, coors.y
        if not (0 <= x <= W_full * CELL and 0 <= y <= H_full * CELL):
            continue
        row, col = coor_to_pixel(x, y, H_full, W_full, row_off, col_off)
        if 0 <= row < H and 0 <= col < W and not bldg[row, col]:
            locations[node_id] = (row, col)
    return locations


def get_link_full_depths(swmm_inp):
    """Read the first geometric depth from SWMM [XSECTIONS]."""
    depths = {}
    section = None
    with open(swmm_inp, "r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.split(";", 1)[0].strip()
            if not line:
                continue
            if line.startswith("[") and line.endswith("]"):
                section = line[1:-1].upper()
                continue
            if section == "XSECTIONS":
                fields = line.split()
                if len(fields) >= 3:
                    depths[fields[0]] = float(fields[2])
    return depths


def run_simulation(label, dem, bldg, rainfall_3d, swmm_inp=None,
                   H_full=400, W_full=560, row_off=0, col_off=0,
                   save_timeseries=False, infiltration_mmh=0.0,
                   inlet_layout_inp=None, inlet_manning=0.012,
                   building_rainfall_mode="global_redistribute",
                   coupling_relaxation=DefaultValues.RELAXATION_FACTOR,
                   coupling_damping=DefaultValues.DAMPING_FACTOR):
    """
    Run ITZI 2D simulation with optional SWMM coupling.

    Parameters
    ----------
    rainfall_3d : (T, H, W) mm per 5-min step
    """
    H, W = dem.shape
    active = ~bldg
    n_active = int(active.sum())
    duration_s = rainfall_3d.shape[0] * DT_RAIN
    if building_rainfall_mode not in {
        "global_redistribute", "nearest_redistribute", "exclude"
    }:
        raise ValueError(f"Unknown building_rainfall_mode={building_rainfall_mode!r}")
    nearest_active = None
    if building_rainfall_mode == "nearest_redistribute":
        nearest_active = distance_transform_edt(
            bldg, return_distances=False, return_indices=True
        )

    dem_itzi = dem.astype(np.float32).copy()
    dem_itzi[bldg] = 50.0

    mannings = np.full((H, W), 0.015, dtype=np.float32)
    mannings[bldg] = 100.0

    mask = np.zeros((H, W), dtype=bool)
    domain = rasterdomain.RasterDomain(
        dtype=np.float32, arr_mask=mask, cell_shape=(CELL, CELL)
    )
    sim_param = {
        'hmin': 0.001, 'cfl': 0.7, 'theta': 0.9, 'g': G,
        'vrouting': 0.1, 'dtmax': 1.0, 'slmax': 0.1,
    }
    sf_sim = surfaceflow.SurfaceFlowSimulation(domain, sim_param)
    inf_model = InfConstantRate(domain, 60.0) if infiltration_mmh > 0 else InfNull(domain, 60.0)
    hydrology = Hydrology(domain, 60.0, inf_model)

    domain.update_array(k="dem", arr=dem_itzi)
    domain.update_array(k="friction", arr=mannings)
    domain.update_array(k="h", arr=np.zeros((H, W), dtype=np.float32))
    domain.update_array(k="rain", arr=np.zeros((H, W), dtype=np.float32))
    inf_rate = np.zeros((H, W), dtype=np.float32)
    if infiltration_mmh > 0:
        inf_rate[active] = infiltration_mmh / 1000.0 / 3600.0
    domain.update_array(k="in_inf", arr=inf_rate)
    sf_sim.update_flow_dir()

    drainage = None
    node_id_to_loc = {}
    if swmm_inp and os.path.exists(swmm_inp):
        drainage, node_id_to_loc = setup_swmm_drainage(
            swmm_inp, dem, bldg, H, W, H_full, W_full, row_off, col_off,
            coupling_relaxation, coupling_damping,
        )
    friction_locations = node_id_to_loc
    if drainage is None and inlet_layout_inp and os.path.exists(inlet_layout_inp):
        friction_locations = get_surface_node_locations(
            inlet_layout_inp, bldg, H, W, H_full, W_full, row_off, col_off
        )
    if inlet_manning is not None and friction_locations:
        # Apply exactly the same inlet-neighbourhood roughness treatment to a
        # no-SWMM control and its coupled counterpart when requested.
        for row, col in friction_locations.values():
            r0, r1 = max(0, row - 1), min(H, row + 2)
            c0, c1 = max(0, col - 1), min(W, col + 2)
            mannings[r0:r1, c0:c1] = np.minimum(
                mannings[r0:r1, c0:c1], inlet_manning
            )
        domain.update_array(k="friction", arr=mannings)

    swmm_state_nodes = []
    swmm_state_links = []
    swmm_link_full_depth = {}
    swmm_states = None
    if drainage is not None and save_timeseries:
        swmm_state_nodes = sorted(
            [node for node in drainage.nodes if node.node_id in node_id_to_loc],
            key=lambda node: node.node_id,
        )
        swmm_state_links = sorted(drainage.links, key=lambda link: link.link_id)
        swmm_link_full_depth = get_link_full_depths(swmm_inp)
        swmm_states = {
            "node_id": np.asarray([node.node_id for node in swmm_state_nodes], dtype="U32"),
            "node_row": np.asarray([node_id_to_loc[node.node_id][0] for node in swmm_state_nodes], dtype=np.int32),
            "node_col": np.asarray([node_id_to_loc[node.node_id][1] for node in swmm_state_nodes], dtype=np.int32),
            "link_id": np.asarray([link.link_id for link in swmm_state_links], dtype="U32"),
            "node_depth_m": [], "node_head_m": [], "node_fullness": [],
            "node_coupling_flow_m3s": [], "node_overflow_m3s": [],
            "link_flow_m3s": [], "link_depth_m": [], "link_fullness": [],
        }

    t = 0.0
    dt = 0.001
    rain_idx = 0
    next_rain = 0.0
    next_hydrology = 0.0
    next_record = RECORD_STEP
    next_series = DT_RAIN
    cumulative_drain = 0.0
    cumulative_swmm_net_exchange = 0.0
    cumulative_surface_held_exchange = 0.0
    cumulative_infiltration = 0.0
    h_series = []

    records = {
        'time_h': [], 'vol_m3': [], 'hmax_m': [],
        'flooded_cells': [], 'drained_m3': [], 'infiltrated_m3': [],
        'swmm_node_flow': [], 'swmm_net_exchange_m3': [],
        'surface_held_exchange_m3': [],
    }
    t0 = time.time()

    while t < duration_s:
        # --- Rainfall (2D heterogeneous, explicit building-runoff treatment) ---
        if t >= next_rain and rain_idx < rainfall_3d.shape[0]:
            rain_2d = rainfall_3d[rain_idx]
            effective_mm = np.zeros((H, W), dtype=np.float32)
            effective_mm[active] = rain_2d[active]
            if building_rainfall_mode == "global_redistribute":
                bldg_rain = float(np.sum(rain_2d[bldg], dtype=np.float64))
                if n_active > 0:
                    effective_mm[active] += bldg_rain / n_active
            elif building_rainfall_mode == "nearest_redistribute" and nearest_active is not None:
                building_rows, building_cols = np.where(bldg)
                target_rows = nearest_active[0, building_rows, building_cols]
                target_cols = nearest_active[1, building_rows, building_cols]
                np.add.at(
                    effective_mm,
                    (target_rows, target_cols),
                    rain_2d[building_rows, building_cols],
                )
            rain_field = effective_mm / 1000.0 / DT_RAIN
            domain.update_array(k="rain", arr=rain_field)
            rain_idx += 1
            next_rain += DT_RAIN

        # --- Hydrology (infiltration and effective rainfall) ---
        if t >= next_hydrology:
            hydrology.solve_dt()
            hydrology.step()
            hydro_dt = hydrology.dt.total_seconds()
            if infiltration_mmh > 0:
                cumulative_infiltration += float(
                    np.sum(domain.get_array("inf")[active], dtype=np.float64)
                    * CELL_AREA * hydro_dt
                )
            next_hydrology += hydro_dt

        # --- SWMM drainage coupling (bi-directional weir/orifice exchange) ---
        if drainage is not None:
            if t >= drainage.elapsed_time and drainage.elapsed_time < duration_s - 5:
                try:
                    drainage.step()
                    surface_states = {}
                    arr_z = domain.get_array("dem")
                    arr_h = domain.get_array("h")
                    for nid, (row, col) in node_id_to_loc.items():
                        surface_states[nid] = {
                            'z': float(arr_z[row, col]),
                            'h': float(arr_h[row, col]),
                        }
                    flows = drainage.apply_coupling_to_nodes(
                        surface_states, CELL_AREA
                    )
                    arr_qd = domain.get_array("n_drain")
                    arr_qd[:] = 0.0
                    step_drain = 0.0
                    drainage_dt_s = drainage.dt.total_seconds()
                    for nid, q in flows.items():
                        if nid in node_id_to_loc:
                            row, col = node_id_to_loc[nid]
                            arr_qd[row, col] = q / CELL_AREA
                            # ITZI defines negative coupling flow as water leaving
                            # the 2D surface and entering the drainage network.
                            step_drain += max(-q, 0) * drainage_dt_s
                    cumulative_drain += step_drain
                    cumulative_swmm_net_exchange += (
                        sum(-q for q in flows.values()) * drainage_dt_s
                    )
                except Exception as exc:
                    if rain_idx <= 1:
                        print(f"    SWMM step warning: {exc}")

        # --- 2D surface flow (partial inertia) ---
        # Match Itzi Simulation.find_dt(): the shared time step must end at
        # the next hydrology, rainfall, drainage, output or CFL event. Without
        # this synchronization, a SWMM flow calculated for a short routing
        # interval can be held too long by the 2D solver and violate mass
        # conservation.
        future_events = [t + dt, duration_s, next_rain, next_hydrology, next_record]
        if save_timeseries:
            future_events.append(next_series)
        if drainage is not None:
            future_events.append(drainage.elapsed_time)
        step_end = min(value for value in future_events if value > t + 1e-10)
        step_dt = step_end - t
        domain.update_ext_array()
        cumulative_surface_held_exchange += float(
            -np.sum(domain.get_array("n_drain"), dtype=np.float64)
            * CELL_AREA * step_dt
        )
        sf_sim.dt = timedelta(seconds=step_dt)
        sf_sim.step()
        sf_sim.solve_dt()
        t = step_end
        dt = sf_sim.dt.total_seconds()

        if save_timeseries and t >= next_series:
            h_series.append(domain.get_array("h").copy())
            if swmm_states is not None:
                node_depth = np.asarray([node.pyswmm_node.depth for node in swmm_state_nodes], dtype=np.float32)
                node_full_depth = np.asarray([node.pyswmm_node.full_depth for node in swmm_state_nodes], dtype=np.float32)
                link_depth = np.asarray([link.pyswmm_link.depth for link in swmm_state_links], dtype=np.float32)
                link_full_depth = np.asarray([
                    max(float(swmm_link_full_depth.get(link.link_id, 0.0)), 1e-6)
                    for link in swmm_state_links
                ], dtype=np.float32)
                swmm_states["node_depth_m"].append(node_depth)
                swmm_states["node_head_m"].append(np.asarray(
                    [node.pyswmm_node.head for node in swmm_state_nodes], dtype=np.float32
                ))
                swmm_states["node_fullness"].append(node_depth / np.maximum(node_full_depth, 1e-6))
                swmm_states["node_coupling_flow_m3s"].append(np.asarray(
                    [node.coupling_flow for node in swmm_state_nodes], dtype=np.float32
                ))
                swmm_states["node_overflow_m3s"].append(np.asarray(
                    [node.get_overflow() for node in swmm_state_nodes], dtype=np.float32
                ))
                swmm_states["link_flow_m3s"].append(np.asarray(
                    [link.pyswmm_link.flow for link in swmm_state_links], dtype=np.float32
                ))
                swmm_states["link_depth_m"].append(link_depth)
                swmm_states["link_fullness"].append(link_depth / link_full_depth)
            next_series += DT_RAIN

        if t >= next_record:
            h_arr = domain.get_array("h")
            vol = float(np.sum(h_arr[active], dtype=np.float64) * CELL_AREA)
            hmax = float(np.max(h_arr))
            flooded = int(np.sum(h_arr > 0.03))
            records['time_h'].append(t / 3600)
            records['vol_m3'].append(vol)
            records['hmax_m'].append(hmax)
            records['flooded_cells'].append(flooded)
            records['drained_m3'].append(cumulative_drain)
            records['swmm_net_exchange_m3'].append(cumulative_swmm_net_exchange)
            records['surface_held_exchange_m3'].append(cumulative_surface_held_exchange)
            records['infiltrated_m3'].append(cumulative_infiltration)
            print(f"  [{label}] t={t/3600:.1f}h vol={vol:.0f}m3 hmax={hmax:.3f}m "
                  f"flooded={flooded} drained={cumulative_drain:.0f}m3 "
                  f"infiltrated={cumulative_infiltration:.0f}m3", flush=True)
            next_record += RECORD_STEP

    h_final = domain.get_array("h").copy()

    if drainage is not None:
        # SWMM 5.2's text report attributes every failed global routing step
        # to outfalls because their convergence flags are initialised FALSE
        # and outfalls are skipped by findNodeDepths().  Query raw toolkit
        # node statistics before closing the engine and retain the actual
        # non-outfall counts for diagnosis.
        outfall_ids = get_swmm_outfall_ids(swmm_inp)
        node_nonconvergence = {}
        outfall_nonconvergence = {}
        for node in drainage.nodes:
            try:
                index = drainage.swmm_model.getObjectIDIndex(
                    tka.ObjectType.NODE.value, node.node_id
                )
                stats = swmm_solver.node_get_stats(index)
                count = int(stats.nonConvergedCount)
            except Exception:
                count = -1
            if node.node_id in outfall_ids:
                outfall_nonconvergence[node.node_id] = count
            else:
                node_nonconvergence[node.node_id] = count
        records["swmm_node_nonconverged_count"] = node_nonconvergence
        records["swmm_outfall_attribution_artifact_count"] = outfall_nonconvergence
        records["swmm_global_failed_steps_from_outfall_artifact"] = max(
            outfall_nonconvergence.values(), default=0
        )
        records["swmm_top_nonconverging_junctions"] = sorted(
            (
                {"node_id": node_id, "count": count}
                for node_id, count in node_nonconvergence.items()
                if count > 0
            ),
            key=lambda item: (-item["count"], item["node_id"]),
        )[:20]
        try:
            drainage.swmm_model.swmm_end()
        except Exception:
            pass
        try:
            drainage.swmm_model.swmm_close()
        except Exception:
            pass
        drainage.swmm_model = None
        drainage.swmm_sim = None
        del drainage
        gc.collect()

    elapsed = time.time() - t0
    records['runtime_s'] = elapsed
    if swmm_states is not None:
        for key, value in list(swmm_states.items()):
            if isinstance(value, list):
                swmm_states[key] = np.asarray(value, dtype=np.float32)
        records['swmm_states'] = swmm_states
    print(f"  [{label}] Done in {elapsed:.0f}s", flush=True)
    if save_timeseries:
        if len(h_series) != rainfall_3d.shape[0]:
            print(
                f"  [{label}] warning: saved {len(h_series)} frames, "
                f"expected {rainfall_3d.shape[0]}",
                flush=True,
            )
        return records, h_final, np.asarray(h_series, dtype=np.float32)
    return records, h_final, None


def load_domain(domain_mode):
    """Load DEM/building mask for full or sub domain."""
    dem_full = np.load(DEM_PATH, allow_pickle=True).astype(np.float32)
    bldg_full = dem_full >= 49.9
    if domain_mode == 'full':
        return dem_full, bldg_full, (slice(None), slice(None)), dem_full.shape
    ys, xs = SUB_SLICE
    dem = dem_full[ys, xs].copy()
    bldg = bldg_full[ys, xs].copy()
    return dem, bldg, (ys, xs), dem_full.shape


def main():
    parser = argparse.ArgumentParser(description="ITZI-SWMM full domain coupled simulation")
    parser.add_argument('--domain', choices=['full', 'sub'], default='full',
                        help='full=8km×11.2km, sub=4km×5.6km study area')
    parser.add_argument('--events', nargs='+', default=ALL_EVENTS,
                        help='Flood events to simulate')
    parser.add_argument('--build-swmm', action='store_true',
                        help='Rebuild SWMM inp from OSM network')
    parser.add_argument('--save-timeseries', action='store_true',
                        help='Save 5-min surface and SWMM water-depth series')
    parser.add_argument('--infiltration-mmh', type=float, default=0.0,
                        help='Constant active-cell infiltration rate in mm/h. Default 0 keeps the previous InfNull behavior.')
    args = parser.parse_args()

    print("=" * 70)
    print("  ITZI 2D + SWMM Full Coupling — Shenzhen Region1")
    print(f"  Domain: {args.domain.upper()} | Events: {args.events}")
    print(f"  Infiltration: {args.infiltration_mmh:.3f} mm/h on non-building active cells")
    print(f"  Reference: MIKE+ only  |  Output: {OUT_DIR}")
    print("=" * 70)

    swmm_inp = SWMM_INP_FULL if args.domain == 'full' else SWMM_INP_SUB
    if args.build_swmm or not os.path.exists(swmm_inp):
        from build_swmm_from_osm import build_swmm_inp, SUB_SLICE
        if args.domain == 'full':
            build_swmm_inp(out_path=SWMM_INP_FULL, region_slice=None,
                           main_trunk_only=True, subsample=True)
        else:
            build_swmm_inp(out_path=SWMM_INP_SUB, region_slice=SUB_SLICE,
                           main_trunk_only=True, subsample=True)

    dem, bldg, slc, full_shape = load_domain(args.domain)
    H, W = dem.shape
    H_full, W_full = full_shape
    row_off = slc[0].start if isinstance(slc[0], slice) else 0
    col_off = slc[1].start if isinstance(slc[1], slice) else 0
    print(f"\nSimulation grid: {H}×{W} = {H*CELL/1000:.1f}km × {W*CELL/1000:.1f}km")
    print(f"Active cells: {(~bldg).sum():,}  |  Buildings: {bldg.sum():,}")

    all_summary = {}

    for evt in args.events:
        evt_dir = os.path.join(FLOOD_DIR, evt)
        rain_path = os.path.join(evt_dir, 'rainfall.npy')
        h_path = os.path.join(evt_dir, 'h.npy')
        if not os.path.exists(rain_path):
            print(f"\nSkip {evt}: rainfall not found")
            continue

        print(f"\n{'='*70}\n  {evt}\n{'='*70}")

        rainfall_full = np.load(rain_path)
        h_ref_full = np.load(h_path)
        rainfall = rainfall_full[(slice(None),) + slc]
        h_ref = h_ref_full[(slice(None),) + slc]
        peak_ref = np.max(h_ref, axis=0)

        print(f"  MIKE+ peak={peak_ref.max():.3f}m  flooded(>3cm)={np.sum(peak_ref>0.03):,}")

        # A) Surface only
        print("\n  [A] ITZI Surface Only")
        geo_kw = dict(H_full=H_full, W_full=W_full, row_off=row_off, col_off=col_off)

        rec_s, h_s, h_s_series = run_simulation(
            f"{evt} surface", dem.copy(), bldg.copy(), rainfall,
            swmm_inp=None, save_timeseries=args.save_timeseries,
            infiltration_mmh=args.infiltration_mmh, **geo_kw
        )

        # B) SWMM coupled
        print("\n  [B] ITZI + SWMM Full Coupling")
        rec_sw, h_sw, h_sw_series = run_simulation(
            f"{evt} swmm", dem.copy(), bldg.copy(), rainfall,
            swmm_inp=swmm_inp, save_timeseries=args.save_timeseries,
            infiltration_mmh=args.infiltration_mmh, **geo_kw
        )

        summary = {
            'mike_peak': float(peak_ref.max()),
            'mike_flooded': int(np.sum(peak_ref > 0.03)),
            'mike_vol_final': float(
                np.sum(h_ref[-1][~bldg], dtype=np.float64) * CELL_AREA
            ),
            'surf_peak': float(h_s.max()),
            'surf_flooded': int(np.sum(h_s > 0.03)),
            'surf_vol': float(rec_s['vol_m3'][-1]) if rec_s['vol_m3'] else 0,
            'swmm_peak': float(h_sw.max()),
            'swmm_flooded': int(np.sum(h_sw > 0.03)),
            'swmm_vol': float(rec_sw['vol_m3'][-1]) if rec_sw['vol_m3'] else 0,
            'swmm_drained': float(rec_sw['drained_m3'][-1]) if rec_sw['drained_m3'] else 0,
            'surf_infiltrated': float(rec_s['infiltrated_m3'][-1]) if rec_s['infiltrated_m3'] else 0,
            'swmm_infiltrated': float(rec_sw['infiltrated_m3'][-1]) if rec_sw['infiltrated_m3'] else 0,
            'infiltration_mmh': float(args.infiltration_mmh),
        }
        all_summary[evt] = summary

        out_npz = os.path.join(OUT_DIR, f'{evt}_coupled.npz')
        np.savez(
            out_npz,
            peak_ref=peak_ref, h_ref_final=h_ref[-1],
            h_surf=h_s, h_swmm=h_sw,
            h_surf_series=h_s_series if h_s_series is not None else np.empty((0,), dtype=np.float32),
            h_swmm_series=h_sw_series if h_sw_series is not None else np.empty((0,), dtype=np.float32),
            rec_surf=rec_s, rec_swmm=rec_sw,
            domain=args.domain,
        )
        print(f"  Saved {out_npz}", flush=True)

        sp = summary['surf_peak'] / summary['mike_peak'] * 100
        swp = summary['swmm_peak'] / summary['mike_peak'] * 100
        print(f"\n  Summary {evt}:")
        print(f"    MIKE+     peak={summary['mike_peak']:.3f}m")
        print(f"    ITZI-S    peak={summary['surf_peak']:.3f}m ({sp:.0f}% of MIKE+)")
        print(f"    ITZI-SWMM peak={summary['swmm_peak']:.3f}m ({swp:.0f}% of MIKE+)")
        print(f"    SWMM drained={summary['swmm_drained']:.0f} m3")

    # Final table
    print(f"\n{'='*70}")
    print(f"  FINAL — ITZI vs MIKE+ ({args.domain} domain)")
    print(f"{'='*70}")
    print(f"{'Event':<10} {'MIKE+':>7} {'Surf':>7} {'SWMM':>7} "
          f"{'S%':>5} {'W%':>5} {'Drained':>12}")
    print("-" * 60)
    for evt, s in all_summary.items():
        sm = s['surf_peak'] / s['mike_peak'] * 100
        wm = s['swmm_peak'] / s['mike_peak'] * 100
        print(f"{evt:<10} {s['mike_peak']:>7.3f} {s['surf_peak']:>7.3f} "
              f"{s['swmm_peak']:>7.3f} {sm:>4.0f}% {wm:>4.0f}% "
              f"{s['swmm_drained']:>12.0f}")

    np.savez(os.path.join(OUT_DIR, 'summary.npz'), summary=all_summary, domain=args.domain)
    print(f"\nResults: {OUT_DIR}")
    sys.stdout.flush()


if __name__ == "__main__":
    try:
        main()
    finally:
        # Avoid SWMM native crash on interpreter shutdown (Windows)
        os._exit(0)
