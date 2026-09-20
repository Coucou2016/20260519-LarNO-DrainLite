#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ITZI Coupled Surface-Drainage Urban Flood Simulation
=====================================================
Demonstrates bi-directional coupling between 2D surface flow and
1D underground storm drainage network (powered by SWMM via pyswmm).

Runs TWO simulations for comparison:
  A) Surface-only (no drainage) — baseline
  B) Surface + Drainage (SWMM coupled) — with underground pipes

Urban layout: 400m x 400m, 2m resolution, 24 buildings, street grid
Drainage: Storm sewer along main avenue + branch lines at cross streets
"""
import os, sys, time
import numpy as np
from datetime import timedelta

import itzi.rasterdomain as rasterdomain
import itzi.surfaceflow as surfaceflow
from itzi.hydrology import Hydrology
from itzi.infiltration import InfNull
from itzi.drainage import DrainageSimulation, DrainageNode, DrainageLink, CouplingTypes
from itzi.swmm_input_parser import SwmmInputParser
from itzi.const import DefaultValues
import pyswmm

# ============================================================
# Configuration
# ============================================================
CELL_SIZE = 2.0
GRID_SIZE = 200
DOMAIN_SIZE = GRID_SIZE * CELL_SIZE
SIM_DURATION_S = 7200
RECORD_STEP_S = 600
DT_MAX = 1.0
CFL = 0.7; THETA = 0.9; HMIN = 0.001; G = 9.80665; DT_INF = 60.0

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "core_sim_output")
VIZ_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "visualization_output")
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(VIZ_DIR, exist_ok=True)


def create_swmm_inp():
    """
    Create a SWMM input file for the urban district drainage network.

    Network layout:
      - Main trunk sewer along central avenue (N→S, 6 manholes)
      - 4 branch sewers along each cross street (E→W into main)
      - Inlets/catch basins at each manhole
      - Outfall at the southern boundary
    """
    inp_path = os.path.join(TEST_DIR, "drainage_network.inp")

    # Node coordinates (matching the urban district layout)
    # Main avenue runs N-S at x=200m (GRID_SIZE/2 * CELL_SIZE)
    main_x = DOMAIN_SIZE / 2  # 200m

    # Manhole spacing along main avenue
    main_nodes_y = [350, 290, 230, 170, 110, 50]  # meters from south
    main_inverts = [46.0, 46.3, 46.6, 46.9, 47.2, 47.5]  # invert elevations

    # Branch nodes on each side of the main avenue
    branch_x_left = [80, 140, 80, 140, 80, 140]
    branch_x_right = [260, 320, 260, 320, 260, 320]
    branch_y = [320, 320, 200, 200, 80, 80]

    with open(inp_path, 'w') as f:
        f.write("""
[TITLE]
Urban District Storm Drainage Network

[OPTIONS]
FLOW_UNITS           CMS
INFILTRATION         HORTON
FLOW_ROUTING         DYNWAVE
LINK_OFFSETS         DEPTH
MIN_SLOPE            0.001
ALLOW_PONDING        NO
SKIP_STEADY_STATE    NO
START_DATE           01/01/2026
START_TIME           00:00:00
REPORT_START_DATE    01/01/2026
REPORT_START_TIME    00:00:00
END_DATE             01/01/2026
END_TIME             02:00:00
SWEEP_START          01/01
SWEEP_END            12/31
DRY_DAYS             0
REPORT_STEP          00:10:00
WET_STEP             00:01:00
DRY_STEP             00:05:00
ROUTING_STEP         0:00:05
INERTIAL_DAMPING     PARTIAL
NORMAL_FLOW_LIMITED  BOTH
FORCE_MAIN_EQUATION  H-W
VARIABLE_STEP        0.75
LENGTHENING_STEP     0
MAX_TRIALS           8
HEAD_TOLERANCE       0.005
SYS_FLOW_TOL         5
LAT_FLOW_TOL         5
MINIMUM_STEP         0.5
THREADS              1

[RAINGAGES]
;; Rain is provided by ITZI surface simulation (not used here)
RainGage    INTENSITY 0:05     1.0     TIMESERIES DummyRain

[TIMESERIES]
DummyRain   0:00    0.0
DummyRain   2:00    0.0

[EVAPORATION]
CONSTANT     0.0
DRY_ONLY     NO

""")

        # --- JUNCTION NODES (Manholes) ---
        f.write("[JUNCTIONS]\n")
        f.write(";; Main trunk line manholes (N→S along central avenue)\n")
        node_id = 1
        for i, (y, inv) in enumerate(zip(main_nodes_y, main_inverts)):
            max_d = 2.0  # max depth
            init_d = 0.0
            sur_d = 0.0
            ponded_a = 0.0
            f.write(f"  M_{node_id:02d}    {inv:.2f}    {max_d:.1f}    {init_d:.1f}    {sur_d:.1f}    {ponded_a:.0f}\n")
            node_id += 1

        f.write("\n;; Branch line manholes\n")
        for i, (x, y) in enumerate(zip(branch_x_left + branch_x_right, branch_y + branch_y)):
            inv = 47.5 - y * 0.005 + 0.3  # slightly above main invert
            max_d = 1.5
            f.write(f"  B_{node_id:02d}    {inv:.2f}    {max_d:.1f}    0.0    0.0    0\n")
            node_id += 1

        # --- OUTFALL ---
        f.write("\n[OUTFALLS]\n")
        f.write(";; Southern boundary outfall\n")
        f.write(f"  Outfall_01    46.0    FREE    NO\n")

        # --- CONDUITS (Pipes) ---
        f.write("\n[CONDUITS]\n")
        f.write(";; Main trunk sewer (N→S)\n")
        f.write(f"  C_M01    M_01    M_02    60    0.013    0    0    0    0\n")
        f.write(f"  C_M02    M_02    M_03    60    0.013    0    0    0    0\n")
        f.write(f"  C_M03    M_03    M_04    60    0.013    0    0    0    0\n")
        f.write(f"  C_M04    M_04    M_05    60    0.013    0    0    0    0\n")
        f.write(f"  C_M05    M_05    M_06    60    0.013    0    0    0    0\n")
        f.write(f"  C_out    M_06    Outfall_01    50    0.013    0    0    0    0\n")

        f.write("\n;; Branch sewers (into main trunk)\n")
        f.write(f"  C_B01    B_07    M_02    80    0.013    0    0    0    0\n")
        f.write(f"  C_B02    B_08    M_02    80    0.013    0    0    0    0\n")
        f.write(f"  C_B03    B_09    M_03    80    0.013    0    0    0    0\n")
        f.write(f"  C_B04    B_10    M_03    80    0.013    0    0    0    0\n")
        f.write(f"  C_B05    B_11    M_05    80    0.013    0    0    0    0\n")
        f.write(f"  C_B06    B_12    M_05    80    0.013    0    0    0    0\n")

        # --- XSECTIONS ---
        f.write("\n[XSECTIONS]\n")
        f.write(";; Main line: 600mm circular pipes\n")
        f.write(f"  C_M01    CIRCULAR    0.6    0    0    0    1.0\n")
        f.write(f"  C_M02    CIRCULAR    0.6    0    0    0    1.0\n")
        f.write(f"  C_M03    CIRCULAR    0.6    0    0    0    1.0\n")
        f.write(f"  C_M04    CIRCULAR    0.6    0    0    0    1.0\n")
        f.write(f"  C_M05    CIRCULAR    0.6    0    0    0    1.0\n")
        f.write(f"  C_out    CIRCULAR    0.8    0    0    0    1.0\n")
        f.write(";; Branch lines: 400mm circular pipes\n")
        f.write(f"  C_B01    CIRCULAR    0.4    0    0    0    1.0\n")
        f.write(f"  C_B02    CIRCULAR    0.4    0    0    0    1.0\n")
        f.write(f"  C_B03    CIRCULAR    0.4    0    0    0    1.0\n")
        f.write(f"  C_B04    CIRCULAR    0.4    0    0    0    1.0\n")
        f.write(f"  C_B05    CIRCULAR    0.4    0    0    0    1.0\n")
        f.write(f"  C_B06    CIRCULAR    0.4    0    0    0    1.0\n")

        # --- COORDINATES (critical: enables surface coupling) ---
        f.write("\n[COORDINATES]\n")
        f.write(";; Main trunk manholes\n")
        for i, y in enumerate(main_nodes_y):
            f.write(f"  M_{i+1:02d}    {main_x:.1f}    {y:.1f}\n")
        f.write(";; Branch manholes (left side, then right side)\n")
        for i, (x, y) in enumerate(zip(branch_x_left, branch_y)):
            f.write(f"  B_{i+7:02d}    {x:.1f}    {y:.1f}\n")
        for i, (x, y) in enumerate(zip(branch_x_right, branch_y)):
            f.write(f"  B_{i+10:02d}    {x:.1f}    {y:.1f}\n")
        f.write(f"  Outfall_01    {main_x:.1f}    10.0\n")

        # --- SUBCATCHMENTS (minimal, ITZI handles surface hydrology) ---
        f.write("\n[SUBCATCHMENTS]\n")
        f.write(";; Minimal subcatchment for SWMM initialization\n")
        f.write("  SC_01    RainGage    M_01    10    50    0.5    0.5    0.5    OUTLET\n")

        f.write("\n[SUBAREAS]\n")
        f.write("  SC_01    0.5    0.03    0.5    0.5    0.0    OUTLET\n")

        f.write("\n[INFILTRATION]\n")
        f.write("  SC_01    80.0    20.0    6.0    10.0    0\n")

        f.write("\n[TAGS]\n")
        f.write("\n[MAP]\n\n")
        f.write("[REPORT]\n")
        f.write("INPUT      NO\n")
        f.write("CONTROLS   NO\n")
        f.write("NODES ALL\n")
        f.write("LINKS ALL\n")

    print(f"  SWMM input file created: {inp_path}")
    print(f"    Network: 6 main manholes + 12 branch manholes + 1 outfall")
    print(f"    Pipes: 600mm main trunk, 400mm branches")
    return inp_path


def create_urban_terrain():
    """Same urban terrain as the urban_district case."""
    x = np.linspace(0, DOMAIN_SIZE, GRID_SIZE)
    y = np.linspace(0, DOMAIN_SIZE, GRID_SIZE)
    xx, yy = np.meshgrid(x, y)

    dem = (50.0 - yy * 0.005).astype(np.float32)
    np.random.seed(12345)
    micro_relief = np.random.randn(GRID_SIZE, GRID_SIZE) * 0.03

    n_cols, n_rows = 6, 4
    street_w, main_street_w = 4, 6
    building_w, building_d = 15, 20
    sidewalk_w = 1
    block_pitch_x = building_w + 2 * sidewalk_w + street_w
    block_pitch_y = building_d + 2 * sidewalk_w + street_w
    total_w = n_cols * block_pitch_x - street_w
    total_h = n_rows * block_pitch_y - street_w
    offset_x = (GRID_SIZE - total_w) // 2
    offset_y = (GRID_SIZE - total_h) // 2 + main_street_w // 2

    building_mask = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
    np.random.seed(42)
    for col in range(n_cols):
        for row in range(n_rows):
            bx = offset_x + col * block_pitch_x + sidewalk_w
            by = offset_y + row * block_pitch_y + sidewalk_w
            height = np.random.uniform(8, 18)
            x0, x1 = bx, min(bx + building_w, GRID_SIZE)
            y0, y1 = by, min(by + building_d, GRID_SIZE)
            if x1 < GRID_SIZE and y1 < GRID_SIZE:
                dem[y0:y1, x0:x1] += height
                building_mask[y0:y1, x0:x1] = True

    # Streets
    main_avenue_center = GRID_SIZE // 2
    avenue_x0 = main_avenue_center - main_street_w // 2
    avenue_x1 = main_avenue_center + main_street_w // 2
    dem[:, avenue_x0:avenue_x1] -= 0.2

    for row in range(n_rows + 1):
        street_y = offset_y + row * block_pitch_y - street_w // 2 - sidewalk_w
        if 0 <= street_y < GRID_SIZE:
            y0, y1 = max(0, street_y - street_w // 2), min(GRID_SIZE, street_y + street_w // 2)
            dem[y0:y1, :] -= 0.15

    park_cx, park_cy = GRID_SIZE // 2, GRID_SIZE // 2
    park_radius = 25
    park_dist = np.sqrt((xx - park_cx * CELL_SIZE)**2 + (yy - park_cy * CELL_SIZE)**2)
    park_mask = park_dist < park_radius * CELL_SIZE
    park_lowering = 0.5 * np.exp(-park_dist**2 / (2 * (park_radius * CELL_SIZE / 2)**2))
    dem[park_mask & ~building_mask] -= park_lowering[park_mask & ~building_mask]

    # Manning's n
    mannings = np.full((GRID_SIZE, GRID_SIZE), 0.015, dtype=np.float32)
    mannings[building_mask] = 0.5
    mannings[park_mask] = 0.04

    dem[~building_mask] += micro_relief[~building_mask]

    return dem, mannings, building_mask, park_mask


def run_simulation(name, dem, mannings, building_mask, park_mask, swmm_inp=None):
    """
    Run ITZI simulation. If swmm_inp is provided, couple with drainage.
    Returns results dict.
    """
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    mask = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
    domain = rasterdomain.RasterDomain(dtype=np.float32, arr_mask=mask, cell_shape=(CELL_SIZE, CELL_SIZE))
    sim_param = {'hmin': HMIN, 'cfl': CFL, 'theta': THETA, 'g': G, 'vrouting': 0.1, 'dtmax': DT_MAX, 'slmax': 0.1}
    sf_sim = surfaceflow.SurfaceFlowSimulation(domain, sim_param)
    inf_model = InfNull(domain, DT_INF)
    hydrology = Hydrology(domain, DT_INF, inf_model)

    domain.update_array("dem", dem)
    domain.update_array("friction", mannings)
    domain.update_array("h", np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32))
    sf_sim.update_flow_dir()

    # Drainage setup (if enabled)
    drainage = None
    nodes_list = None
    node_id_to_loc = {}
    if swmm_inp and os.path.exists(swmm_inp):
        print("  Setting up SWMM drainage network...")
        swmm_sim = pyswmm.Simulation(swmm_inp)
        swmm_inp_parser = SwmmInputParser(swmm_inp)

        all_nodes = pyswmm.Nodes(swmm_sim)
        nodes_coors_dict = swmm_inp_parser.get_nodes_id_as_dict()

        # Create drainage nodes
        drainage_params = {
            'orifice_coeff': DefaultValues.ORIFICE_COEFF,
            'free_weir_coeff': DefaultValues.FREE_WEIR_COEFF,
            'submerged_weir_coeff': DefaultValues.SUBMERGED_WEIR_COEFF,
        }

        nodes_list = []
        for pyswmm_node in all_nodes:
            coors = nodes_coors_dict.get(pyswmm_node.nodeid)
            node = DrainageNode(
                node_object=pyswmm_node,
                coordinates=coors,
                coupling_type=CouplingTypes.NOT_COUPLED,
                orifice_coeff=drainage_params['orifice_coeff'],
                free_weir_coeff=drainage_params['free_weir_coeff'],
                submerged_weir_coeff=drainage_params['submerged_weir_coeff'],
                g=G,
            )

            if coors is not None:
                # Check if node is inside the domain
                x, y = coors.x, coors.y
                if 0 <= x <= DOMAIN_SIZE and 0 <= y <= DOMAIN_SIZE:
                    node.coupling_type = CouplingTypes.COUPLED_NO_FLOW
                    col = int(x / CELL_SIZE)
                    row = int((DOMAIN_SIZE - y) / CELL_SIZE)
                    row = max(0, min(GRID_SIZE - 1, row))
                    col = max(0, min(GRID_SIZE - 1, col))
                    node_id_to_loc[pyswmm_node.nodeid] = (row, col)
                    # Lower Manning's n slightly at inlet locations (grate)
                    mannings[row-1:row+2, col-1:col+2] = 0.012

            nodes_list.append((pyswmm_node.nodeid, node, coors.x if coors else None, coors.y if coors else None, row if coors and 0 <= x <= DOMAIN_SIZE and 0 <= y <= DOMAIN_SIZE else None, col if coors and 0 <= x <= DOMAIN_SIZE and 0 <= y <= DOMAIN_SIZE else None))

        # Create links
        links_vertices_dict = swmm_inp_parser.get_links_id_as_dict()
        pyswmm_links = pyswmm.Links(swmm_sim)
        links_list = []
        for pyswmm_link in pyswmm_links:
            in_coor = nodes_coors_dict.get(pyswmm_link.inlet_node)
            out_coor = nodes_coors_dict.get(pyswmm_link.outlet_node)
            vertices = [in_coor] if in_coor else []
            vdata = links_vertices_dict.get(pyswmm_link.linkid)
            if vdata and vdata.vertices:
                vertices.extend(vdata.vertices)
            if out_coor:
                vertices.append(out_coor)
            links_list.append(DrainageLink(link_object=pyswmm_link, vertices=vertices))

        node_objects = [n[1] for n in nodes_list]
        drainage = DrainageSimulation(swmm_sim, node_objects, links_list)

        # Re-update friction with inlet modifications
        domain.update_array("friction", mannings)

        n_coupled = sum(1 for n in nodes_list if n[1].coupling_type != CouplingTypes.NOT_COUPLED)
        print(f"    {len(nodes_list)} nodes, {n_coupled} coupled to surface, {len(links_list)} links")
    else:
        print("  No drainage (surface only)")

    # Update domain with modified mannings
    domain.update_array("friction", mannings)

    # Rainfall
    dt_rain = 300
    n_rain_steps = SIM_DURATION_S // dt_rain
    t_rain = np.arange(0, SIM_DURATION_S, dt_rain)
    t_peak = 2700
    intensity = np.where(t_rain < t_peak, 50 * (t_rain / t_peak) ** 0.5,
                         50 * ((SIM_DURATION_S - t_rain) / (SIM_DURATION_S - t_peak)) ** 1.5)
    intensity = intensity * 50 / (np.sum(intensity) * dt_rain / 3600)
    rain_ms = (intensity / (1000 * 3600)).astype(np.float32)

    # Run
    print(f"  Running simulation...")
    start_time = time.time()
    sim_time_s = 0.0
    dt = 0.001
    next_record = RECORD_STEP_S
    next_rain = dt_rain
    rain_idx = 0
    record_count = 0
    next_drainage_step = 0.0  # Track when to call drainage.step()

    results = {'time': [], 'volume': [], 'hmax': [], 'flooded': [], 'h_record': {},
               'drain_flow': [] if drainage else None}

    while sim_time_s < SIM_DURATION_S:
        # Update rainfall
        if sim_time_s >= next_rain and rain_idx < n_rain_steps:
            rain_arr = np.full((GRID_SIZE, GRID_SIZE), rain_ms[rain_idx], dtype=np.float32)
            domain.update_array("rain", rain_arr)
            rain_idx += 1
            next_rain += dt_rain

        # Hydrology
        hydrology.solve_dt()
        hydrology.step()

        # Drainage step (only when ITZI time catches up to SWMM time)
        if drainage and sim_time_s >= drainage.elapsed_time and drainage.elapsed_time < SIM_DURATION_S - 5:
            try:
                drainage.step()
                # Apply coupling
                surface_states = {}
                arr_z = domain.get_array("dem")
                arr_h = domain.get_array("h")
                cell_surf = CELL_SIZE * CELL_SIZE
                for node_id, (row, col) in node_id_to_loc.items():
                    surface_states[node_id] = {'z': arr_z[row, col], 'h': arr_h[row, col]}
                coupling_flows = drainage.apply_coupling_to_nodes(surface_states, cell_surf)
                arr_qd = domain.get_array("n_drain")
                for node_id, flow in coupling_flows.items():
                    if node_id in node_id_to_loc:
                        row, col = node_id_to_loc[node_id]
                        arr_qd[row, col] = flow / cell_surf
            except Exception:
                pass

        # Surface flow
        domain.update_ext_array()
        sf_sim.dt = timedelta(seconds=dt)
        sf_sim.step()
        sf_sim.solve_dt()

        sim_time_s += dt
        dt = sf_sim.dt.total_seconds()

        # Record
        if sim_time_s >= next_record:
            record_count += 1
            h = domain.get_array("h")
            results['time'].append(sim_time_s / 3600)
            results['volume'].append(np.sum(h) * CELL_SIZE * CELL_SIZE)
            results['hmax'].append(np.max(h))
            results['flooded'].append(np.sum(h > 0.005))
            results['h_record'][record_count] = h.copy()
            if drainage:
                total_drain = np.sum(domain.get_array("n_drain")) * cell_surf
                results['drain_flow'].append(total_drain)

            elapsed = time.time() - start_time
            drain_str = f"  drain={total_drain:+.1f}m3/s" if drainage else ""
            print(f"  {sim_time_s/3600:>5.2f}h  vol={results['volume'][-1]:.0f}m3  hmax={results['hmax'][-1]:.3f}m  flooded={results['flooded'][-1]}{drain_str}")
            next_record += RECORD_STEP_S

    elapsed = timedelta(seconds=int(time.time() - start_time))
    print(f"  Completed in {elapsed}, {record_count} records")

    # Clean up drainage
    if drainage:
        try:
            drainage.swmm_model.swmm_report()
            drainage.swmm_model.swmm_close()
        except:
            pass

    return results


def main():
    print("=" * 60)
    print("  ITZI Coupled Surface-Drainage Urban Flood Simulation")
    print("=" * 60)

    # Create terrain and SWMM input
    print("\nStep 1: Creating urban terrain...")
    dem, mannings, building_mask, park_mask = create_urban_terrain()
    print(f"  DEM: {dem.min():.1f}m - {dem.max():.1f}m")

    print("\nStep 2: Creating SWMM drainage network...")
    import create_swmm_network as csn
    swmm_inp = csn.OUTPUT if hasattr(csn, 'OUTPUT') else os.path.join(TEST_DIR, "drainage_network.inp")
    if not os.path.exists(swmm_inp):
        create_swmm_inp()
    else:
        print(f"  Using existing: {swmm_inp}")

    # Run BOTH simulations
    print("\n" + "=" * 60)
    print("  Running paired simulations...")

    # A) Surface only
    results_no_drain = run_simulation(
        "A) SURFACE ONLY (No Drainage)",
        dem.copy(), mannings.copy(), building_mask, park_mask, swmm_inp=None
    )

    # B) Surface + Drainage
    results_drain = run_simulation(
        "B) SURFACE + DRAINAGE (SWMM Coupled)",
        dem.copy(), mannings.copy(), building_mask, park_mask, swmm_inp=swmm_inp
    )

    # ================================================================
    # Save and compare results
    # ================================================================
    print("\n" + "=" * 60)
    print("  Results Summary")
    print("=" * 60)

    for label, res in [("No Drainage", results_no_drain), ("With Drainage", results_drain)]:
        print(f"\n  {label}:")
        print(f"    Final volume:   {res['volume'][-1]:.0f} m3")
        print(f"    Final max depth: {res['hmax'][-1]:.3f} m")
        print(f"    Final flooded:   {res['flooded'][-1]} cells")

    # Save all results
    np.savez(os.path.join(OUTPUT_DIR, "coupled_results.npz"),
             time=results_no_drain['time'],
             volume_no_drain=results_no_drain['volume'],
             volume_drain=results_drain['volume'],
             hmax_no_drain=results_no_drain['hmax'],
             hmax_drain=results_drain['hmax'],
             flooded_no_drain=results_no_drain['flooded'],
             flooded_drain=results_drain['flooded'],
             drain_flow=results_drain['drain_flow'] if results_drain['drain_flow'] else [],
             dem=dem, mannings=mannings,
             building_mask=building_mask, park_mask=park_mask,
             h_final_no_drain=results_no_drain['h_record'][max(results_no_drain['h_record'].keys())],
             h_final_drain=results_drain['h_record'][max(results_drain['h_record'].keys())],
             cell_size=CELL_SIZE, domain_size=DOMAIN_SIZE)

    # Save depth snapshots for both
    for label, res in [("nodrain", results_no_drain), ("drain", results_drain)]:
        for rec_id, h in res['h_record'].items():
            if rec_id <= len(res['time']):
                t_h = res['time'][rec_id - 1]
                np.savetxt(os.path.join(OUTPUT_DIR, f"depth_{label}_t{t_h:.1f}h.asc"), h, fmt='%.4f')

    print(f"\n  Results saved to: {OUTPUT_DIR}")
    print("=" * 60)
    print("  Done!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
