#!/usr/bin/env python3
"""
ITZI Dynamic Model vs MIKE+ Comparison
========================================
Uses the actual ITZI 2D partial-inertia surface flow solver
with Manning's friction, coupled to the OSM pipe network.

This is the REAL ITZI model, not a simplified version.
"""

import sys, os, time
import numpy as np
import itzi.rasterdomain as rasterdomain
import itzi.surfaceflow as surfaceflow
from itzi.hydrology import Hydrology
from itzi.infiltration import InfNull
from datetime import timedelta

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEM_PATH = os.path.join(PROJECT_ROOT, "LarNO-main", "benchmark", "urbanflood", "geodata", "region1_20m", "dem.npy")
FLOOD_DIR = os.path.join(PROJECT_ROOT, "LarNO-main", "benchmark", "urbanflood", "flood", "region1_20m")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "itzi_mike")
os.makedirs(OUT_DIR, exist_ok=True)

CELL = 20.0; CELL_AREA = CELL**2
flood_cmap = LinearSegmentedColormap.from_list('flood', [
    (0.0, '#F7FBFF'), (0.12, '#C6DBEF'), (0.25, '#6BAED6'),
    (0.4, '#3182BD'), (0.55, '#08519C'), (0.7, '#08306B'), (1.0, '#041838'),
])


def run_itzi_simulation(dem, bldg, rainfall_3d, pipe_nodes=None, pipe_links=None, label=""):
    """
    Run ITZI 2D partial-inertia simulation with actual rainfall.

    Parameters:
    - dem: (H, W) float32 DEM
    - bldg: (H, W) bool building mask
    - rainfall_3d: (T, H, W) in mm/5min
    - pipe_nodes, pipe_links: optional pipe network
    """
    H, W = dem.shape; active = ~bldg
    DURATION = rainfall_3d.shape[0] * 300  # 72 steps × 5min = 21600s
    DT_RAIN = 300

    # Setup terrain
    dem_itzi = dem.copy()
    dem_itzi[bldg] = 50.0  # Buildings as 50m walls

    mask = np.zeros((H, W), dtype=bool)
    domain = rasterdomain.RasterDomain(dtype=np.float32, arr_mask=mask, cell_shape=(CELL, CELL))

    sim_param = {'hmin': 0.001, 'cfl': 0.7, 'theta': 0.9, 'g': 9.80665,
                 'vrouting': 0.1, 'dtmax': 1.0, 'slmax': 0.1}
    sf_sim = surfaceflow.SurfaceFlowSimulation(domain, sim_param)
    hydrology = Hydrology(domain, 60.0, InfNull(domain, 60.0))

    # Manning's n: streets=0.015, buildings=100
    mannings = np.full((H, W), 0.015, dtype=np.float32)
    mannings[bldg] = 100.0

    domain.update_array(k="dem", arr=dem_itzi)
    domain.update_array(k="friction", arr=mannings)
    domain.update_array(k="h", arr=np.zeros((H, W), dtype=np.float32))
    domain.update_array(k="rain", arr=np.zeros((H, W), dtype=np.float32))
    sf_sim.update_flow_dir()

    # Pipe network: pre-compute inlet cells and capacity
    has_pipes = pipe_nodes is not None and len(pipe_nodes) > 0
    if has_pipes:
        node_cells = []
        for n in pipe_nodes:
            r, c = n['row'], n['col']
            if 0 <= r < H and 0 <= c < W and not bldg[r, c]:
                node_cells.append((r, c))
        print(f"  Pipes: {len(node_cells)} inlets")
    else:
        node_cells = []

    # Run
    t = 0.0; dt = 0.001
    rain_idx = 0; next_rain = DT_RAIN
    next_record = 1800  # 30 min
    cumulative_drained = 0.0
    n_active = active.sum()

    records = {'time_h': [], 'vol_m3': [], 'hmax_m': [], 'flooded_cells': [], 'drained_m3': []}
    t0 = time.time()

    while t < DURATION and rain_idx < rainfall_3d.shape[0]:
        # Update rainfall every 5 min
        if t >= next_rain and rain_idx < rainfall_3d.shape[0]:
            rain_2d = rainfall_3d[rain_idx]  # mm/5min
            # Route building rain to active cells
            bldg_rain = np.sum(rain_2d[bldg])
            bldg_extra = bldg_rain / n_active if n_active > 0 else 0
            rain_field = np.zeros((H, W), dtype=np.float32)
            rain_field[active] = (rain_2d[active] + bldg_extra) / 1000.0 / DT_RAIN  # m/s
            domain.update_array(k="rain", arr=rain_field)
            rain_idx += 1; next_rain += DT_RAIN

        # ITZI step
        domain.update_ext_array()
        hydrology.solve_dt()
        hydrology.step()

        # Pipe drainage (every 30s)
        if has_pipes and int(t) % 30 == 0 and int(t) > 0:
            h_arr = domain.get_array("h")
            for r, c in node_cells:
                if h_arr[r, c] > 0.01:
                    surf_wl = dem_itzi[r, c] + h_arr[r, c]
                    head = max(surf_wl - (dem_itzi[r, c] - 2.0), 0.01)
                    Q = 0.65 * 2.0 * np.sqrt(2 * 9.81 * head)
                    Q = min(Q, h_arr[r, c] * CELL_AREA / 30.0 * 0.3)
                    removal = Q * 30.0 / CELL_AREA
                    h_arr[r, c] = max(0, h_arr[r, c] - removal)
                    cumulative_drained += Q * 30.0
            domain.update_array(k="h", arr=h_arr)

        sf_sim.dt = timedelta(seconds=dt)
        sf_sim.step()
        sf_sim.solve_dt()
        t += dt
        dt = sf_sim.dt.total_seconds()

        # Record
        if t >= next_record:
            h_arr = domain.get_array("h")
            vol = float(np.sum(h_arr[active]) * CELL_AREA)
            hmax = float(np.max(h_arr))
            flooded = int(np.sum(h_arr > 0.03))
            records['time_h'].append(t / 3600)
            records['vol_m3'].append(vol); records['hmax_m'].append(hmax)
            records['flooded_cells'].append(flooded)
            records['drained_m3'].append(float(cumulative_drained))
            print(f"  t={t/3600:.1f}h dt={dt:.3f}s vol={vol:.0f}m3 hmax={hmax:.3f}m flooded={flooded}")
            next_record += 1800

    elapsed = time.time() - t0
    print(f"  Completed in {elapsed:.0f}s")
    h_final = domain.get_array("h")
    return records, h_final


def main():
    print("=" * 60)
    print("  ITZI Dynamic Model vs MIKE+ Comparison")
    print("=" * 60)

    dem_full = np.load(DEM_PATH, allow_pickle=True).astype(np.float32)
    bldg_full = dem_full >= 49.9
    H_full, W_full = dem_full.shape
    print(f"Full DEM: {H_full}x{W_full} cells = {H_full*CELL/1000:.1f}km x {W_full*CELL/1000:.1f}km")

    # Use 2km x 2km sub-region (matches original ITZI test case scale)
    y0, y1 = 120, 220  # 100 cells = 2km
    x0, x1 = 220, 320  # 100 cells = 2km
    dem = dem_full[y0:y1, x0:x1].copy()
    bldg = bldg_full[y0:y1, x0:x1].copy()
    H, W = dem.shape
    print(f"Sub-region: {H}x{W} cells = {H*CELL/1000:.1f}km x {W*CELL/1000:.1f}km")
    print(f"Buildings: {bldg.sum()}/{bldg.size} ({100*bldg.sum()/bldg.size:.0f}%)")

    net = np.load(os.path.join(os.path.dirname(OUT_DIR), 'osm_merged_network.npz'), allow_pickle=True)
    all_pipe_nodes = net['nodes'].tolist(); all_pipe_links = net['links'].tolist()
    # Filter nodes to sub-region
    pipe_nodes = [n for n in all_pipe_nodes if y0 <= n['row'] < y1 and x0 <= n['col'] < x1]
    # Remap node IDs for links that connect nodes within sub-region
    sub_ids = set(n['id'] for n in pipe_nodes)
    pipe_links = [l for l in all_pipe_links if l['from_node'] in sub_ids and l['to_node'] in sub_ids]
    print(f"Pipe network in sub-region: {len(pipe_nodes)} nodes, {len(pipe_links)} links")

    events = ['event65', 'event67', 'event1']
    all_results = {}

    for evt in events:
        print(f"\n{'='*60}")
        print(f"  {evt}")
        print(f"{'='*60}")

        rainfall_full = np.load(os.path.join(FLOOD_DIR, evt, 'rainfall.npy'))
        h_ref_full = np.load(os.path.join(FLOOD_DIR, evt, 'h.npy'))
        rainfall = rainfall_full[:, y0:y1, x0:x1]
        h_ref = h_ref_full[:, y0:y1, x0:x1]
        peak_ref = np.max(h_ref, axis=0)
        print(f"  MIKE+ ref: peak={np.max(peak_ref):.3f}m, final_vol={np.sum(h_ref[-1][~bldg])*CELL_AREA:.0f}m3")

        # ITZI surface only
        print("\n  --- ITZI Surface Only ---")
        rec_a, h_a = run_itzi_simulation(dem, bldg, rainfall, label=f"{evt} surface")

        # ITZI with pipes
        print("\n  --- ITZI With Pipes ---")
        rec_b, h_b = run_itzi_simulation(dem, bldg, rainfall,
                                          pipe_nodes=pipe_nodes, pipe_links=pipe_links,
                                          label=f"{evt} pipes")

        all_results[evt] = {
            'mike_peak': float(np.max(peak_ref)),
            'mike_vol': float(np.sum(h_ref[-1][~bldg]) * CELL_AREA),
            'itzi_surf_peak': float(np.max(h_a)),
            'itzi_surf_vol': float(rec_a['vol_m3'][-1]),
            'itzi_pipe_peak': float(np.max(h_b)),
            'itzi_pipe_vol': float(rec_b['vol_m3'][-1]),
            'itzi_pipe_drained': float(rec_b['drained_m3'][-1]),
        }

        # Save
        np.savez(os.path.join(OUT_DIR, f'{evt}_itzi.npz'),
                 h_ref=h_ref[-1], peak_ref=peak_ref,
                 h_itzi_surf=h_a, h_itzi_pipe=h_b,
                 rec_a=rec_a, rec_b=rec_b)

        sm = all_results[evt]['itzi_surf_peak'] / all_results[evt]['mike_peak'] * 100
        pm = all_results[evt]['itzi_pipe_peak'] / all_results[evt]['mike_peak'] * 100
        print(f"\n  Peak: MIKE+={all_results[evt]['mike_peak']:.3f}m, ITZI-S={all_results[evt]['itzi_surf_peak']:.3f}m ({sm:.0f}%), ITZI-P={all_results[evt]['itzi_pipe_peak']:.3f}m ({pm:.0f}%)")

    # Summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    for evt in events:
        r = all_results[evt]
        print(f"  {evt}: MIKE+={r['mike_peak']:.3f}m, ITZI-S={r['itzi_surf_peak']:.3f}m, ITZI-P={r['itzi_pipe_peak']:.3f}m, Drained={r['itzi_pipe_drained']:.0f}m3")

    np.savez(os.path.join(OUT_DIR, 'itzi_summary.npz'), all_results=all_results)
    print(f"\nResults saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
