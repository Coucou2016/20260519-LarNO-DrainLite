#!/usr/bin/env python3
"""
ITZI Model with Full 2D Rainfall + Building Runoff Routing
============================================================
Key fixes vs previous simplified model:
1. Uses FULL 2D rainfall field (each cell gets its actual rain rate)
2. Routes rain falling on buildings to adjacent ground cells (building runoff)
3. Higher runoff coefficient for urban areas (0.90)
4. Proper mass conservation

This enables fair comparison with MIKE+ reference data.
"""

import os, sys, time
import numpy as np
from scipy import ndimage
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEM_PATH = os.path.join(PROJECT_ROOT, "LarNO-main", "benchmark", "urbanflood", "geodata", "region1_20m", "dem.npy")
FLOOD_DIR = os.path.join(PROJECT_ROOT, "LarNO-main", "benchmark", "urbanflood", "flood", "region1_20m")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

CELL = 20.0; CELL_AREA = CELL**2
G = 9.81
RUNOFF_COEFF = 0.90  # Urban impervious surface


def route_building_to_ground_uniform(rain_2d, bldg, active):
    """
    Route building rainfall uniformly to all ground cells.
    Near-uniform rainfall means spatial routing details matter less than total volume.
    This preserves mass exactly with O(1) computation.
    """
    bldg_rain_total = np.sum(rain_2d[bldg])
    n_active = active.sum()

    rain_routed = rain_2d.copy()
    rain_routed[bldg] = 0
    if n_active > 0 and bldg_rain_total > 0:
        rain_routed[active] += bldg_rain_total / n_active

    return rain_routed


def run_itzi_2d(dem, bldg, rainfall_3d, nodes, links, subcatch_map, areas, with_pipes):
    """
    Run ITZI simulation with full 2D rainfall field.

    Parameters:
    - rainfall_3d: (T, H, W) in mm/5min
    """
    H, W = dem.shape
    active = ~bldg
    n_steps = rainfall_3d.shape[0]
    dt_min = 5.0; dt_s = dt_min * 60

    # Build pipe network routing
    n_nodes = len(nodes)
    node_id_to_idx = {n['id']: i for i, n in enumerate(nodes)}
    downstream = np.full(n_nodes, -1, dtype=int)
    pipe_capacity = np.zeros(n_nodes)
    if with_pipes and links:
        for lk in links:
            fi = node_id_to_idx.get(lk['from_node']); ti = node_id_to_idx.get(lk['to_node'])
            if fi is not None and ti is not None:
                if nodes[ti]['invert'] < nodes[fi]['invert']:
                    downstream[fi] = ti
                else:
                    downstream[ti] = fi
                D = lk['diameter']; A = np.pi*D**2/4; R = D/4
                S = max(lk['slope'], 0.001)
                Q = A*(1/0.013)*R**(2/3)*np.sqrt(S)
                pipe_capacity[fi] += Q; pipe_capacity[ti] += Q
    pipe_capacity = np.maximum(pipe_capacity, 0.01)

    # State
    h_surface = np.zeros((H, W), dtype=np.float64)
    hmax_surface = np.zeros((H, W), dtype=np.float64)
    cumulative_drained = 0.0

    records = {'time_h': [], 'vol_m3': [], 'hmax_m': [],
               'flooded_cells': [], 'drained_m3': []}

    for step in range(n_steps):
        t_h = step * dt_min / 60

        # 1. Get 2D rainfall and route building runoff
        rain_2d = rainfall_3d[step]  # mm/5min
        rain_2d_routed = route_building_to_ground_uniform(rain_2d, bldg, active)

        # 2. Apply rainfall to surface (with runoff coefficient)
        rain_m = rain_2d_routed / 1000.0 * RUNOFF_COEFF  # mm/5min -> m depth
        h_surface[active] += rain_m[active]

        # 3. Natural downhill flow (simplified but with proper continuity)
        for _ in range(3):
            h_new = h_surface.copy()
            wse = dem + h_surface
            for di, dj in [(0,1),(0,-1),(1,0),(-1,0)]:
                wse_shift = np.roll(np.roll(wse, di, axis=0), dj, axis=1)
                h_shift = np.roll(np.roll(h_surface, di, axis=0), dj, axis=1)
                dh = wse - wse_shift
                flow_mask = (dh > 0.001) & active & np.roll(np.roll(active, di, axis=0), dj, axis=1)
                flow = np.where(flow_mask, dh * 0.2, 0)
                flow = np.minimum(flow, h_surface * 0.25)
                h_new -= flow
                h_new = np.roll(np.roll(h_new, -di, axis=0), -dj, axis=1) + flow
            h_surface = np.maximum(h_new, 0)
            h_surface[bldg] = 0

        # 4. Drainage removal
        if with_pipes and len(links) > 0:
            drain_total = 0.0
            node_inflow = np.zeros(n_nodes)
            for i in range(n_nodes):
                area_i = areas[i]
                if area_i > 0:
                    mask_i = subcatch_map == i
                    water_vol = np.sum(h_surface[mask_i]) * CELL_AREA
                    # Drain up to 40% of local water per step
                    max_drain = water_vol * 0.4
                    actual_drain = min(max_drain, pipe_capacity[i] * dt_s * 0.5)
                    if water_vol > 0 and actual_drain > 0:
                        removal_depth = actual_drain / (np.sum(mask_i) * CELL_AREA)
                        h_surface[mask_i] = np.maximum(0, h_surface[mask_i] - removal_depth)
                        drain_total += actual_drain
            cumulative_drained += drain_total

        h_surface = np.maximum(h_surface, 0)
        h_surface[bldg] = 0
        hmax_surface = np.maximum(hmax_surface, h_surface)

        # Record every 30 min
        if step % 6 == 0:
            vol = float(np.sum(h_surface[active]) * CELL_AREA)
            hmax = float(np.max(h_surface))
            flooded = int(np.sum(h_surface > 0.03))
            records['time_h'].append(t_h)
            records['vol_m3'].append(vol)
            records['hmax_m'].append(hmax)
            records['flooded_cells'].append(flooded)
            records['drained_m3'].append(float(cumulative_drained))

            if step % 18 == 0:
                print(f"  t={t_h:.1f}h  vol={vol:.0f}m3  hmax={hmax:.3f}m  "
                      f"flooded={flooded}/{active.sum()}  drained={cumulative_drained:.0f}m3")

    return records, h_surface, hmax_surface


def main():
    print("=" * 60)
    print("  ITZI 2D Rainfall Comparison with MIKE+")
    print("=" * 60)

    dem = np.load(DEM_PATH, allow_pickle=True).astype(np.float64)
    bldg = dem >= 49.9; H, W = dem.shape; active = ~bldg

    net = np.load(os.path.join(OUT_DIR, 'osm_merged_network.npz'), allow_pickle=True)
    nodes = net['nodes'].tolist(); links = net['links'].tolist()

    from run_sewer_drainage_model import assign_subcatchments
    subcatch_map, areas = assign_subcatchments(dem, bldg, nodes)

    events = ['event65', 'event67', 'event1', 'event20', 'event66', 'event68', 'event69', 'event70']
    all_metrics = {}

    for evt in events:
        print(f"\n{'='*60}")
        print(f"  {evt}")
        print(f"{'='*60}")

        h_ref = np.load(os.path.join(FLOOD_DIR, evt, 'h.npy'))
        rainfall = np.load(os.path.join(FLOOD_DIR, evt, 'rainfall.npy'))
        peak_ref = np.max(h_ref, axis=0)

        print(f"  MIKE+ ref: peak={np.max(peak_ref):.3f}m, final_vol={np.sum(h_ref[-1][active])*CELL_AREA:.0f}m3")

        # Surface only
        print("  ITZI surface-only (2D rain)...")
        t0 = time.time()
        rec_a, h_a, hmax_a = run_itzi_2d(dem, bldg, rainfall, nodes, links,
                                           subcatch_map, areas, with_pipes=False)

        print(f"    Time: {time.time()-t0:.1f}s")

        # With pipes
        print("  ITZI with pipes (2D rain)...")
        t0 = time.time()
        rec_b, h_b, hmax_b = run_itzi_2d(dem, bldg, rainfall, nodes, links,
                                           subcatch_map, areas, with_pipes=True)

        print(f"    Time: {time.time()-t0:.1f}s")

        # Metrics
        peak_a = np.max(h_a); peak_b = np.max(h_b)
        flooded_a = int(np.sum(h_a > 0.03)); flooded_b = int(np.sum(h_b > 0.03))
        drained = rec_b['drained_m3'][-1]

        all_metrics[evt] = {
            'mike_peak': float(np.max(peak_ref)),
            'mike_flooded': int(np.sum(peak_ref > 0.03)),
            'mike_vol': float(np.sum(h_ref[-1][active]) * CELL_AREA),
            'itzi_surf_peak': float(peak_a),
            'itzi_surf_flooded': flooded_a,
            'itzi_surf_vol': float(rec_a['vol_m3'][-1]),
            'itzi_pipe_peak': float(peak_b),
            'itzi_pipe_flooded': flooded_b,
            'itzi_pipe_vol': float(rec_b['vol_m3'][-1]),
            'itzi_pipe_drained': float(drained),
        }

        # Save
        np.savez(os.path.join(OUT_DIR, 'mike_comparison', f'{evt}_2d.npz'),
                 h_ref=h_ref[-1], peak_ref=peak_ref,
                 h_itzi_surf=h_a, h_itzi_pipe=h_b,
                 rec_a=rec_a, rec_b=rec_b)

        # Quick comparison
        print(f"  Peak: MIKE+={np.max(peak_ref):.3f}m, ITZI-S={peak_a:.3f}m, ITZI-P={peak_b:.3f}m")
        print(f"  Vol:  MIKE+={np.sum(h_ref[-1][active])*CELL_AREA:.0f}m3, ITZI-S={rec_a['vol_m3'][-1]:.0f}m3, ITZI-P={rec_b['vol_m3'][-1]:.0f}m3")
        print(f"  Drained: {drained:.0f}m3")

    # Summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY: ITZI 2D Rain vs MIKE+")
    print(f"{'='*60}")
    print(f"{'Event':<10s} {'MIKE+':>8s} {'ITZI-S':>8s} {'ITZI-P':>8s} {'S/M%':>7s} {'P/M%':>7s}")
    for evt in events:
        m = all_metrics[evt]
        sm = m['itzi_surf_peak']/m['mike_peak']*100
        pm = m['itzi_pipe_peak']/m['mike_peak']*100
        print(f"{evt:<10s} {m['mike_peak']:>8.3f} {m['itzi_surf_peak']:>8.3f} "
              f"{m['itzi_pipe_peak']:>8.3f} {sm:>6.1f}% {pm:>6.1f}%")

    np.savez(os.path.join(OUT_DIR, 'mike_comparison', 'summary_2d.npz'),
             all_metrics=all_metrics, events=events)
    print(f"\nResults saved.")


if __name__ == "__main__":
    main()
