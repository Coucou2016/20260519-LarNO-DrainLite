#!/usr/bin/env python3
"""
ITZI Dynamic Simulation — Shenzhen Region1 (LarNO dataset)
===========================================================
Uses copied LarNO benchmark data: DEM, rainfall events, OSM pipe network.
Runs surface-only and pipe-coupled scenarios for comparison with MIKE+ reference.

Data source: E:\\Projects\\20260519-LarNO (copied locally, originals unchanged)
"""
import sys, os, time
import numpy as np

sys.path.insert(0, r'E:\Miniconda3\Lib\site-packages')
import itzi.rasterdomain as rasterdomain
import itzi.surfaceflow as surfaceflow
from itzi.hydrology import Hydrology
from itzi.infiltration import InfNull
from datetime import timedelta

CASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(CASE_DIR, "input_data")
DEM_PATH = os.path.join(INPUT_DIR, "geodata", "region1_20m", "dem.npy")
FLOOD_DIR = os.path.join(INPUT_DIR, "flood", "region1_20m")
NET_PATH = os.path.join(INPUT_DIR, "networks", "osm_merged_network.npz")
OUT_DIR = os.path.join(CASE_DIR, "output")
os.makedirs(OUT_DIR, exist_ok=True)

CELL = 20.0
CELL_AREA = CELL ** 2
G = 9.81

# Sub-region: 4km x 5.6km (same as LarNO extended study)
Y0, Y1 = 80, 280
X0, X1 = 120, 400


def run_itzi_event(dem, bldg, rainfall_3d, pipe_nodes=None, label=""):
    """Run ITZI 2D partial-inertia solver on sub-region."""
    H, W = dem.shape
    active = ~bldg
    DURATION = rainfall_3d.shape[0] * 300
    DT_RAIN = 300

    dem_itzi = dem.astype(np.float32).copy()
    dem_itzi[bldg] = 50.0

    mask = np.zeros((H, W), dtype=bool)
    domain = rasterdomain.RasterDomain(
        dtype=np.float32, arr_mask=mask, cell_shape=(CELL, CELL)
    )

    sim_param = {
        'hmin': 0.001, 'cfl': 0.7, 'theta': 0.9, 'g': 9.80665,
        'vrouting': 0.1, 'dtmax': 1.0, 'slmax': 0.1,
    }
    sf_sim = surfaceflow.SurfaceFlowSimulation(domain, sim_param)
    hydrology = Hydrology(domain, 60.0, InfNull(domain, 60.0))

    mannings = np.full((H, W), 0.015, dtype=np.float32)
    mannings[bldg] = 100.0

    domain.update_array(k="dem", arr=dem_itzi)
    domain.update_array(k="friction", arr=mannings)
    domain.update_array(k="h", arr=np.zeros((H, W), dtype=np.float32))
    domain.update_array(k="rain", arr=np.zeros((H, W), dtype=np.float32))
    sf_sim.update_flow_dir()

    has_pipes = pipe_nodes is not None and len(pipe_nodes) > 0
    if has_pipes:
        node_inlets = [
            (n['row'], n['col'], n['invert'])
            for n in pipe_nodes
            if 0 <= n['row'] < H and 0 <= n['col'] < W and not bldg[n['row'], n['col']]
        ]
    else:
        node_inlets = []

    t = 0.0
    dt = 0.001
    rain_idx = 0
    next_rain = DT_RAIN
    next_record = 1800
    n_active = active.sum()
    cumulative_drained = 0.0

    records = {
        'time_h': [], 'vol_m3': [], 'hmax_m': [],
        'flooded_cells': [], 'drained_m3': [],
    }
    t0 = time.time()

    while t < DURATION and rain_idx < rainfall_3d.shape[0]:
        if t >= next_rain and rain_idx < rainfall_3d.shape[0]:
            rain_2d = rainfall_3d[rain_idx]
            bldg_rain = np.sum(rain_2d[bldg])
            bldg_extra = bldg_rain / n_active if n_active > 0 else 0
            rain_field = np.zeros((H, W), dtype=np.float32)
            rain_field[active] = (rain_2d[active] + bldg_extra) / 1000.0 / DT_RAIN
            domain.update_array(k="rain", arr=rain_field)
            rain_idx += 1
            next_rain += DT_RAIN

        domain.update_ext_array()
        hydrology.solve_dt()
        hydrology.step()

        if has_pipes and int(t) % 30 == 0 and int(t) > 0:
            h_arr = domain.get_array("h")
            for r, c, invert in node_inlets:
                if h_arr[r, c] > 0.01:
                    surf_wl = dem_itzi[r, c] + h_arr[r, c]
                    head = max(surf_wl - invert, 0.01)
                    Q = 0.65 * 2.0 * np.sqrt(2 * G * head)
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

        if t >= next_record:
            h_arr = domain.get_array("h")
            vol = float(np.sum(h_arr[active]) * CELL_AREA)
            hmax = float(np.max(h_arr))
            flooded = int(np.sum(h_arr > 0.03))
            records['time_h'].append(t / 3600)
            records['vol_m3'].append(vol)
            records['hmax_m'].append(hmax)
            records['flooded_cells'].append(flooded)
            records['drained_m3'].append(float(cumulative_drained))
            print(f"  [{label}] t={t/3600:.1f}h vol={vol:.0f}m3 hmax={hmax:.3f}m "
                  f"flooded={flooded} drained={cumulative_drained:.0f}m3")
            next_record += 1800

    elapsed = time.time() - t0
    print(f"  [{label}] Done in {elapsed:.0f}s")
    return records, domain.get_array("h")


def load_pipe_nodes(dem, bldg):
    """Load OSM pipe network and remap to sub-region coordinates."""
    net = np.load(NET_PATH, allow_pickle=True)
    all_nodes = net['nodes'].tolist()

    H, W = dem.shape
    pipe_nodes = []
    for n in all_nodes:
        nr, nc = n['row'] - Y0, n['col'] - X0
        if 0 <= nr < H and 0 <= nc < W and not bldg[nr, nc]:
            pipe_nodes.append({
                'row': nr, 'col': nc,
                'invert': n['invert'],
                'type': n.get('type', 'junction'),
            })
    return pipe_nodes


def main():
    print("=" * 60)
    print("  ITZI Shenzhen Region1 — Surface + Pipe Network")
    print(f"  Data: {INPUT_DIR}")
    print(f"  Output: {OUT_DIR}")
    print("=" * 60)

    dem_full = np.load(DEM_PATH, allow_pickle=True).astype(np.float32)
    bldg_full = dem_full >= 49.9
    Hf, Wf = dem_full.shape
    print(f"Full DEM: {Hf}x{Wf} = {Hf*CELL/1000:.1f}km x {Wf*CELL/1000:.1f}km")

    dem = dem_full[Y0:Y1, X0:X1].copy()
    bldg = bldg_full[Y0:Y1, X0:X1].copy()
    H, W = dem.shape
    print(f"Sub-region: {H*CELL/1000:.1f}km x {W*CELL/1000:.1f}km, active={(~bldg).sum():,} cells")

    pipe_nodes = load_pipe_nodes(dem, bldg)
    print(f"Pipe network: {len(pipe_nodes)} inlet nodes in sub-region")

    events = ['event1', 'event20', 'event65', 'event66',
              'event67', 'event68', 'event69', 'event70']
    all_results = {}

    for evt in events:
        evt_dir = os.path.join(FLOOD_DIR, evt)
        if not os.path.exists(os.path.join(evt_dir, 'rainfall.npy')):
            print(f"Skipping {evt} — data not found")
            continue

        print(f"\n{'='*60}\n  {evt}\n{'='*60}")

        rainfall_full = np.load(os.path.join(evt_dir, 'rainfall.npy'))
        h_ref_full = np.load(os.path.join(evt_dir, 'h.npy'))
        rainfall = rainfall_full[:, Y0:Y1, X0:X1]
        h_ref = h_ref_full[:, Y0:Y1, X0:X1]
        peak_ref = np.max(h_ref, axis=0)
        print(f"  MIKE+ ref: peak={np.max(peak_ref):.3f}m, "
              f"flooded={np.sum(peak_ref > 0.03):,}")

        print("  --- ITZI Surface Only ---")
        rec_s, h_s = run_itzi_event(
            dem.copy(), bldg.copy(), rainfall, label=f"{evt} surface"
        )

        print("  --- ITZI With Pipes ---")
        rec_p, h_p = run_itzi_event(
            dem.copy(), bldg.copy(), rainfall,
            pipe_nodes=pipe_nodes, label=f"{evt} pipes"
        )

        all_results[evt] = {
            'mike_peak': float(np.max(peak_ref)),
            'mike_flooded': int(np.sum(peak_ref > 0.03)),
            'itzi_surf_peak': float(np.max(h_s)),
            'itzi_surf_flooded': int(np.sum(h_s > 0.03)),
            'itzi_pipe_peak': float(np.max(h_p)),
            'itzi_pipe_flooded': int(np.sum(h_p > 0.03)),
            'itzi_pipe_drained': float(rec_p['drained_m3'][-1]),
        }

        np.savez(
            os.path.join(OUT_DIR, f'{evt}_itzi.npz'),
            h_ref=h_ref[-1], peak_ref=peak_ref,
            h_itzi_surf=h_s, h_itzi_pipe=h_p,
            rec_s=rec_s, rec_p=rec_p,
        )

        sp = all_results[evt]['itzi_surf_peak'] / all_results[evt]['mike_peak'] * 100
        pp = all_results[evt]['itzi_pipe_peak'] / all_results[evt]['mike_peak'] * 100
        print(f"  Peak: MIKE+={all_results[evt]['mike_peak']:.3f}m, "
              f"ITZI-S={all_results[evt]['itzi_surf_peak']:.3f}m ({sp:.0f}%), "
              f"ITZI-P={all_results[evt]['itzi_pipe_peak']:.3f}m ({pp:.0f}%)")

    print(f"\n{'='*80}")
    print("  SUMMARY — ITZI vs MIKE+ (Shenzhen 4km x 5.6km)")
    print(f"{'='*80}")
    print(f"{'Event':<10s} {'MIKE+':>8s} {'ITZI-S':>8s} {'ITZI-P':>8s} "
          f"{'S/M%':>7s} {'P/M%':>7s} {'Drained':>10s}")
    print("-" * 65)
    for evt, r in all_results.items():
        sm = r['itzi_surf_peak'] / r['mike_peak'] * 100
        pm = r['itzi_pipe_peak'] / r['mike_peak'] * 100
        print(f"{evt:<10s} {r['mike_peak']:>8.3f} {r['itzi_surf_peak']:>8.3f} "
              f"{r['itzi_pipe_peak']:>8.3f} {sm:>6.0f}% {pm:>6.0f}% "
              f"{r['itzi_pipe_drained']:>10.0f}")

    np.savez(os.path.join(OUT_DIR, 'all_summary.npz'), all_results=all_results)
    print(f"\nResults saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
