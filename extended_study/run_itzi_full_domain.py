#!/usr/bin/env python3
"""
ITZI Dynamic Model — Full Domain (400×560) Comparison with MIKE+
===============================================================
Runs ITZI 2D partial-inertia solver on the entire Shenzhen Futian domain.
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
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "itzi_full")
os.makedirs(OUT_DIR, exist_ok=True)

CELL = 20.0; CELL_AREA = CELL**2

flood_cmap = LinearSegmentedColormap.from_list('flood', [
    (0.0, '#F7FBFF'), (0.12, '#C6DBEF'), (0.25, '#6BAED6'),
    (0.4, '#3182BD'), (0.55, '#08519C'), (0.7, '#08306B'), (1.0, '#041838'),
])


def run_itzi_region(dem, bldg, rainfall_3d, label=""):
    """Run ITZI on full 400x560 domain."""
    H, W = dem.shape; active = ~bldg
    DURATION = rainfall_3d.shape[0] * 300
    DT_RAIN = 300

    dem_itzi = dem.astype(np.float32).copy()
    dem_itzi[bldg] = 50.0

    mask = np.zeros((H, W), dtype=bool)
    domain = rasterdomain.RasterDomain(dtype=np.float32, arr_mask=mask, cell_shape=(CELL, CELL))

    sim_param = {'hmin': 0.001, 'cfl': 0.7, 'theta': 0.9, 'g': 9.80665,
                 'vrouting': 0.1, 'dtmax': 1.0, 'slmax': 0.1}
    sf_sim = surfaceflow.SurfaceFlowSimulation(domain, sim_param)
    hydrology = Hydrology(domain, 60.0, InfNull(domain, 60.0))

    mannings = np.full((H, W), 0.015, dtype=np.float32)
    mannings[bldg] = 100.0

    domain.update_array(k="dem", arr=dem_itzi)
    domain.update_array(k="friction", arr=mannings)
    domain.update_array(k="h", arr=np.zeros((H, W), dtype=np.float32))
    domain.update_array(k="rain", arr=np.zeros((H, W), dtype=np.float32))

    print("  Computing flow directions...")
    sf_sim.update_flow_dir()
    print("  Flow directions computed.")

    t = 0.0; dt = 0.001
    rain_idx = 0; next_rain = DT_RAIN
    next_record = 1800
    n_active = active.sum()
    records = {'time_h': [], 'vol_m3': [], 'hmax_m': [], 'flooded_cells': []}
    t0 = time.time()
    step_count = 0

    while t < DURATION and rain_idx < rainfall_3d.shape[0]:
        # Update rainfall
        if t >= next_rain and rain_idx < rainfall_3d.shape[0]:
            rain_2d = rainfall_3d[rain_idx]
            bldg_rain = np.sum(rain_2d[bldg])
            bldg_extra = bldg_rain / n_active if n_active > 0 else 0
            rain_field = np.zeros((H, W), dtype=np.float32)
            rain_field[active] = (rain_2d[active] + bldg_extra) / 1000.0 / DT_RAIN  # m/s
            domain.update_array(k="rain", arr=rain_field)
            rain_idx += 1; next_rain += DT_RAIN

        # ITZI steps
        domain.update_ext_array()
        hydrology.solve_dt()
        hydrology.step()
        sf_sim.dt = timedelta(seconds=dt)
        sf_sim.step()
        sf_sim.solve_dt()

        t += dt
        dt = sf_sim.dt.total_seconds()
        step_count += 1

        # Record every 30 min
        if t >= next_record:
            h_arr = domain.get_array("h")
            vol = float(np.sum(h_arr[active]) * CELL_AREA)
            hmax = float(np.max(h_arr))
            flooded = int(np.sum(h_arr > 0.03))
            records['time_h'].append(t / 3600)
            records['vol_m3'].append(vol); records['hmax_m'].append(hmax)
            records['flooded_cells'].append(flooded)
            elapsed = time.time() - t0
            print(f"  t={t/3600:.1f}h dt={dt:.4f}s vol={vol:.0f}m3 hmax={hmax:.3f}m flooded={flooded} elapsed={elapsed:.0f}s")
            next_record += 1800

    elapsed = time.time() - t0
    print(f"  Completed: {step_count} steps in {elapsed:.0f}s")
    h_final = domain.get_array("h")
    return records, h_final


def main():
    print("=" * 60)
    print("  ITZI Dynamic Model — 4km x 5.6km Region")
    print("=" * 60)

    dem_full = np.load(DEM_PATH, allow_pickle=True).astype(np.float32)
    bldg_full = dem_full >= 49.9
    Hf, Wf = dem_full.shape
    print(f"Full DEM: {Hf}x{Wf} = {Hf*CELL/1000:.1f}km x {Wf*CELL/1000:.1f}km")

    # Use 4km x 5.6km sub-region (200x280 cells) — large enough for "global" view
    y0, y1 = 80, 280  # 200 rows = 4km
    x0, x1 = 120, 400  # 280 cols = 5.6km
    dem = dem_full[y0:y1, x0:x1].copy()
    bldg = bldg_full[y0:y1, x0:x1].copy()
    H, W = dem.shape
    print(f"Sub-region: {H}x{W} = {H*CELL/1000:.1f}km x {W*CELL/1000:.1f}km")
    print(f"Active cells: {(~bldg).sum():,} / {bldg.size:,}")

    events = ['event65', 'event1', 'event67']
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
        print(f"  MIKE+ ref: peak={np.max(peak_ref):.3f}m")

        rec, h_itzi = run_itzi_region(dem.copy(), bldg.copy(), rainfall, label=evt)

        all_results[evt] = {
            'mike_peak': float(np.max(peak_ref)),
            'mike_flooded': int(np.sum(peak_ref > 0.03)),
            'mike_vol': float(np.sum(h_ref[-1][~bldg]) * CELL_AREA),
            'itzi_peak': float(np.max(h_itzi)),
            'itzi_flooded': int(np.sum(h_itzi > 0.03)),
            'itzi_vol': float(rec['vol_m3'][-1]),
        }

        np.savez(os.path.join(OUT_DIR, f'{evt}_full.npz'),
                 h_ref=h_ref[-1], peak_ref=peak_ref,
                 h_itzi=h_itzi, rec=rec)

        pct = all_results[evt]['itzi_peak'] / all_results[evt]['mike_peak'] * 100
        print(f"  Result: MIKE+={all_results[evt]['mike_peak']:.3f}m, ITZI={all_results[evt]['itzi_peak']:.3f}m ({pct:.0f}%)")

    # Summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    for evt in events:
        r = all_results[evt]
        pct = r['itzi_peak'] / r['mike_peak'] * 100
        print(f"  {evt}: MIKE+={r['mike_peak']:.3f}m, ITZI={r['itzi_peak']:.3f}m ({pct:.0f}%), "
              f"Flooded: MIKE+={r['mike_flooded']} vs ITZI={r['itzi_flooded']}")

    np.savez(os.path.join(OUT_DIR, 'full_summary.npz'), all_results=all_results)
    print(f"\nResults saved to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
