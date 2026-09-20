#!/usr/bin/env python3
"""
ITZI Core Simulation Demo - runs the ITZI computational engine without GRASS GIS.

This demonstrates the full flood simulation workflow:
  1. Create synthetic terrain and forcing data
  2. Run ITZI's core 2D surface flow simulation
  3. Generate visualization outputs

Uses ITZI's RasterDomain, SurfaceFlowSimulation, and flow C extension directly.
"""
import os
import sys
import time
import numpy as np
from datetime import datetime, timedelta

# Add ITZI to path

import itzi.rasterdomain as rasterdomain
import itzi.surfaceflow as surfaceflow
from itzi.hydrology import Hydrology
from itzi.infiltration import InfNull

# ============================================================
# Configuration
# ============================================================
GRID_SIZE = 100
CELL_SIZE = 5.0
DOMAIN_SIZE = GRID_SIZE * CELL_SIZE  # 500m

SIM_DURATION_S = 7200  # 2 hours
RECORD_STEP_S = 600    # 10 minutes
DT_MAX = 1.0           # maximum timestep
CFL = 0.7
THETA = 0.9
HMIN = 0.001
G = 9.80665

DT_INF = 60.0  # infiltration timestep

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "core_sim_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def gaussian_filter(arr, sigma):
    """Pure numpy gaussian filter."""
    size = int(4 * sigma + 1) | 1
    x = np.arange(-(size // 2), size // 2 + 1)
    kernel = np.exp(-x**2 / (2 * sigma**2))
    kernel /= kernel.sum()
    result = arr.copy()
    for axis in range(arr.ndim):
        result = np.apply_along_axis(
            lambda r: np.convolve(r, kernel, mode='same'), axis, result
        )
    return result


def create_terrain():
    """Create a synthetic DEM with depression and channel."""
    x = np.linspace(0, DOMAIN_SIZE, GRID_SIZE)
    y = np.linspace(0, DOMAIN_SIZE, GRID_SIZE)
    xx, yy = np.meshgrid(x, y)

    # Base slope 2%
    base = 50.0 - yy * 0.02

    # Central depression
    cx, cy = GRID_SIZE // 2, GRID_SIZE // 2
    dist = np.sqrt((xx - cx * CELL_SIZE)**2 + (yy - cy * CELL_SIZE)**2)
    depression = -1.5 * np.exp(-dist**2 / (2 * 30**2))

    # Terrain roughness
    np.random.seed(42)
    roughness = gaussian_filter(np.random.randn(GRID_SIZE, GRID_SIZE) * 0.15, 2)

    # Channel
    channel_dist = np.abs(xx - cx * CELL_SIZE)
    channel = -1.0 * np.exp(-channel_dist**2 / (2 * 8**2))
    channel = gaussian_filter(channel, 1.5)

    dem = (base + depression + roughness + channel).astype(np.float32)
    return dem, xx, yy


def create_rainfall_series():
    """Create 2-hour hyetograph (50mm total, peak at 45min)."""
    dt_rain = 300  # 5 minutes in seconds
    n_steps = SIM_DURATION_S // dt_rain
    t = np.arange(0, SIM_DURATION_S, dt_rain)
    t_peak = 2700  # 45 min

    # Chicago design storm
    intensity = np.where(
        t < t_peak,
        50 * (t / t_peak) ** 0.5,
        50 * ((SIM_DURATION_S - t) / (SIM_DURATION_S - t_peak)) ** 1.5
    )
    # Scale to 50mm total
    total = np.sum(intensity) * dt_rain / 3600
    intensity = intensity * 50 / total
    # Convert mm/h to m/s
    rain_ms = (intensity / (1000 * 3600)).astype(np.float32)
    return rain_ms, dt_rain, n_steps


def main():
    print("=" * 60)
    print("  ITZI Core Engine - Flood Simulation Demo")
    print("=" * 60)
    print(f"  Grid: {GRID_SIZE}x{GRID_SIZE} @ {CELL_SIZE}m")
    print(f"  Domain: {DOMAIN_SIZE}m x {DOMAIN_SIZE}m")
    print(f"  Duration: {SIM_DURATION_S}s")
    print(f"  Output: {OUTPUT_DIR}")
    print()

    # ----------------------------------------------------------
    # Step 1: Create terrain and initial conditions
    # ----------------------------------------------------------
    print("Step 1: Creating terrain...")
    dem, xx, yy = create_terrain()
    mannings = np.full((GRID_SIZE, GRID_SIZE), 0.03, dtype=np.float32)
    mannings[np.abs(xx - DOMAIN_SIZE // 2) > 100] = 0.06
    mannings[np.abs(xx - DOMAIN_SIZE // 2) < 25] = 0.02

    # Mask: all cells active (no mask)
    mask = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)

    print(f"  DEM: min={dem.min():.2f}m, max={dem.max():.2f}m")

    # ----------------------------------------------------------
    # Step 2: Create rainfall forcing
    # ----------------------------------------------------------
    print("Step 2: Creating rainfall forcing...")
    rain_intensity, dt_rain, n_rain_steps = create_rainfall_series()
    print(f"  {n_rain_steps} rainfall steps, total rainfall: {np.sum(rain_intensity) * dt_rain * 1000:.1f}mm")

    # ----------------------------------------------------------
    # Step 3: Initialize ITZI core components
    # ----------------------------------------------------------
    print("Step 3: Initializing ITZI core components...")

    # RasterDomain - stores all simulation arrays
    domain = rasterdomain.RasterDomain(
        dtype=np.float32,
        arr_mask=mask,
        cell_shape=(CELL_SIZE, CELL_SIZE),
    )
    print(f"  RasterDomain: {domain.shape[0]}x{domain.shape[1]} cells")

    # Surface flow simulation
    sim_param = {
        'hmin': HMIN,
        'cfl': CFL,
        'theta': THETA,
        'g': G,
        'vrouting': 0.1,
        'dtmax': DT_MAX,
        'slmax': 0.1,
    }
    sf_sim = surfaceflow.SurfaceFlowSimulation(domain, sim_param)
    print(f"  SurfaceFlowSimulation initialized")

    # Infiltration (null - no infiltration for simplicity)
    inf_model = InfNull(domain, DT_INF)

    # Hydrology
    hydrology = Hydrology(domain, DT_INF, inf_model)
    print(f"  Hydrology and Infiltration initialized")

    # ----------------------------------------------------------
    # Step 4: Load initial data
    # ----------------------------------------------------------
    print("Step 4: Loading initial conditions...")

    # Set DEM
    domain.update_array("dem", dem)

    # Set Manning's n
    domain.update_array("friction", mannings)

    # Set initial water depth (all zeros = dry start)
    h0 = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
    domain.update_array("h", h0)

    # Update flow directions based on DEM
    sf_sim.update_flow_dir()

    print(f"  DEM loaded, flow directions computed")

    # ----------------------------------------------------------
    # Step 5: Run simulation
    # ----------------------------------------------------------
    print("Step 5: Running simulation...")
    print(f"  {'Time':<10s} {'Step(s)':<10s} {'Vol(m3)':<15s} {'hmax(m)':<12s}")

    start_time = time.time()
    sim_time_s = 0.0
    dt = 0.001  # Initial timestep (forced to be small for first step)
    next_record = RECORD_STEP_S
    record_count = 0

    # Storage for results
    results = {
        'time': [],
        'volume': [],
        'hmax': [],
        'h_record': {},
    }

    rain_idx = 0
    next_rain_update = dt_rain

    while sim_time_s < SIM_DURATION_S:
        # Update rainfall if needed
        if sim_time_s >= next_rain_update and rain_idx < n_rain_steps:
            rain_arr = np.full((GRID_SIZE, GRID_SIZE), rain_intensity[rain_idx], dtype=np.float32)
            domain.update_array("rain", rain_arr)
            rain_idx += 1
            next_rain_update += dt_rain

        # Update external arrays (combines rain, inflow, drainage)
        domain.update_ext_array()

        # Hydrology step
        hydrology.solve_dt()
        hydrology.step()

        # Surface flow step
        sf_sim.dt = timedelta(seconds=dt)
        sf_sim.step()
        sf_sim.solve_dt()

        # Next timestep
        sim_time_s += dt
        dt = sf_sim.dt.total_seconds()

        # Recording
        if sim_time_s >= next_record:
            record_count += 1
            h = domain.get_array("h")
            v = domain.get_array("v")
            water_vol = np.sum(h[~mask]) * CELL_SIZE * CELL_SIZE
            hmax_val = np.max(h)

            results['time'].append(sim_time_s / 3600)  # hours
            results['volume'].append(water_vol)
            results['hmax'].append(hmax_val)
            results['h_record'][record_count] = h.copy()

            elapsed = time.time() - start_time
            print(f"  {sim_time_s/3600:>6.2f}h    {dt:>6.3f}s    {water_vol:>10.1f}    {hmax_val:>8.4f}")

            next_record += RECORD_STEP_S

    elapsed = timedelta(seconds=int(time.time() - start_time))
    print(f"\n  Simulation completed in {elapsed}")
    print(f"  {record_count} records written")

    # ----------------------------------------------------------
    # Step 6: Save results
    # ----------------------------------------------------------
    print("\nStep 6: Saving results...")

    # Save numpy results
    np.savez(os.path.join(OUTPUT_DIR, "simulation_results.npz"),
             time=np.array(results['time']),
             volume=np.array(results['volume']),
             hmax=np.array(results['hmax']),
             dem=dem,
             mannings=mannings,
             cell_size=CELL_SIZE,
             domain_size=DOMAIN_SIZE)

    # Save depth maps at key times
    for t_idx, (rec_id, h_arr) in enumerate(sorted(results['h_record'].items())):
        if t_idx < len(results['time']):
            t_h = results['time'][t_idx]
            np.savetxt(os.path.join(OUTPUT_DIR, f"depth_t{t_h:.1f}h.asc"),
                       h_arr, fmt='%.4f')

    # Save CSV summary
    csv_path = os.path.join(OUTPUT_DIR, "simulation_summary.csv")
    with open(csv_path, 'w') as f:
        f.write("time_h,water_volume_m3,max_depth_m\n")
        for t, v, h in zip(results['time'], results['volume'], results['hmax']):
            f.write(f"{t:.4f},{v:.2f},{h:.6f}\n")
    print(f"  CSV summary: {csv_path}")

    # Save final state
    h_final = domain.get_array("h")
    v_final = domain.get_array("v")
    np.savetxt(os.path.join(OUTPUT_DIR, "depth_final.asc"), h_final, fmt='%.4f')
    np.savetxt(os.path.join(OUTPUT_DIR, "velocity_final.asc"), v_final, fmt='%.4f')
    np.savetxt(os.path.join(OUTPUT_DIR, "dem.asc"), dem, fmt='%.3f')

    print(f"\n  Results saved to: {OUTPUT_DIR}")
    print("=" * 60)
    print("  Simulation Complete!")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
