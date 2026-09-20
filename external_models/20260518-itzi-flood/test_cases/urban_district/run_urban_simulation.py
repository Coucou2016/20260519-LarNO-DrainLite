#!/usr/bin/env python3
"""
ITZI Urban District Flood Simulation
=====================================
Models a small urban district with explicit building blocks, streets,
and drainage features. Buildings are represented by raised DEM cells
and elevated Manning's n values.

Urban layout (400m x 400m at 2m resolution = 200x200 cells):
  - 24 buildings arranged in 6 cols x 4 rows
  - 8m streets between buildings
  - 12m main avenue through center
  - Central park/open space
  - Natural terrain slope of 0.5% N to S
"""
import os, sys, time
import numpy as np
from datetime import timedelta

import itzi.rasterdomain as rasterdomain
import itzi.surfaceflow as surfaceflow
from itzi.hydrology import Hydrology
from itzi.infiltration import InfNull

# ============================================================
# Configuration
# ============================================================
CELL_SIZE = 2.0          # 2m resolution (captures building details)
GRID_SIZE = 200          # 200x200 = 400m domain
DOMAIN_SIZE = GRID_SIZE * CELL_SIZE
SIM_DURATION_S = 7200    # 2 hours
RECORD_STEP_S = 600      # 10 minutes
DT_MAX = 1.0
CFL = 0.7
THETA = 0.9
HMIN = 0.001
G = 9.80665
DT_INF = 60.0

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "core_sim_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def create_urban_terrain():
    """
    Create a realistic urban district terrain.

    Returns: dem, mannings, xx, yy (meshgrid), building_mask
    """
    x = np.linspace(0, DOMAIN_SIZE, GRID_SIZE)
    y = np.linspace(0, DOMAIN_SIZE, GRID_SIZE)
    xx, yy = np.meshgrid(x, y)

    # ================================================================
    # Base terrain: gentle 0.5% slope from north to south
    # ================================================================
    dem = (50.0 - yy * 0.005).astype(np.float32)

    # Add micro-topography: natural ground roughness
    np.random.seed(12345)
    micro_relief = np.random.randn(GRID_SIZE, GRID_SIZE) * 0.03

    # ================================================================
    # Building layout definition
    # ================================================================
    # 6 columns x 4 rows of buildings
    n_cols, n_rows = 6, 4
    street_width = 4     # cells (8m)
    main_street_width = 6  # cells (12m)
    building_w = 15       # cells (30m)
    building_d = 20       # cells (40m)
    sidewalk_w = 1        # cells (2m) on each side of buildings

    # Calculate layout positions
    # Total block width = building_w + 2*sidewalk_w + street_width
    block_pitch_x = building_w + 2 * sidewalk_w + street_width   # 15+2+4=21 cells
    block_pitch_y = building_d + 2 * sidewalk_w + street_width   # 20+2+4=26 cells

    # Start offset to center the blocks
    total_w = n_cols * block_pitch_x - street_width
    total_h = n_rows * block_pitch_y - street_width
    offset_x = (GRID_SIZE - total_w) // 2
    offset_y = (GRID_SIZE - total_h) // 2 + main_street_width // 2

    building_mask = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
    building_info = []  # (col, row, cx, cy, height)

    np.random.seed(42)
    for col in range(n_cols):
        for row in range(n_rows):
            bx = offset_x + col * block_pitch_x + sidewalk_w
            by = offset_y + row * block_pitch_y + sidewalk_w

            # Building height: 8-18m (2-5 stories)
            height = np.random.uniform(8, 18)

            # Add building to DEM
            x0 = bx
            x1 = bx + building_w
            y0 = by
            y1 = by + building_d

            # Ensure within bounds
            if x1 < GRID_SIZE and y1 < GRID_SIZE:
                dem[y0:y1, x0:x1] += height
                building_mask[y0:y1, x0:x1] = True
                building_info.append((col, row, bx + building_w/2, by + building_d/2, height))

    # ================================================================
    # Street network: lower elevation to channel flow along streets
    # ================================================================
    # Main avenue (north-south, through center)
    main_avenue_center = GRID_SIZE // 2
    avenue_x0 = main_avenue_center - main_street_width // 2
    avenue_x1 = main_avenue_center + main_street_width // 2
    dem[:, avenue_x0:avenue_x1] -= 0.2  # Street is 0.2m below surroundings

    # Side streets (east-west, between building rows)
    for row in range(n_rows + 1):
        street_y = offset_y + row * block_pitch_y - street_width // 2 - sidewalk_w
        if 0 <= street_y < GRID_SIZE:
            y0 = max(0, street_y - street_width // 2)
            y1 = min(GRID_SIZE, street_y + street_width // 2)
            dem[y0:y1, :] -= 0.15

    # Cross streets (north-south, between building columns)
    for col in range(n_cols + 1):
        street_x = offset_x + col * block_pitch_x - street_width // 2 - sidewalk_w
        if 0 <= street_x < GRID_SIZE:
            x0 = max(0, street_x - street_width // 2)
            x1 = min(GRID_SIZE, street_x + street_width // 2)
            dem[:, x0:x1] -= 0.15

    # ================================================================
    # Central plaza/park: open depression for water accumulation
    # ================================================================
    park_cx, park_cy = GRID_SIZE // 2, GRID_SIZE // 2
    park_radius = 25  # cells (50m)
    park_dist = np.sqrt((xx - park_cx * CELL_SIZE)**2 + (yy - park_cy * CELL_SIZE)**2)
    park_mask = park_dist < park_radius * CELL_SIZE

    # Lower the park area (but not buildings)
    park_lowering = 0.5 * np.exp(-park_dist**2 / (2 * (park_radius * CELL_SIZE / 2)**2))
    dem[park_mask & ~building_mask] -= park_lowering[park_mask & ~building_mask]

    # ================================================================
    # Manning's n: different values for different urban surfaces
    # ================================================================
    mannings = np.full((GRID_SIZE, GRID_SIZE), 0.015, dtype=np.float32)  # default: streets

    # Buildings: very high resistance (effectively blocks flow)
    mannings[building_mask] = 0.5

    # Park/plaza: grass
    mannings[park_mask] = 0.04

    # Add sidewalk strips around buildings (medium roughness)
    for col in range(n_cols):
        for row in range(n_rows):
            bx = offset_x + col * block_pitch_x
            by = offset_y + row * block_pitch_y
            x0 = max(0, bx)
            x1 = min(GRID_SIZE, bx + building_w + 2 * sidewalk_w)
            y0 = max(0, by)
            y1 = min(GRID_SIZE, by + building_d + 2 * sidewalk_w)
            # Sidewalk = area around building not yet assigned
            sidewalk = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
            sidewalk[y0:y1, x0:x1] = True
            sidewalk[building_mask] = False
            mannings[sidewalk] = 0.025

    # ================================================================
    # Add natural micro-relief to non-building areas
    # ================================================================
    dem[~building_mask] += micro_relief[~building_mask]

    print(f"  Buildings: {len(building_info)} blocks")
    print(f"  Building heights: {min(b[4] for b in building_info):.0f}m - {max(b[4] for b in building_info):.0f}m")
    print(f"  DEM range: {dem.min():.2f}m - {dem.max():.2f}m")
    print(f"  Manning's n range: {mannings.min():.3f} - {mannings.max():.3f}")

    return dem, mannings, xx, yy, building_mask, park_mask


def create_rainfall_series():
    """2-hour design storm, 50mm total, peak at 45min."""
    dt_rain = 300  # 5 minutes
    n_steps = SIM_DURATION_S // dt_rain
    t = np.arange(0, SIM_DURATION_S, dt_rain)
    t_peak = 2700

    intensity = np.where(
        t < t_peak,
        50 * (t / t_peak) ** 0.5,
        50 * ((SIM_DURATION_S - t) / (SIM_DURATION_S - t_peak)) ** 1.5
    )
    total = np.sum(intensity) * dt_rain / 3600
    intensity = intensity * 50 / total
    rain_ms = (intensity / (1000 * 3600)).astype(np.float32)
    return rain_ms, dt_rain, n_steps


def main():
    print("=" * 60)
    print("  ITZI Urban District Flood Simulation")
    print("=" * 60)
    print(f"  Domain: {DOMAIN_SIZE}m x {DOMAIN_SIZE}m")
    print(f"  Resolution: {CELL_SIZE}m ({GRID_SIZE}x{GRID_SIZE} cells)")
    print(f"  Duration: {SIM_DURATION_S / 3600:.0f}h")
    print()

    # ----------------------------------------------------------------
    # Step 1: Create urban terrain
    # ----------------------------------------------------------------
    print("Step 1: Creating urban terrain with buildings...")
    dem, mannings, xx, yy, building_mask, park_mask = create_urban_terrain()

    # Mask: all cells active
    mask = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)

    # ----------------------------------------------------------------
    # Step 2: Create rainfall
    # ----------------------------------------------------------------
    print("Step 2: Creating rainfall forcing...")
    rain_intensity, dt_rain, n_rain_steps = create_rainfall_series()
    print(f"  {n_rain_steps} rainfall steps, {np.sum(rain_intensity) * dt_rain * 1000:.1f}mm total")

    # ----------------------------------------------------------------
    # Step 3: Initialize ITZI
    # ----------------------------------------------------------------
    print("Step 3: Initializing ITZI core components...")
    domain = rasterdomain.RasterDomain(
        dtype=np.float32, arr_mask=mask, cell_shape=(CELL_SIZE, CELL_SIZE),
    )

    sim_param = {
        'hmin': HMIN, 'cfl': CFL, 'theta': THETA, 'g': G,
        'vrouting': 0.1, 'dtmax': DT_MAX, 'slmax': 0.1,
    }
    sf_sim = surfaceflow.SurfaceFlowSimulation(domain, sim_param)
    inf_model = InfNull(domain, DT_INF)
    hydrology = Hydrology(domain, DT_INF, inf_model)

    # ----------------------------------------------------------------
    # Step 4: Load initial conditions
    # ----------------------------------------------------------------
    print("Step 4: Loading initial conditions...")
    domain.update_array("dem", dem)
    domain.update_array("friction", mannings)
    domain.update_array("h", np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32))
    sf_sim.update_flow_dir()

    # ----------------------------------------------------------------
    # Step 5: Run simulation
    # ----------------------------------------------------------------
    print("Step 5: Running urban flood simulation...")
    print(f"  {'Time':<8s} {'dt(s)':<8s} {'Vol(m3)':<12s} {'hmax(m)':<10s} {'Flooded':<10s}")

    start_time = time.time()
    sim_time_s = 0.0
    dt = 0.001
    next_record = RECORD_STEP_S
    record_count = 0

    results = {'time': [], 'volume': [], 'hmax': [], 'flooded_cells': [], 'h_record': {}}
    rain_idx = 0
    next_rain_update = dt_rain

    while sim_time_s < SIM_DURATION_S:
        if sim_time_s >= next_rain_update and rain_idx < n_rain_steps:
            rain_arr = np.full((GRID_SIZE, GRID_SIZE), rain_intensity[rain_idx], dtype=np.float32)
            domain.update_array("rain", rain_arr)
            rain_idx += 1
            next_rain_update += dt_rain

        domain.update_ext_array()
        hydrology.solve_dt()
        hydrology.step()

        sf_sim.dt = timedelta(seconds=dt)
        sf_sim.step()
        sf_sim.solve_dt()

        sim_time_s += dt
        dt = sf_sim.dt.total_seconds()

        if sim_time_s >= next_record:
            record_count += 1
            h = domain.get_array("h")
            water_vol = np.sum(h[~mask]) * CELL_SIZE * CELL_SIZE
            hmax_val = np.max(h)
            flooded = np.sum(h > 0.005)

            results['time'].append(sim_time_s / 3600)
            results['volume'].append(water_vol)
            results['hmax'].append(hmax_val)
            results['flooded_cells'].append(flooded)
            results['h_record'][record_count] = h.copy()

            elapsed = time.time() - start_time
            print(f"  {sim_time_s/3600:>5.2f}h   {dt:>5.3f}s   {water_vol:>8.1f}    {hmax_val:>7.3f}    {flooded:>6d}")

            next_record += RECORD_STEP_S

    elapsed = timedelta(seconds=int(time.time() - start_time))
    print(f"\n  Simulation completed in {elapsed}")
    print(f"  {record_count} records written")

    # ----------------------------------------------------------------
    # Step 6: Save results
    # ----------------------------------------------------------------
    print("\nStep 6: Saving results...")
    h_final = domain.get_array("h")
    v_final = domain.get_array("v")
    hmax_arr = domain.get_array("hmax")

    np.savez(os.path.join(OUTPUT_DIR, "urban_simulation_results.npz"),
             time=np.array(results['time']),
             volume=np.array(results['volume']),
             hmax=np.array(results['hmax']),
             flooded_cells=np.array(results['flooded_cells']),
             dem=dem, mannings=mannings,
             building_mask=building_mask, park_mask=park_mask,
             h_final=h_final, v_final=v_final, hmax_arr=hmax_arr,
             cell_size=CELL_SIZE, domain_size=DOMAIN_SIZE)

    np.savetxt(os.path.join(OUTPUT_DIR, "dem_urban.asc"), dem, fmt='%.3f')
    np.savetxt(os.path.join(OUTPUT_DIR, "mannings_urban.asc"), mannings, fmt='%.4f')
    np.savetxt(os.path.join(OUTPUT_DIR, "depth_final_urban.asc"), h_final, fmt='%.4f')
    np.savetxt(os.path.join(OUTPUT_DIR, "velocity_final_urban.asc"), v_final, fmt='%.4f')

    # Save depth snapshots
    for t_h, rec_id in zip(results['time'], sorted(results['h_record'].keys())):
        np.savetxt(os.path.join(OUTPUT_DIR, f"depth_urban_t{t_h:.1f}h.asc"),
                   results['h_record'][rec_id], fmt='%.4f')

    # CSV summary
    with open(os.path.join(OUTPUT_DIR, "urban_summary.csv"), 'w') as f:
        f.write("time_h,water_volume_m3,max_depth_m,flooded_cells\n")
        for t, v, h, c in zip(results['time'], results['volume'],
                               results['hmax'], results['flooded_cells']):
            f.write(f"{t:.4f},{v:.2f},{h:.6f},{c}\n")

    print(f"  Results saved to: {OUTPUT_DIR}")
    print("=" * 60)
    print("  Urban District Simulation Complete!")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
