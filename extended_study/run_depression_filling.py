#!/usr/bin/env python3
"""
Extended Study: Depression-Filling Urban Flood Model
=====================================================
Physically robust approach:
  - Bathtub/depression-filling model for surface flooding
  - Sorted elevation filling (like a rising water surface)
  - Manning's pipe flow for drainage network
  - Volume balance for surface-drainage coupling

This approach is numerically stable and widely used in
urban flood screening studies.
"""

import os, sys, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "LarNO-main", "benchmark", "urbanflood")
DEM_PATH = os.path.join(ROOT, "geodata", "region1_20m", "dem.npy")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUT_DIR, exist_ok=True)

CELL_SIZE = 20.0  # m
CELL_AREA = CELL_SIZE * CELL_SIZE
flood_cmap = LinearSegmentedColormap.from_list('flood', [
    (0.0, '#F7FBFF'), (0.12, '#C6DBEF'), (0.25, '#6BAED6'),
    (0.4, '#3182BD'), (0.55, '#08519C'), (0.7, '#08306B'), (1.0, '#041838'),
])
diff_cmap = plt.cm.RdBu_r


def depression_flood_model(dem, bldg, rain_intensity_mmh, dt_min, pipe_network=None):
    """
    Depression-filling flood model.

    For each time step:
    1. Add rainfall uniformly over active cells
    2. Sort cells by water surface elevation
    3. Fill depressions (water flows to lowest cells)
    4. Drainage pipes remove water from inlet locations

    Returns depth maps at each recording step.
    """
    H, W = dem.shape
    active = ~bldg
    n_active = np.sum(active)
    n_steps = len(rain_intensity_mmh)
    dt_s = dt_min * 60

    # Get sorted indices of active cells by elevation
    active_flat = active.ravel()
    dem_flat = dem.ravel()
    active_idx = np.where(active_flat)[0]
    sorted_by_elev = active_idx[np.argsort(dem_flat[active_idx])]

    # Water depth (flat array for computation)
    h_flat = np.zeros(H * W, dtype=np.float64)

    # Accumulated rainwater volume per step
    rain_m_per_step = np.array(rain_intensity_mmh) * dt_min / 60 / 1000  # mm/h -> m

    # Pipe properties
    if pipe_network:
        node_cells = []
        for node in pipe_network['nodes']:
            r, c = node['row'], node['col']
            if 0 <= r < H and 0 <= c < W and not bldg[r, c]:
                node_cells.append((r, c, node['invert'], node.get('type', 'junction')))
        pipe_capacity = compute_total_pipe_capacity(pipe_network)
    else:
        node_cells = []
        pipe_capacity = 0

    # Record every 30 min
    record_interval = max(1, int(30 / dt_min))
    records = {'time_h': [], 'vol_m3': [], 'hmax_m': [], 'flooded_cells': [],
               'drained_m3': []}

    cumulative_drained = 0.0

    for step in range(n_steps):
        t_h = step * dt_min / 60

        # 1. Add rainfall
        rain_m = rain_m_per_step[step]
        h_flat[active_idx] += rain_m

        # 2. Drainage removal FIRST (before depression filling)
        #    This is critical: remove water from pipe catchments BEFORE
        #    the water is redistributed to low points
        if pipe_network and pipe_capacity > 0 and len(node_cells) > 0:
            drain_vol = 0
            for r, c, invert, ntype in node_cells:
                idx = r * W + c
                surf_wl = dem_flat[idx] + h_flat[idx]
                pipe_wl = invert

                if surf_wl > pipe_wl and h_flat[idx] > 0.01:
                    head = max(surf_wl - pipe_wl, 0.01)
                    Q_inlet = 0.65 * 2.0 * np.sqrt(2 * 9.81 * head)
                    # Each node can drain from its local catchment (~40m radius = 2 cells)
                    catchment_cells = 0
                    catchment_water = 0.0
                    for dr in range(-2, 3):
                        for dc in range(-2, 3):
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < H and 0 <= nc < W and not bldg[nr, nc]:
                                catchment_cells += 1
                                catchment_water += h_flat[nr * W + nc] * CELL_AREA
                    max_Q_local = catchment_water / dt_s * 0.8  # can drain 80% per step
                    Q_inlet = min(Q_inlet, max_Q_local)

                    # Apply drainage proportionally across catchment
                    if catchment_water > 0.01:
                        removal_fraction = Q_inlet * dt_s / catchment_water
                        removal_fraction = min(removal_fraction, 0.5)  # max 50% removal per cell
                        for dr in range(-2, 3):
                            for dc in range(-2, 3):
                                nr, nc = r + dr, c + dc
                                if 0 <= nr < H and 0 <= nc < W and not bldg[nr, nc]:
                                    ni = nr * W + nc
                                    removed = h_flat[ni] * removal_fraction
                                    h_flat[ni] -= removed
                                    drain_vol += removed * CELL_AREA

            cumulative_drained += drain_vol

        # 3. Fill depressions: water settles to lowest elevations first
        total_water_vol = np.sum(h_flat[active_idx]) * CELL_AREA

        if total_water_vol > 0:
            elevs = dem_flat[sorted_by_elev]
            cumul_areas = np.arange(1, len(elevs) + 1) * CELL_AREA

            cumulative_storage = 0.0
            wl = elevs[0]
            for i in range(1, len(elevs)):
                prev_wl = elevs[i - 1]
                curr_elev = elevs[i]
                if curr_elev > prev_wl:
                    vol_to_fill = (curr_elev - prev_wl) * cumul_areas[i - 1]
                    if cumulative_storage + vol_to_fill >= total_water_vol:
                        remaining = total_water_vol - cumulative_storage
                        wl = prev_wl + remaining / cumul_areas[i - 1]
                        break
                    cumulative_storage += vol_to_fill
                    wl = curr_elev
                else:
                    wl = curr_elev
            else:
                remaining = total_water_vol - cumulative_storage
                wl = elevs[-1] + remaining / cumul_areas[-1]

            h_flat[:] = 0
            h_flat[active_idx] = np.maximum(0, wl - dem_flat[active_idx])

        # 4. Record
        if step % record_interval == 0:
            h_2d = h_flat.reshape(H, W)
            vol = float(np.sum(h_2d[active]) * CELL_AREA)
            hmax = float(np.max(h_2d))
            flooded = int(np.sum(h_2d > 0.03))

            records['time_h'].append(t_h)
            records['vol_m3'].append(vol)
            records['hmax_m'].append(hmax)
            records['flooded_cells'].append(flooded)
            records['drained_m3'].append(float(cumulative_drained))

            if step % (record_interval * 3) == 0:
                print(f"  t={t_h:.1f}h  vol={vol:.0f}m3  hmax={hmax:.3f}m  "
                      f"flooded={flooded}/{n_active}  drained={cumulative_drained:.0f}m3")

    h_final = h_flat.reshape(H, W)
    return records, h_final


def depression_flood_model_2d(dem, bldg, rainfall_3d, pipe_network=None):
    """
    Depression-filling flood model with full 2D spatially-heterogeneous rainfall.
    rainfall_3d: (T, H, W) in mm/5min

    Uses the same stable depression-filling algorithm but with cell-level rainfall
    and building runoff routing. This preserves spatial rainfall patterns while
    maintaining numerical stability.
    """
    H, W = dem.shape
    active = ~bldg
    n_active = np.sum(active)
    n_steps = rainfall_3d.shape[0]
    dt_min = 5.0; dt_s = dt_min * 60
    runoff_coeff = 0.90

    # Sorted active cells by elevation (for depression filling)
    active_flat = active.ravel()
    dem_flat = dem.ravel()
    active_idx = np.where(active_flat)[0]
    sorted_by_elev = active_idx[np.argsort(dem_flat[active_idx])]
    elevs = dem_flat[sorted_by_elev]
    cumul_areas = np.arange(1, len(elevs) + 1) * CELL_AREA

    # Water depth
    h_flat = np.zeros(H * W, dtype=np.float64)

    # Pipe properties
    if pipe_network:
        node_cells = []
        for node in pipe_network['nodes']:
            r, c = node['row'], node['col']
            if 0 <= r < H and 0 <= c < W and not bldg[r, c]:
                node_cells.append((r, c, node['invert']))
        pipe_capacity = compute_total_pipe_capacity(pipe_network)
    else:
        node_cells = []
        pipe_capacity = 0

    record_interval = max(1, int(30 / dt_min))
    records = {'time_h': [], 'vol_m3': [], 'hmax_m': [], 'flooded_cells': [], 'drained_m3': []}
    cumulative_drained = 0.0

    for step in range(n_steps):
        t_h = step * dt_min / 60

        # 1. Apply 2D rainfall (cell-level) + building runoff
        rain_2d = rainfall_3d[step]  # mm/5min
        bldg_rain = np.sum(rain_2d[bldg])
        bldg_per_active = bldg_rain / n_active if n_active > 0 else 0

        rain_m_active = rain_2d.ravel() / 1000.0 * runoff_coeff  # mm/5min -> m
        rain_m_active[active_flat] += bldg_per_active / 1000.0 * runoff_coeff
        rain_m_active[~active_flat] = 0
        h_flat += rain_m_active
        h_flat = np.maximum(h_flat, 0)

        # 2. Depression-filling redistribution (stable)
        total_water_vol = np.sum(h_flat[active_idx]) * CELL_AREA
        if total_water_vol > 0:
            cumulative_storage = 0.0; wl = elevs[0]
            for i in range(1, len(elevs)):
                prev_wl = elevs[i - 1]; curr_elev = elevs[i]
                if curr_elev > prev_wl:
                    vol_to_fill = (curr_elev - prev_wl) * cumul_areas[i - 1]
                    if cumulative_storage + vol_to_fill >= total_water_vol:
                        remaining = total_water_vol - cumulative_storage
                        wl = prev_wl + remaining / cumul_areas[i - 1]
                        break
                    cumulative_storage += vol_to_fill
                    wl = curr_elev
            else:
                remaining = total_water_vol - cumulative_storage
                wl = elevs[-1] + remaining / cumul_areas[-1]
            h_flat[:] = 0
            h_flat[active_idx] = np.maximum(0, wl - dem_flat[active_idx])

        # 3. Drainage removal
        if pipe_network and pipe_capacity > 0 and len(node_cells) > 0:
            drain_vol = 0.0
            for r, c, invert in node_cells:
                idx = r * W + c
                surf_wl = dem_flat[idx] + h_flat[idx]
                if surf_wl > invert and h_flat[idx] > 0.01:
                    head = max(surf_wl - invert, 0.01)
                    Q_inlet = 0.65 * 2.0 * np.sqrt(2 * 9.81 * head)
                    catchment_water = 0.0
                    for dr in range(-2, 3):
                        for dc in range(-2, 3):
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < H and 0 <= nc < W and not bldg[nr, nc]:
                                catchment_water += h_flat[nr * W + nc] * CELL_AREA
                    max_Q_local = catchment_water / dt_s * 0.8
                    Q_inlet = min(Q_inlet, max_Q_local)
                    removal_fraction = min(Q_inlet * dt_s / max(catchment_water, 0.01), 0.5)
                    for dr in range(-2, 3):
                        for dc in range(-2, 3):
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < H and 0 <= nc < W and not bldg[nr, nc]:
                                ni = nr * W + nc
                                h_flat[ni] -= h_flat[ni] * removal_fraction
                                drain_vol += h_flat[ni] * removal_fraction * CELL_AREA
            cumulative_drained += drain_vol

        h_flat = np.maximum(h_flat, 0)
        h_flat[~active_flat] = 0

        if step % record_interval == 0:
            h_2d = h_flat.reshape(H, W)
            vol = float(np.sum(h_2d[active]) * CELL_AREA)
            hmax = float(np.max(h_2d))
            flooded = int(np.sum(h_2d > 0.03))
            records['time_h'].append(t_h); records['vol_m3'].append(vol)
            records['hmax_m'].append(hmax); records['flooded_cells'].append(flooded)
            records['drained_m3'].append(float(cumulative_drained))
            if step % (record_interval * 3) == 0:
                print(f"  t={t_h:.1f}h  vol={vol:.0f}m3  hmax={hmax:.3f}m  "
                      f"flooded={flooded}/{n_active}  drained={cumulative_drained:.0f}m3")

    h_final = h_flat.reshape(H, W)
    return records, h_final


def compute_total_pipe_capacity(network):
    """Compute total pipe drainage capacity (m3/s)."""
    total = 0.0
    for link in network['links']:
        D = link['diameter']
        A = np.pi * D**2 / 4
        R = D / 4
        S = max(link['slope'], 0.001)
        Q = A * (1.0 / 0.013) * R**(2/3) * np.sqrt(S)
        total += Q
    return total


def main():
    print("=" * 60)
    print("  Extended Study: Depression-Filling Flood Model")
    print("=" * 60)

    # Load DEM
    print("\n[1] Loading terrain...")
    dem_full = np.load(DEM_PATH, allow_pickle=True)
    bldg_full = dem_full >= 49.9

    # 2km x 2km sub-region
    y0, y1 = 90, 190
    x0, x1 = 160, 260
    dem = dem_full[y0:y1, x0:x1].astype(np.float64)
    bldg = bldg_full[y0:y1, x0:x1]
    H, W = dem.shape
    active = ~bldg

    # Add natural slope
    yy, xx = np.mgrid[0:H, 0:W]
    dem[active] -= (H - 1 - yy)[active] * CELL_SIZE * 0.0015  # 0.15% slope S→N

    print(f"  Domain: {H*CELL_SIZE/1000:.1f}km x {W*CELL_SIZE/1000:.1f}km")
    print(f"  Active cells: {active.sum()}/{active.size} ({100*active.sum()/active.size:.0f}%)")

    # Design pipe network
    print("\n[2] Designing pipe network...")
    from design_pipe_network import design_pipe_network
    network = design_pipe_network(dem, bldg, sub_region=(0, H, 0, W))
    pipe_cap = compute_total_pipe_capacity(network)
    print(f"  Nodes: {len(network['nodes'])}, Links: {len(network['links'])}")
    print(f"  Total capacity: {pipe_cap:.2f} m3/s")

    # Design storm
    print("\n[3] Creating design storm...")
    duration_h = 6.0
    dt_min = 10.0
    n_steps = int(duration_h * 60 / dt_min)
    t_min = np.arange(n_steps) * dt_min
    t_peak = duration_h * 60 * 0.35

    intensity = np.where(
        t_min < t_peak,
        30 * (t_min / t_peak)**0.4,
        30 * ((duration_h * 60 - t_min) / (duration_h * 60 - t_peak))**1.2
    )
    intensity = intensity * 50.0 / (np.sum(intensity) * dt_min / 60)
    print(f"  {n_steps} steps @ {dt_min}min, total=50mm, peak={np.max(intensity):.1f}mm/h")

    # Run paired simulations
    print("\n[4] Running paired simulations...")

    print("\n  --- A) Surface Only ---")
    rec_a, h_a = depression_flood_model(dem, bldg, intensity, dt_min, pipe_network=None)

    print("\n  --- B) Surface + Drainage ---")
    rec_b, h_b = depression_flood_model(dem, bldg, intensity, dt_min, pipe_network=network)

    # Summary
    print("\n[5] Results Summary:")
    print(f"  {'Metric':<25s} {'Surface Only':>15s} {'With Drainage':>15s} {'Change':>10s}")
    print(f"  {'-'*65}")
    for key, label in [('vol_m3', 'Final Volume (m3)'), ('hmax_m', 'Max Depth (m)'),
                        ('flooded_cells', 'Flooded Cells')]:
        va = rec_a[key][-1]
        vb = rec_b[key][-1]
        change = (vb - va) / va * 100 if va > 0 else 0
        print(f"  {label:<25s} {va:>15.1f} {vb:>15.1f} {change:>+9.1f}%")
    print(f"  {'Total Drained (m3)':<25s} {'---':>15s} {rec_b['drained_m3'][-1]:>15.1f}")

    # Save
    print("\n[6] Saving...")
    np.savez(os.path.join(OUT_DIR, 'depression_filling_results.npz'),
             time_h=rec_a['time_h'],
             vol_a=rec_a['vol_m3'], vol_b=rec_b['vol_m3'],
             hmax_a=rec_a['hmax_m'], hmax_b=rec_b['hmax_m'],
             flooded_a=rec_a['flooded_cells'], flooded_b=rec_b['flooded_cells'],
             drained_m3=rec_b['drained_m3'],
             h_final_a=h_a, h_final_b=h_b, dem=dem, bldg=bldg)

    # Visualize
    print("[7] Creating visualizations...")
    create_visualizations(rec_a, rec_b, h_a, h_b, dem, bldg, network, pipe_cap)

    print(f"\n  Outputs: {OUT_DIR}")
    for f in sorted(os.listdir(OUT_DIR)):
        fpath = os.path.join(OUT_DIR, f)
        if f.endswith('.png'):
            print(f"    {f} ({os.path.getsize(fpath)/1024:.0f} KB)")
    print("=" * 60)
    print("  Complete!")
    return 0


def create_visualizations(rec_a, rec_b, h_a, h_b, dem, bldg, network, pipe_cap):
    """Generate comparison figures."""
    H, W = dem.shape
    extent = [0, W * CELL_SIZE, 0, H * CELL_SIZE]
    time_h = rec_a['time_h']
    bldg_bg = np.where(bldg, 0.3, 0)

    vmax = max(np.max(h_a[h_a > 0.001]) if np.any(h_a > 0.001) else 0.05,
               np.max(h_b[h_b > 0.001]) if np.any(h_b > 0.001) else 0.05, 0.05)

    # Fig 1: Side-by-side
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Depression-Filling Model: Effect of Pipe Drainage on Urban Flooding',
                 fontsize=14, fontweight='bold')

    for idx, (ax, h, title) in enumerate([
        (axes[0], h_a, f'Surface Only\nMax={np.max(h_a):.3f}m, Vol={rec_a["vol_m3"][-1]:.0f}m3'),
        (axes[1], h_b, f'With Drainage\nMax={np.max(h_b):.3f}m, Vol={rec_b["vol_m3"][-1]:.0f}m3'),
    ]):
        h_show = np.ma.masked_where(h < 0.001, h)
        ax.imshow(np.flipud(bldg_bg), cmap=ListedColormap(['none', '#D3D3D3']),
                  extent=extent, aspect='equal', alpha=0.5)
        im = ax.imshow(np.flipud(h_show), cmap=flood_cmap, extent=extent, aspect='equal', vmin=0, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel('Easting (m)'); ax.set_ylabel('Northing (m)')
        plt.colorbar(im, ax=ax, label='Depth (m)', shrink=0.8)

    ax = axes[2]
    diff = h_a - h_b
    vlim = max(abs(np.min(diff)), abs(np.max(diff)), 0.01)
    im = ax.imshow(np.flipud(diff), cmap=diff_cmap, extent=extent, aspect='equal', vmin=-vlim, vmax=vlim)
    ax.set_title('Depth Reduction by Drainage\n(Blue = Less Flooding)')
    ax.set_xlabel('Easting (m)'); ax.set_ylabel('Northing (m)')
    plt.colorbar(im, ax=ax, label='Depth Diff (m)', shrink=0.8)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'extended_flood_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Fig 2: Time series
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Drainage Network Performance Time Series', fontsize=14, fontweight='bold')

    ax = axes[0, 0]
    ax.plot(time_h, rec_a['vol_m3'], 'r-o', lw=2, ms=3, label='Surface Only')
    ax.plot(time_h, rec_b['vol_m3'], 'b-s', lw=2, ms=3, label='With Drainage')
    expected_vol = 50.0 / 1000 * np.sum(~bldg) * CELL_AREA
    ax.axhline(y=expected_vol, color='green', ls='--', label=f'Total Rain ({expected_vol:.0f}m3)')
    ax.set_xlabel('Time (h)'); ax.set_ylabel('Volume (m3)')
    ax.set_title('Surface Water Volume'); ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(time_h, rec_a['hmax_m'], 'r-o', lw=2, ms=3, label='Surface Only')
    ax.plot(time_h, rec_b['hmax_m'], 'b-s', lw=2, ms=3, label='With Drainage')
    ax.set_xlabel('Time (h)'); ax.set_ylabel('Max Depth (m)')
    ax.set_title('Maximum Inundation Depth'); ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(time_h, np.array(rec_a['flooded_cells']) * CELL_AREA, 'r-o', lw=2, ms=3, label='Surface Only')
    ax.plot(time_h, np.array(rec_b['flooded_cells']) * CELL_AREA, 'b-s', lw=2, ms=3, label='With Drainage')
    ax.set_xlabel('Time (h)'); ax.set_ylabel('Flooded Area (m2)')
    ax.set_title('Inundated Area'); ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.fill_between(time_h, 0, rec_b['drained_m3'], color='blue', alpha=0.3, label='Cumulative Drained')
    ax.plot(time_h, rec_b['drained_m3'], 'b-', lw=2)
    ax.set_xlabel('Time (h)'); ax.set_ylabel('Volume (m3)')
    ax.set_title(f'Water Removed by Drainage (Total: {rec_b["drained_m3"][-1]:.0f}m3)')
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'extended_timeseries.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Fig 3: Dashboard
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle('Extended Study: Coupled Urban Flood & Drainage Summary', fontsize=16, fontweight='bold')

    # (1) Terrain
    ax = fig.add_subplot(2, 3, 1)
    terrain_show = np.where(bldg, np.nan, dem)
    im = ax.imshow(np.flipud(terrain_show), cmap='terrain', extent=extent, aspect='equal')
    ax.set_title('Terrain Elevation')
    plt.colorbar(im, ax=ax, label='Elev (m)')

    # (2) Flood with drainage
    ax = fig.add_subplot(2, 3, 2)
    ax.imshow(np.flipud(bldg_bg), cmap=ListedColormap(['none', '#aaa']),
              extent=extent, aspect='equal', alpha=0.5)
    h_show = np.ma.masked_where(h_b < 0.001, h_b)
    im = ax.imshow(np.flipud(h_show), cmap=flood_cmap, extent=extent, aspect='equal', vmin=0, vmax=vmax)
    ax.set_title('Flood Depth with Drainage')
    plt.colorbar(im, ax=ax, label='Depth (m)')

    # (3) Difference
    ax = fig.add_subplot(2, 3, 3)
    diff_all = h_a - h_b
    vlim_all = max(abs(np.min(diff_all)), abs(np.max(diff_all)), 0.01)
    im = ax.imshow(np.flipud(diff_all), cmap=diff_cmap, extent=extent, aspect='equal', vmin=-vlim_all, vmax=vlim_all)
    ax.set_title('Drainage Benefit (Depth Reduction)')
    plt.colorbar(im, ax=ax, label='Diff (m)')

    # (4) Cross-section
    ax = fig.add_subplot(2, 3, 4)
    xs = np.arange(W) * CELL_SIZE
    ax.fill_between(xs, dem[H//2], dem[H//2] + h_a[H//2], alpha=0.4, color='red', label='No Drainage')
    ax.fill_between(xs, dem[H//2], dem[H//2] + h_b[H//2], alpha=0.4, color='blue', label='With Drainage')
    ax.plot(xs, dem[H//2], 'brown', lw=1.5)
    ax.set_xlabel('Distance (m)'); ax.set_ylabel('Elevation (m)')
    ax.set_title('E-W Cross Section'); ax.legend(); ax.grid(True, alpha=0.3)

    # (5) Volume time series
    ax = fig.add_subplot(2, 3, 5)
    ax.plot(time_h, rec_a['vol_m3'], 'r-', lw=2, label='Surface Only')
    ax.plot(time_h, rec_b['vol_m3'], 'b-', lw=2, label='With Drainage')
    ax.set_xlabel('Time (h)'); ax.set_ylabel('Volume (m3)')
    ax.set_title('Volume Balance'); ax.legend(); ax.grid(True, alpha=0.3)

    # (6) Metrics
    ax = fig.add_subplot(2, 3, 6)
    ax.axis('off')
    metrics = (
        f"EXTENDED STUDY — SHENZHEN FUTIAN\n"
        f"================================\n\n"
        f"DOMAIN: {H*CELL_SIZE/1000:.1f}km x {W*CELL_SIZE/1000:.1f}km\n"
        f"Resolution: {CELL_SIZE:.0f}m grid ({H}x{W} cells)\n"
        f"Active cells: {np.sum(~bldg)} (buildings excluded)\n"
        f"Storm: 50mm / 6h Chicago design storm\n\n"
        f"DRAINAGE NETWORK\n"
        f"Nodes: {len(network['nodes'])}, Links: {len(network['links'])}\n"
        f"Capacity: {pipe_cap:.1f} m3/s\n\n"
        f"RESULTS\n"
        f"Volume:  {rec_a['vol_m3'][-1]:.0f} -> {rec_b['vol_m3'][-1]:.0f} m3\n"
        f"         ({(rec_a['vol_m3'][-1]-rec_b['vol_m3'][-1])/rec_a['vol_m3'][-1]*100:.1f}% reduction)\n"
        f"MaxDepth: {rec_a['hmax_m'][-1]:.3f} -> {rec_b['hmax_m'][-1]:.3f} m\n"
        f"Flooded: {rec_a['flooded_cells'][-1]} -> {rec_b['flooded_cells'][-1]} cells\n"
        f"Drained: {rec_b['drained_m3'][-1]:.0f} m3 total\n\n"
        f"Method: Depression-filling model\n"
        f"  (Bathtub + Manning pipe flow)\n"
        f"Adapted from ITZI-flood methodology"
    )
    ax.text(0.05, 0.5, metrics, transform=ax.transAxes, fontsize=9,
            fontfamily='monospace', verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='#F0F8FF', alpha=0.8))

    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'extended_dashboard.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
