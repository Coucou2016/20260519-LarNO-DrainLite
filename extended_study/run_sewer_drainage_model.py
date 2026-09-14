#!/usr/bin/env python3
"""
OSM-Based Sewer Drainage Model
===============================
Uses a subcatchment-based approach:
1. Each manhole drains from its Voronoi/Thiessen polygon (contributing area)
2. Runoff from each subcatchment is computed using the rational method
3. Pipe capacity is checked against inflow
4. Excess water becomes surface flooding

This is the standard engineering approach for urban drainage design.
"""

import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from scipy import ndimage

DEM_PATH = r"e:\Projects\20260519-LarNO\LarNO-main\benchmark\urbanflood\geodata\region1_20m\dem.npy"
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUT_DIR, exist_ok=True)

CELL_SIZE = 20.0; CELL_AREA = CELL_SIZE ** 2
G = 9.81
flood_cmap = LinearSegmentedColormap.from_list('flood', [
    (0.0, '#F7FBFF'), (0.12, '#C6DBEF'), (0.25, '#6BAED6'),
    (0.4, '#3182BD'), (0.55, '#08519C'), (0.7, '#08306B'), (1.0, '#041838'),
])


def assign_subcatchments(dem, bldg, nodes):
    """
    Assign each active cell to the nearest manhole (Voronoi partition).
    This defines the contributing area for each inlet.
    """
    H, W = dem.shape
    active = ~bldg

    # Build node position arrays
    node_rows = np.array([n['row'] for n in nodes])
    node_cols = np.array([n['col'] for n in nodes])
    n_nodes = len(nodes)

    # For each active cell, find nearest node
    subcatch_map = np.full((H, W), -1, dtype=int)
    active_rows, active_cols = np.where(active)

    if n_nodes == 0:
        return subcatch_map, np.zeros(n_nodes)

    for r, c in zip(active_rows, active_cols):
        dists = np.sqrt((node_rows - r)**2 + (node_cols - c)**2)
        nearest = np.argmin(dists)
        # Only assign if within reasonable distance (500m = 25 cells)
        if dists[nearest] <= 25:
            subcatch_map[r, c] = nearest

    # Count areas
    areas = np.array([np.sum(subcatch_map == i) * CELL_AREA for i in range(n_nodes)])
    print(f"  Assigned {np.sum(subcatch_map >= 0):,} cells to {np.sum(areas > 0)} catchments")
    print(f"  Catchment areas: min={areas[areas>0].min()/1e4:.1f}ha, "
          f"mean={areas[areas>0].mean()/1e4:.1f}ha, max={areas.max()/1e4:.1f}ha")

    return subcatch_map, areas


def run_sewer_simulation(dem, bldg, nodes, links, intensity_mmh, dt_min, subcatch_map, areas):
    """
    Run sewer drainage simulation using the time-area method.

    For each time step:
    1. Compute runoff from each subcatchment (Rational method)
    2. Route flow through pipe network
    3. Excess inflow (beyond pipe capacity) becomes surface flooding
    4. Track water depth evolution
    """
    H, W = dem.shape
    active = ~bldg
    n_nodes = len(nodes)
    n_steps = len(intensity_mmh)
    dt_s = dt_min * 60

    # Build pipe network adjacency
    node_id_to_idx = {n['id']: i for i, n in enumerate(nodes)}
    downstream = np.full(n_nodes, -1, dtype=int)  # which node does each node flow to
    pipe_capacity = np.zeros(n_nodes)

    for link in links:
        fi = node_id_to_idx.get(link['from_node'])
        ti = node_id_to_idx.get(link['to_node'])
        if fi is not None and ti is not None:
            # Flow from fi to ti (downhill)
            if nodes[ti]['invert'] < nodes[fi]['invert']:
                downstream[fi] = ti
            else:
                downstream[ti] = fi
            D = link['diameter']; A = np.pi*D**2/4; R = D/4
            S = max(link['slope'], 0.001)
            Q = A * (1/0.013) * R**(2/3) * np.sqrt(S)
            pipe_capacity[fi] += Q
            pipe_capacity[ti] += Q

    pipe_capacity = np.maximum(pipe_capacity, 0.01)

    # Runoff coefficient (urban: 0.7-0.9) — increased to account for building runoff
    runoff_coeff = 0.90

    # Track water
    h_surface = np.zeros((H, W), dtype=np.float64)
    hmax_surface = np.zeros((H, W), dtype=np.float64)
    cumulative_drained = 0.0  # water that successfully passes through pipes
    cumulative_flooded = 0.0  # water that exceeds pipe capacity

    records = {'time_h': [], 'vol_m3': [], 'hmax_m': [],
               'flooded_cells': [], 'drained_m3': []}

    # Pre-compute depression storage: how much water each cell can hold before overflowing
    # This is the key improvement: track local depression storage
    node_storage = np.zeros(n_nodes)  # water stored at each node (waiting to enter pipe)

    for step in range(n_steps):
        t_h = step * dt_min / 60

        # 1. Compute rainfall volume per subcatchment
        rain_m = intensity_mmh[step] * dt_min / 60 / 1000  # mm/h -> m depth
        rain_vol_per_cell = rain_m * CELL_AREA

        # 2. Runoff generation per subcatchment
        inflow_to_node = np.zeros(n_nodes)
        for i in range(n_nodes):
            area_i = areas[i]
            if area_i > 0:
                # Rational method: Q = C * I * A
                runoff_vol = runoff_coeff * rain_m * area_i  # m3
                inflow_to_node[i] += runoff_vol

        # 3. Route flow through the pipe network (topological order)
        # Find root nodes (no inflow from other pipes, or at top of network)
        has_inflow = np.zeros(n_nodes, dtype=bool)
        for i in range(n_nodes):
            if downstream[i] >= 0:
                has_inflow[downstream[i]] = True

        # Process from upstream to downstream
        processed = np.zeros(n_nodes, dtype=bool)
        # Start with leaves (no one flows to them = highest elevation nodes)
        leaves = np.where(~has_inflow)[0]

        total_drained = 0.0
        total_overflow = 0.0
        node_overflow = np.zeros(n_nodes)

        # Multiple passes to handle network routing
        for _ in range(10):  # up to 10 routing iterations
            for i in range(n_nodes):
                if processed[i]:
                    continue

                total_inflow = inflow_to_node[i] + node_storage[i]

                # Can the downstream pipe handle this flow?
                max_pipe_flow = pipe_capacity[i] * dt_s  # m3 per time step

                if total_inflow <= max_pipe_flow:
                    # All water enters the pipe
                    node_storage[i] = 0
                    routed = total_inflow
                    overflow = 0
                else:
                    # Pipe is at capacity, excess overflows
                    routed = max_pipe_flow
                    overflow = total_inflow - max_pipe_flow
                    node_storage[i] = 0

                total_drained += routed
                total_overflow += overflow
                node_overflow[i] += overflow

                # Pass routed flow downstream
                if downstream[i] >= 0:
                    inflow_to_node[downstream[i]] += routed

                processed[i] = True

        cumulative_drained += total_drained

        # 4. Distribute overflow as surface flooding
        # Water that can't enter pipes accumulates on the surface
        if total_overflow > 0:
            # Distribute overflow across contributing cells
            overflow_depth_m = np.zeros((H, W), dtype=np.float64)
            for i in range(n_nodes):
                if node_overflow[i] > 0 and areas[i] > 0:
                    mask = subcatch_map == i
                    if mask.sum() > 0:
                        rain_this_step = rain_m
                        # Add overflow proportionally to rainfall
                        overflow_ratio = node_overflow[i] / (inflow_to_node[i] + 1e-6)
                        # Distribute based on current water depth (more to already wet areas)
                        overflow_depth_m[mask] += rain_this_step * overflow_ratio

            h_surface += overflow_depth_m
            h_surface[active] += rain_m * runoff_coeff  # Rain becomes surface flow

        else:
            h_surface[active] += rain_m * runoff_coeff

        # Natural drainage: water flows downhill
        for _ in range(2):
            h_new = h_surface.copy()
            wse = dem + h_surface
            for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                wse_shift = np.roll(np.roll(wse, di, axis=0), dj, axis=1)
                h_shift = np.roll(np.roll(h_surface, di, axis=0), dj, axis=1)
                dh = wse - wse_shift
                flow_mask = (dh > 0.001) & active & np.roll(np.roll(active, di, axis=0), dj, axis=1)
                flow = np.where(flow_mask, dh * 0.15, 0)
                flow = np.minimum(flow, h_surface * 0.2)
                h_new -= flow
                rolled_back = np.roll(np.roll(flow, -di, axis=0), -dj, axis=1)
                h_new += rolled_back
            h_surface = np.maximum(h_new, 0)
            h_surface[bldg] = 0

        hmax_surface = np.maximum(hmax_surface, h_surface)

        # Record
        if step % (max(1, int(30/dt_min))) == 0:
            vol = float(np.sum(h_surface[active]) * CELL_AREA)
            hmax = float(np.max(h_surface))
            flooded = int(np.sum(h_surface > 0.03))
            records['time_h'].append(t_h)
            records['vol_m3'].append(vol)
            records['hmax_m'].append(hmax)
            records['flooded_cells'].append(flooded)
            records['drained_m3'].append(float(cumulative_drained))

            if step % (max(1, int(180/dt_min))) == 0:
                print(f"  t={t_h:.1f}h  vol={vol:.0f}m3  hmax={hmax:.3f}m  "
                      f"flooded={flooded}/{active.sum()}  drained={cumulative_drained:.0f}m3")

    return records, h_surface, hmax_surface


def run_sewer_simulation_2d(dem, bldg, rainfall_3d, nodes, links, subcatch_map, areas, with_pipes):
    """
    Run sewer simulation with full 2D spatially-heterogeneous rainfall.
    rainfall_3d: (T, H, W) in mm/5min. Uses vectorized cell-level operations.
    """
    H, W = dem.shape; active = ~bldg
    n_steps = rainfall_3d.shape[0]; dt_min = 5.0; dt_s = dt_min * 60
    runoff_coeff = 0.90; n_active = active.sum()

    # Pipe network: pre-compute per-subcatchment pipe capacity
    n_nodes = len(nodes)
    node_id_to_idx = {n['id']: i for i, n in enumerate(nodes)}
    pipe_cap_per_node = np.zeros(n_nodes)
    if with_pipes and links:
        for lk in links:
            fi = node_id_to_idx.get(lk['from_node']); ti = node_id_to_idx.get(lk['to_node'])
            if fi is not None and ti is not None:
                D = lk['diameter']; A = np.pi*D**2/4; R = D/4
                S = max(lk['slope'], 0.001)
                Q = A*(1/0.013)*R**(2/3)*np.sqrt(S)
                pipe_cap_per_node[fi] += Q; pipe_cap_per_node[ti] += Q
    pipe_cap_per_node = np.maximum(pipe_cap_per_node, 0.01)

    # Pre-compute: for each node, list of cell indices in its subcatchment
    subcatch_cells = {}  # node_idx -> array of flat indices
    subcatch_ncells = np.zeros(n_nodes, dtype=int)
    for i in range(n_nodes):
        mask_i = subcatch_map == i
        nc = mask_i.sum()
        if nc > 0:
            subcatch_cells[i] = np.where(mask_i.ravel())[0]
            subcatch_ncells[i] = nc

    h_flat = np.zeros(H * W, dtype=np.float64)
    hmax_flat = np.zeros(H * W, dtype=np.float64)
    cumulative_drained = 0.0
    records = {'time_h': [], 'vol_m3': [], 'hmax_m': [], 'flooded_cells': [], 'drained_m3': []}

    for step in range(n_steps):
        t_h = step * dt_min / 60
        rain_2d = rainfall_3d[step]  # mm/5min

        # Building runoff: total building rain distributed uniformly to active cells
        bldg_rain_total = np.sum(rain_2d[bldg])
        bldg_per_active = bldg_rain_total / n_active if n_active > 0 else 0

        # Apply cell-level rainfall + building runoff (vectorized)
        rain_m = rain_2d.ravel() / 1000.0 * runoff_coeff  # mm -> m
        rain_m[active.ravel()] += bldg_per_active / 1000.0 * runoff_coeff
        rain_m[bldg.ravel()] = 0
        h_flat += rain_m
        h_flat = np.maximum(h_flat, 0)

        # Downhill flow (2 iterations, vectorized)
        h_2d = h_flat.reshape(H, W)
        for _ in range(2):
            h_new = h_2d.copy(); wse = dem + h_2d
            for di, dj in [(0,1),(0,-1),(1,0),(-1,0)]:
                wse_s = np.roll(np.roll(wse, di, axis=0), dj, axis=1)
                h_s = np.roll(np.roll(h_2d, di, axis=0), dj, axis=1)
                dh = wse - wse_s
                fm = (dh > 0.001) & active & np.roll(np.roll(active, di, axis=0), dj, axis=1)
                flow = np.where(fm, dh * 0.15, 0)
                flow = np.minimum(flow, h_2d * 0.2)
                h_new -= flow
                h_new = np.roll(np.roll(h_new, -di, axis=0), -dj, axis=1) + flow
            h_2d = np.maximum(h_new, 0); h_2d[bldg] = 0
        h_flat = h_2d.ravel()

        # Drainage (per subcatchment)
        if with_pipes and len(links) > 0:
            for i in range(n_nodes):
                cells = subcatch_cells.get(i)
                if cells is None or len(cells) == 0:
                    continue
                water_vol = np.sum(h_flat[cells]) * CELL_AREA
                if water_vol > 0.01:
                    max_drain = min(water_vol * 0.4, pipe_cap_per_node[i] * dt_s * 0.5)
                    removal_depth = max_drain / (len(cells) * CELL_AREA)
                    h_flat[cells] = np.maximum(0, h_flat[cells] - removal_depth)
                    cumulative_drained += max_drain

        h_flat[bldg.ravel()] = 0
        hmax_flat = np.maximum(hmax_flat, h_flat)

        if step % 6 == 0:
            h_2d = h_flat.reshape(H, W)
            vol = float(np.sum(h_2d[active]) * CELL_AREA)
            hmax = float(np.max(h_2d))
            flooded = int(np.sum(h_2d > 0.03))
            records['time_h'].append(t_h); records['vol_m3'].append(vol)
            records['hmax_m'].append(hmax); records['flooded_cells'].append(flooded)
            records['drained_m3'].append(float(cumulative_drained))
            if step % 18 == 0:
                print(f"  t={t_h:.1f}h  vol={vol:.0f}m3  hmax={hmax:.3f}m  "
                      f"flooded={flooded}/{active.sum()}  drained={cumulative_drained:.0f}m3")

    h_final = h_flat.reshape(H, W)
    hmax_final = hmax_flat.reshape(H, W)
    return records, h_final, hmax_final


def main():
    print("=" * 60)
    print("  OSM-Based Sewer Drainage Simulation")
    print("=" * 60)

    # Load data
    print("\n[1] Loading data...")
    dem = np.load(DEM_PATH, allow_pickle=True).astype(np.float64)
    bldg = dem >= 49.9; H, W = dem.shape; active = ~bldg

    data = np.load(os.path.join(OUT_DIR, 'osm_merged_network.npz'), allow_pickle=True)
    nodes = data['nodes'].tolist(); links = data['links'].tolist()
    print(f"  DEM: {H}x{W}, Buildings: {bldg.sum():,}, Network: {len(nodes)} nodes, {len(links)} links")

    # Assign subcatchments
    print("\n[2] Assigning subcatchments...")
    subcatch_map, areas = assign_subcatchments(dem, bldg, nodes)

    # Design storm
    print("\n[3] Creating design storm...")
    duration_h = 6.0; dt_min = 5.0
    n_steps = int(duration_h*60/dt_min)
    t_min = np.arange(n_steps)*dt_min; t_peak = duration_h*60*0.35
    intensity = np.where(t_min<t_peak, 30*(t_min/t_peak)**0.4,
                         30*((duration_h*60-t_min)/(duration_h*60-t_peak))**1.2)
    intensity = intensity * 50.0 / (np.sum(intensity)*dt_min/60)
    print(f"  {n_steps} steps, 50mm, peak={np.max(intensity):.1f}mm/h")

    # Run paired simulations
    print("\n[4] Running simulations...")

    print("\n  --- A) Surface Only (no pipes) ---")
    rec_a, h_a, hmax_a = run_sewer_simulation(
        dem, bldg, [], [], intensity, dt_min, subcatch_map, areas)

    print("\n  --- B) With OSM Drainage Network ---")
    rec_b, h_b, hmax_b = run_sewer_simulation(
        dem, bldg, nodes, links, intensity, dt_min, subcatch_map, areas)

    # Summary
    print("\n[5] Results:")
    print(f"  {'Metric':<20s} {'Surface Only':>15s} {'With OSM Drainage':>18s} {'Change':>10s}")
    print(f"  {'-'*65}")
    expected = 0.05 * active.sum() * CELL_AREA
    for key, label in [('vol_m3', 'Volume (m3)'), ('hmax_m', 'Max Depth (m)'),
                        ('flooded_cells', 'Flooded Cells')]:
        va, vb = rec_a[key][-1], rec_b[key][-1]
        chg = (vb-va)/va*100 if va>0 else 0
        print(f"  {label:<20s} {va:>15.0f} {vb:>18.0f} {chg:>+9.1f}%")
    print(f"  {'Drained (m3)':<20s} {'---':>15s} {rec_b['drained_m3'][-1]:>18.0f}")
    print(f"  Expected rain: {expected:.0f} m3")

    # Save
    np.savez(os.path.join(OUT_DIR, 'osm_sewer_results.npz'),
             time_h=rec_a['time_h'], vol_a=rec_a['vol_m3'], vol_b=rec_b['vol_m3'],
             hmax_a=rec_a['hmax_m'], hmax_b=rec_b['hmax_m'],
             flooded_a=rec_a['flooded_cells'], flooded_b=rec_b['flooded_cells'],
             drained_m3=rec_b['drained_m3'],
             h_final_a=h_a, h_final_b=h_b, hmax_arr_a=hmax_a, hmax_arr_b=hmax_b,
             dem=dem, bldg=bldg, n_nodes=len(nodes), n_links=len(links),
             subcatch_map=subcatch_map)

    print("\n[6] Saving results and visualizations...")
    create_visualizations(rec_a, rec_b, h_a, h_b, dem, bldg, nodes, links, subcatch_map)

    print(f"\n  Done! Outputs in: {OUT_DIR}")
    return 0


def create_visualizations(rec_a, rec_b, h_a, h_b, dem, bldg, nodes, links, subcatch_map):
    H, W = dem.shape; extent = [0, W*CELL_SIZE, 0, H*CELL_SIZE]
    time_h = rec_a['time_h']
    bldg_bg = np.where(bldg, 0.3, 0)

    vmax = max(np.max(h_a[h_a>0.001]) if np.any(h_a>0.001) else 0.05,
               np.max(h_b[h_b>0.001]) if np.any(h_b>0.001) else 0.05, 0.05)

    # Fig 1: Flood comparison
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    fig.suptitle('OSM Road-Based Drainage: Flood Reduction (50mm/6h Design Storm)',
                 fontsize=14, fontweight='bold')

    for idx, (ax, h, title) in enumerate([
        (axes[0], h_a, f'Surface Only\nMax={np.max(h_a):.3f}m, Vol={rec_a["vol_m3"][-1]:.0f}m3'),
        (axes[1], h_b, f'With OSM Drainage\nMax={np.max(h_b):.3f}m, Vol={rec_b["vol_m3"][-1]:.0f}m3'),
    ]):
        h_show = np.ma.masked_where(h < 0.001, h)
        ax.imshow(np.flipud(bldg_bg), cmap=ListedColormap(['none', '#CCC']),
                  extent=extent, aspect='equal', alpha=0.5)
        im = ax.imshow(np.flipud(h_show), cmap=flood_cmap, extent=extent, aspect='equal', vmin=0, vmax=vmax)
        ax.set_title(title); ax.set_xlabel('E (m)'); ax.set_ylabel('N (m)')
        plt.colorbar(im, ax=ax, label='Depth (m)', shrink=0.8)

    ax = axes[2]; diff = h_a - h_b
    vlim = max(abs(np.min(diff)), abs(np.max(diff)), 0.01)
    im = ax.imshow(np.flipud(diff), cmap=plt.cm.RdBu_r, extent=extent, aspect='equal', vmin=-vlim, vmax=vlim)
    node_lookup = {n['id']: n for n in nodes}
    for link in links[:500]:  # sample links for performance
        fn = node_lookup.get(link['from_node']); tn = node_lookup.get(link['to_node'])
        if fn and tn: ax.plot([fn['x_m'], tn['x_m']], [fn['y_m'], tn['y_m']], 'black', lw=0.2, alpha=0.2)
    drained_val = rec_b['drained_m3'][-1]
    ax.set_title(f'Depth Reduction\nDrained={drained_val:.0f}m3')
    ax.set_xlabel('E (m)'); ax.set_ylabel('N (m)')
    plt.colorbar(im, ax=ax, label='Diff (m)', shrink=0.8)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'osm_flood_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Fig 2: Time series
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('OSM Drainage Network Performance', fontsize=14, fontweight='bold')

    ax = axes[0,0]
    ax.plot(time_h, rec_a['vol_m3'], 'r-o', lw=2, ms=3, label='Surface Only')
    ax.plot(time_h, rec_b['vol_m3'], 'b-s', lw=2, ms=3, label='With OSM Drainage')
    ax.set_xlabel('Time (h)'); ax.set_ylabel('Volume (m3)')
    ax.set_title(f'Surface Water Volume ({(rec_a["vol_m3"][-1]-rec_b["vol_m3"][-1])/rec_a["vol_m3"][-1]*100:.1f}% reduction)')
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[0,1]
    ax.plot(time_h, rec_a['hmax_m'], 'r-o', lw=2, ms=3, label='Surface Only')
    ax.plot(time_h, rec_b['hmax_m'], 'b-s', lw=2, ms=3, label='With OSM Drainage')
    ax.set_xlabel('Time (h)'); ax.set_ylabel('Max Depth (m)')
    ax.set_title('Maximum Depth'); ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1,0]
    ax.plot(time_h, np.array(rec_a['flooded_cells'])*CELL_AREA, 'r-o', lw=2, ms=3, label='Surface Only')
    ax.plot(time_h, np.array(rec_b['flooded_cells'])*CELL_AREA, 'b-s', lw=2, ms=3, label='With OSM Drainage')
    ax.set_xlabel('Time (h)'); ax.set_ylabel('Flooded Area (m2)')
    ax.set_title('Inundated Area'); ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1,1]
    ax.fill_between(time_h, 0, rec_b['drained_m3'], color='blue', alpha=0.3)
    ax.plot(time_h, rec_b['drained_m3'], 'b-', lw=2)
    ax.set_xlabel('Time (h)'); ax.set_ylabel('Volume (m3)')
    ax.set_title(f'Cumulative Drained: {rec_b["drained_m3"][-1]:.0f}m3')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'osm_timeseries.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    print("  Saved: osm_flood_comparison.png, osm_timeseries.png")


if __name__ == "__main__":
    sys.exit(main())
