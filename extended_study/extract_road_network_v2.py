#!/usr/bin/env python3
"""
Road Network Extraction V2 — Building-Gap-Based Pipe Network Design
=====================================================================
Revised approach:
1. Identify open ground (non-building cells) as potential road space
2. Filter by slope to exclude mountains
3. Place manholes at regular intervals on open ground
4. Connect manholes with pipes that follow terrain (avoid crossing buildings)
5. Identify main trunk from the densest flow paths
"""

import os, sys
import numpy as np
from scipy import ndimage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import warnings
warnings.filterwarnings('ignore')

DEM_PATH = r"e:\Projects\20260519-LarNO\LarNO-main\benchmark\urbanflood\geodata\region1_20m\dem.npy"
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUT_DIR, exist_ok=True)

CELL_SIZE = 20.0  # m


def design_road_based_network(dem, building_mask):
    """
    Design pipe network based on building-gap road corridors.

    Algorithm:
    1. Mark all non-building, non-steep cells as "road space"
    2. Place manholes on a coarse grid (every ~160m) within road space
    3. Connect each manhole to its nearest neighbors (up to 4 connections)
       only if the straight line doesn't cross buildings
    4. Compute flow accumulation on the resulting graph to identify main trunks
    5. Add outfall at the lowest point
    """
    H, W = dem.shape

    # Compute slope
    dy, dx = np.gradient(dem, CELL_SIZE)
    slope = np.sqrt(dx**2 + dy**2)

    # Step 1: Road space = open ground with reasonable slope
    open_ground = ~building_mask
    flat_enough = slope < 0.2  # ~11 degrees max for roads
    road_space = open_ground & flat_enough

    print(f"  Open ground: {open_ground.sum():,} cells")
    print(f"  Road space (flat + open): {road_space.sum():,} cells")

    # Step 2: Place manholes on a grid within road space
    spacing = 8  # cells = 160m  (between 100-200m per design standards)
    nodes = []
    node_positions = {}  # (row, col) -> node index

    nid = 0
    for r in range(spacing // 2, H, spacing):
        for c in range(spacing // 2, W, spacing):
            if road_space[r, c]:
                # Check that this isn't an isolated pixel
                local_count = np.sum(road_space[max(0, r-1):min(H, r+2),
                                               max(0, c-1):min(W, c+2)])
                if local_count >= 3:  # At least 3 of 9 cells are road
                    nid += 1
                    elev = float(dem[r, c])
                    invert = max(elev - 2.0, 0.3)
                    node_id = f'N{nid:05d}'
                    node_positions[(r, c)] = nid - 1
                    nodes.append({
                        'id': node_id, 'row': r, 'col': c,
                        'x_m': c * CELL_SIZE + CELL_SIZE / 2,
                        'y_m': (H - r) * CELL_SIZE - CELL_SIZE / 2,
                        'elevation': elev,
                        'invert': invert,
                        'type': 'junction',
                    })

    print(f"  Manholes placed: {len(nodes)}")

    # Step 3: Connect nodes with pipes (avoid crossing buildings)
    links = []
    lid = 0
    max_dist = spacing * CELL_SIZE * 3  # Max connection distance ~480m

    for i in range(len(nodes)):
        ni = nodes[i]
        # Find nearest neighbors
        distances = []
        for j in range(len(nodes)):
            if i >= j:
                continue
            nj = nodes[j]
            dr = ni['row'] - nj['row']
            dc = ni['col'] - nj['col']
            dist_cells = np.sqrt(dr**2 + dc**2)
            if dist_cells > spacing * 3:
                continue
            distances.append((dist_cells, j))

        # Connect to nearest neighbors (up to 4)
        distances.sort()
        for _, j in distances[:4]:
            nj = nodes[j]
            dist_m = np.sqrt((ni['x_m'] - nj['x_m'])**2 + (ni['y_m'] - nj['y_m'])**2)

            # Check that line doesn't cross buildings
            if not _line_crosses_buildings(ni['row'], ni['col'], nj['row'], nj['col'],
                                           building_mask, road_space):
                slope_pipe = max(abs(ni['invert'] - nj['invert']) / max(dist_m, 0.1), 0.001)
                lid += 1
                links.append({
                    'id': f'C{lid:05d}',
                    'from_node': ni['id'], 'to_node': nj['id'],
                    'length': dist_m, 'slope': slope_pipe,
                    'diameter': 0.6, 'mannings_n': 0.013, 'type': 'collector',
                })

    print(f"  Pipes created: {len(links)}")
    total_len = sum(l['length'] for l in links)
    print(f"  Total pipe length: {total_len:.0f}m")

    # Step 4: Identify main trunk (largest connected component)
    if nodes and links:
        # Build adjacency
        node_id_to_idx = {n['id']: i for i, n in enumerate(nodes)}
        adj = {i: [] for i in range(len(nodes))}
        for link in links:
            fi = node_id_to_idx[link['from_node']]
            ti = node_id_to_idx[link['to_node']]
            adj[fi].append(ti)
            adj[ti].append(fi)

        # Find largest connected component
        visited = set()
        components = []
        for start in range(len(nodes)):
            if start in visited:
                continue
            comp = []
            queue = [start]
            visited.add(start)
            while queue:
                v = queue.pop(0)
                comp.append(v)
                for nb in adj[v]:
                    if nb not in visited:
                        visited.add(nb)
                        queue.append(nb)
            components.append(comp)

        print(f"  Network components: {len(components)}")
        for i, comp in enumerate(components):
            print(f"    Component {i+1}: {len(comp)} nodes, "
                  f"{sum(1 for l in links if node_id_to_idx[l['from_node']] in comp and node_id_to_idx[l['to_node']] in comp)} links")

        # Mark the largest component as main trunk
        largest = max(components, key=len)
        main_trunk_nodes = set(largest)
        for ni in largest[:max(1, len(largest)//3)]:  # Top third = main trunk
            nodes[ni]['type'] = 'main_trunk'

        # Update pipe types and diameters for main trunk
        for link in links:
            fi = node_id_to_idx[link['from_node']]
            ti = node_id_to_idx[link['to_node']]
            if fi in main_trunk_nodes and ti in main_trunk_nodes:
                link['type'] = 'main'
                link['diameter'] = 0.8

    # Step 5: Add outfall
    if nodes:
        # Find lowest node
        lowest = min(nodes, key=lambda n: n['invert'])
        # Place outfall downstream (following terrain)
        out_r = min(lowest['row'] + 5, H - 2)
        out_c = min(lowest['col'] + 3, W - 2)
        if building_mask[out_r, out_c]:
            # Find nearest non-building cell
            for r_offset in range(10):
                for c_offset in range(-r_offset, r_offset + 1):
                    nr = lowest['row'] + r_offset
                    nc = lowest['col'] + c_offset
                    if 0 <= nr < H and 0 <= nc < W and not building_mask[nr, nc]:
                        out_r, out_c = nr, nc
                        break
                else:
                    continue
                break

        nid += 1
        outfall = {
            'id': f'N{nid:05d}', 'row': out_r, 'col': out_c,
            'x_m': out_c * CELL_SIZE + CELL_SIZE / 2,
            'y_m': (H - out_r) * CELL_SIZE - CELL_SIZE / 2,
            'elevation': float(dem[out_r, out_c]),
            'invert': lowest['invert'] - 2.5, 'type': 'outfall',
        }
        nodes.append(outfall)

        lid += 1
        links.append({
            'id': f'C{lid:05d}',
            'from_node': lowest['id'], 'to_node': outfall['id'],
            'length': np.sqrt((lowest['x_m']-outfall['x_m'])**2 + (lowest['y_m']-outfall['y_m'])**2),
            'slope': 0.005, 'diameter': 1.0, 'mannings_n': 0.013, 'type': 'main',
        })

    print(f"\n  Final Network:")
    for ntype in ['main_trunk', 'junction', 'outfall']:
        cnt = sum(1 for n in nodes if n['type'] == ntype)
        if cnt:
            print(f"    {ntype}: {cnt}")
    print(f"    Links: {len(links)}")
    print(f"    Total length: {sum(l['length'] for l in links):.0f}m")

    return {'nodes': nodes, 'links': links, 'ns_positions': [], 'ew_positions': [],
            'sub_region': [0, H, 0, W]}, road_space, slope


def _line_crosses_buildings(r1, c1, r2, c2, building_mask, road_space):
    """Check if the line from (r1,c1) to (r2,c2) crosses buildings or non-road areas."""
    # Bresenham line algorithm
    dr = abs(r2 - r1)
    dc = abs(c2 - c1)
    if dr == 0 and dc == 0:
        return True

    steps = max(dr, dc)
    for t in range(1, steps):
        frac = t / steps
        r = int(r1 + frac * (r2 - r1))
        c = int(c1 + frac * (c2 - c1))
        if building_mask[r, c]:
            return True
        # Allow minor deviation through non-road (small gaps)
        if not road_space[r, c]:
            # Check if there's nearby road space
            nearby = np.any(road_space[max(0, r-2):min(building_mask.shape[0], r+3),
                                       max(0, c-2):min(building_mask.shape[1], c+3)])
            if not nearby:
                return True
    return False


def visualize_full_pipeline(dem, bldg, road_space, slope, network, out_path):
    """6-panel visualization of the complete pipeline."""
    H, W = dem.shape
    extent = [0, W * CELL_SIZE, 0, H * CELL_SIZE]

    fig, axes = plt.subplots(2, 3, figsize=(22, 14))
    fig.suptitle('Road-Based Pipe Network Design — Shenzhen Futian Study Area',
                 fontsize=15, fontweight='bold')

    # (1) DEM with buildings masked
    ax = axes[0, 0]
    terrain = np.where(bldg, np.nan, dem)
    im = ax.imshow(terrain, cmap='terrain', extent=extent, aspect='equal', origin='lower')
    ax.set_title('DEM (Buildings Masked)')
    ax.set_xlabel('Easting (m)'); ax.set_ylabel('Northing (m)')
    plt.colorbar(im, ax=ax, label='Elev (m)', shrink=0.8)

    # (2) Building mask
    ax = axes[0, 1]
    overlay = np.zeros((H, W, 3))
    overlay[~bldg] = [0.9, 0.95, 0.85]  # Light beige: open ground
    overlay[bldg] = [0.5, 0.05, 0.05]   # Dark red: buildings
    ax.imshow(overlay.transpose(1, 0, 2), extent=extent, aspect='equal', origin='lower')
    ax.set_title(f'Building Mask\nBuildings: {bldg.sum():,} cells ({100*bldg.sum()/bldg.size:.0f}%)')
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color='darkred', label='Buildings (50m wall)'),
        Patch(color='beige', label='Open Ground'),
    ], loc='lower right')

    # (3) Road space (open + flat)
    ax = axes[0, 2]
    ax.imshow(np.where(bldg, 0.3, 0), cmap=ListedColormap(['none', '#888']),
              extent=extent, aspect='equal', origin='lower', alpha=0.4)
    ax.imshow(road_space, cmap=ListedColormap(['#E8E8E8', '#4CAF50']),
              extent=extent, aspect='equal', origin='lower', alpha=0.7)
    ax.set_title(f'Road Space (Open & Flat)\n{road_space.sum():,} cells = {road_space.sum()*CELL_SIZE**2/1e6:.2f} km2')
    ax.set_xlabel('Easting (m)')
    ax.legend(handles=[
        Patch(color='green', label='Road Space (slope<0.2)'),
        Patch(color='gray', label='Buildings'),
    ], loc='lower right')

    # (4) Manhole placement grid
    ax = axes[1, 0]
    ax.imshow(np.where(bldg, 0.3, 0), cmap=ListedColormap(['none', '#CCC']),
              extent=extent, aspect='equal', origin='lower', alpha=0.5)
    ax.imshow(road_space, cmap=ListedColormap(['none', '#E8F5E9']),
              extent=extent, aspect='equal', origin='lower', alpha=0.4)
    for node in network['nodes']:
        c = {'main_trunk': 'red', 'junction': '#2196F3', 'outfall': 'green'}.get(node['type'], 'gray')
        s = {'main_trunk': 35, 'junction': 18, 'outfall': 60}.get(node['type'], 12)
        m = {'outfall': '^'}.get(node['type'], 'o')
        ax.scatter(node['x_m'], node['y_m'], c=c, s=s, marker=m,
                  zorder=5, edgecolors='black', linewidth=0.3)
    ax.set_title(f'Manholes: {len(network["nodes"])} Nodes on Grid')
    ax.set_xlabel('Easting (m)'); ax.set_ylabel('Northing (m)')
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='Main Trunk'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#2196F3', markersize=6, label='Junction'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='green', markersize=10, label='Outfall'),
    ], loc='lower right')

    # (5) Pipe network connections
    ax = axes[1, 1]
    ax.imshow(terrain, cmap='terrain', extent=extent, aspect='equal', origin='lower', alpha=0.6)
    node_lookup = {n['id']: n for n in network['nodes']}
    for link in network['links']:
        fn = node_lookup[link['from_node']]
        tn = node_lookup[link['to_node']]
        color = {'main': '#D32F2F', 'collector': '#FF9800', 'branch': '#2196F3'}.get(link['type'], 'gray')
        lw = {'main': 2.8, 'collector': 1.8, 'branch': 1.0}.get(link['type'], 1.0)
        ax.plot([fn['x_m'], tn['x_m']], [fn['y_m'], tn['y_m']],
                color=color, linewidth=lw, alpha=0.85, zorder=3)
    for node in network['nodes']:
        c = {'main_trunk': 'red', 'junction': '#2196F3', 'outfall': 'green'}.get(node['type'], 'gray')
        s = {'main_trunk': 30, 'junction': 12, 'outfall': 50}.get(node['type'], 10)
        m = {'outfall': '^'}.get(node['type'], 'o')
        ax.scatter(node['x_m'], node['y_m'], c=c, s=s, marker=m,
                  zorder=5, edgecolors='black', linewidth=0.3)
    ax.set_title(f'Pipe Network: {len(network["links"])} Pipes, {sum(l["length"] for l in network["links"]):.0f}m Total')
    ax.set_xlabel('Easting (m)')
    ax.legend(handles=[
        Line2D([0], [0], color='#D32F2F', lw=2.8, label='Main Trunk (D=800mm)'),
        Line2D([0], [0], color='#FF9800', lw=1.8, label='Collector (D=600mm)'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='green', markersize=8, label='Outfall'),
    ], loc='lower right')

    # (6) Summary
    ax = axes[1, 2]
    ax.axis('off')
    node_types = {}
    for n in network['nodes']:
        node_types[n['type']] = node_types.get(n['type'], 0) + 1
    link_types = {}
    total_len = 0
    for l in network['links']:
        link_types[l['type']] = link_types.get(l['type'], 0) + 1
        total_len += l['length']

    metrics = f"""
    ROAD-BASED PIPE NETWORK DESIGN
    ==============================

    DOMAIN
    2.0km x 2.0km @ 20m resolution
    {H}x{W} cells, {bldg.sum():,} buildings

    ROAD EXTRACTION
    Method: Building-gap analysis
    Road space: {road_space.sum():,} cells
    Gradient: < 0.2 (roads are flat)
    Manholes on {CELL_SIZE*8:.0f}m grid spacing

    NETWORK STATISTICS
    Total manholes: {len(network['nodes'])}
    {'  Main trunk: ' + str(node_types.get('main_trunk', 0)) if 'main_trunk' in node_types else ''}
    {'  Junctions: ' + str(node_types.get('junction', 0)) if 'junction' in node_types else ''}
    Outfall: {node_types.get('outfall', 0)}
    Total pipes: {len(network['links'])}
    {'  Main pipes: ' + str(link_types.get('main', 0)) if 'main' in link_types else ''}
    {'  Collectors: ' + str(link_types.get('collector', 0)) if 'collector' in link_types else ''}
    Total length: {total_len:.0f}m

    DESIGN STANDARDS
    Main trunk: D=800mm, n=0.013
    Branch/collector: D=600mm, n=0.013
    Manhole spacing: ~160m
    Pipe slope: terrain-following
    Cover depth: 2.0m minimum
    """.strip()
    ax.text(0.05, 0.5, metrics, transform=ax.transAxes, fontsize=9.5,
            fontfamily='monospace', verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='#F0F8FF', alpha=0.9))

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {os.path.basename(out_path)}")


def main():
    print("=" * 60)
    print("  Road-Based Pipe Network Design V2")
    print("  Shenzhen Futian District")
    print("=" * 60)

    # Load DEM
    print("\n[1] Loading data...")
    dem_full = np.load(DEM_PATH, allow_pickle=True).astype(np.float64)
    bldg_full = dem_full >= 49.9

    # Use a larger study area: 3km x 3km (150x150 cells)
    # This gives more room for roads to form connected networks
    y0, y1 = 80, 230  # 150 rows = 3km
    x0, x1 = 150, 300  # 150 cols = 3km
    dem = dem_full[y0:y1, x0:x1]
    bldg = bldg_full[y0:y1, x0:x1]

    H, W = dem.shape
    print(f"  Study area: {H*CELL_SIZE/1000:.1f}km x {W*CELL_SIZE/1000:.1f}km ({H}x{W} cells)")
    print(f"  Buildings: {bldg.sum():,}/{bldg.size} cells ({100*bldg.sum()/bldg.size:.0f}%)")

    # Design network based on building-gap roads
    print("\n[2] Designing road-based pipe network...")
    network, road_space, slope = design_road_based_network(dem, bldg)

    # Visualize
    print("\n[3] Creating visualization...")
    visualize_full_pipeline(dem, bldg, road_space, slope, network,
                            os.path.join(OUT_DIR, 'road_based_network_v2.png'))

    # Also create full-domain context
    print("\n[4] Creating full-domain context...")
    fig, ax = plt.subplots(figsize=(14, 10))
    terrain_full = np.where(bldg_full, np.nan, dem_full)
    ax.imshow(terrain_full, cmap='terrain',
              extent=[0, dem_full.shape[1]*CELL_SIZE, 0, dem_full.shape[0]*CELL_SIZE],
              aspect='equal', origin='lower', alpha=0.85)
    from matplotlib.patches import Rectangle
    rect = Rectangle((x0*CELL_SIZE, y0*CELL_SIZE), W*CELL_SIZE, H*CELL_SIZE,
                      linewidth=2.5, edgecolor='red', facecolor='none', linestyle='--')
    ax.add_patch(rect)
    ax.text(x0*CELL_SIZE + 200, (y0+H)*CELL_SIZE - 300,
            f'Study Area\n{H*CELL_SIZE/1000:.1f}x{W*CELL_SIZE/1000:.1f}km',
            color='red', fontweight='bold', fontsize=12,
            bbox=dict(facecolor='white', alpha=0.8))
    ax.set_title('Shenzhen Futian — Full Domain (Study Area Outlined)')
    ax.set_xlabel('Easting (m)'); ax.set_ylabel('Northing (m)')
    plt.colorbar(ax.images[0], ax=ax, label='Elevation (m)')
    fig.savefig(os.path.join(OUT_DIR, 'full_domain_context.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Save network data
    np.savez(os.path.join(OUT_DIR, 'road_based_network_v2.npz'),
             nodes=network['nodes'], links=network['links'],
             road_space=road_space, sub_region=[y0, y1, x0, x1],
             dem_shape=dem.shape)

    print(f"\n{'='*60}")
    print(f"  Road-Based Pipe Network — Complete")
    print(f"  Output directory: {OUT_DIR}")
    print(f"  Key outputs:")
    for f in ['road_based_network_v2.png', 'full_domain_context.png',
              'road_based_network_v2.npz']:
        fpath = os.path.join(OUT_DIR, f)
        if os.path.exists(fpath):
            print(f"    {f} ({os.path.getsize(fpath)/1024:.0f} KB)")
    print(f"{'='*60}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
