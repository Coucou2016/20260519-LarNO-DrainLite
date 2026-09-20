#!/usr/bin/env python3
"""
OSM-Based Pipe Network Design for Shenzhen Futian District
============================================================
Uses OpenStreetMap road network data to design the storm drainage
pipe network. The OSM data is aligned to the DEM grid based on the
known extent of both datasets.

Pipeline:
1. Download OSM roads for Futian District (osmnx)
2. Filter to major roads (motorway, trunk, primary, secondary)
3. Align OSM coordinates to DEM grid cells
4. Place manholes along major road centerlines at design spacing
5. Connect manholes with pipes following road (terrain) slope
6. Add outfall at lowest point
"""

import os, sys
import numpy as np
from scipy import ndimage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import warnings
warnings.filterwarnings('ignore')

DEM_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "LarNO-main", "benchmark", "urbanflood", "geodata", "region1_20m", "dem.npy")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUT_DIR, exist_ok=True)

CELL_SIZE = 20.0  # m

# OSM bounding box (from actual download)
OSM_X_MIN, OSM_X_MAX = 113.9907, 114.0997  # X_MIN +0.0025deg (total shift ~250m west)
OSM_Y_MIN, OSM_Y_MAX = 22.5130, 22.5820  # Y_MIN shifted north by ~1km to align with DEM

# Rotation: 3 degrees clockwise around DEM center
ROT_DEG = -3.0  # negative = clockwise
ROT_CENTER_LON = (OSM_X_MIN + OSM_X_MAX) / 2.0
ROT_CENTER_LAT = (OSM_Y_MIN + OSM_Y_MAX) / 2.0

def rotate_osm_coords(lon, lat):
    import math
    theta = math.radians(ROT_DEG)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    dlon = lon - ROT_CENTER_LON
    dlat = lat - ROT_CENTER_LAT
    new_lon = ROT_CENTER_LON + dlon * cos_t - dlat * sin_t
    new_lat = ROT_CENTER_LAT + dlon * sin_t + dlat * cos_t
    return new_lon, new_lat


def download_osm_roads():
    """Download OSM road network for Futian District."""
    import osmnx as ox

    print("  Downloading OSM roads...")
    G = ox.graph_from_place('Futian District, Shenzhen, China', network_type='drive')
    nodes_gdf, edges_gdf = ox.graph_to_gdfs(G)

    # Filter to major roads
    major_types = ['motorway', 'trunk', 'primary', 'secondary']
    major_edges = edges_gdf[edges_gdf['highway'].isin(major_types)].copy()

    # Also include tertiary for better coverage
    all_significant = edges_gdf[
        edges_gdf['highway'].isin(major_types + ['tertiary', 'primary_link', 'trunk_link'])
    ].copy()

    print(f"  All roads: {len(edges_gdf)} segments, {len(nodes_gdf)} nodes")
    print(f"  Major roads: {len(major_edges)} segments")
    print(f"  Significant roads: {len(all_significant)} segments")

    return nodes_gdf, major_edges, all_significant


def osm_to_grid(lon, lat, dem_shape):
    """
    Convert OSM (lon, lat) coordinates to DEM grid (row, col).

    OSM: lon 113.9882-114.0997, lat 22.5037-22.5820
    DEM: 400 rows (N-S) x 560 cols (E-W) at 20m
    """
    H, W = dem_shape

    # Linear mapping
    col_frac = (lon - OSM_X_MIN) / (OSM_X_MAX - OSM_X_MIN)
    row_frac = (lat - OSM_Y_MIN) / (OSM_Y_MAX - OSM_Y_MIN)

    col = int(np.clip(col_frac * W, 0, W - 1))
    # DEM row 0 = north (high latitude)
    row = int(np.clip((1.0 - row_frac) * H, 0, H - 1))

    return row, col


def grid_to_plot_xy(r, c, dem_shape):
    """Grid (row, col) -> plot Easting/Northing (m), same as manhole x_m/y_m."""
    H, _ = dem_shape
    return c * CELL_SIZE + CELL_SIZE / 2, (H - r) * CELL_SIZE - CELL_SIZE / 2


def osm_lonlat_to_grid(lon, lat, dem_shape):
    """Apply rotation + bbox alignment (same as extract_road_centerlines)."""
    lon, lat = rotate_osm_coords(lon, lat)
    return osm_to_grid(lon, lat, dem_shape)


def iter_geom_lonlat(geom):
    """Yield (lon, lat) along a LineString or MultiLineString."""
    if geom is None or geom.is_empty:
        return
    if geom.geom_type == 'LineString':
        yield from geom.coords
    elif geom.geom_type == 'MultiLineString':
        for line in geom.geoms:
            yield from line.coords


def _highway_type(edge):
    hw = edge.get('highway', '')
    if isinstance(hw, (list, tuple)):
        hw = hw[0] if hw else ''
    return hw


def draw_osm_roads(ax, edges_gdf, dem_shape, style='colored', zorder=4):
    """
    Overlay OSM roads using the same lon/lat transform as pipe-network extraction.

    style: 'colored' (by highway class) or 'single' (one color for building-mask panel)
    """
    H, W = dem_shape
    road_colors = {
        'motorway': '#8B0000', 'trunk': '#FF0000', 'primary': '#FF6600',
        'secondary': '#FF9900', 'tertiary': '#FFCC00',
        'primary_link': '#FF6600', 'trunk_link': '#FF0000',
    }
    road_lw = {
        'motorway': 2.5, 'trunk': 2.0, 'primary': 1.5, 'secondary': 1.2,
        'tertiary': 0.9, 'primary_link': 1.2, 'trunk_link': 1.5,
    }

    for _, edge in edges_gdf.iterrows():
        geom = edge.geometry
        if geom is None or geom.is_empty:
            continue

        hw = _highway_type(edge)
        if style == 'colored':
            color = road_colors.get(hw, '#888888')
            lw = road_lw.get(hw, 0.6)
            alpha = 0.85
        else:
            color, lw, alpha = '#E53935', 1.1, 0.75

        xs, ys = [], []
        for lon, lat in iter_geom_lonlat(geom):
            r, c = osm_lonlat_to_grid(lon, lat, dem_shape)
            if 0 <= r < H and 0 <= c < W:
                x, y = grid_to_plot_xy(r, c, dem_shape)
                xs.append(x)
                ys.append(y)
            elif len(xs) >= 2:
                ax.plot(xs, ys, color=color, linewidth=lw, alpha=alpha,
                        solid_capstyle='round', zorder=zorder)
                xs, ys = [], []
        if len(xs) >= 2:
            ax.plot(xs, ys, color=color, linewidth=lw, alpha=alpha,
                    solid_capstyle='round', zorder=zorder)


def extract_road_centerlines(edges_gdf, dem_shape, min_length_m=200):
    """
    Extract road centerlines from OSM edges, mapped to DEM grid.

    Returns list of road segments, each as list of (row, col) tuples.
    """
    H, W = dem_shape
    segments = []

    for _, edge in edges_gdf.iterrows():
        geom = edge.geometry
        if geom is None or geom.is_empty:
            continue

        # Extract coordinates along the road
        if geom.geom_type == 'LineString':
            coords = list(geom.coords)
        elif geom.geom_type == 'MultiLineString':
            coords = []
            for line in geom.geoms:
                coords.extend(list(line.coords))
        else:
            continue

        if len(coords) < 2:
            continue

        # Map to grid
        grid_points = []
        for lon, lat in coords:
            r, c = osm_lonlat_to_grid(lon, lat, dem_shape)
            if 0 <= r < H and 0 <= c < W:
                if not grid_points or (r, c) != grid_points[-1]:
                    grid_points.append((r, c))

        if len(grid_points) >= 2:
            # Calculate approximate length
            length_m = edge.get('length', 0)
            if length_m >= min_length_m or len(grid_points) >= 3:
                segments.append(grid_points)

    print(f"  Extracted {len(segments)} road segments (>= {min_length_m}m)")
    return segments


def place_manholes_along_roads(segments, dem, building_mask, spacing_cells=8):
    """
    Place manholes along road centerlines at regular intervals.

    Args:
        segments: list of road segments [(r,c), ...]
        dem: DEM array
        building_mask: boolean mask of buildings
        spacing_cells: manhole spacing in cells (~160m at 20m resolution)
    """
    nodes = []
    links = []
    node_positions = {}  # (r, c) -> node index
    nid, lid = 0, 0

    for seg in segments:
        if len(seg) < 3:
            continue

        # Cumulative distance along segment
        seg_nodes = []
        seg_dist = 0
        prev_pt = None

        for i, (r, c) in enumerate(seg):
            if prev_pt is not None:
                dr = r - prev_pt[0]
                dc = c - prev_pt[1]
                seg_dist += np.sqrt(dr**2 + dc**2)

            # Skip if on building
            if building_mask[r, c]:
                # Try to shift to nearest non-building cell
                shifted = False
                for radius in range(1, 5):
                    for dr in range(-radius, radius + 1):
                        for dc in range(-radius, radius + 1):
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < dem.shape[0] and 0 <= nc < dem.shape[1]:
                                if not building_mask[nr, nc]:
                                    r, c = nr, nc
                                    shifted = True
                                    break
                        if shifted:
                            break
                    if shifted:
                        break
                if building_mask[r, c]:
                    prev_pt = (r, c)
                    continue

            # Place manhole at this position if far enough from previous
            pos_key = (r, c)
            if pos_key in node_positions:
                seg_nodes.append(node_positions[pos_key])
            else:
                # Check if we should place a new manhole here
                should_place = (len(seg_nodes) == 0) or (
                    seg_dist >= spacing_cells * len(seg_nodes)
                )

                if should_place and seg_dist >= 0:
                    # Check that this is not too close to existing node on this segment
                    too_close = False
                    if seg_nodes:
                        last_node = nodes[seg_nodes[-1]]
                        dr = r - last_node['row']
                        dc = c - last_node['col']
                        if np.sqrt(dr**2 + dc**2) < spacing_cells * 0.5:
                            too_close = True
                    if not too_close:
                        nid += 1
                        elev = float(dem[r, c])
                        node_id = f'N{nid:05d}'
                        node_positions[pos_key] = nid - 1
                        nodes.append({
                            'id': node_id, 'row': r, 'col': c,
                            'x_m': c * CELL_SIZE + CELL_SIZE / 2,
                            'y_m': (dem.shape[0] - r) * CELL_SIZE - CELL_SIZE / 2,
                            'elevation': elev,
                            'invert': max(elev - 2.0, 0.3),
                            'type': 'junction',
                        })
                        seg_nodes.append(nid - 1)

            prev_pt = (r, c)

        # Connect consecutive nodes on this segment
        for i in range(len(seg_nodes) - 1):
            fn = nodes[seg_nodes[i]]
            tn = nodes[seg_nodes[i + 1]]
            dist_m = np.sqrt((fn['x_m'] - tn['x_m'])**2 + (fn['y_m'] - tn['y_m'])**2)
            if dist_m < 10 or dist_m > 500:
                continue

            # Check line doesn't cross buildings
            if _line_crosses_buildings(fn['row'], fn['col'], tn['row'], tn['col'],
                                       building_mask, min_open=0.7):
                continue

            slope_pipe = max(abs(fn['invert'] - tn['invert']) / max(dist_m, 0.1), 0.001)
            lid += 1
            links.append({
                'id': f'C{lid:05d}',
                'from_node': fn['id'], 'to_node': tn['id'],
                'length': dist_m, 'slope': slope_pipe,
                'diameter': 0.6, 'mannings_n': 0.013, 'type': 'collector',
            })

    return nodes, links


def _line_crosses_buildings(r1, c1, r2, c2, building_mask, min_open=0.7):
    """Check if > (1-min_open) fraction of line crosses buildings."""
    dr = abs(r2 - r1)
    dc = abs(c2 - c1)
    steps = max(dr, dc, 1)
    building_count = 0
    total = 0
    for t in range(steps + 1):
        frac = t / steps
        r = int(r1 + frac * (r2 - r1))
        c = int(c1 + frac * (c2 - c1))
        if 0 <= r < building_mask.shape[0] and 0 <= c < building_mask.shape[1]:
            total += 1
            if building_mask[r, c]:
                building_count += 1
    if total == 0:
        return True
    return (building_count / total) > (1.0 - min_open)


def identify_main_trunk(nodes, links):
    """Identify the main trunk line from the largest connected component."""
    if not nodes or not links:
        return

    node_id_to_idx = {n['id']: i for i, n in enumerate(nodes)}
    adj = {i: [] for i in range(len(nodes))}
    for link in links:
        fi = node_id_to_idx.get(link['from_node'])
        ti = node_id_to_idx.get(link['to_node'])
        if fi is not None and ti is not None:
            adj[fi].append(ti)
            adj[ti].append(fi)

    # Find connected components
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

    if not components:
        return

    # Largest component = main network
    largest = max(components, key=len)
    main_set = set(largest)

    # Mark main trunk nodes (nodes with highest connectivity in largest component)
    for ni in largest:
        if len(adj[ni]) >= 3:  # well-connected nodes
            nodes[ni]['type'] = 'main_trunk'
        elif len(adj[ni]) >= 2:
            nodes[ni]['type'] = 'main_trunk'

    # Update main trunk pipe diameters
    for link in links:
        fi = node_id_to_idx.get(link['from_node'])
        ti = node_id_to_idx.get(link['to_node'])
        if fi is not None and ti is not None:
            if fi in main_set and ti in main_set:
                fn_type = nodes[fi]['type']
                tn_type = nodes[ti]['type']
                if fn_type == 'main_trunk' or tn_type == 'main_trunk':
                    link['type'] = 'main'
                    link['diameter'] = 0.8

    print(f"  Network components: {len(components)}")
    for i, comp in enumerate(components[:5]):
        print(f"    Component {i+1}: {len(comp)} nodes")


def add_outfall(nodes, links, dem, building_mask):
    """Add outfall at the lowest point of the main network."""
    if not nodes:
        return

    # Find lowest node in main trunk
    main_nodes = [n for n in nodes if n['type'] == 'main_trunk']
    if not main_nodes:
        main_nodes = nodes

    lowest = min(main_nodes, key=lambda n: n['invert'])

    # Place outfall downstream
    out_r = min(lowest['row'] + 5, dem.shape[0] - 2)
    out_c = min(lowest['col'] + 3, dem.shape[1] - 2)

    # Ensure outfall is not on a building
    if building_mask[out_r, out_c]:
        for radius in range(1, 15):
            found = False
            for dr in range(-radius, radius + 1):
                nr = lowest['row'] + radius
                nc = lowest['col'] + dr
                if 0 <= nr < dem.shape[0] and 0 <= nc < dem.shape[1]:
                    if not building_mask[nr, nc]:
                        out_r, out_c = nr, nc
                        found = True
                        break
            if found:
                break

    nid = len(nodes) + 1
    outfall = {
        'id': f'N{nid:05d}', 'row': out_r, 'col': out_c,
        'x_m': out_c * CELL_SIZE + CELL_SIZE / 2,
        'y_m': (dem.shape[0] - out_r) * CELL_SIZE - CELL_SIZE / 2,
        'elevation': float(dem[out_r, out_c]),
        'invert': lowest['invert'] - 2.5, 'type': 'outfall',
    }
    nodes.append(outfall)

    lid = len(links) + 1
    links.append({
        'id': f'C{lid:05d}',
        'from_node': lowest['id'], 'to_node': outfall['id'],
        'length': np.sqrt((lowest['x_m']-outfall['x_m'])**2 + (lowest['y_m']-outfall['y_m'])**2),
        'slope': 0.005, 'diameter': 1.0, 'mannings_n': 0.013, 'type': 'main',
    })

    return outfall


def visualize_osm_network(dem, bldg, nodes, links, edges_gdf, out_path):
    """Create comprehensive visualization of OSM-based pipe network."""
    H, W = dem.shape
    extent = [0, W * CELL_SIZE, 0, H * CELL_SIZE]
    # DEM row 0 = north. Use imshow default origin='upper' so raster matches y_m / OSM (north up).
    terrain = np.where(bldg, np.nan, dem)

    fig, axes = plt.subplots(2, 3, figsize=(24, 15))
    fig.suptitle('OSM Road-Based Pipe Network Design — Shenzhen Futian',
                 fontsize=15, fontweight='bold')

    # (1) DEM + OSM roads overlay (same rotate/bbox as pipe extraction)
    ax = axes[0, 0]
    im = ax.imshow(terrain, cmap='terrain', extent=extent, aspect='equal', alpha=0.8)
    draw_osm_roads(ax, edges_gdf, (H, W), style='colored', zorder=4)
    ax.set_title('DEM with OSM Roads (aligned)')
    ax.set_xlabel('Easting (m)'); ax.set_ylabel('Northing (m)')
    plt.colorbar(im, ax=ax, label='Elev (m)', shrink=0.8)
    ax.legend(handles=[
        Line2D([0],[0], color='#8B0000', lw=2.5, label='Motorway'),
        Line2D([0],[0], color='#FF0000', lw=2.0, label='Trunk'),
        Line2D([0],[0], color='#FF6600', lw=1.5, label='Primary'),
        Line2D([0],[0], color='#FF9900', lw=1.2, label='Secondary'),
        Line2D([0],[0], color='#FFCC00', lw=0.9, label='Tertiary'),
    ], loc='lower right', fontsize=7)

    # (2) Buildings + OSM Roads
    ax = axes[0, 1]
    ax.imshow(np.where(bldg, 0.4, 0), cmap=ListedColormap(['none', '#999']),
              extent=extent, aspect='equal', alpha=0.5)
    draw_osm_roads(ax, edges_gdf, (H, W), style='single', zorder=4)
    ax.set_title(f'Building Mask ({bldg.sum():,} cells) + OSM Roads (aligned)')
    ax.set_xlabel('Easting (m)')

    # (3) Manhole placement
    ax = axes[0, 2]
    ax.imshow(np.where(bldg, 0.3, 0), cmap=ListedColormap(['none', '#DDD']),
              extent=extent, aspect='equal', alpha=0.5)
    ax.imshow(terrain, cmap='terrain', extent=extent, aspect='equal', alpha=0.3)
    for node in nodes:
        c = {'main_trunk': 'red', 'junction': '#2196F3', 'outfall': 'green'}.get(node['type'], 'gray')
        s = {'main_trunk': 30, 'junction': 15, 'outfall': 60}.get(node['type'], 10)
        m = {'outfall': '^'}.get(node['type'], 'o')
        ax.scatter(node['x_m'], node['y_m'], c=c, s=s, marker=m,
                  zorder=5, edgecolors='black', linewidth=0.2)
    ax.set_title(f'Manholes: {len(nodes)} Nodes')
    ax.set_xlabel('Easting (m)'); ax.set_ylabel('Northing (m)')

    # (4) Pipe network
    ax = axes[1, 0]
    ax.imshow(terrain, cmap='terrain', extent=extent, aspect='equal', alpha=0.5)
    node_lookup = {n['id']: n for n in nodes}
    for link in links:
        fn = node_lookup[link['from_node']]; tn = node_lookup[link['to_node']]
        color = {'main': '#D32F2F', 'collector': '#FF9800', 'branch': '#2196F3'}.get(link['type'], '#999')
        lw = {'main': 2.5, 'collector': 1.5, 'branch': 0.8}.get(link['type'], 1.0)
        ax.plot([fn['x_m'], tn['x_m']], [fn['y_m'], tn['y_m']],
                color=color, linewidth=lw, alpha=0.8, zorder=3)
    for node in nodes:
        c = {'main_trunk': 'red', 'junction': '#2196F3', 'outfall': 'green'}.get(node['type'], 'gray')
        s = {'main_trunk': 25, 'junction': 10, 'outfall': 50}.get(node['type'], 8)
        m = {'outfall': '^'}.get(node['type'], 'o')
        ax.scatter(node['x_m'], node['y_m'], c=c, s=s, marker=m,
                  zorder=5, edgecolors='black', linewidth=0.2)
    ax.set_title(f'Pipe Network: {len(links)} Pipes')
    ax.set_xlabel('Easting (m)')
    ax.legend(handles=[
        Line2D([0],[0], color='#D32F2F', lw=2.5, label='Main (D=800mm)'),
        Line2D([0],[0], color='#FF9800', lw=1.5, label='Collector (D=600mm)'),
        Line2D([0],[0], marker='^', color='w', markerfacecolor='green', markersize=8, label='Outfall'),
    ], loc='lower right', fontsize=7)

    # (5) OSM roads vs pipe network comparison
    ax = axes[1, 1]
    ax.imshow(terrain, cmap='terrain', extent=extent, aspect='equal', alpha=0.45)
    # Draw OSM roads faint (aligned)
    for _, edge in edges_gdf.iterrows():
        xs, ys = [], []
        for lon, lat in iter_geom_lonlat(edge.geometry):
            r, c = osm_lonlat_to_grid(lon, lat, (H, W))
            if 0 <= r < H and 0 <= c < W:
                x, y = grid_to_plot_xy(r, c, (H, W))
                xs.append(x)
                ys.append(y)
        if len(xs) >= 2:
            ax.plot(xs, ys, color='#AAAAAA', linewidth=0.5, alpha=0.4, zorder=2)
    # Draw pipe network on top
    for link in links:
        fn = node_lookup[link['from_node']]; tn = node_lookup[link['to_node']]
        color = 'red' if link['type'] == 'main' else '#FF9800'
        lw = 2.0 if link['type'] == 'main' else 1.2
        ax.plot([fn['x_m'], tn['x_m']], [fn['y_m'], tn['y_m']],
                color=color, linewidth=lw, alpha=0.85, zorder=4)
    ax.set_title('Pipe Network Overlay on OSM Roads')
    ax.set_xlabel('Easting (m)')

    # (6) Statistics
    ax = axes[1, 2]; ax.axis('off')
    n_main = sum(1 for n in nodes if n['type'] == 'main_trunk')
    n_junc = sum(1 for n in nodes if n['type'] == 'junction')
    n_out = sum(1 for n in nodes if n['type'] == 'outfall')
    n_main_links = sum(1 for l in links if l['type'] == 'main')
    n_col_links = sum(1 for l in links if l['type'] == 'collector')
    total_len = sum(l['length'] for l in links)

    metrics = f"""
    OSM-BASED PIPE NETWORK DESIGN
    =============================

    DATA SOURCE
    OpenStreetMap road network
    Futian District, Shenzhen
    Major roads: motorway/trunk/primary/secondary

    DOMAIN
    {dem.shape[0]*CELL_SIZE/1000:.1f}km x {dem.shape[1]*CELL_SIZE/1000:.1f}km
    {dem.shape[0]}x{dem.shape[1]} cells @ {CELL_SIZE:.0f}m resolution
    Buildings: {bldg.sum():,} cells ({100*bldg.sum()/bldg.size:.0f}%)

    PIPE NETWORK
    Manholes: {len(nodes)} total
      Main trunk: {n_main}
      Junctions: {n_junc}
      Outfalls: {n_out}
    Pipes: {len(links)} total
      Main (D=800mm): {n_main_links}
      Collector (D=600mm): {n_col_links}
    Total length: {total_len:.0f}m

    DESIGN STANDARDS
    Manhole spacing: ~160m on roads
    Pipe slope: follows terrain
    Cover depth: 2.0m minimum
    Manning's n: 0.013 (concrete)
    """.strip()
    ax.text(0.05, 0.5, metrics, transform=ax.transAxes, fontsize=10,
            fontfamily='monospace', verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='#F0F8FF', alpha=0.9))

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {os.path.basename(out_path)}")


def main():
    print("=" * 60)
    print("  OSM Road-Based Pipe Network Design")
    print("=" * 60)

    # Load DEM
    print("\n[1] Loading DEM...")
    dem_full = np.load(DEM_PATH, allow_pickle=True).astype(np.float64)
    # Buildings = cells with elevation >= 49.9 (wall_height ~50m)
    building_mask = dem_full >= 49.9

    H_full, W_full = dem_full.shape
    print(f"  Full DEM: {H_full}x{W_full} cells = {H_full*CELL_SIZE/1000:.1f}km x {W_full*CELL_SIZE/1000:.1f}km")

    # Download OSM roads
    print("\n[2] Getting OSM road data...")
    try:
        nodes_gdf, major_edges, all_significant = download_osm_roads()
    except Exception as e:
        print(f"  OSM download failed: {e}")
        print("  Using cached data...")
        import geopandas as gpd
        major_edges = gpd.read_file(os.path.join(OUT_DIR, 'osm_roads_major.geojson'))
        all_significant = major_edges

    # Extract centerlines
    print("\n[3] Extracting road centerlines...")
    road_segments = extract_road_centerlines(all_significant, dem_full.shape, min_length_m=150)
    print(f"  Total OSM road segments: {len(road_segments)}")

    # Design pipe network
    print("\n[4] Placing manholes along road centerlines...")
    nodes, links = place_manholes_along_roads(road_segments, dem_full, building_mask, spacing_cells=8)

    # Identify main trunk
    print("\n[5] Identifying main trunk...")
    identify_main_trunk(nodes, links)

    # Add outfall
    print("\n[6] Adding outfall...")
    add_outfall(nodes, links, dem_full, building_mask)

    # Summary
    total_len = sum(l['length'] for l in links)
    print(f"\n{'='*60}")
    print(f"  NETWORK SUMMARY")
    print(f"{'='*60}")
    print(f"  Total manholes: {len(nodes)}")
    for t in ['main_trunk', 'junction', 'outfall']:
        cnt = sum(1 for n in nodes if n['type'] == t)
        if cnt:
            print(f"    {t}: {cnt}")
    print(f"  Total pipes: {len(links)}")
    for t in ['main', 'collector']:
        cnt = sum(1 for l in links if l['type'] == t)
        if cnt:
            print(f"    {t}: {cnt}")
    print(f"  Total pipe length: {total_len:.0f}m")

    # Visualize
    print("\n[7] Creating visualization...")
    network = {'nodes': nodes, 'links': links,
               'ns_positions': [], 'ew_positions': [],
               'sub_region': [0, H_full, 0, W_full]}
    visualize_osm_network(dem_full, building_mask, nodes, links, all_significant,
                          os.path.join(OUT_DIR, 'osm_pipe_network.png'))

    # Save
    print("\n[8] Saving network data...")
    np.savez(os.path.join(OUT_DIR, 'osm_pipe_network.npz'),
             nodes=nodes, links=links,
             dem_shape=dem_full.shape)

    print(f"\n  Outputs saved to: {OUT_DIR}")
    print(f"{'='*60}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
