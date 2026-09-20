#!/usr/bin/env python3
"""
Extract Road Network from Building Gaps in DEM
===============================================
Since the DEM has no CRS/projection metadata, we extract the road
network from the spatial structure of the data itself:

1. Buildings are marked with wall_height=50m in the DEM
2. The spaces BETWEEN buildings are potential roads
3. Use morphological skeletonization to extract road centerlines
4. Filter by slope to exclude mountain/hill areas
5. Build a graph of connected road segments for pipe routing
"""

import os, sys
import numpy as np
from scipy import ndimage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from skimage import morphology as skimorph
import warnings
warnings.filterwarnings('ignore')

DEM_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "LarNO-main", "benchmark", "urbanflood", "geodata", "region1_20m", "dem.npy")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUT_DIR, exist_ok=True)

CELL_SIZE = 20.0  # m


def load_and_preprocess():
    """Load DEM and extract building mask, terrain features."""
    dem = np.load(DEM_PATH, allow_pickle=True).astype(np.float64)
    H, W = dem.shape

    # Building mask: wall_height = 50m
    building_mask = dem >= 49.9

    # Valid terrain (not building)
    valid = ~building_mask
    valid_dem = np.where(building_mask, np.nan, dem)

    # Compute slope
    dy, dx = np.gradient(dem, CELL_SIZE)
    slope = np.sqrt(dx**2 + dy**2)

    print(f"DEM: {H}x{W} cells @ 20m = {H*CELL_SIZE/1000:.1f}km x {W*CELL_SIZE/1000:.1f}km")
    print(f"Buildings: {building_mask.sum():,} cells ({100*building_mask.sum()/building_mask.size:.1f}%)")
    print(f"Valid terrain: {(~building_mask).sum():,} cells")
    print(f"Slope: mean={np.nanmean(slope[~building_mask]):.4f}, median={np.nanmedian(slope[~building_mask]):.4f}")

    return dem, building_mask, slope


def extract_road_candidates(building_mask, slope, dem):
    """
    Extract candidate road areas using distance transform.

    Strategy (revised for dense urban areas):
    1. Compute distance transform of building mask
       -> Each cell gets distance to nearest building
    2. The local maxima of this distance field are road centerlines
       (furthest from buildings = middle of road corridors)
    3. Threshold the distance to get wide-enough corridors
    4. Filter by slope to exclude mountain roads
    """
    H, W = building_mask.shape

    # Distance to nearest building
    # Invert: buildings=0 (obstacle), open=1 (free space)
    free_space = (~building_mask).astype(np.uint8)
    dist_to_bldg = ndimage.distance_transform_edt(free_space)  # cells

    # Road centerline candidates: local maxima of distance field
    # Use morphological dilation + comparison to find ridges
    footprint = np.ones((3, 3))
    dilated = ndimage.maximum_filter(dist_to_bldg, footprint=footprint)
    ridges = (dist_to_bldg == dilated) & free_space.astype(bool)

    # Filter: roads should have at least 1 cell distance from buildings
    # (i.e., road corridor is at least 2 cells = 40m wide)
    wide_enough = dist_to_bldg >= 1.0

    # Slope filter
    gentle_slope = slope < 0.3

    # Combine
    road_centerlines = ridges & wide_enough & gentle_slope

    # Road area: all open ground near centerlines
    road_area = free_space.astype(bool) & gentle_slope

    print(f"\nRoad extraction (distance transform method):")
    print(f"  Open ground cells: {free_space.sum():,}")
    print(f"  Distance to building: min={dist_to_bldg[free_space.astype(bool)].min():.1f}, "
          f"max={dist_to_bldg[free_space.astype(bool)].max():.1f}, "
          f"mean={dist_to_bldg[free_space.astype(bool)].mean():.1f}")
    print(f"  Ridge points (centerline candidates): {ridges.sum():,}")
    print(f"  After width filter: { (ridges & wide_enough).sum():,}")
    print(f"  After slope filter: {road_centerlines.sum():,}")

    return road_area, road_centerlines, dist_to_bldg


def trace_road_segments(centerline_mask, dist_to_bldg):
    """
    Extract connected road segments from centerline mask.

    Uses the centerline mask directly (already skeleton-like from ridge detection).
    Connects nearby centerline points to form segments.
    """
    H, W = centerline_mask.shape

    # Thin the centerlines to single-pixel width
    if centerline_mask.sum() > 0:
        skeleton = skimorph.skeletonize(centerline_mask)
    else:
        skeleton = np.zeros((H, W), dtype=bool)

    # Remove short isolated branches
    # First label connected components in skeleton
    labeled_skel, n_comp = ndimage.label(skeleton)
    comp_sizes = ndimage.sum(skeleton, labeled_skel, range(n_comp + 1))
    keep_comp = comp_sizes >= 5  # Keep segments with >=5 pixels
    skeleton = keep_comp[labeled_skel]

    print(f"  Skeleton pixels: {skeleton.sum():,} (from {centerline_mask.sum():,} centerline points)")
    print(f"  Connected components: {n_comp} -> {np.sum(keep_comp)-1} kept (>=5px)")

    # For very fragmented skeletons, use morphological closing to connect nearby pieces
    if skeleton.sum() < 100:
        print("  Skeleton too fragmented, applying morphological closing...")
        skeleton = ndimage.binary_closing(skeleton, structure=np.ones((5, 5)), iterations=2)

    return skeleton


def build_road_graph(skeleton):
    """
    Build a graph from skeleton pixels: identify junctions and segments.

    Returns:
        junctions: list of (row, col) junction points
        segments: list of lists of (row, col) segment points
    """
    H, W = skeleton.shape

    # Count neighbors for each skeleton pixel
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
    neighbor_count = ndimage.convolve(skeleton.astype(np.uint8), kernel,
                                       mode='constant', cval=0)

    # Junctions: pixels with >2 neighbors
    junctions_mask = (neighbor_count > 2) & skeleton
    # Endpoints: pixels with exactly 1 neighbor
    endpoints_mask = (neighbor_count == 1) & skeleton
    # Regular points: pixels with exactly 2 neighbors
    regular_mask = (neighbor_count == 2) & skeleton

    # Get junction coordinates
    jy, jx = np.where(junctions_mask)
    junctions = list(zip(jy, jx))

    print(f"  Junctions: {len(junctions)}, Endpoints: {np.sum(endpoints_mask)}")

    # Trace segments between junctions and endpoints
    all_special = junctions_mask | endpoints_mask
    visited = np.zeros((H, W), dtype=bool)
    segments = []

    # Start tracing from each junction
    for j_row, j_col in junctions:
        # Check 8 neighbors
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = j_row + dr, j_col + dc
                if 0 <= nr < H and 0 <= nc < W:
                    if skeleton[nr, nc] and not visited[nr, nc] and not all_special[nr, nc]:
                        # Trace this segment
                        seg = trace_segment(skeleton, visited, all_special, nr, nc, H, W)
                        if len(seg) >= 3:  # minimum segment length
                            segments.append([(j_row, j_col)] + seg)

    # Also trace from endpoints
    ey, ex = np.where(endpoints_mask)
    for er, ec in zip(ey, ex):
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = er + dr, ec + dc
                if 0 <= nr < H and 0 <= nc < W:
                    if skeleton[nr, nc] and not visited[nr, nc] and not all_special[nr, nc]:
                        seg = trace_segment(skeleton, visited, all_special, nr, nc, H, W)
                        if len(seg) >= 3:
                            segments.append([(er, ec)] + seg)

    print(f"  Road segments: {len(segments)}")

    return junctions, segments


def trace_segment(skeleton, visited, special, r, c, H, W):
    """Trace a road segment from a starting point until hitting a junction or endpoint."""
    segment = []
    cr, cc = r, c
    prev_r, prev_c = -1, -1

    max_steps = 500  # safety limit
    for _ in range(max_steps):
        if cr < 0 or cr >= H or cc < 0 or cc >= W:
            break
        if not skeleton[cr, cc] or visited[cr, cc]:
            break

        segment.append((cr, cc))
        visited[cr, cc] = True

        if special[cr, cc]:
            break  # reached another junction/endpoint

        # Find next pixel (prefer straight ahead)
        neighbors = []
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < H and 0 <= nc < W:
                    if skeleton[nr, nc] and not visited[nr, nc]:
                        # Prefer going straight
                        if (cr - prev_r) == dr and (cc - prev_c) == dc:
                            weight = 0
                        else:
                            weight = 1
                        neighbors.append((weight, nr, nc))

        if not neighbors:
            break

        neighbors.sort()
        _, next_r, next_c = neighbors[0]
        prev_r, prev_c = cr, cc
        cr, cc = next_r, next_c

    return segment


def design_pipe_network_from_roads(segments, dem, building_mask):
    """
    Design pipe network nodes and links along road centerlines.

    Strategy:
    - Place manholes every ~150m along road segments
    - Connect manholes within the same segment
    - Connect segments that share junctions
    - Pipe slope follows road (terrain) slope
    """
    nodes = []
    links = []
    nid, lid = 0, 0
    node_positions = {}  # (row, col) -> node_id

    spacing = 7  # cells between manholes (~140m at 20m resolution)

    # Place nodes along each segment
    for seg in segments:
        if len(seg) < spacing:
            continue

        seg_nodes = []
        for i in range(0, len(seg), spacing):
            r, c = seg[min(i, len(seg) - 1)]
            if building_mask[r, c]:
                # Shift to nearest non-building cell
                for dr in range(1, 5):
                    for dc_off in range(-dr, dr + 1):
                        nc = c + dc_off
                        nr = r + dr
                        if 0 <= nr < dem.shape[0] and 0 <= nc < dem.shape[1]:
                            if not building_mask[nr, nc]:
                                r, c = nr, nc
                                break
                    else:
                        continue
                    break

            if building_mask[r, c]:
                continue

            pos_key = (r, c)
            if pos_key in node_positions:
                seg_nodes.append(node_positions[pos_key])
            else:
                nid += 1
                elev = float(dem[r, c])
                invert = max(elev - 2.0, 0.3)
                node_id = f'N{nid:05d}'
                node_positions[pos_key] = nid - 1  # 0-indexed into nodes list
                nodes.append({
                    'id': node_id, 'row': r, 'col': c,
                    'x_m': c * CELL_SIZE + CELL_SIZE / 2,
                    'y_m': (dem.shape[0] - r) * CELL_SIZE - CELL_SIZE / 2,
                    'elevation': elev, 'invert': invert,
                    'type': 'junction',
                })
                seg_nodes.append(nid - 1)

        # Link consecutive nodes along segment
        for i in range(len(seg_nodes) - 1):
            fn = nodes[seg_nodes[i]]
            tn = nodes[seg_nodes[i + 1]]
            dist = np.sqrt((fn['x_m'] - tn['x_m'])**2 + (fn['y_m'] - tn['y_m'])**2)
            if dist < 10 or dist > 400:
                continue
            slope_pipe = max(abs(fn['invert'] - tn['invert']) / max(dist, 0.1), 0.001)
            lid += 1
            links.append({
                'id': f'C{lid:05d}',
                'from_node': fn['id'], 'to_node': tn['id'],
                'length': dist, 'slope': slope_pipe,
                'diameter': 0.6, 'mannings_n': 0.013, 'type': 'collector',
            })

    # Assign main trunk: pick the longest connected chain
    if nodes:
        # Build adjacency
        adj = {i: set() for i in range(len(nodes))}
        for link in links:
            fi = next(i for i, n in enumerate(nodes) if n['id'] == link['from_node'])
            ti = next(i for i, n in enumerate(nodes) if n['id'] == link['to_node'])
            adj[fi].add(ti)
            adj[ti].add(fi)

        # Simple BFS to find largest component
        visited = set()
        largest_component = []
        for start in range(len(nodes)):
            if start in visited:
                continue
            component = []
            queue = [start]
            visited.add(start)
            while queue:
                v = queue.pop(0)
                component.append(v)
                for neighbor in adj[v]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            if len(component) > len(largest_component):
                largest_component = component

        # Mark nodes in largest component as main trunk candidates
        main_component = set(largest_component)

        # Find the longest path in the component (diameter) as main trunk
        if len(main_component) > 5:
            for ni in largest_component[:max(1, len(largest_component)//4)]:
                nodes[ni]['type'] = 'main_trunk'

            # Update pipe types
            for link in links:
                fi = next(i for i, n in enumerate(nodes) if n['id'] == link['from_node'])
                ti = next(i for i, n in enumerate(nodes) if n['id'] == link['to_node'])
                if fi in main_component and ti in main_component:
                    link['type'] = 'main'
                    link['diameter'] = 0.8

    # Add outfall at lowest node
    if nodes:
        valid_nodes = [n for n in nodes if 'DEM' not in str(n)]
        lowest = min(nodes, key=lambda n: n['invert'])
        nid += 1
        outfall_row = min(lowest['row'] + 3, dem.shape[0] - 2)
        outfall_col = min(lowest['col'] + 2, dem.shape[1] - 2)
        if building_mask[outfall_row, outfall_col]:
            outfall_col = lowest['col']

        outfall = {
            'id': f'N{nid:05d}', 'row': outfall_row, 'col': outfall_col,
            'x_m': outfall_col * CELL_SIZE + CELL_SIZE / 2,
            'y_m': (dem.shape[0] - outfall_row) * CELL_SIZE - CELL_SIZE / 2,
            'elevation': float(dem[outfall_row, outfall_col]),
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

    print(f"\nPipe Network from Road Extraction:")
    print(f"  Nodes: {len(nodes)}")
    for ntype in ['main_trunk', 'junction', 'outfall', 'collector']:
        cnt = sum(1 for n in nodes if n['type'] == ntype)
        if cnt:
            print(f"    {ntype}: {cnt}")
    print(f"  Links: {len(links)}")
    total_len = sum(l['length'] for l in links)
    print(f"  Total length: {total_len:.0f}m")

    return {'nodes': nodes, 'links': links, 'ns_positions': [], 'ew_positions': [],
            'sub_region': [0, dem.shape[0], 0, dem.shape[1]]}


def _fallback_grid_network(dem, bldg, dist_to_bldg):
    """Fallback: grid network using distance-to-building as road indicator."""
    nodes, links = [], []
    nid, lid = 0, 0
    H, W = dem.shape

    # Sample points every N cells in areas far from buildings
    spacing = 8
    for r in range(spacing, H - spacing, spacing):
        for c in range(spacing, W - spacing, spacing):
            if not bldg[r, c] and dist_to_bldg[r, c] >= 1:
                nid += 1
                elev = float(dem[r, c])
                nodes.append({
                    'id': f'N{nid:05d}', 'row': r, 'col': c,
                    'x_m': c * CELL_SIZE + CELL_SIZE / 2,
                    'y_m': (H - r) * CELL_SIZE - CELL_SIZE / 2,
                    'elevation': elev, 'invert': max(elev - 2.0, 0.3),
                    'type': 'junction',
                })

    # Connect nearby nodes
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            ni, nj = nodes[i], nodes[j]
            dist = np.sqrt((ni['x_m'] - nj['x_m'])**2 + (ni['y_m'] - nj['y_m'])**2)
            if dist < 300:
                mid_r = (ni['row'] + nj['row']) // 2
                mid_c = (ni['col'] + nj['col']) // 2
                if not bldg[mid_r, mid_c]:
                    slope_pipe = max(abs(ni['invert'] - nj['invert']) / max(dist, 0.1), 0.001)
                    lid += 1
                    links.append({
                        'id': f'C{lid:05d}',
                        'from_node': ni['id'], 'to_node': nj['id'],
                        'length': dist, 'slope': slope_pipe,
                        'diameter': 0.6, 'mannings_n': 0.013, 'type': 'collector',
                    })

    # Outfall
    if nodes:
        lowest = min(nodes, key=lambda n: n['invert'])
        nid += 1
        nodes.append({
            'id': f'N{nid:05d}', 'row': min(lowest['row'] + 3, H - 2),
            'col': min(lowest['col'] + 2, W - 2),
            'x_m': (min(lowest['col'] + 2, W - 2)) * CELL_SIZE + CELL_SIZE / 2,
            'y_m': (H - min(lowest['row'] + 3, H - 2)) * CELL_SIZE - CELL_SIZE / 2,
            'elevation': float(dem[min(lowest['row'] + 3, H - 2), min(lowest['col'] + 2, W - 2)]),
            'invert': lowest['invert'] - 2.5, 'type': 'outfall',
        })
        lid += 1
        links.append({
            'id': f'C{lid:05d}',
            'from_node': lowest['id'], 'to_node': nodes[-1]['id'],
            'length': 100, 'slope': 0.005, 'diameter': 1.0,
            'mannings_n': 0.013, 'type': 'main',
        })

    return {'nodes': nodes, 'links': links, 'ns_positions': [], 'ew_positions': [],
            'sub_region': [0, H, 0, W]}


def visualize_road_extraction(dem, building_mask, slope, road_area, skeleton,
                                junctions, segments, network, out_path):
    """Comprehensive visualization of the road extraction and pipe network."""
    H, W = dem.shape
    extent = [0, W * CELL_SIZE, 0, H * CELL_SIZE]

    fig, axes = plt.subplots(2, 3, figsize=(22, 14))
    fig.suptitle('Road Network Extraction from Building Gaps — Shenzhen Futian',
                 fontsize=15, fontweight='bold')

    # (1) DEM with buildings
    ax = axes[0, 0]
    terrain = np.where(building_mask, np.nan, dem)
    im = ax.imshow(terrain, cmap='terrain', extent=extent, aspect='equal', origin='lower')
    ax.set_title('DEM (Buildings Masked)')
    plt.colorbar(im, ax=ax, label='Elev (m)', shrink=0.8)

    # (2) Building mask + open ground
    ax = axes[0, 1]
    open_ground = ~building_mask
    overlay = np.zeros((H, W, 4))
    overlay[building_mask] = [0.55, 0, 0, 0.85]  # Dark red: buildings
    overlay[open_ground] = [0.9, 0.9, 0.85, 0.6]  # Beige: open ground
    ax.imshow(overlay.transpose(1, 0, 2), extent=extent, aspect='equal', origin='lower')
    ax.set_title(f'Building Mask ({building_mask.sum():,} cells)\nOpen Ground = Potential Roads')
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color='darkred', label='Buildings (50m wall)'),
        Patch(color='beige', label='Open Ground'),
    ], loc='lower right')

    # (3) Road candidates (open ground with slope filter)
    ax = axes[0, 2]
    ax.imshow(road_area, cmap='Greens', extent=extent, aspect='equal', origin='lower', alpha=0.7)
    ax.imshow(np.where(building_mask, 0.5, 0), cmap=ListedColormap(['none', '#888']),
              extent=extent, aspect='equal', origin='lower', alpha=0.4)
    ax.set_title(f'Road Candidates ({road_area.sum():,} cells)')
    ax.set_xlabel('Easting (m)')

    # (4) Skeleton (road centerlines)
    ax = axes[1, 0]
    ax.imshow(road_area, cmap=ListedColormap(['#F0F0F0', '#E0E0E0']),
              extent=extent, aspect='equal', origin='lower')
    sy, sx = np.where(skeleton)
    ax.scatter(sx * CELL_SIZE + CELL_SIZE/2, (H - 1 - sy) * CELL_SIZE + CELL_SIZE/2,
              c='red', s=1, alpha=0.5)
    # Junctions
    jy, jx = zip(*junctions) if junctions else ([], [])
    ax.scatter([x * CELL_SIZE + CELL_SIZE/2 for x in jx],
              [(H - 1 - y) * CELL_SIZE + CELL_SIZE/2 for y in jy],
              c='blue', s=30, marker='s', zorder=5)
    ax.set_title(f'Road Centerlines ({skeleton.sum():,} pixels) + {len(junctions)} Junctions')
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=4, label='Skeleton'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='blue', markersize=8, label='Junction'),
    ], loc='lower right')

    # (5) Road segments (traced)
    ax = axes[1, 1]
    ax.imshow(road_area, cmap=ListedColormap(['#F5F5F5', '#E8E8E8']),
              extent=extent, aspect='equal', origin='lower')
    colors_seg = plt.cm.tab20(np.linspace(0, 1, max(len(segments), 1)))
    for i, seg in enumerate(segments):
        pts = np.array(seg)
        ax.plot(pts[:, 1] * CELL_SIZE + CELL_SIZE/2,
                (H - 1 - pts[:, 0]) * CELL_SIZE + CELL_SIZE/2,
                color=colors_seg[i % 20], linewidth=2, alpha=0.8)
    ax.set_title(f'Traced Segments ({len(segments)} segments)')

    # (6) Final pipe network
    ax = axes[1, 2]
    ax.imshow(terrain, cmap='terrain', extent=extent, aspect='equal', origin='lower', alpha=0.6)
    node_lookup = {n['id']: n for n in network['nodes']}
    for link in network['links']:
        fn = node_lookup[link['from_node']]
        tn = node_lookup[link['to_node']]
        color = 'red' if link['type'] == 'main' else 'orange'
        lw = 2.5 if link['type'] == 'main' else 1.5
        ax.plot([fn['x_m'], tn['x_m']], [fn['y_m'], tn['y_m']],
                color=color, linewidth=lw, alpha=0.8)
    for node in network['nodes']:
        c = {'main_trunk': 'red', 'junction': 'orange', 'outfall': 'green'}.get(node['type'], 'gray')
        s = {'main_trunk': 30, 'junction': 15, 'outfall': 50}.get(node['type'], 10)
        m = {'outfall': '^'}.get(node['type'], 'o')
        ax.scatter(node['x_m'], node['y_m'], c=c, s=s, marker=m, zorder=5, edgecolors='black', linewidth=0.3)
    ax.set_title(f'Pipe Network: {len(network["nodes"])} Nodes, {len(network["links"])} Links')
    ax.set_xlabel('Easting (m)')
    ax.legend(handles=[
        Line2D([0], [0], color='red', lw=2.5, label='Main Trunk (D=800mm)'),
        Line2D([0], [0], color='orange', lw=1.5, label='Collector (D=600mm)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=6, label='Manhole'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='green', markersize=8, label='Outfall'),
    ], loc='lower right')

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved: {os.path.basename(out_path)}")


def main():
    print("=" * 60)
    print("  Road Network Extraction for Pipe Network Design")
    print("=" * 60)

    # Load data
    print("\n[1] Loading DEM and extracting features...")
    dem, building_mask, slope = load_and_preprocess()

    # For the sub-region used in extended study
    y0, y1 = 90, 190
    x0, x1 = 160, 260
    dem_sub = dem[y0:y1, x0:x1]
    bldg_sub = building_mask[y0:y1, x0:x1]
    slope_sub = slope[y0:y1, x0:x1]

    H_sub, W_sub = dem_sub.shape
    print(f"\n  Study sub-region: {H_sub*CELL_SIZE/1000:.1f}km x {W_sub*CELL_SIZE/1000:.1f}km")

    # Extract roads using distance transform
    print("\n[2] Extracting road candidates (distance transform)...")
    road_area, centerlines, dist_to_bldg = extract_road_candidates(bldg_sub, slope_sub, dem_sub)

    # Trace segments from centerlines
    print("\n[3] Tracing road segments...")
    skeleton = trace_road_segments(centerlines, dist_to_bldg)

    # Build graph
    print("\n[4] Building road graph from skeleton...")
    junctions, segments = build_road_graph(skeleton)

    # Design pipe network
    print("\n[5] Designing pipe network from road graph...")
    if len(segments) == 0:
        print("  WARNING: No road segments extracted!")
        print("  Falling back to grid-based network from building gaps...")
        network = _fallback_grid_network(dem_sub, bldg_sub, dist_to_bldg)
    else:
        network = design_pipe_network_from_roads(segments, dem_sub, bldg_sub)

    # Visualize
    print("\n[6] Creating visualization...")
    visualize_road_extraction(
        dem_sub, bldg_sub, slope_sub, road_area, skeleton,
        junctions, segments, network,
        os.path.join(OUT_DIR, 'road_extraction_network.png')
    )

    # Also do the full domain (for context)
    print("\n[7] Creating full-domain context map...")
    fig, ax = plt.subplots(figsize=(14, 10))
    terrain_full = np.where(building_mask, np.nan, dem)
    ax.imshow(terrain_full, cmap='terrain',
              extent=[0, dem.shape[1]*CELL_SIZE, 0, dem.shape[0]*CELL_SIZE],
              aspect='equal', origin='lower', alpha=0.8)
    # Draw study area rectangle
    from matplotlib.patches import Rectangle
    rect = Rectangle(
        (x0*CELL_SIZE, y0*CELL_SIZE),
        W_sub*CELL_SIZE, H_sub*CELL_SIZE,
        linewidth=2, edgecolor='red', facecolor='none', linestyle='--'
    )
    ax.add_patch(rect)
    ax.text(x0*CELL_SIZE + 100, (y0 + H_sub)*CELL_SIZE - 100,
            'Study Area', color='red', fontweight='bold', fontsize=12)
    ax.set_title('Shenzhen Futian — Full Domain (Study Area in Red)')
    ax.set_xlabel('Easting (m)'); ax.set_ylabel('Northing (m)')
    plt.colorbar(ax.images[0], ax=ax, label='Elevation (m)')
    fig.savefig(os.path.join(OUT_DIR, 'full_domain_context.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Save network
    np.savez(os.path.join(OUT_DIR, 'road_based_network.npz'),
             nodes=network['nodes'], links=network['links'],
             skeleton=skeleton, road_area=road_area,
             junctions=junctions, segments=np.array(segments, dtype=object),
             sub_region=[y0, y1, x0, x1])

    print(f"\n{'='*60}")
    print(f"  Road-based pipe network design complete")
    print(f"  Output: {OUT_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
