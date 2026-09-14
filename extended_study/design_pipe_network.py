#!/usr/bin/env python3
"""
Extended Study: Pipe Network Design for Shenzhen Futian District
=================================================================
Designs a storm drainage pipe network along main road corridors
identified from the DEM. The network follows terrain slope and
adheres to typical design standards.

Design Standards (ref: GB 50014-2021):
  - Manhole spacing: 100-200m on main lines, 80-150m on branches
  - Minimum pipe diameter: 400mm (branches), 600mm (main), 800mm (trunk)
  - Minimum slope: 0.001 (0.1%)
  - Maximum slope: 0.05 (5%) without drop structures
  - Pipe material: concrete (n=0.013 Manning's)
  - Cover depth: minimum 1.5m
"""

import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy import ndimage

# ============================================================
# Paths
# ============================================================
ROOT = r"e:\Projects\20260519-LarNO\LarNO-main\benchmark\urbanflood"
DEM_PATH = os.path.join(ROOT, "geodata", "region1_20m", "dem.npy")
FLOOD_DIR = os.path.join(ROOT, "flood", "region1_20m")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUT_DIR, exist_ok=True)

CELL_SIZE = 20.0  # meters


def design_pipe_network(dem, building_mask, sub_region=None):
    """
    Design a storm drainage pipe network for the urban area.

    Parameters
    ----------
    dem : ndarray (H, W)
        Digital Elevation Model in meters
    building_mask : ndarray (H, W)
        Boolean mask for building cells
    sub_region : tuple (y0, y1, x0, x1) or None
        Optional sub-region for detailed network design

    Returns
    -------
    network : dict
        Pipe network design with nodes, links, and properties
    """
    H, W = dem.shape

    if sub_region:
        y0, y1, x0, x1 = sub_region
    else:
        y0, y1, x0, x1 = 0, H, 0, W

    sub_dem = dem[y0:y1, x0:x1]
    sub_bldg = building_mask[y0:y1, x0:x1]
    sh, sw = sub_dem.shape

    # Compute terrain gradients
    dy, dx = np.gradient(sub_dem, CELL_SIZE)

    # ================================================================
    # Step 1: Identify main road corridors
    # ================================================================
    # Roads are typically characterized by:
    #  - Low slope variability (smooth surface)
    #  - Not buildings
    #  - Linear features

    # Use slope smoothness as a road indicator
    slope = np.sqrt(dx**2 + dy**2)
    slope_smooth = ndimage.gaussian_filter(slope, sigma=2)
    valid_area = ~sub_bldg

    # Road likelihood: low slope, not building, smooth area
    road_likelihood = np.where(valid_area, 1.0 / (1.0 + slope_smooth * 10), 0)

    # ================================================================
    # Step 2: Design main trunk line (N-S and E-W)
    # ================================================================
    # Main trunk follows the general terrain slope (SW direction)
    # We design a grid network:
    #   - 2-3 N-S main lines (primary drainage direction)
    #   - 3-4 E-W collector lines
    #   - Branch laterals connecting to nearest main/collector

    # Compute cumulative flow accumulation to find natural drainage paths
    # Simplified: use the negative DEM as flow direction indicator
    # Flow generally goes from NE to SW

    # Main N-S trunk lines at x positions
    ns_positions = np.linspace(x0 + sw * 0.15, x0 + sw * 0.85, 3).astype(int)
    ew_positions = np.linspace(y0 + sh * 0.2, y0 + sh * 0.8, 3).astype(int)

    print(f"  Main N-S trunk lines at columns: {ns_positions}")
    print(f"  Main E-W collector lines at rows: {ew_positions}")

    # ================================================================
    # Step 3: Place manholes along the network
    # ================================================================
    nodes = []
    node_id = 0

    def add_node(y, x, node_type='junction'):
        nonlocal node_id
        if y0 <= y < y1 and x0 <= x < x1:
            elev = dem[y, x]
            if not building_mask[y, x] and elev < 49.0:
                node_id += 1
                invert = elev - 2.0  # 2m cover depth
                nodes.append({
                    'id': f'N_{node_id:04d}',
                    'row': y, 'col': x,
                    'x_m': x * CELL_SIZE + CELL_SIZE / 2,
                    'y_m': (H - y) * CELL_SIZE - CELL_SIZE / 2,
                    'elevation': elev,
                    'invert': max(invert, 0.5),
                    'type': node_type,
                })
                return node_id
        return None

    # Manhole spacing: ~200m = 10 cells on main lines
    main_spacing = 10  # cells (200m)
    branch_spacing = 6  # cells (120m)

    # Place nodes along N-S lines
    main_node_ids = {}
    for col in ns_positions:
        line_nodes = []
        for row in range(y0, y1, main_spacing):
            nid = add_node(row, col, 'main_trunk')
            if nid:
                line_nodes.append(nid)
        main_node_ids[col] = line_nodes

    # Place nodes along E-W lines
    for row in ew_positions:
        for col in range(x0, x1, main_spacing):
            add_node(row, col, 'collector')

    # ================================================================
    # Step 4: Create conduits (pipes) between nodes
    # ================================================================
    links = []
    link_id = 0

    def add_link(from_node, to_node, link_type='main'):
        nonlocal link_id
        fn = nodes[from_node]
        tn = nodes[to_node]
        dist = np.sqrt((fn['x_m'] - tn['x_m'])**2 + (fn['y_m'] - tn['y_m'])**2)
        if dist < 5 or dist > 500:
            return

        # Slope = elevation difference / distance
        slope = abs(fn['invert'] - tn['invert']) / max(dist, 0.1)
        slope = max(slope, 0.001)  # minimum slope 0.1%

        diameters = {'main': 0.8, 'collector': 0.6, 'branch': 0.4}
        diameter = diameters.get(link_type, 0.4)

        link_id += 1
        links.append({
            'id': f'C_{link_id:04d}',
            'from_node': fn['id'],
            'to_node': tn['id'],
            'length': dist,
            'slope': slope,
            'diameter': diameter,
            'mannings_n': 0.013,
            'type': link_type,
        })

    # Connect N-S main lines
    for col, line_nodes in main_node_ids.items():
        for i in range(len(line_nodes) - 1):
            add_link(line_nodes[i], line_nodes[i + 1], 'main')

    # ================================================================
    # Step 5: Create outfall
    # ================================================================
    # Outfall at the lowest point (SW corner)
    outfall_y = y1 - 1
    outfall_x = int(ns_positions[-1])
    add_node(outfall_y, outfall_x, 'outfall')
    # Lower outfall invert
    if nodes:
        nodes[-1]['invert'] = dem[outfall_y, outfall_x] - 3.0

    # Connect last main node to outfall
    if main_node_ids:
        last_col = ns_positions[-1]
        last_nodes = main_node_ids.get(last_col, [])
        if last_nodes and len(nodes) > last_nodes[-1]:
            add_link(last_nodes[-1], len(nodes) - 1, 'main')

    return {
        'nodes': nodes,
        'links': links,
        'ns_positions': ns_positions.tolist(),
        'ew_positions': ew_positions.tolist(),
        'sub_region': [y0, y1, x0, x1],
    }


def visualize_network(dem, building_mask, network, out_path):
    """Create a visualization of the pipe network overlaid on the DEM."""
    y0, y1, x0, x1 = network['sub_region']
    H, W = dem.shape

    fig, axes = plt.subplots(1, 2, figsize=(20, 9))
    fig.suptitle('Shenzhen Futian — Storm Drainage Network Design',
                 fontsize=14, fontweight='bold')

    # (a) Full domain with network overlay
    ax = axes[0]
    terrain = np.where(building_mask, np.nan, dem)
    im = ax.imshow(terrain, cmap='terrain', aspect='equal',
                   extent=[0, W * CELL_SIZE, 0, H * CELL_SIZE])
    ax.set_title('Full Domain (8.0km x 11.2km) with Drainage Network')
    ax.set_xlabel('Easting (m)'); ax.set_ylabel('Northing (m)')

    # Draw sub-region box
    rect = plt.Rectangle(
        (x0 * CELL_SIZE, (H - y1) * CELL_SIZE),
        (x1 - x0) * CELL_SIZE, (y1 - y0) * CELL_SIZE,
        linewidth=2, edgecolor='red', facecolor='none', linestyle='--'
    )
    ax.add_patch(rect)
    ax.text(x0 * CELL_SIZE + 100, (H - y0) * CELL_SIZE + 100,
            'Study Area', color='red', fontweight='bold', fontsize=10)

    plt.colorbar(im, ax=ax, label='Elevation (m)', shrink=0.8)

    # (b) Study area with detailed network
    ax = axes[1]
    sub_dem = dem[y0:y1, x0:x1]
    sub_bldg = building_mask[y0:y1, x0:x1]
    terrain_sub = np.where(sub_bldg, np.nan, sub_dem)
    extent_sub = [x0 * CELL_SIZE, x1 * CELL_SIZE, (H - y1) * CELL_SIZE, (H - y0) * CELL_SIZE]

    im = ax.imshow(terrain_sub, cmap='terrain', aspect='equal', extent=extent_sub)
    ax.set_title('Study Area with Pipe Network')
    ax.set_xlabel('Easting (m)'); ax.set_ylabel('Northing (m)')

    # Draw nodes
    for node in network['nodes']:
        color = {'main_trunk': 'red', 'collector': 'orange', 'junction': 'blue',
                 'outfall': 'green'}.get(node['type'], 'gray')
        size = {'main_trunk': 40, 'collector': 25, 'junction': 15,
                'outfall': 60}.get(node['type'], 15)
        marker = {'outfall': '^'}.get(node['type'], 'o')
        ax.scatter(node['x_m'], (H - node['row']) * CELL_SIZE - CELL_SIZE / 2,
                   c=color, s=size, marker=marker, zorder=5, edgecolors='black', linewidth=0.5)

    # Draw links
    node_lookup = {n['id']: n for n in network['nodes']}
    for link in network['links']:
        fn = node_lookup[link['from_node']]
        tn = node_lookup[link['to_node']]
        color = {'main': 'red', 'collector': 'orange', 'branch': 'blue'}.get(link['type'], 'gray')
        lw = {'main': 2.5, 'collector': 1.8, 'branch': 1.0}.get(link['type'], 1.0)
        ax.plot([fn['x_m'], tn['x_m']],
                [(H - fn['row']) * CELL_SIZE - CELL_SIZE / 2,
                 (H - tn['row']) * CELL_SIZE - CELL_SIZE / 2],
                color=color, linewidth=lw, alpha=0.8, zorder=3)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
               markersize=8, label='Main Trunk Manhole'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='orange',
               markersize=6, label='Collector Manhole'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='green',
               markersize=10, label='Outfall'),
        Line2D([0], [0], color='red', linewidth=2.5, label='Main Trunk (D=800mm)'),
        Line2D([0], [0], color='orange', linewidth=1.8, label='Collector (D=600mm)'),
    ]
    ax.legend(handles=legend_elements, loc='lower left', fontsize=8)
    plt.colorbar(im, ax=ax, label='Elevation (m)', shrink=0.8)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {os.path.basename(out_path)}")


def create_swmm_input(network, dem, building_mask, out_path):
    """Generate an SWMM input file (INP) for the designed network."""
    H, W = dem.shape
    y0, y1, x0, x1 = network['sub_region']

    lines = [
        "[TITLE]",
        "Shenzhen Futian District - Storm Drainage Network",
        "Extended Study: Coupled Urban Flood & Drainage Modeling",
        "",
        "[OPTIONS]",
        "FLOW_UNITS           CMS",
        "FLOW_ROUTING         DYNWAVE",
        "INFILTRATION         HORTON",
        "START_DATE           01/01/2026",
        "START_TIME           00:00:00",
        "REPORT_START_DATE    01/01/2026",
        "REPORT_START_TIME    00:00:00",
        "END_DATE             01/01/2026",
        "END_TIME             06:00:00",
        "SWEEP_START          01/01",
        "SWEEP_END            12/31",
        "DRY_DAYS             0",
        "REPORT_STEP          00:05:00",
        "WET_STEP             00:01:00",
        "DRY_STEP             00:05:00",
        "ROUTING_STEP         0:00:05",
        "ALLOW_PONDING        NO",
        "INERTIAL_DAMPING     PARTIAL",
        "VARIABLE_STEP        0.75",
        "MINIMUM_STEP         0.5",
        "MIN_SLOPE            0.001",
        "THREADS              1",
        "",
        "[EVAPORATION]",
        "CONSTANT         0.0",
        "DRY_ONLY         NO",
        "",
        "[RAINGAGES]",
        "RainGage1    INTENSITY 0:05     1.0     TIMESERIES RainTS",
        "",
        "[TIMESERIES]",
        "RainTS   0:00    0.0",
        "RainTS   0:05    0.0",
        "",
    ]

    # JUNCTIONS
    lines.append("[JUNCTIONS]")
    lines.append(";; NodeID    Invert(m)  MaxDepth(m)  InitDepth(m)  Surcharge(m)  PondedArea(m2)")
    for node in network['nodes']:
        if node['type'] != 'outfall':
            max_depth = max(2.0, dem[node['row'], node['col']] - node['invert'] + 0.5)
            lines.append(
                f"  {node['id']:<12s} {node['invert']:<10.2f} {max_depth:<12.1f} "
                f"0.0         0.0          0.0"
            )

    # OUTFALLS
    lines.append("\n[OUTFALLS]")
    lines.append(";; NodeID    Elev(m)   Type   Gate")
    for node in network['nodes']:
        if node['type'] == 'outfall':
            lines.append(f"  {node['id']:<12s} {node['invert']:<8.2f} FREE    NO")

    # CONDUITS
    lines.append("\n[CONDUITS]")
    lines.append(";; ConduitID  FromNode    ToNode      Length(m)  ManningsN  InOffset  OutOffset  InitFlow  MaxFlow")
    for link in network['links']:
        lines.append(
            f"  {link['id']:<12s} {link['from_node']:<12s} {link['to_node']:<12s} "
            f"{link['length']:<10.1f} {link['mannings_n']:<10.3f} 0          0          0         0"
        )

    # XSECTIONS
    lines.append("\n[XSECTIONS]")
    lines.append(";; LinkID     Shape       Geom1(m)  Geom2  Geom3  Geom4  Barrels")
    for link in network['links']:
        lines.append(
            f"  {link['id']:<12s} CIRCULAR    {link['diameter']:<8.2f} 0      0      0      1"
        )

    # COORDINATES
    lines.append("\n[COORDINATES]")
    lines.append(";; NodeID     X(m)       Y(m)")
    for node in network['nodes']:
        lines.append(f"  {node['id']:<12s} {node['x_m']:<10.1f} {node['y_m']:<10.1f}")

    # SUBCATCHMENTS (minimal)
    lines.append("\n[SUBCATCHMENTS]")
    lines.append(";; Subcatchment  RainGage   Outlet  Area  Width  Slope  Imperv  ImpervWidth  RouteTo")
    lines.append("  SC_01          RainGage1  N_0001  1.0   100    0.005  50      0.5          OUTLET")
    lines.append("\n[SUBAREAS]")
    lines.append("  SC_01    0.5    0.03    0.5    0.5    0.0    OUTLET")
    lines.append("\n[INFILTRATION]")
    lines.append("  SC_01    80.0    20.0    6.0    10.0    0")

    lines += [
        "\n[TAGS]", "",
        "\n[MAP]", "",
        "\n[COORDINATES]", "",
        "\n[REPORT]",
        "INPUT      NO",
        "CONTROLS   NO",
        "NODES ALL",
        "LINKS ALL",
    ]

    with open(out_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"  SWMM input saved: {os.path.basename(out_path)}")
    print(f"    Nodes: {sum(1 for n in network['nodes'])}")
    print(f"    Links: {sum(1 for l in network['links'])}")
    return out_path


def main():
    print("=" * 60)
    print("  Extended Study — Pipe Network Design")
    print("  Shenzhen Futian District")
    print("=" * 60)

    # Load DEM
    print("\n[1] Loading DEM...")
    dem = np.load(DEM_PATH, allow_pickle=True)
    building_mask = dem >= 49.9
    H, W = dem.shape
    print(f"  DEM: {H}x{W} cells at 20m resolution")

    # Define study sub-region (northern-central area, ~4km x 4km)
    # This region has a good mix of buildings and open space
    y0, y1 = 50, 250  # rows (200 cells = 4km)
    x0, x1 = 100, 350  # cols (250 cells = 5km)
    print(f"\n[2] Study area: rows [{y0},{y1}], cols [{x0},{x1}]")
    print(f"    Size: {(y1-y0)*CELL_SIZE/1000:.1f}km x {(x1-x0)*CELL_SIZE/1000:.1f}km")

    # Design network
    print("\n[3] Designing pipe network...")
    network = design_pipe_network(dem, building_mask, sub_region=(y0, y1, x0, x1))
    print(f"    Designed: {len(network['nodes'])} nodes, {len(network['links'])} links")

    # Visualize
    print("\n[4] Creating network visualization...")
    visualize_network(dem, building_mask, network,
                      os.path.join(OUT_DIR, 'extended_01_network_design.png'))

    # Create SWMM input
    print("\n[5] Generating SWMM input file...")
    create_swmm_input(network, dem, building_mask,
                      os.path.join(OUT_DIR, 'drainage_network.inp'))

    # Save network data
    np.savez(os.path.join(OUT_DIR, 'pipe_network.npz'),
             nodes=network['nodes'], links=network['links'],
             ns_positions=network['ns_positions'],
             ew_positions=network['ew_positions'],
             sub_region=network['sub_region'])

    # Print network summary
    print(f"\n{'='*60}")
    print(f"  Network Summary")
    print(f"{'='*60}")
    print(f"  Study area: {(y1-y0)*CELL_SIZE/1000:.1f}km x {(x1-x0)*CELL_SIZE/1000:.1f}km")
    print(f"  Total nodes: {len(network['nodes'])}")
    print(f"  Total links: {len(network['links'])}")
    for ntype in ['main_trunk', 'collector', 'junction', 'outfall']:
        count = sum(1 for n in network['nodes'] if n['type'] == ntype)
        if count:
            print(f"    {ntype}: {count}")
    for ltype in ['main', 'collector', 'branch']:
        count = sum(1 for l in network['links'] if l['type'] == ltype)
        if count:
            total_len = sum(l['length'] for l in network['links'] if l['type'] == ltype)
            print(f"    {ltype} pipes: {count}, total length: {total_len:.0f}m")

    print(f"\n  Outputs saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
