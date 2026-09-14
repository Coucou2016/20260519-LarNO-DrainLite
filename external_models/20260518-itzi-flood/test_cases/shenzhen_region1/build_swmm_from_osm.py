#!/usr/bin/env python3
"""
Build SWMM input from OSM pipe network — format matches urban_drainage/create_swmm_network.py
(verified working with pyswmm + ITZI coupling).
"""
import os
import numpy as np

CASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(CASE_DIR, "input_data")
NET_PATH = os.path.join(INPUT_DIR, "networks", "osm_merged_network.npz")
DEM_PATH = os.path.join(INPUT_DIR, "geodata", "region1_20m", "dem.npy")
OUT_FULL = os.path.join(INPUT_DIR, "networks", "swmm_osm_full.inp")
OUT_SUB = os.path.join(INPUT_DIR, "networks", "swmm_osm_sub.inp")

CELL = 20.0
SIM_HOURS = 6
SUB_SLICE = (slice(80, 280), slice(120, 400))


def filter_network(nodes, links, dem, bldg, region_slice=None, main_trunk_only=False):
    """Filter nodes/links; optionally clip to sub-region."""
    H, W = dem.shape
    valid_nodes = []
    for n in nodes:
        if main_trunk_only and n.get('type') not in ('main_trunk', 'outfall'):
            continue
        r, c = n['row'], n['col']
        if region_slice:
            ys, xs = region_slice
            if r < ys.start or r >= ys.stop or c < xs.start or c >= xs.stop:
                continue
        if 0 <= r < H and 0 <= c < W and not bldg[r, c]:
            elev = float(dem[r, c])
            inv = float(n.get('invert', elev - 2.0))
            inv = max(min(inv, elev - 0.5), 0.5)
            valid_nodes.append({
                **n,
                'invert': inv,
                'x_m': float(n.get('x_m', c * CELL + CELL / 2)),
                'y_m': float(n.get('y_m', (H - r) * CELL - CELL / 2)),
            })

    ids = {n['id'] for n in valid_nodes}
    valid_links = []
    for l in links:
        if l['from_node'] not in ids or l['to_node'] not in ids:
            continue
        if float(l['length']) < 1.0 or float(l.get('diameter', 0.6)) < 0.2:
            continue
        valid_links.append(l)

    # Ensure at least one outfall
    has_outfall = any(n.get('type') == 'outfall' for n in valid_nodes)
    if not has_outfall and valid_nodes:
        lowest = min(valid_nodes, key=lambda x: x['invert'])
        valid_nodes.append({
            'id': 'Outfall_01',
            'row': lowest['row'], 'col': lowest['col'],
            'x_m': lowest['x_m'], 'y_m': lowest['y_m'] - 50,
            'invert': max(lowest['invert'] - 1.0, 0.1),
            'type': 'outfall',
        })
        ids.add('Outfall_01')

    return valid_nodes, valid_links


def subsample_for_coupling(nodes, links, min_spacing_m=400):
    """Reduce nodes and rebuild trunk links along N-S/E-W corridors."""
    from collections import defaultdict

    junctions = [n for n in nodes if n.get('type') != 'outfall']
    outfalls = [n for n in nodes if n.get('type') == 'outfall']

    kept = []
    kept_pos = []
    for n in sorted(junctions, key=lambda x: (x['col'], x['row'])):
        pos = (n['row'], n['col'])
        if any(np.hypot((pos[0] - p[0]) * CELL, (pos[1] - p[1]) * CELL) < min_spacing_m
               for p in kept_pos):
            continue
        kept.append(n)
        kept_pos.append(pos)

    # Rebuild links along columns and rows
    kept_links = []
    lid = 0
    by_col = defaultdict(list)
    by_row = defaultdict(list)
    for n in kept:
        by_col[n['col']].append(n)
        by_row[n['row']].append(n)

    def add_chain(nlist, link_type='main'):
        nonlocal lid
        nlist.sort(key=lambda x: x['row'])
        for i in range(len(nlist) - 1):
            a, b = nlist[i], nlist[i + 1]
            dist = float(np.hypot((a['row'] - b['row']) * CELL, (a['col'] - b['col']) * CELL))
            if dist < 5:
                continue
            slope = max(abs(a['invert'] - b['invert']) / dist, 0.001)
            lid += 1
            kept_links.append({
                'id': f'C_{lid:05d}', 'from_node': a['id'], 'to_node': b['id'],
                'length': dist, 'slope': slope,
                'diameter': 0.8 if link_type == 'main' else 0.6,
                'mannings_n': 0.013, 'type': link_type,
            })

    for col, nlist in by_col.items():
        if len(nlist) >= 2:
            add_chain(nlist, 'main')
    for row, nlist in by_row.items():
        if len(nlist) >= 2:
            add_chain(nlist, 'collector')

    # Outfall at lowest invert node
    if not outfalls and kept:
        lowest = min(kept, key=lambda x: x['invert'])
        outfalls = [{
            'id': 'Outfall_01', 'row': lowest['row'], 'col': lowest['col'],
            'x_m': lowest['x_m'], 'y_m': lowest['y_m'] - 30,
            'invert': max(lowest['invert'] - 1.0, 0.1), 'type': 'outfall',
        }]
        lid += 1
        dist = 30.0
        kept_links.append({
            'id': f'C_{lid:05d}', 'from_node': lowest['id'], 'to_node': 'Outfall_01',
            'length': dist, 'slope': 0.005, 'diameter': 1.0,
            'mannings_n': 0.013, 'type': 'main',
        })
    kept.extend(outfalls)

    print(f"  Subsampled: {len(nodes)} -> {len(kept)} nodes, "
          f"{len(links)} -> {len(kept_links)} rebuilt links")
    return kept, kept_links


def write_swmm_inp(nodes, links, out_path, title="Shenzhen OSM Drainage"):
    """Write SWMM inp using proven urban_drainage format."""
    junctions = [n for n in nodes if n.get('type') != 'outfall']
    outfalls = [n for n in nodes if n.get('type') == 'outfall']
    if not junctions:
        raise ValueError("No junction nodes")

    lines = [
        "[TITLE]",
        title,
        "",
        "[OPTIONS]",
        "FLOW_UNITS           CMS",
        "FLOW_ROUTING         DYNWAVE",
        "START_DATE           01/01/2026",
        "START_TIME           00:00:00",
        "REPORT_START_DATE    01/01/2026",
        "REPORT_START_TIME    00:00:00",
        "END_DATE             01/01/2026",
        f"END_TIME             {SIM_HOURS:02d}:00:00",
        "SWEEP_START          01/01",
        "SWEEP_END            12/31",
        "DRY_DAYS             0",
        "REPORT_STEP          00:10:00",
        "WET_STEP             00:01:00",
        "DRY_STEP             00:05:00",
        "ROUTING_STEP         0:00:05",
        "ALLOW_PONDING        NO",
        "INERTIAL_DAMPING     PARTIAL",
        "VARIABLE_STEP        0.75",
        "MINIMUM_STEP         0.5",
        "THREADS              1",
        "",
        "[EVAPORATION]",
        "CONSTANT         0.0",
        "DRY_ONLY         NO",
        "",
    ]

    lines.append("[JUNCTIONS]")
    for n in junctions:
        max_d = 3.0
        lines.append(
            f"{n['id']:<12s} {n['invert']:.2f}    {max_d:.1f}       "
            f"0.0        0.0         0.0"
        )
    lines.append("")

    lines.append("[OUTFALLS]")
    for n in outfalls:
        lines.append(f"{n['id']:<12s} {n['invert']:.2f}    FREE    NO")
    lines.append("")

    lines.append("[CONDUITS]")
    for l in links:
        length = float(l['length'])
        n_val = float(l.get('mannings_n', 0.013))
        lines.append(
            f"{l['id']:<12s} {l['from_node']:<12s} {l['to_node']:<12s} "
            f"{length:<8.1f}  {n_val:.3f}      0      0       0         0"
        )
    lines.append("")

    lines.append("[XSECTIONS]")
    for l in links:
        diam = float(l.get('diameter', 0.6))
        lines.append(
            f"{l['id']:<12s} CIRCULAR    {diam:.2f}    0      0      0      1"
        )
    lines.append("")

    lines.append("[COORDINATES]")
    for n in nodes:
        lines.append(f"{n['id']:<12s} {n['x_m']:.1f}       {n['y_m']:.1f}")
    lines.append("")

    lines += [
        "[TAGS]", "", "[MAP]", "",
        "[REPORT]",
        "INPUT      NO",
        "CONTROLS   NO",
        "NODES ALL",
        "LINKS ALL",
    ]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print(f"SWMM: {out_path}")
    print(f"  Junctions={len(junctions)} Outfalls={len(outfalls)} Conduits={len(links)}")
    return out_path


def validate_swmm(inp_path):
    import pyswmm
    try:
        with pyswmm.Simulation(inp_path):
            pass
        print("  Validation: OK")
        return True
    except Exception as e:
        print(f"  Validation FAILED: {e}")
        return False


OUT_COUPLED_SUB = os.path.join(INPUT_DIR, "networks", "swmm_coupled_sub.inp")
OUT_COUPLED_FULL = os.path.join(INPUT_DIR, "networks", "swmm_coupled_full.inp")


def build_swmm_inp(net_path=NET_PATH, dem_path=DEM_PATH, out_path=OUT_FULL,
                   region_slice=None, main_trunk_only=False, subsample=False):
    dem = np.load(dem_path, allow_pickle=True).astype(np.float64)
    bldg = dem >= 49.9
    net = np.load(net_path, allow_pickle=True)
    nodes, links = filter_network(
        net['nodes'].tolist(), net['links'].tolist(), dem, bldg, region_slice,
        main_trunk_only=main_trunk_only,
    )
    title = "Shenzhen OSM"
    if main_trunk_only:
        title += " Main Trunk"
    if region_slice is None:
        title += " Full"
    else:
        title += " Sub"
    if subsample:
        nodes, links = subsample_for_coupling(nodes, links, min_spacing_m=400)
    path = write_swmm_inp(nodes, links, out_path, title)
    validate_swmm(path)
    return path


if __name__ == "__main__":
    print("Building coupled sub network (main trunk, for ITZI-SWMM)...")
    build_swmm_inp(out_path=OUT_COUPLED_SUB, region_slice=SUB_SLICE,
                   main_trunk_only=True, subsample=True)
    print("\nBuilding coupled full network (main trunk)...")
    build_swmm_inp(out_path=OUT_COUPLED_FULL, region_slice=None,
                   main_trunk_only=True, subsample=True)
