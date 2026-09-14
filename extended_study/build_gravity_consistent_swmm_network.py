#!/usr/bin/env python3
"""Build a duplicate-free, gravity-consistent conceptual SWMM network.

The source graph is the road-derived OSM graph copied with the formal Shenzhen
Itzi-SWMM case. The reconstructed network deliberately uses a directed forest:
each junction has exactly one downstream path to one of several boundary
outfalls. This removes parallel duplicate conduits, cycles, zero slopes and
ambiguous stored directions while retaining the largest connected road graph.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import networkx as nx
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CASE = ROOT / "external_models" / "20260518-itzi-flood" / "test_cases" / "shenzhen_region1"
SOURCE = CASE / "input_data" / "networks" / "osm_merged_network.npz"
DEM_PATH = CASE / "input_data" / "geodata" / "region1_20m" / "dem.npy"
OUT_DIR = ROOT / "extended_study" / "output" / "reviewer_major_revision_v3" / "network"
CELL_M = 20.0


def node_xy(row: int, col: int, height: int) -> tuple[float, float]:
    return col * CELL_M + CELL_M / 2.0, (height - row) * CELL_M - CELL_M / 2.0


def nearest_boundary_side(node: dict, height: int, width: int) -> str:
    distances = {
        "north": node["row"],
        "south": height - 1 - node["row"],
        "west": node["col"],
        "east": width - 1 - node["col"],
    }
    return min(distances, key=distances.get)


def outside_location(node: dict, height: int, width: int, offset_cells: int = 3) -> tuple[int, int]:
    side = nearest_boundary_side(node, height, width)
    if side == "north":
        return -offset_cells, node["col"]
    if side == "south":
        return height - 1 + offset_cells, node["col"]
    if side == "west":
        return node["row"], -offset_cells
    return node["row"], width - 1 + offset_cells


def choose_roots(graph: nx.Graph, nodes: dict[str, dict], height: int, width: int, count: int) -> list[str]:
    """Select spatially distributed low-elevation receiving points.

    The public data do not identify receiving rivers or municipal outfalls.
    Restricting every conceptual outfall to the rectangular outer edge forced
    pipes tens of metres below ground across terrain divides. We instead use
    distributed low points as synthetic receiving locations and omit their map
    coordinates from the coupling interface, so they cannot exchange water
    directly with the two-dimensional surface.
    """

    selected = [min(nodes, key=lambda nid: (nodes[nid]["elevation"], nid))]
    while len(selected) < count:
        distance = nx.multi_source_dijkstra_path_length(graph, selected, weight="weight")
        maximum = max(distance.values())
        remote = [nid for nid, value in distance.items() if value >= 0.85 * maximum and nid not in selected]
        if not remote:
            remote = [nid for nid in nodes if nid not in selected]
        selected.append(min(remote, key=lambda nid: (nodes[nid]["elevation"], -distance[nid], nid)))
    return selected


def diameter_for_upstream_nodes(count: int) -> float:
    """Size pipes without allowing their crowns above 1.5 m-cover nodes."""
    if count < 8:
        return 0.8
    if count < 30:
        return 1.0
    if count < 100:
        return 1.2
    if count < 300:
        return 1.5
    return 2.0


def build(
    min_slope: float,
    max_slope: float,
    max_cover: float,
    smoothing_steps: int,
    base_cover: float,
    direct_outfalls: bool,
    outlet_length_m: float,
) -> tuple[list[dict], list[dict], dict]:
    dem = np.load(DEM_PATH).astype(np.float64)
    height, width = dem.shape
    raw = np.load(SOURCE, allow_pickle=True)

    nodes: dict[str, dict] = {}
    for item in raw["nodes"].tolist():
        row, col = int(item["row"]), int(item["col"])
        if not (0 <= row < height and 0 <= col < width) or dem[row, col] >= 49.9:
            continue
        x_m, y_m = node_xy(row, col, height)
        nodes[item["id"]] = {
            "id": item["id"],
            "row": row,
            "col": col,
            "x_m": float(item.get("x_m", x_m)),
            "y_m": float(item.get("y_m", y_m)),
            "elevation": float(dem[row, col]),
        }

    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    duplicate_counter: Counter[tuple[str, str]] = Counter()
    best_edges: dict[tuple[str, str], dict] = {}
    for item in raw["links"].tolist():
        a, b = item["from_node"], item["to_node"]
        if a not in nodes or b not in nodes or a == b:
            continue
        pair = tuple(sorted((a, b)))
        duplicate_counter[pair] += 1
        length = float(item.get("length", 0.0))
        if not math.isfinite(length) or length < CELL_M:
            na, nb = nodes[a], nodes[b]
            length = math.hypot(na["x_m"] - nb["x_m"], na["y_m"] - nb["y_m"])
        candidate = {**item, "length": max(length, CELL_M)}
        current = best_edges.get(pair)
        if current is None or float(candidate.get("diameter", 0.0)) > float(current.get("diameter", 0.0)):
            best_edges[pair] = candidate

    for (a, b), item in best_edges.items():
        graph.add_edge(a, b, weight=float(item["length"]), source=item)

    components = sorted(nx.connected_components(graph), key=len, reverse=True)
    retained_ids = set(components[0])
    graph = graph.subgraph(retained_ids).copy()
    nodes = {nid: nodes[nid] for nid in retained_ids}

    # Suppress cell-scale DEM noise on the road graph before choosing flow
    # directions. Every directed edge then descends in the smoothed terrain
    # key, so the resulting topology is acyclic by construction.
    routing_elevation = {nid: node["elevation"] for nid, node in nodes.items()}
    for _ in range(smoothing_steps):
        routing_elevation = {
            nid: 0.5 * routing_elevation[nid]
            + 0.5 * float(np.mean([routing_elevation[other] for other in graph[nid]]))
            for nid in graph
        }

    parent: dict[str, str] = {}
    for nid in graph:
        lower = [
            other for other in graph[nid]
            if (routing_elevation[other], other) < (routing_elevation[nid], nid)
        ]
        if lower:
            parent[nid] = min(lower, key=lambda other: (routing_elevation[other], other))
    initial_outfalls = len(nodes) - len(parent)

    def solve_inverts() -> tuple[dict[str, float], dict[str, list[str]]]:
        children: dict[str, list[str]] = defaultdict(list)
        for child, downstream in parent.items():
            children[downstream].append(child)
        invert: dict[str, float] = {}
        # Children always have a larger routing key than their parent. Process
        # the graph in descending order so every child invert is available.
        ordered = sorted(graph, key=lambda nid: (routing_elevation[nid], nid), reverse=True)
        for nid in ordered:
            candidates = [nodes[nid]["elevation"] - base_cover]
            candidates.extend(
                invert[child] - min_slope * float(graph[nid][child]["weight"])
                for child in children[nid]
            )
            invert[nid] = min(candidates)
        # Very steep terrain-following conduits make Dynamic Wave routing
        # unnecessarily stiff. Lower the upstream invert where needed so each
        # directed pipe remains between the prescribed slope limits.
        for nid in reversed(ordered):
            if nid not in parent:
                continue
            downstream = parent[nid]
            length = float(graph[nid][downstream]["weight"])
            invert[nid] = min(invert[nid], invert[downstream] + max_slope * length)
        return invert, children

    # If a descending branch still drives a receiving node below the maximum
    # cover, detach the controlling branch as a separate receiving subcatchment.
    # Unlike global repartitioning, this operation cannot recruit low terrain
    # from a neighbouring basin and therefore monotonically reduces overdepth.
    cover_splits = 0
    while True:
        invert, children = solve_inverts()
        cover_by_node = {
            nid: nodes[nid]["elevation"] - invert[nid]
            for nid in nodes
        }
        deepest = max(cover_by_node, key=cover_by_node.get)
        if cover_by_node[deepest] <= max_cover + 1e-9:
            break
        controlling = [
            child for child in children[deepest]
            if abs(
                invert[child]
                - min_slope * float(graph[deepest][child]["weight"])
                - invert[deepest]
            ) < 1e-8
        ]
        cut = controlling[0] if controlling else deepest
        if cut not in parent:
            raise RuntimeError(f"Cannot resolve cover exceedance at root {deepest}")
        del parent[cut]
        cover_splits += 1

    roots = sorted(nid for nid in nodes if nid not in parent)
    edge_length = {
        child: float(graph[child][downstream]["weight"])
        for child, downstream in parent.items()
    }

    junctions: list[dict] = []
    for nid, node in nodes.items():
        cover = node["elevation"] - invert[nid]
        junctions.append({
            **node,
            "invert": invert[nid],
            "cover": cover,
            "max_depth": cover,
            "kind": "junction",
        })

    children: dict[str, list[str]] = defaultdict(list)
    for child, downstream in parent.items():
        children[downstream].append(child)
    upstream_count: dict[str, int] = {}

    def count_upstream(nid: str) -> int:
        if nid not in upstream_count:
            upstream_count[nid] = 1 + sum(count_upstream(child) for child in children[nid])
        return upstream_count[nid]

    for root in roots:
        count_upstream(root)

    conduits: list[dict] = []
    for index, (child, downstream) in enumerate(sorted(parent.items()), start=1):
        length = edge_length[child]
        conduits.append({
            "id": f"G{index:05d}",
            "from_node": child,
            "to_node": downstream,
            "length": length,
            "diameter": diameter_for_upstream_nodes(upstream_count[child]),
            "roughness": 0.013,
            "slope": (invert[child] - invert[downstream]) / length,
        })

    outfall_nodes: list[dict] = []
    node_lookup = {node["id"]: node for node in junctions}
    if direct_outfalls:
        multi_inlet_roots = [root for root in roots if len(children[root]) > 1]
        if multi_inlet_roots:
            raise ValueError(
                "SWMM outfalls permit only one inlet link; direct outfalls are "
                f"invalid for {len(multi_inlet_roots)} branched roots"
            )
        root_set = set(roots)
        retained_junctions = []
        for node in junctions:
            if node["id"] in root_set:
                outfall_nodes.append({
                    **node,
                    "kind": "outfall",
                    "write_coordinate": True,
                })
            else:
                retained_junctions.append(node)
        junctions = retained_junctions
    else:
        for index, root in enumerate(roots, start=1):
            root_node = node_lookup[root]
            length = outlet_length_m
            outfall_id = f"OUT{index:02d}"
            outfall_invert = root_node["invert"] - min_slope * length
            outfall_nodes.append({
                "id": outfall_id,
                "invert": outfall_invert,
                "x_m": root_node["x_m"] + 1.0,
                "y_m": root_node["y_m"] + 1.0,
                "write_coordinate": False,
                "kind": "outfall",
            })
            conduits.append({
                "id": f"O{index:05d}",
                "from_node": root,
                "to_node": outfall_id,
                "length": length,
                "diameter": diameter_for_upstream_nodes(upstream_count[root]),
                "roughness": 0.013,
                "slope": min_slope,
            })

    covers = np.array([node["cover"] for node in junctions])
    lengths = np.array([link["length"] for link in conduits])
    diameters = Counter(link["diameter"] for link in conduits)
    audit = {
        "source_nodes": int(len(raw["nodes"])),
        "source_links": int(len(raw["links"])),
        "valid_nodes": len(retained_ids),
        "source_components": len(components),
        "retained_component_nodes": len(retained_ids),
        "removed_nodes_outside_largest_component": sum(len(component) for component in components[1:]),
        "unique_source_endpoint_pairs": len(best_edges),
        "removed_parallel_excess_links": int(sum(value - 1 for value in duplicate_counter.values())),
        "junctions": len(junctions),
        "outfalls": len(outfall_nodes),
        "initial_outfalls": initial_outfalls,
        "auto_added_outfalls_for_cover_control": cover_splits,
        "maximum_allowed_cover_m": max_cover,
        "junctions_above_cover_limit": int(np.count_nonzero(covers > max_cover + 1e-9)),
        "graph_elevation_smoothing_steps": smoothing_steps,
        "base_cover_m": base_cover,
        "conduits": len(conduits),
        "directed_junctions_reaching_outfall": len(junctions),
        "direct_terminal_outfalls": direct_outfalls,
        "removed_synthetic_outlet_conduits": len(roots) if direct_outfalls else 0,
        "synthetic_outlet_conduit_length_m": 0.0 if direct_outfalls else outlet_length_m,
        "cycles": 0,
        "zero_slope_conduits": 0,
        "minimum_design_slope": min(link["slope"] for link in conduits),
        "maximum_design_slope": max(link["slope"] for link in conduits),
        "cover_depth_m": {
            "min": float(covers.min()),
            "median": float(np.median(covers)),
            "p95": float(np.quantile(covers, 0.95)),
            "max": float(covers.max()),
        },
        "conduit_length_m": {
            "min": float(lengths.min()),
            "median": float(np.median(lengths)),
            "p95": float(np.quantile(lengths, 0.95)),
            "max": float(lengths.max()),
        },
        "diameter_counts": {str(key): value for key, value in sorted(diameters.items())},
        "outfall_boundary": (
            f"{len(outfall_nodes)} distributed synthetic receiving points; "
            + (
                "terminal low-point nodes are direct outfalls and are excluded from surface coupling by node type"
                if direct_outfalls
                else "virtual outfall coordinates are omitted from surface coupling"
            )
        ),
    }
    return junctions + outfall_nodes, conduits, audit


def write_inp(
    nodes: list[dict],
    links: list[dict],
    output: Path,
    routing_step: float,
    variable_step: float,
    trials: int,
    lengthening_step: float,
    outfall_type: str,
    surcharge_method: str,
    head_tolerance: float,
    surcharge_depth_m: float,
) -> None:
    junctions = [node for node in nodes if node.get("kind", "junction") == "junction"]
    outfalls = [node for node in nodes if node.get("kind") == "outfall"]
    routing_step_value = (
        f"{routing_step:.3f}" if routing_step < 1.0
        else f"0:00:{routing_step:05.2f}"
    )
    lines = [
        "[TITLE]", "Gravity-consistent road-derived conceptual drainage network", "",
        "[OPTIONS]",
        "FLOW_UNITS           CMS",
        "FLOW_ROUTING         DYNWAVE",
        "START_DATE           01/01/2026",
        "START_TIME           00:00:00",
        "REPORT_START_DATE    01/01/2026",
        "REPORT_START_TIME    00:00:00",
        "END_DATE             01/01/2026",
        "END_TIME             06:00:00",
        "REPORT_STEP          00:05:00",
        "WET_STEP             00:00:30",
        "DRY_STEP             00:05:00",
        f"ROUTING_STEP         {routing_step_value}",
        # Surface ponding is represented by the coupled Itzi raster.  Enabling
        # SWMM ponding would create a second above-ground storage at every
        # coupled junction and departs from Itzi's bundled coupling examples.
        "ALLOW_PONDING        NO",
        "INERTIAL_DAMPING     PARTIAL",
        f"VARIABLE_STEP        {variable_step:.2f}",
        "MINIMUM_STEP         0.05",
        f"LENGTHENING_STEP     {lengthening_step:.1f}",
        "NORMAL_FLOW_LIMITED  BOTH",
        "FORCE_MAIN_EQUATION  H-W",
        f"SURCHARGE_METHOD     {surcharge_method}",
        f"MAX_TRIALS           {trials}",
        f"HEAD_TOLERANCE       {head_tolerance:.4f}",
        "THREADS              1",
        "", "[EVAPORATION]", "CONSTANT 0.0", "DRY_ONLY NO", "",
        "[JUNCTIONS]",
    ]
    for node in junctions:
        lines.append(
            f"{node['id']:<16s} {node['invert']:<12.4f} {node['max_depth']:<10.4f} "
            f"0 {surcharge_depth_m:.3f} 0"
        )
    lines.extend(["", "[OUTFALLS]"])
    for node in outfalls:
        if outfall_type == "FIXED":
            lines.append(
                f"{node['id']:<16s} {node['invert']:<12.4f} FIXED "
                f"{node['invert']:<12.4f} NO"
            )
        else:
            lines.append(f"{node['id']:<16s} {node['invert']:<12.4f} {outfall_type} NO")
    lines.extend(["", "[CONDUITS]"])
    for link in links:
        lines.append(
            f"{link['id']:<16s} {link['from_node']:<16s} {link['to_node']:<16s} "
            f"{link['length']:<10.3f} {link['roughness']:<8.4f} 0 0 0 0"
        )
    lines.extend(["", "[XSECTIONS]"])
    for link in links:
        lines.append(f"{link['id']:<16s} CIRCULAR {link['diameter']:<8.3f} 0 0 0 1")
    lines.extend(["", "[COORDINATES]"])
    for node in nodes:
        if node.get("write_coordinate") is False:
            continue
        lines.append(f"{node['id']:<16s} {node['x_m']:<12.3f} {node['y_m']:<12.3f}")
    lines.extend(["", "[REPORT]", "INPUT NO", "CONTROLS NO", "NODES ALL", "LINKS ALL", ""])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


def validate(output: Path) -> str:
    sys.path.insert(0, r"E:\Miniconda3\Lib\site-packages")
    import pyswmm

    simulation = pyswmm.Simulation(str(output))
    try:
        simulation.start()
    finally:
        simulation.close()
    return "ok"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-slope", type=float, default=0.0005)
    parser.add_argument("--max-slope", type=float, default=0.10)
    parser.add_argument("--max-cover", type=float, default=8.0)
    parser.add_argument("--smoothing-steps", type=int, default=2)
    parser.add_argument("--base-cover", type=float, default=1.5)
    parser.add_argument("--routing-step", type=float, default=1.0)
    parser.add_argument("--variable-step", type=float, default=0.25)
    parser.add_argument("--max-trials", type=int, default=50)
    parser.add_argument("--lengthening-step", type=float, default=5.0)
    parser.add_argument("--outfall-type", choices=["FIXED", "NORMAL", "FREE"], default="FIXED")
    parser.add_argument("--surcharge-method", choices=["SLOT", "EXTRAN"], default="SLOT")
    parser.add_argument("--head-tolerance", type=float, default=0.0015)
    parser.add_argument(
        "--direct-outfalls",
        action="store_true",
        help="Convert terminal low-point nodes directly to outfalls and omit artificial outlet conduits",
    )
    parser.add_argument(
        "--surcharge-depth-m",
        type=float,
        default=5.0,
        help="Head range above the junction rim available for bidirectional surcharge exchange",
    )
    parser.add_argument(
        "--outlet-length-m",
        type=float,
        default=100.0,
        help="Length of each terminal junction-to-outfall conduit",
    )
    parser.add_argument("--output", type=Path, default=OUT_DIR / "swmm_gravity_full.inp")
    args = parser.parse_args()
    nodes, links, audit = build(
        args.min_slope,
        args.max_slope,
        args.max_cover,
        args.smoothing_steps,
        args.base_cover,
        args.direct_outfalls,
        args.outlet_length_m,
    )
    write_inp(
        nodes, links, args.output, args.routing_step, args.variable_step,
        args.max_trials, args.lengthening_step, args.outfall_type,
        args.surcharge_method, args.head_tolerance,
        args.surcharge_depth_m,
    )
    audit["swmm_validation"] = validate(args.output)
    audit["outfall_boundary"] = (
        f"{args.outfall_type} at {audit['outfalls']} distributed synthetic receiving interfaces; "
        "FIXED stage equals invert elevation when selected; outfall IDs are excluded from surface coupling"
    )
    audit["routing_options"] = {
        "routing_step_s": args.routing_step,
        "variable_step": args.variable_step,
        "max_trials": args.max_trials,
        "lengthening_step_s": args.lengthening_step,
        "outfall_type": args.outfall_type,
        "surcharge_method": args.surcharge_method,
        "head_tolerance_m": args.head_tolerance,
        "allow_swmm_ponding": False,
        "ponded_area_m2": 0.0,
        "junction_surcharge_depth_m": args.surcharge_depth_m,
    }
    audit_path = args.output.with_suffix(".audit.json")
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(args.output)
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
