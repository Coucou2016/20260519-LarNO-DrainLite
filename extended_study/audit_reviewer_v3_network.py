#!/usr/bin/env python3
"""Independently audit a generated SWMM network from its INP text."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import networkx as nx
import numpy as np


def sections(path: Path) -> dict[str, list[list[str]]]:
    parsed: dict[str, list[list[str]]] = {}
    current = None
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.split(";", 1)[0].strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            current = line[1:-1].upper()
            parsed.setdefault(current, [])
            continue
        if current:
            parsed[current].append(line.split())
    return parsed


def quantiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "median": None, "p95": None, "max": None}
    array = np.asarray(values, dtype=float)
    return {
        "min": float(array.min()),
        "median": float(np.median(array)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(array.max()),
    }


def audit(path: Path) -> dict:
    data = sections(path)
    options = {row[0].upper(): row[1] for row in data.get("OPTIONS", []) if len(row) >= 2}
    junctions = {
        row[0]: {
            "invert": float(row[1]),
            "max_depth": float(row[2]),
            "surcharge_depth": float(row[4]) if len(row) > 4 else 0.0,
            "ponded_area": float(row[5]) if len(row) > 5 else 0.0,
        }
        for row in data.get("JUNCTIONS", [])
    }
    outfalls = {
        row[0]: {
            "invert": float(row[1]),
            "type": row[2].upper(),
            "stage": float(row[3]) if row[2].upper() == "FIXED" and len(row) > 3 else None,
        }
        for row in data.get("OUTFALLS", [])
    }
    node_invert = {key: value["invert"] for key, value in junctions.items()}
    node_invert.update({key: value["invert"] for key, value in outfalls.items()})
    diameters = {row[0]: float(row[2]) for row in data.get("XSECTIONS", [])}

    graph = nx.DiGraph()
    graph.add_nodes_from(node_invert)
    undirected_pairs: Counter[tuple[str, str]] = Counter()
    slopes = []
    lengths = []
    pipe_diameters = []
    reverse_or_zero = 0
    crown_above_rim: list[dict[str, object]] = []
    for row in data.get("CONDUITS", []):
        link, upstream, downstream = row[0], row[1], row[2]
        length = float(row[3])
        inlet_offset = float(row[5])
        outlet_offset = float(row[6])
        upstream_invert = node_invert[upstream] + inlet_offset
        downstream_invert = node_invert[downstream] + outlet_offset
        slope = (upstream_invert - downstream_invert) / length
        graph.add_edge(upstream, downstream, link=link)
        undirected_pairs[tuple(sorted((upstream, downstream)))] += 1
        slopes.append(slope)
        lengths.append(length)
        pipe_diameters.append(diameters.get(link, float("nan")))
        reverse_or_zero += int(slope <= 0)
        diameter = diameters.get(link, float("nan"))
        for node, offset, endpoint in [
            (upstream, inlet_offset, "upstream"),
            (downstream, outlet_offset, "downstream"),
        ]:
            if node in junctions and np.isfinite(diameter):
                excess = offset + diameter - junctions[node]["max_depth"]
                if excess > 1e-6:
                    crown_above_rim.append(
                        {
                            "link": link,
                            "node": node,
                            "endpoint": endpoint,
                            "excess_m": float(excess),
                        }
                    )

    reachable = set(outfalls)
    reverse = graph.reverse(copy=False)
    for outfall in outfalls:
        reachable.update(nx.descendants(reverse, outfall))
    reachable_junctions = set(junctions) & reachable
    cycles = list(nx.simple_cycles(graph))
    coordinates = {row[0] for row in data.get("COORDINATES", [])}
    duplicate_groups = {pair: count for pair, count in undirected_pairs.items() if count > 1}
    return {
        "input_file": str(path.resolve()),
        "junctions": len(junctions),
        "outfalls": len(outfalls),
        "conduits": len(data.get("CONDUITS", [])),
        "junctions_with_coordinates": len(set(junctions) & coordinates),
        "outfalls_with_coordinates": len(set(outfalls) & coordinates),
        "outfall_type_counts": dict(Counter(value["type"] for value in outfalls.values())),
        "allow_ponding": options.get("ALLOW_PONDING"),
        "junctions_with_nonzero_ponded_area": sum(
            value["ponded_area"] > 0 for value in junctions.values()
        ),
        "junction_surcharge_depth_m": quantiles(
            [value["surcharge_depth"] for value in junctions.values()]
        ),
        "fixed_outfall_stage_equals_invert": (
            all(
                value["stage"] is not None
                and abs(value["stage"] - value["invert"]) <= 1e-6
                for value in outfalls.values()
                if value["type"] == "FIXED"
            )
            if any(value["type"] == "FIXED" for value in outfalls.values())
            else None
        ),
        "directed_junctions_reaching_outfall": len(reachable_junctions),
        "directed_outfall_reachability_pct": 100.0 * len(reachable_junctions) / max(len(junctions), 1),
        "junctions_without_directed_outfall_path": len(junctions) - len(reachable_junctions),
        "isolated_junctions": sum(graph.in_degree(node) + graph.out_degree(node) == 0 for node in junctions),
        "directed_cycle_count": len(cycles),
        "duplicate_undirected_endpoint_groups": len(duplicate_groups),
        "excess_parallel_conduits": int(sum(count - 1 for count in duplicate_groups.values())),
        "zero_or_reverse_slope_conduits": reverse_or_zero,
        "pipe_crown_above_junction_rim_endpoints": len(crown_above_rim),
        "pipe_crown_above_junction_rim_examples": crown_above_rim[:20],
        "slope": quantiles(slopes),
        "length_m": quantiles(lengths),
        "cover_depth_m": quantiles([value["max_depth"] for value in junctions.values()]),
        "diameter_m": quantiles([value for value in pipe_diameters if np.isfinite(value)]),
        "junction_in_degree": quantiles([graph.in_degree(node) for node in junctions]),
        "junction_out_degree": quantiles([graph.out_degree(node) for node in junctions]),
        "acceptance": {
            "all_junctions_reach_outfall": len(reachable_junctions) == len(junctions),
            "no_isolated_junctions": all(graph.in_degree(node) + graph.out_degree(node) > 0 for node in junctions),
            "no_cycles": len(cycles) == 0,
            "no_parallel_endpoint_duplicates": not duplicate_groups,
            "all_conduits_positive_slope": reverse_or_zero == 0,
            "all_pipe_crowns_below_junction_rims": not crown_above_rim,
            "cover_not_above_8m": max(value["max_depth"] for value in junctions.values()) <= 8.0 + 1e-9,
            # The revised coupling runner excludes IDs listed in [OUTFALLS]
            # explicitly, so coordinates can be retained for pipe geometry.
            "outfall_nodes_identifiable_by_type": bool(outfalls),
            "outfall_boundary_defined": bool(outfalls) and all(
                value["type"] in {"FIXED", "NORMAL", "FREE"}
                for value in outfalls.values()
            ),
            "fixed_tailwater_consistent_when_used": all(
                value["type"] != "FIXED"
                or (
                    value["stage"] is not None
                    and abs(value["stage"] - value["invert"]) <= 1e-6
                )
                for value in outfalls.values()
            ),
            "swmm_ponding_disabled_for_2d_coupling": (
                options.get("ALLOW_PONDING", "").upper() == "NO"
                and all(value["ponded_area"] == 0.0 for value in junctions.values())
            ),
            "positive_surcharge_headroom_for_bidirectional_exchange": all(
                value["surcharge_depth"] > 0.0 for value in junctions.values()
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.input)
    output = args.output or args.input.with_suffix(".independent_audit.json")
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if all(result["acceptance"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
