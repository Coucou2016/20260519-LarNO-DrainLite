#!/usr/bin/env python3
"""Reproducible topology and numerical-stability audit for the SWMM network."""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import networkx as nx
import numpy as np

from publication_plot_style import add_panel_labels, configure_publication_style


configure_publication_style()

ROOT = Path(__file__).resolve().parents[1]
INP = ROOT / "external_models" / "20260518-itzi-flood" / "test_cases" / "shenzhen_region1" / "input_data" / "networks" / "swmm_connected_sub.inp"
ROAD_GRAPH = ROOT / "external_models" / "20260518-itzi-flood" / "test_cases" / "shenzhen_region1" / "input_data" / "networks" / "osm_merged_network.npz"
RUN = ROOT / "extended_study" / "output" / "connected_itzi_swmm_inf_1mmh"
GEO = ROOT / "LarNO-main" / "benchmark" / "urbanflood" / "geodata" / "region1_20m_connected_swmm_v2_inf1mmh"
OUT = ROOT / "extended_study" / "output" / "larno_drainlite_major_revision" / "network_audit"
EVENTS = ["event1", "event20", "event65", "event66", "event67", "event68", "event69", "event70"]


def read_sections(path: Path) -> dict[str, list[list[str]]]:
    sections: dict[str, list[list[str]]] = {}
    current = ""
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith(";"):
            continue
        if line.startswith("[") and line.endswith("]"):
            current = line.upper()
            sections.setdefault(current, [])
        elif current:
            sections[current].append(line.split())
    return sections


def parse_report(path: Path) -> dict[str, object]:
    text = path.read_text(encoding="utf-8", errors="ignore")

    def grab(pattern: str) -> float:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        return float(match.group(1)) if match else float("nan")

    top_nodes = []
    block = re.search(
        r"Most Frequent Nonconverging Nodes\s*\n[-\s]+\n(.*?)(?:\n\s*Highest Flow Instability Indexes|\Z)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if block:
        for line in block.group(1).splitlines():
            parts = line.split()
            if len(parts) >= 2:
                try:
                    top_nodes.append({"node": parts[0], "nonconverging_pct": float(parts[-1])})
                except ValueError:
                    pass
    fii = []
    fii_block = re.search(
        r"Highest Flow Instability Indexes\s*\n[-\s]+\n(.*?)(?:\n\s*Analysis begun|\Z)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if fii_block:
        for line in fii_block.group(1).splitlines():
            parts = line.split()
            if len(parts) >= 2:
                try:
                    fii.append({"link": parts[0], "fii": float(parts[-1])})
                except ValueError:
                    pass
    return {
        "continuity_error_pct": grab(r"Continuity Error \(%\)\s+\.*\s+([-+\d.]+)"),
        "nonconverging_steps_pct": grab(r"% of Steps Not Converging\s*:\s*([-+\d.]+)"),
        "top_nonconverging_nodes": top_nodes[:10],
        "highest_flow_instability_indexes": fii[:10],
    }


def graph_audit(sections: dict[str, list[list[str]]]) -> dict[str, object]:
    junctions = {row[0]: float(row[1]) for row in sections.get("[JUNCTIONS]", [])}
    outfalls = {row[0]: {"invert": float(row[1]), "type": row[2]} for row in sections.get("[OUTFALLS]", [])}
    conduits = sections.get("[CONDUITS]", [])
    xsections = {row[0]: float(row[2]) for row in sections.get("[XSECTIONS]", [])}
    undirected = nx.MultiGraph()
    directed = nx.MultiDiGraph()
    undirected.add_nodes_from(junctions)
    undirected.add_nodes_from(outfalls)
    directed.add_nodes_from(junctions)
    directed.add_nodes_from(outfalls)
    endpoint_groups: Counter[tuple[str, str]] = Counter()
    directed_groups: Counter[tuple[str, str]] = Counter()
    identical_groups: Counter[tuple[object, ...]] = Counter()
    zero_slope = []
    adverse_slope = []
    lengths = []
    diameters = []
    for row in conduits:
        link, source, target = row[0], row[1], row[2]
        length, roughness = float(row[3]), float(row[4])
        source_invert = junctions.get(source, outfalls.get(source, {}).get("invert", np.nan))
        target_invert = junctions.get(target, outfalls.get(target, {}).get("invert", np.nan))
        slope = (float(source_invert) - float(target_invert)) / max(length, 1e-9)
        if abs(slope) < 1e-12:
            zero_slope.append(link)
        elif slope < 0:
            adverse_slope.append(link)
        diameter = xsections.get(link, float("nan"))
        lengths.append(length)
        diameters.append(diameter)
        undirected.add_edge(source, target, key=link)
        directed.add_edge(source, target, key=link)
        endpoint_groups[tuple(sorted((source, target)))] += 1
        directed_groups[(source, target)] += 1
        identical_groups[(source, target, round(length, 6), round(roughness, 6), round(diameter, 6))] += 1

    simple_undirected = nx.Graph(undirected)
    simple_directed = nx.DiGraph(directed)
    components = list(nx.connected_components(simple_undirected))
    outfall_nodes = set(outfalls)
    outfall_components = set().union(
        *(nx.node_connected_component(simple_undirected, outfall) for outfall in outfall_nodes)
    )
    undirected_to_outfall = sum(node in outfall_components for node in junctions)
    directed_reachable = set().union(
        *(nx.ancestors(simple_directed, outfall) | {outfall} for outfall in outfall_nodes)
    )
    directed_to_outfall = sum(node in directed_reachable for node in junctions)
    degree_counts = Counter(dict(undirected.degree()).values())
    options = {row[0]: " ".join(row[1:]) for row in sections.get("[OPTIONS]", []) if len(row) > 1}
    return {
        "junctions": len(junctions),
        "outfalls": len(outfalls),
        "conduits": len(conduits),
        "undirected_components": len(components),
        "cyclomatic_number": len(conduits) - (len(junctions) + len(outfalls)) + len(components),
        "junctions_with_undirected_path_to_outfall": undirected_to_outfall,
        "junctions_with_directed_path_to_outfall": directed_to_outfall,
        "zero_degree_nodes": int(sum(1 for _, degree in undirected.degree() if degree == 0)),
        "degree_distribution": {str(key): value for key, value in sorted(degree_counts.items())},
        "parallel_endpoint_groups": int(sum(value > 1 for value in endpoint_groups.values())),
        "parallel_endpoint_excess_links": int(sum(max(value - 1, 0) for value in endpoint_groups.values())),
        "same_direction_duplicate_groups": int(sum(value > 1 for value in directed_groups.values())),
        "same_direction_excess_links": int(sum(max(value - 1, 0) for value in directed_groups.values())),
        "identical_attribute_duplicate_groups": int(sum(value > 1 for value in identical_groups.values())),
        "zero_slope_conduits": len(zero_slope),
        "adverse_slope_conduits": len(adverse_slope),
        "length_m": {"min": float(np.min(lengths)), "median": float(np.median(lengths)), "max": float(np.max(lengths))},
        "diameter_counts": {str(key): value for key, value in sorted(Counter(diameters).items())},
        "outfall_types": Counter(row[2] for row in sections.get("[OUTFALLS]", [])),
        "routing_options": options,
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def plot_stability(rows: list[dict[str, object]]) -> None:
    events = [str(row["event"]) for row in rows]
    continuity = np.array([float(row["continuity_error_pct"]) for row in rows])
    nonconverging = np.array([float(row["nonconverging_steps_pct"]) for row in rows])
    x = np.arange(len(events))
    fig, axes = plt.subplots(2, 1, figsize=(7.1, 4.9), sharex=True, constrained_layout=True)
    colors = ["#15803d" if abs(value) <= 2 else "#ca8a04" if abs(value) <= 8 else "#dc2626" for value in continuity]
    axes[0].bar(x, continuity, color=colors, alpha=0.85)
    for threshold, style in [(2, "--"), (-2, "--"), (8, ":"), (-8, ":")]:
        axes[0].axhline(threshold, color="#374151", lw=0.8, ls=style)
    axes[0].set_ylabel("Continuity error (%)")
    axes[0].set_title("Routing continuity error")
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(x, nonconverging, color="#7c3aed", alpha=0.8)
    axes[1].set_ylabel("Routing steps not converging (%)")
    axes[1].set_title("Non-converging Dynamic Wave steps")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(events, rotation=28)
    axes[1].grid(axis="y", alpha=0.25)
    add_panel_labels(axes, x=-0.09, y=1.03)
    fig.savefig(OUT / "network_stability_audit.png")
    plt.close(fig)


def plot_network_map() -> None:
    dem = np.load(GEO / "dem.npy").astype(float)
    inlet = np.load(GEO / "drain_inlet_mask.npy") > 0
    pipe = np.load(GEO / "pipe_mask.npy") > 0
    wall = ~np.isfinite(dem) | (dem >= 49.9)
    terrain = np.ma.array(dem, mask=wall)
    road_segments: list[tuple[tuple[float, float], tuple[float, float]]] = []
    if ROAD_GRAPH.exists():
        road_data = np.load(ROAD_GRAPH, allow_pickle=True)
        nodes = {item["id"]: item for item in road_data["nodes"]}
        for link in road_data["links"]:
            start = nodes.get(link["from_node"])
            end = nodes.get(link["to_node"])
            if start is None or end is None:
                continue
            # The analysed window is rows 80:280 and columns 120:400 of the
            # released 400 x 560 grid.
            r0, c0 = float(start["row"]) - 80.0, float(start["col"]) - 120.0
            r1, c1 = float(end["row"]) - 80.0, float(end["col"]) - 120.0
            road_segments.append(((c0 * 0.02, r0 * 0.02), (c1 * 0.02, r1 * 0.02)))
    fig = plt.figure(figsize=(7.1, 5.4), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, width_ratios=[1.65, 1.0])
    ax = fig.add_subplot(grid[:, 0])
    zoom1 = fig.add_subplot(grid[0, 1])
    zoom2 = fig.add_subplot(grid[1, 1])
    cmap = plt.get_cmap("terrain").copy()
    cmap.set_bad("#d1d5db")

    def draw(axis: plt.Axes, row_slice: slice, col_slice: slice, title: str) -> None:
        local = terrain[row_slice, col_slice]
        r0, r1 = row_slice.start or 0, row_slice.stop or dem.shape[0]
        c0, c1 = col_slice.start or 0, col_slice.stop or dem.shape[1]
        extent = [c0 * 0.02, c1 * 0.02, r1 * 0.02, r0 * 0.02]
        axis.imshow(local, cmap=cmap, origin="upper", extent=extent, aspect="equal")
        for (x0, y0), (x1, y1) in road_segments:
            if min(x0, x1) >= extent[0] and max(x0, x1) <= extent[1] and min(y0, y1) >= extent[3] and max(y0, y1) <= extent[2]:
                axis.plot([x0, x1], [y0, y1], color="#4b5563", lw=0.55, alpha=0.62, zorder=2)
        pipe_local = pipe[row_slice, col_slice]
        inlet_local = inlet[row_slice, col_slice]
        x = (np.arange(c0, c1) + 0.5) * 0.02
        y = (np.arange(r0, r1) + 0.5) * 0.02
        axis.contour(x, y, pipe_local.astype(float), levels=[0.5], colors=["#1d4ed8"], linewidths=0.65, zorder=3)
        rr, cc = np.where(inlet_local)
        axis.scatter((cc + c0 + 0.5) * 0.02, (rr + r0 + 0.5) * 0.02, s=4, color="#dc2626", edgecolors="none", zorder=4)
        axis.set_title(title)
        axis.set_xlabel("Distance east (km)")
        axis.set_ylabel("Distance south (km)")

    draw(ax, slice(0, dem.shape[0]), slice(0, dem.shape[1]), "Aligned conceptual network and DEM")
    draw(zoom1, slice(25, 95), slice(20, 115), "North-west zoom")
    draw(zoom2, slice(95, 180), slice(145, 270), "South-east zoom")
    ax.annotate("N", xy=(0.35, 0.25), xytext=(0.35, 0.85), ha="center", va="bottom", fontsize=8,
                arrowprops={"arrowstyle": "-|>", "color": "#111827", "lw": 1.0})
    ax.plot([0.35, 1.35], [3.72, 3.72], color="#111827", lw=2.0)
    ax.text(0.85, 3.64, "1 km", ha="center", va="bottom", fontsize=7)
    handles = [
        Line2D([0], [0], color="#4b5563", lw=1.5, label="Road-derived candidate graph"),
        Line2D([0], [0], color="#1d4ed8", lw=1.5, label="Rasterised SWMM pipe"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#dc2626", markersize=4, label="Coupled junction"),
    ]
    ax.legend(handles=handles, loc="lower left", fontsize=6.5, frameon=True, facecolor="white", edgecolor="0.7")
    add_panel_labels([ax, zoom1, zoom2], x=-0.07, y=1.02)
    fig.savefig(OUT / "aligned_network_dem_map.png")
    plt.close(fig)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    sections = read_sections(INP)
    topology = graph_audit(sections)
    event_rows = []
    report_details = {}
    for event in EVENTS:
        report = parse_report(RUN / f"{event}_swmm_connected.rpt")
        event_rows.append(
            {
                "event": event,
                "continuity_error_pct": report["continuity_error_pct"],
                "nonconverging_steps_pct": report["nonconverging_steps_pct"],
                "continuity_screen": "pass_2pct" if abs(float(report["continuity_error_pct"])) <= 2 else "warning_8pct" if abs(float(report["continuity_error_pct"])) <= 8 else "fail_8pct",
                "numerical_stability": "unacceptable_high_nonconvergence" if float(report["nonconverging_steps_pct"]) > 10 else "review",
            }
        )
        report_details[event] = report

    inlet_count = np.load(GEO / "drain_inlet_count.npy")
    pipe_count = np.load(GEO / "pipe_segment_count.npy")
    raster = {
        "coupled_junction_count": int(inlet_count.sum()),
        "unique_coupled_junction_cells": int(np.count_nonzero(inlet_count)),
        "multi_junction_cells": int(np.count_nonzero(inlet_count > 1)),
        "pipe_cells": int(np.count_nonzero(pipe_count)),
        "overlapping_pipe_cells": int(np.count_nonzero(pipe_count > 1)),
        "maximum_segments_per_cell": int(pipe_count.max()),
        "mapping_rule": "identical int() coordinate-to-cell rule as the ITZI coupling runner",
    }
    audit = {
        "network_file": str(INP),
        "topology": topology,
        "rasterization": raster,
        "event_stability": event_rows,
        "report_details": report_details,
        "interpretation": {
            "continuity_thresholds": "The 2% and 8% cutoffs are study-defined screening thresholds, not EPA acceptance criteria.",
            "stability": "Continuity-screened events still have high percentages of nonconverging Dynamic Wave routing steps and are exploratory labels, not numerically validated hydraulic truth.",
            "network": "The road-aligned network is conceptual and uses free synthetic outfalls; it is not a surveyed municipal drainage network.",
        },
    }
    (OUT / "connected_swmm_major_revision_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2, default=dict), encoding="utf-8")
    write_csv(OUT / "event_numerical_stability.csv", event_rows)
    plot_stability(event_rows)
    plot_network_map()
    print(json.dumps({"topology": topology, "rasterization": raster, "event_stability": event_rows}, ensure_ascii=False, indent=2, default=dict))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
