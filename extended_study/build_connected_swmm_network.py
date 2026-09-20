"""Build a connected SWMM network for the copied Shenzhen ITZI-SWMM case.

The previous `swmm_coupled_sub.inp` was over-subsampled and its junctions were
not connected to the outfall. This script creates a new sub-region SWMM input
where every retained junction has at least one graph path to `Outfall_01`.

It writes only inside the copied model under the current project:
`external_models/20260518-itzi-flood/...`; it does not modify the user's
original `E:/Projects/20260518-itzi-flood` folder.
"""

from __future__ import annotations

import csv
import heapq
import json
import math
from collections import defaultdict, deque
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CASE = ROOT / "external_models" / "20260518-itzi-flood" / "test_cases" / "shenzhen_region1"
NET_DIR = CASE / "input_data" / "networks"
NET_PATH = NET_DIR / "osm_merged_network.npz"
DEM_PATH = CASE / "input_data" / "geodata" / "region1_20m" / "dem.npy"
OUT_INP = NET_DIR / "swmm_connected_sub.inp"
OUT_AUDIT = ROOT / "extended_study" / "output" / "connected_swmm_network"

CELL = 20.0
SIM_HOURS = 6
SUB_SLICE = (slice(80, 280), slice(120, 400))


def normalize_node(raw: dict, dem: np.ndarray) -> dict:
    r = int(raw["row"])
    c = int(raw["col"])
    elev = float(dem[r, c])
    inv = float(raw.get("invert", elev - 2.0))
    inv = max(min(inv, elev - 0.5), 0.1)
    return {
        **raw,
        "row": r,
        "col": c,
        "elevation": elev,
        "invert": inv,
        "x_m": float(raw.get("x_m", c * CELL + CELL / 2.0)),
        "y_m": float(raw.get("y_m", (dem.shape[0] - r) * CELL - CELL / 2.0)),
    }


def in_subregion(node: dict) -> bool:
    ys, xs = SUB_SLICE
    return ys.start <= int(node["row"]) < ys.stop and xs.start <= int(node["col"]) < xs.stop


def build_graph(nodes: list[dict], links: list[dict]) -> tuple[dict[str, dict], dict[str, list[tuple[str, float, dict]]]]:
    node_by_id = {n["id"]: n for n in nodes}
    adj: dict[str, list[tuple[str, float, dict]]] = defaultdict(list)
    for link in links:
        a = link["from_node"]
        b = link["to_node"]
        if a not in node_by_id or b not in node_by_id:
            continue
        length = float(link.get("length", 0.0))
        if not math.isfinite(length) or length <= 0:
            na, nb = node_by_id[a], node_by_id[b]
            length = float(math.hypot((na["row"] - nb["row"]) * CELL, (na["col"] - nb["col"]) * CELL))
        if length <= 0:
            continue
        adj[a].append((b, length, link))
        adj[b].append((a, length, link))
    return node_by_id, adj


def dijkstra_to_root(root: str, adj: dict[str, list[tuple[str, float, dict]]]) -> tuple[dict[str, float], dict[str, tuple[str, dict]]]:
    dist = {root: 0.0}
    parent: dict[str, tuple[str, dict]] = {}
    pq = [(0.0, root)]
    while pq:
        d, u = heapq.heappop(pq)
        if d != dist.get(u):
            continue
        for v, w, link in adj.get(u, []):
            nd = d + w
            if nd < dist.get(v, float("inf")):
                dist[v] = nd
                parent[v] = (u, link)
                heapq.heappush(pq, (nd, v))
    return dist, parent


def make_link(link_id: str, a: dict, b: dict, diameter: float = 0.8, link_type: str = "main") -> dict:
    length = float(math.hypot((a["row"] - b["row"]) * CELL, (a["col"] - b["col"]) * CELL))
    length = max(length, CELL)
    slope = max(abs(float(a["invert"]) - float(b["invert"])) / length, 0.001)
    return {
        "id": link_id,
        "from_node": a["id"],
        "to_node": b["id"],
        "length": length,
        "slope": slope,
        "diameter": diameter,
        "mannings_n": 0.013,
        "type": link_type,
    }


def connected_component(start: str, adj: dict[str, list[tuple[str, float, dict]]]) -> set[str]:
    seen = {start}
    dq = deque([start])
    while dq:
        u = dq.popleft()
        for v, _, _ in adj.get(u, []):
            if v not in seen:
                seen.add(v)
                dq.append(v)
    return seen


def select_connected_network() -> tuple[list[dict], list[dict], dict]:
    dem = np.load(DEM_PATH, allow_pickle=True).astype(np.float64)
    bldg = dem >= 49.9
    raw = np.load(NET_PATH, allow_pickle=True)
    raw_nodes = raw["nodes"].tolist()
    raw_links = raw["links"].tolist()

    nodes_all = []
    for n in raw_nodes:
        r, c = int(n["row"]), int(n["col"])
        if not (0 <= r < dem.shape[0] and 0 <= c < dem.shape[1]):
            continue
        if bldg[r, c]:
            continue
        nodes_all.append(normalize_node(n, dem))

    node_by_id_all, adj_all = build_graph(nodes_all, raw_links)
    sub_node_ids = {n["id"] for n in nodes_all if in_subregion(n)}
    main_sub_ids = {n["id"] for n in nodes_all if in_subregion(n) and n.get("type") in ("main_trunk", "outfall")}

    if not main_sub_ids:
        raise RuntimeError("No main-trunk nodes found in sub-region.")

    # Use the lowest main-trunk node in the sub-region as the physical outfall
    # anchor, then attach a synthetic outfall just outside/near that cell.
    root_id = min(main_sub_ids, key=lambda nid: node_by_id_all[nid]["invert"])
    root = node_by_id_all[root_id]
    outfall_row = SUB_SLICE[0].stop + 5
    outfall_col = int(root["col"])
    outfall = {
        "id": "Outfall_01",
        "row": outfall_row,
        "col": outfall_col,
        "elevation": root["elevation"],
        "invert": max(float(root["invert"]) - 5.0, -10.0),
        "x_m": float(root["x_m"]),
        "y_m": float((dem.shape[0] - outfall_row) * CELL - CELL / 2.0),
        "type": "outfall",
    }

    dist, parent = dijkstra_to_root(root_id, adj_all)
    reachable_sub_main = [nid for nid in main_sub_ids if nid in dist]
    reachable_sub_all = [nid for nid in sub_node_ids if nid in dist]

    # Keep a connected main-trunk drainage skeleton first. This avoids the old
    # 82-node/29-link network while still staying tractable for ITZI-SWMM.
    candidate_ids = set(reachable_sub_main)

    # Keep only a modest number of non-main junctions in this stability-focused
    # build. The previous attempt retained >1100 coupled nodes and generated
    # large lateral inflow spikes. The main-trunk set itself already has >1000
    # coupled nodes in this case; a small extra budget is enough to keep local
    # connectors without turning the network into an unstable all-street graph.
    extra_budget = 80
    extras = sorted(
        [nid for nid in reachable_sub_all if nid not in candidate_ids],
        key=lambda nid: dist[nid],
    )[:extra_budget]
    candidate_ids.update(extras)

    # Build the shortest-path tree from every candidate to the root.
    parent_edge_by_child: dict[str, tuple[str, dict]] = {}
    expanded_ids = set(candidate_ids)
    for nid in list(candidate_ids):
        cur = nid
        while cur != root_id:
            if cur not in parent:
                break
            nxt, link = parent[cur]
            expanded_ids.add(cur)
            expanded_ids.add(nxt)
            parent_edge_by_child.setdefault(cur, (nxt, link))
            cur = nxt

    # Keep every intermediate graph node on the shortest paths. Some valid
    # drainage paths briefly leave the cropped ITZI window before returning to
    # the selected outfall/root. If those intermediate nodes are clipped out, the
    # SWMM graph becomes disconnected again. Nodes outside the ITZI sub-window
    # remain SWMM transfer nodes but are not coupled to the 2D surface by
    # `setup_swmm_drainage`.
    retained_ids = set(expanded_ids)
    retained_ids.add(root_id)

    retained_nodes = [dict(node_by_id_all[nid]) for nid in sorted(retained_ids)]
    retained_by_id = {n["id"]: n for n in retained_nodes}

    # Enforce a gentle monotonic invert profile toward the outfall. The OSM-road
    # derived nodes were not surveyed pipe inverts, so using raw DEM-derived
    # invert elevations can create adverse or flat conduits that make SWMM
    # unstable. We keep the original values where possible but force every
    # retained child node to sit at least a small drop above its downstream
    # parent.
    min_pipe_slope = 1.0e-5
    min_drop = 0.001
    outfall["invert"] = max(float(root["invert"]) - 5.0, -10.0)
    retained_by_id[root_id]["invert"] = max(float(outfall["invert"]) + 0.20, 0.2)
    for nid in sorted(retained_ids, key=lambda x: dist.get(x, 0.0)):
        if nid == root_id or nid not in parent_edge_by_child:
            continue
        downstream_id, link = parent_edge_by_child[nid]
        if downstream_id not in retained_by_id:
            continue
        downstream_inv = float(retained_by_id[downstream_id]["invert"])
        length = float(link.get("length", 0.0))
        if not math.isfinite(length) or length <= 0:
            a = retained_by_id[nid]
            b = retained_by_id[downstream_id]
            length = float(math.hypot((a["row"] - b["row"]) * CELL, (a["col"] - b["col"]) * CELL))
        required_inv = downstream_inv + max(length * min_pipe_slope, min_drop)
        original_inv = float(retained_by_id[nid]["invert"])
        ground_limit = float(retained_by_id[nid]["elevation"]) - 0.2
        adjusted = max(original_inv, required_inv)
        # If the required invert would be above the DEM, accept a shallow cover
        # rather than reintroducing an adverse slope. Such nodes are flagged in
        # the audit by the minimum cover statistic.
        retained_by_id[nid]["invert"] = adjusted if adjusted <= ground_limit else adjusted

    retained_links: list[dict] = []
    lid = 0
    for child_id, (downstream_id, link) in sorted(parent_edge_by_child.items()):
        if child_id not in retained_by_id or downstream_id not in retained_by_id:
            continue
        a, b = retained_by_id[child_id], retained_by_id[downstream_id]
        lid += 1
        diameter = max(float(link.get("diameter", 0.8)), 1.0 if a.get("type") == "main_trunk" or b.get("type") == "main_trunk" else 0.8)
        retained_links.append(make_link(f"C_CONN_{lid:05d}", a, b, diameter=diameter))

    # Attach outfall to root. This was missing in the previous inp.
    retained_nodes.append(outfall)
    retained_links.append(make_link(f"C_CONN_{lid + 1:05d}", retained_by_id[root_id], outfall, diameter=2.0))

    node_by_id, adj = build_graph(retained_nodes, retained_links)
    seen = connected_component("Outfall_01", adj)
    connected_junctions = [n for n in retained_nodes if n["id"] != "Outfall_01" and n["id"] in seen]
    if len(connected_junctions) != len(retained_nodes) - 1:
        missing = [n["id"] for n in retained_nodes if n["id"] != "Outfall_01" and n["id"] not in seen]
        raise RuntimeError(f"Connected network still has disconnected nodes: {missing[:20]}")

    audit = {
        "source_nodes_total": len(raw_nodes),
        "source_links_total": len(raw_links),
        "valid_nonbuilding_nodes_total": len(nodes_all),
        "subregion_nodes_total": len(sub_node_ids),
        "subregion_main_nodes_total": len(main_sub_ids),
        "reachable_subregion_nodes_total": len(reachable_sub_all),
        "reachable_subregion_main_nodes_total": len(reachable_sub_main),
        "retained_junctions": len(retained_nodes) - 1,
        "retained_subregion_junctions": len([n for n in retained_nodes if n["id"] != "Outfall_01" and n["id"] in sub_node_ids]),
        "retained_transfer_junctions_outside_subregion": len([n for n in retained_nodes if n["id"] != "Outfall_01" and n["id"] not in sub_node_ids]),
        "retained_outfalls": 1,
        "retained_conduits": len(retained_links),
        "root_node": root_id,
        "root_row": int(root["row"]),
        "root_col": int(root["col"]),
        "root_invert": float(root["invert"]),
        "adjusted_root_invert": float(retained_by_id[root_id]["invert"]),
        "outfall_invert": float(outfall["invert"]),
        "minimum_cover_depth_after_adjustment": float(
            min(float(n["elevation"]) - float(n["invert"]) for n in retained_nodes if n["id"] != "Outfall_01")
        ),
        "zero_degree_junctions": 0,
        "junctions_connected_to_outfall": len(connected_junctions),
        "all_junctions_connected_to_outfall": True,
    }
    return retained_nodes, retained_links, audit


def write_swmm(nodes: list[dict], links: list[dict], path: Path) -> None:
    junctions = [n for n in nodes if n["id"] != "Outfall_01"]
    outfalls = [n for n in nodes if n["id"] == "Outfall_01"]
    lines = [
        "[TITLE]",
        "Shenzhen connected OSM drainage subnetwork",
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
        "ROUTING_STEP         0:00:02",
        "ALLOW_PONDING        YES",
        "INERTIAL_DAMPING     PARTIAL",
        "VARIABLE_STEP        0.50",
        "MINIMUM_STEP         0.2",
        "NORMAL_FLOW_LIMITED  BOTH",
        "THREADS              1",
        "",
        "[EVAPORATION]",
        "CONSTANT         0.0",
        "DRY_ONLY         NO",
        "",
        "[JUNCTIONS]",
    ]
    for n in junctions:
        max_depth = 3.0
        lines.append(f"{n['id']:<16s} {float(n['invert']):<10.3f} {max_depth:<8.3f} 0.0 0.0 10.0")
    lines.extend(["", "[OUTFALLS]"])
    for n in outfalls:
        lines.append(f"{n['id']:<16s} {float(n['invert']):<10.3f} FREE    NO")
    lines.extend(["", "[CONDUITS]"])
    for l in links:
        lines.append(
            f"{l['id']:<16s} {l['from_node']:<16s} {l['to_node']:<16s} "
            f"{float(l['length']):<10.2f} {float(l.get('mannings_n', 0.013)):<8.4f} 0 0 0 0"
        )
    lines.extend(["", "[XSECTIONS]"])
    for l in links:
        lines.append(f"{l['id']:<16s} CIRCULAR    {float(l.get('diameter', 0.8)):<8.3f} 0 0 0 1")
    lines.extend(["", "[COORDINATES]"])
    for n in nodes:
        lines.append(f"{n['id']:<16s} {float(n['x_m']):<12.2f} {float(n['y_m']):<12.2f}")
    lines.extend(["", "[REPORT]", "INPUT      NO", "CONTROLS   NO", "NODES ALL", "LINKS ALL", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_connectivity(path: Path) -> dict:
    sections: dict[str, list[list[str]]] = {}
    cur = ""
    for line in path.read_text(errors="ignore").splitlines():
        s = line.strip()
        if not s or s.startswith(";"):
            continue
        if s.startswith("[") and s.endswith("]"):
            cur = s.upper()
            sections[cur] = []
            continue
        if cur:
            sections[cur].append(s.split())
    junctions = [r[0] for r in sections.get("[JUNCTIONS]", [])]
    outfalls = [r[0] for r in sections.get("[OUTFALLS]", [])]
    conduits = sections.get("[CONDUITS]", [])
    adj: dict[str, set[str]] = defaultdict(set)
    for r in conduits:
        if len(r) < 3:
            continue
        a, b = r[1], r[2]
        adj[a].add(b)
        adj[b].add(a)
    seen = set(outfalls)
    dq = deque(outfalls)
    while dq:
        u = dq.popleft()
        for v in adj[u]:
            if v not in seen:
                seen.add(v)
                dq.append(v)
    return {
        "junctions": len(junctions),
        "outfalls": len(outfalls),
        "conduits": len(conduits),
        "junctions_connected_to_outfall": len([n for n in junctions if n in seen]),
        "disconnected_junctions": len([n for n in junctions if n not in seen]),
        "zero_degree_junctions": len([n for n in junctions if len(adj[n]) == 0]),
    }


def validate_with_pyswmm(path: Path) -> dict:
    try:
        import pyswmm

        with pyswmm.Simulation(str(path)):
            pass
        return {"pyswmm_validation": "ok"}
    except Exception as exc:
        return {"pyswmm_validation": "failed", "pyswmm_error": str(exc)}


def main() -> None:
    OUT_AUDIT.mkdir(parents=True, exist_ok=True)
    nodes, links, audit = select_connected_network()
    write_swmm(nodes, links, OUT_INP)
    connectivity = parse_connectivity(OUT_INP)
    validation = validate_with_pyswmm(OUT_INP)
    audit.update(connectivity)
    audit.update(validation)

    (OUT_AUDIT / "connected_swmm_network_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    with (OUT_AUDIT / "connected_swmm_network_audit.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(audit.keys()))
        writer.writeheader()
        writer.writerow(audit)
    print(f"Wrote {OUT_INP}")
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
