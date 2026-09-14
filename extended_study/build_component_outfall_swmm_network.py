"""Build a stable connected SWMM subnetwork with one outfall per component.

The road-derived OSM network is not a surveyed sewer network. Forcing all nodes
to one synthetic outfall creates impossible invert profiles over terrain highs.
This builder instead:

- clips the OSM network to the ITZI sub-region,
- deletes isolated/tiny components,
- adds one synthetic outfall to each retained connected component,
- orients conduits from higher invert to lower invert where possible,
- places synthetic outfalls just outside the ITZI window so they are not coupled
  as surface inlet cells.

The output path is the same working connected-network file used by the
comparison script: `swmm_connected_sub.inp`.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict, deque
from pathlib import Path

import numpy as np


ROOT = Path(r"E:\Projects\20260519-LarNO")
CASE = ROOT / "external_models" / "20260518-itzi-flood" / "test_cases" / "shenzhen_region1"
NET_DIR = CASE / "input_data" / "networks"
NET_PATH = NET_DIR / "osm_merged_network.npz"
DEM_PATH = CASE / "input_data" / "geodata" / "region1_20m" / "dem.npy"
OUT_INP = NET_DIR / "swmm_connected_sub.inp"
OUT_AUDIT = ROOT / "extended_study" / "output" / "connected_swmm_network"

CELL = 20.0
SIM_HOURS = 6
SUB_Y0, SUB_Y1 = 80, 280
SUB_X0, SUB_X1 = 120, 400
MIN_COMPONENT_SIZE = 3
COUPLED_NODE_SPACING_M = 0.0
CONDUIT_INLET_OFFSET_M = 0.0


def node_xy(row: int, col: int, h_full: int) -> tuple[float, float]:
    return col * CELL + CELL / 2.0, (h_full - row) * CELL - CELL / 2.0


def normalize_node(raw: dict, dem: np.ndarray) -> dict:
    r = int(raw["row"])
    c = int(raw["col"])
    elev = float(dem[r, c])
    inv = float(raw.get("invert", elev - 2.0))
    inv = max(min(inv, elev - 0.5), 0.1)
    x, y = node_xy(r, c, dem.shape[0])
    return {
        **raw,
        "row": r,
        "col": c,
        "elevation": elev,
        "invert": inv,
        "x_m": float(raw.get("x_m", x)),
        "y_m": float(raw.get("y_m", y)),
    }


def in_sub(node: dict) -> bool:
    return SUB_Y0 <= node["row"] < SUB_Y1 and SUB_X0 <= node["col"] < SUB_X1


def nearest_outside_point(row: int, col: int, h_full: int) -> tuple[int, int, float, float]:
    distances = {
        "north": row - SUB_Y0,
        "south": SUB_Y1 - 1 - row,
        "west": col - SUB_X0,
        "east": SUB_X1 - 1 - col,
    }
    side = min(distances, key=distances.get)
    if side == "north":
        rr, cc = SUB_Y0 - 5, col
    elif side == "south":
        rr, cc = SUB_Y1 + 5, col
    elif side == "west":
        rr, cc = row, SUB_X0 - 5
    else:
        rr, cc = row, SUB_X1 + 5
    x, y = node_xy(rr, cc, h_full)
    return rr, cc, x, y


def components(ids: set[str], links: list[dict]) -> tuple[list[list[str]], dict[str, set[str]]]:
    adj: dict[str, set[str]] = defaultdict(set)
    for link in links:
        a, b = link["from_node"], link["to_node"]
        if a in ids and b in ids:
            adj[a].add(b)
            adj[b].add(a)
    seen: set[str] = set()
    comps: list[list[str]] = []
    for nid in ids:
        if nid in seen:
            continue
        dq = deque([nid])
        seen.add(nid)
        comp = []
        while dq:
            u = dq.popleft()
            comp.append(u)
            for v in adj[u]:
                if v not in seen:
                    seen.add(v)
                    dq.append(v)
        comps.append(comp)
    return comps, adj


def build() -> tuple[list[dict], list[dict], dict]:
    dem = np.load(DEM_PATH, allow_pickle=True).astype(np.float64)
    bldg = dem >= 49.9
    raw = np.load(NET_PATH, allow_pickle=True)

    all_nodes = []
    for raw_node in raw["nodes"].tolist():
        r, c = int(raw_node["row"]), int(raw_node["col"])
        if not (0 <= r < dem.shape[0] and 0 <= c < dem.shape[1]):
            continue
        if bldg[r, c]:
            continue
        node = normalize_node(raw_node, dem)
        if in_sub(node):
            all_nodes.append(node)

    node_by_id = {n["id"]: n for n in all_nodes}
    ids = set(node_by_id)
    all_links = []
    for raw_link in raw["links"].tolist():
        a, b = raw_link["from_node"], raw_link["to_node"]
        if a in ids and b in ids:
            length = float(raw_link.get("length", 0.0))
            if not math.isfinite(length) or length <= 0:
                na, nb = node_by_id[a], node_by_id[b]
                length = math.hypot((na["row"] - nb["row"]) * CELL, (na["col"] - nb["col"]) * CELL)
            if length >= 1:
                all_links.append({**raw_link, "length": length})

    comps, _ = components(ids, all_links)
    kept_comps = [c for c in comps if len(c) >= MIN_COMPONENT_SIZE]
    kept_ids = {nid for comp in kept_comps for nid in comp}
    kept_nodes = [dict(node_by_id[nid]) for nid in sorted(kept_ids)]
    kept_by_id = {n["id"]: n for n in kept_nodes}

    kept_links = []
    lid = 0
    for raw_link in all_links:
        a_id, b_id = raw_link["from_node"], raw_link["to_node"]
        if a_id not in kept_by_id or b_id not in kept_by_id:
            continue
        a, b = kept_by_id[a_id], kept_by_id[b_id]
        # Orient conduit from higher invert to lower invert to reduce adverse
        # slope warnings. Dynamic-wave routing can still reverse flow.
        if float(a["invert"]) < float(b["invert"]):
            a, b = b, a
        lid += 1
        diam = max(float(raw_link.get("diameter", 0.8)), 1.0 if a.get("type") == "main_trunk" or b.get("type") == "main_trunk" else 0.8)
        kept_links.append(
            {
                "id": f"C_COMP_{lid:05d}",
                "from_node": a["id"],
                "to_node": b["id"],
                "length": float(raw_link["length"]),
                "diameter": diam,
                "mannings_n": float(raw_link.get("mannings_n", 0.013)),
            }
        )

    outfalls = []
    outlet_junction_ids: set[str] = set()
    for idx, comp in enumerate(sorted(kept_comps, key=lambda c: (-len(c), min(c))), start=1):
        comp_nodes = [kept_by_id[nid] for nid in comp if nid in kept_by_id]
        boundary_nodes = [
            n for n in comp_nodes
            if n["row"] in range(SUB_Y0, SUB_Y0 + 5)
            or n["row"] in range(SUB_Y1 - 5, SUB_Y1)
            or n["col"] in range(SUB_X0, SUB_X0 + 5)
            or n["col"] in range(SUB_X1 - 5, SUB_X1)
        ]
        candidates = boundary_nodes if boundary_nodes else comp_nodes
        outlet_node = min(candidates, key=lambda n: float(n["invert"]))
        rr, cc, x, y = nearest_outside_point(outlet_node["row"], outlet_node["col"], dem.shape[0])
        outfall = {
            "id": f"Outfall_{idx:02d}",
            "row": rr,
            "col": cc,
            "elevation": float(outlet_node["elevation"]),
            "invert": max(float(outlet_node["invert"]) - 2.0, -5.0),
            "x_m": x,
            "y_m": y,
            "type": "outfall",
        }
        outfalls.append(outfall)
        outlet_junction_ids.add(outlet_node["id"])
        lid += 1
        length = max(math.hypot((outlet_node["row"] - rr) * CELL, (outlet_node["col"] - cc) * CELL), CELL)
        kept_links.append(
            {
                "id": f"C_COMP_{lid:05d}",
                "from_node": outlet_node["id"],
                "to_node": outfall["id"],
                "length": length,
                "diameter": 2.0,
                "mannings_n": 0.013,
            }
        )

    selected_coupled: list[dict] = []
    for n in sorted(kept_nodes, key=lambda item: (item["id"] not in outlet_junction_ids, item["row"], item["col"])):
        if n["id"] in outlet_junction_ids:
            n["coupled_to_surface"] = True
            selected_coupled.append(n)
            continue
        if all(
            math.hypot((n["row"] - s["row"]) * CELL, (n["col"] - s["col"]) * CELL) >= COUPLED_NODE_SPACING_M
            for s in selected_coupled
        ):
            n["coupled_to_surface"] = True
            selected_coupled.append(n)
        else:
            n["coupled_to_surface"] = False

    nodes = kept_nodes + outfalls
    audit = {
        "source_nodes_total": int(len(raw["nodes"])),
        "source_links_total": int(len(raw["links"])),
        "subregion_valid_nodes": len(all_nodes),
        "subregion_valid_links": len(all_links),
        "source_components": len(comps),
        "kept_components": len(kept_comps),
        "dropped_small_components": len(comps) - len(kept_comps),
        "retained_junctions": len(kept_nodes),
        "retained_outfalls": len(outfalls),
        "retained_conduits": len(kept_links),
        "coupled_surface_junctions": len(selected_coupled),
        "internal_transfer_junctions": len(kept_nodes) - len(selected_coupled),
        "coupled_node_spacing_m": COUPLED_NODE_SPACING_M,
        "minimum_cover_depth": float(min(float(n["elevation"]) - float(n["invert"]) for n in kept_nodes)),
        "maximum_cover_depth": float(max(float(n["elevation"]) - float(n["invert"]) for n in kept_nodes)),
    }
    return nodes, kept_links, audit


def write_swmm(nodes: list[dict], links: list[dict], path: Path) -> None:
    junctions = [n for n in nodes if not str(n["id"]).startswith("Outfall_")]
    outfalls = [n for n in nodes if str(n["id"]).startswith("Outfall_")]
    lines = [
        "[TITLE]",
        "Shenzhen component-outfall connected OSM drainage subnetwork",
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
        "MAX_TRIALS           20",
        "THREADS              1",
        "",
        "[EVAPORATION]",
        "CONSTANT         0.0",
        "DRY_ONLY         NO",
        "",
        "[JUNCTIONS]",
    ]
    for n in junctions:
        lines.append(f"{n['id']:<16s} {float(n['invert']):<10.3f} 3.000    0.0 0.0 20.0")
    lines.extend(["", "[OUTFALLS]"])
    for n in outfalls:
        lines.append(f"{n['id']:<16s} {float(n['invert']):<10.3f} FREE    NO")
    lines.extend(["", "[CONDUITS]"])
    for l in links:
        lines.append(
            f"{l['id']:<16s} {l['from_node']:<16s} {l['to_node']:<16s} "
            f"{float(l['length']):<10.2f} {float(l['mannings_n']):<8.4f} {CONDUIT_INLET_OFFSET_M:.3f} 0 0 0"
        )
    lines.extend(["", "[XSECTIONS]"])
    for l in links:
        lines.append(f"{l['id']:<16s} CIRCULAR    {float(l['diameter']):<8.3f} 0 0 0 1")
    lines.extend(["", "[COORDINATES]"])
    for n in nodes:
        if not str(n["id"]).startswith("Outfall_") and not n.get("coupled_to_surface", False):
            continue
        lines.append(f"{n['id']:<16s} {float(n['x_m']):<12.2f} {float(n['y_m']):<12.2f}")
    lines.extend(["", "[REPORT]", "INPUT      NO", "CONTROLS   NO", "NODES ALL", "LINKS ALL", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def connectivity(path: Path) -> dict:
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
    adj: dict[str, set[str]] = defaultdict(set)
    for r in sections.get("[CONDUITS]", []):
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
        "conduits": len(sections.get("[CONDUITS]", [])),
        "junctions_connected_to_any_outfall": len([n for n in junctions if n in seen]),
        "disconnected_junctions": len([n for n in junctions if n not in seen]),
        "zero_degree_junctions": len([n for n in junctions if len(adj[n]) == 0]),
    }


def pyswmm_validate(path: Path) -> dict:
    try:
        sys.path.insert(0, r"E:\Miniconda3\Lib\site-packages")
        import pyswmm

        with pyswmm.Simulation(str(path)):
            pass
        return {"pyswmm_validation": "ok"}
    except Exception as exc:
        return {"pyswmm_validation": "failed", "pyswmm_error": str(exc)}


def main() -> None:
    OUT_AUDIT.mkdir(parents=True, exist_ok=True)
    nodes, links, audit = build()
    write_swmm(nodes, links, OUT_INP)
    audit.update(connectivity(OUT_INP))
    audit.update(pyswmm_validate(OUT_INP))
    (OUT_AUDIT / "component_outfall_swmm_network_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    with (OUT_AUDIT / "component_outfall_swmm_network_audit.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(audit.keys()))
        writer.writeheader()
        writer.writerow(audit)
    print(f"Wrote {OUT_INP}")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
