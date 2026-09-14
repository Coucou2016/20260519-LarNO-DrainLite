#!/usr/bin/env python3
"""
Build SWMM .inp from UFIM Node/Link shapefiles.

Format follows test_cases/urban_drainage/create_swmm_network.py
([COORDINATES] required for ITZI surface coupling).
"""
from __future__ import annotations

import argparse
import os
from typing import Dict, List, Tuple

import shapefile

from sample_utils import GridMeta, load_dem, proj_to_local, sample_dir

SIM_HOURS = 2


def _assign_node_ids(nodes_raw: List[dict]) -> None:
    """Assign unique sequential SWMM IDs in-place."""
    j = 0
    o = 0
    for n in nodes_raw:
        if n["type"] == "O":
            o += 1
            n["id"] = f"OF{o:04d}"
        else:
            j += 1
            n["id"] = f"J{j:04d}"


def _assign_link_ids(links_raw: List[dict]) -> None:
    for i, l in enumerate(links_raw, start=1):
        l["id"] = f"C{i:05d}"


def read_subcatchments(sample: str, id_map: Dict[str, str]) -> List[dict]:
    """Read Subcatchment.shp for SWMM [SUBCATCHMENTS] (topology only, no SWMM rain)."""
    path = os.path.join(sample_dir(sample), "3-subcatchment", "Subcatchment")
    if not os.path.exists(path + ".shp"):
        return []
    sf = shapefile.Reader(path)
    subs: List[dict] = []
    for i, rec in enumerate(sf.iterShapeRecords(), start=1):
        attrs = rec.record.as_dict() if hasattr(rec.record, "as_dict") else dict(
            zip([f[0] for f in sf.fields[1:]], rec.record)
        )
        outlet_raw = str(attrs.get("Outlet", ""))
        outlet = id_map.get(outlet_raw)
        if not outlet:
            continue
        subs.append({
            "id": f"SC{i:04d}",
            "outlet": outlet,
            "area_ha": max(float(attrs.get("Area", 0.01)), 0.001),
            "width": 10.0,
            "slope": 0.5,
            "imperv": min(max(float(attrs.get("P_IS", 0.0)), 0.0), 100.0),
        })
    return subs


def read_network(sample: str) -> Tuple[List[dict], List[dict], GridMeta]:
    base = sample_dir(sample)
    dmeta, _ = load_dem(sample)

    sf_n = shapefile.Reader(os.path.join(base, "1-node", "Node"))
    nodes: List[dict] = []
    id_map: Dict[str, str] = {}

    for rec in sf_n.iterShapeRecords():
        attrs = rec.record.as_dict() if hasattr(rec.record, "as_dict") else dict(
            zip([f[0] for f in sf_n.fields[1:]], rec.record)
        )
        raw_id = str(attrs.get("NodeID", attrs.get("nodeid", "")))
        x = float(attrs.get("x", rec.shape.points[0][0]))
        y = float(attrs.get("y", rec.shape.points[0][1]))
        xl, yl = proj_to_local(x, y, dmeta)
        ntype = str(attrs.get("Type", "J")).upper()
        nodes.append({
            "raw_id": raw_id,
            "type": ntype,
            "invert": float(attrs.get("Invert_EI", 0.0)),
            "surface": float(attrs.get("Surface_EI", attrs.get("Invert_EI", 0.0) + 1.0)),
            "max_depth": max(float(attrs.get("Max_Depth", 2.0)), 0.5),
            "x_local": xl,
            "y_local": yl,
        })

    _assign_node_ids(nodes)
    for n in nodes:
        id_map[n["raw_id"]] = n["id"]

    sf_l = shapefile.Reader(os.path.join(base, "2-link", "Link"))
    links: List[dict] = []
    for rec in sf_l.iterShapeRecords():
        attrs = rec.record.as_dict() if hasattr(rec.record, "as_dict") else dict(
            zip([f[0] for f in sf_l.fields[1:]], rec.record)
        )
        us = str(attrs.get("Us_Node", ""))
        ds = str(attrs.get("Ds_Node", ""))
        if us not in id_map or ds not in id_map:
            continue
        shape_code = str(attrs.get("Shape_1", "CIRC")).upper()
        width = float(attrs.get("Width", attrs.get("Height", 0.6)))
        height = float(attrs.get("Height", width))
        diam = max(width, height, 0.2)
        length = max(float(attrs.get("Length", 10.0)), 1.0)
        rough = float(attrs.get("Roughness", 0.013))
        links.append({
            "from": id_map[us],
            "to": id_map[ds],
            "length": length,
            "roughness": rough,
            "diameter": diam,
            "shape": shape_code,
        })

    _assign_link_ids(links)
    return nodes, links, dmeta


def write_swmm_inp(
    nodes: List[dict],
    links: List[dict],
    out_path: str,
    title: str,
    sim_hours: float = SIM_HOURS,
    subcatchments: List[dict] | None = None,
) -> None:
    junctions = [n for n in nodes if n["type"] != "O"]
    outfalls = [n for n in nodes if n["type"] == "O"]

    if not outfalls and junctions:
        lowest = min(junctions, key=lambda n: n["invert"])
        outfalls = [{
            **lowest,
            "id": "Outfall_01",
            "type": "O",
            "invert": max(lowest["invert"] - 0.5, 0.1),
            "x_local": lowest["x_local"],
            "y_local": max(lowest["y_local"] - 20.0, 0.0),
        }]
        links = list(links) + [{
            "from": lowest["id"],
            "to": outfalls[0]["id"],
            "length": 20.0,
            "roughness": 0.013,
            "diameter": 0.8,
            "shape": "CIRC",
        }]
        _assign_link_ids(links)

    end_h = int(sim_hours)
    end_m = int(round((sim_hours - end_h) * 60))
    end_time = f"{end_h:02d}:{end_m:02d}:00"

    lines = [
        "[TITLE]",
        title,
        "",
        "[OPTIONS]",
        "FLOW_UNITS           CMS",
        "FLOW_ROUTING         DYNWAVE",
        "START_DATE           05/12/2025",
        "START_TIME           00:00:00",
        "REPORT_START_DATE    05/12/2025",
        "REPORT_START_TIME    00:00:00",
        f"END_DATE             05/12/2025",
        f"END_TIME             {end_time}",
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
        "[RAINGAGES]",
        ";; Rain applied on ITZI surface; dummy gage for SWMM subcatchment init",
        "RainGage         INTENSITY 0:05     1.0     TIMESERIES DummyRain",
        "",
        "[TIMESERIES]",
        "DummyRain        0:00    0.0",
        f"DummyRain        {end_time}    0.0",
        "",
    ]

    lines.append("[JUNCTIONS]")
    for n in junctions:
        inv = max(n["invert"], 0.01)
        md = max(n["max_depth"], 0.5)
        lines.append(f"{n['id']:<16} {inv:10.3f} {md:8.2f} 0.0 0.0 0.0")
    lines.append("")

    lines.append("[OUTFALLS]")
    for n in outfalls:
        inv = max(n["invert"], 0.01)
        lines.append(f"{n['id']:<16} {inv:10.3f} FREE NO")
    lines.append("")

    lines.append("[CONDUITS]")
    for l in links:
        lines.append(
            f"{l['id']:<16} {l['from']:<16} {l['to']:<16} "
            f"{l['length']:8.1f} {l['roughness']:.3f} 0 0 0 0"
        )
    lines.append("")

    lines.append("[XSECTIONS]")
    for l in links:
        d = min(max(l["diameter"], 0.2), 3.0)
        if l["shape"].startswith("RECT"):
            lines.append(f"{l['id']:<16} RECT_CLOSED {d:.3f} {d*0.8:.3f} 0 0 0")
        else:
            lines.append(f"{l['id']:<16} CIRCULAR {d:.3f} 0 0 0 1")
    lines.append("")

    lines.append("[COORDINATES]")
    for n in junctions + outfalls:
        lines.append(f"{n['id']:<16} {n['x_local']:12.2f} {n['y_local']:12.2f}")
    lines.append("")

    subs = subcatchments or []
    if subs:
        lines.append("[SUBCATCHMENTS]")
        lines.append(";;Name           RainGage Outlet           Area     %Imperv  Width    %Slope   CurbLen  SnowPack")
        for s in subs:
            lines.append(
                f"  {s['id']:<14} RainGage   {s['outlet']:<14} "
                f"{s['area_ha']:8.4f} {s['imperv']:8.2f} {s['width']:8.2f} "
                f"{s['slope']:8.4f} 0"
            )
        lines.append("")
        lines.append("[SUBAREAS]")
        for s in subs:
            lines.append(f"  {s['id']:<14} 0.5    0.03    0.5    0.5    0.0    OUTLET")
        lines.append("")
        lines.append("[INFILTRATION]")
        for s in subs:
            lines.append(f"  {s['id']:<14} 80.0    20.0    6.0    10.0    0")
        lines.append("")
    lines += ["[TAGS]", "", "[MAP]", "", "[REPORT]",
              "INPUT      NO", "CONTROLS   NO", "NODES ALL", "LINKS ALL", ""]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"SWMM input written: {out_path}")
    print(
        f"  Junctions: {len(junctions)}, Outfalls: {len(outfalls)}, "
        f"Links: {len(links)}, Subcatchments: {len(subs)}"
    )


def build_swmm_inp(sample: str, out_path: str | None = None, sim_hours: float = SIM_HOURS) -> str:
    if out_path is None:
        out_path = os.path.join(sample_dir(sample), "network", "drainage.inp")
    nodes, links, _ = read_network(sample)
    if not nodes:
        raise ValueError(f"No nodes found for {sample}")
    id_map = {n["raw_id"]: n["id"] for n in nodes}
    subs_shp = read_subcatchments(sample, id_map)
    write_swmm_inp(
        nodes, links, out_path,
        title=f"UFIM {sample} drainage",
        sim_hours=sim_hours,
        subcatchments=subs_shp,
    )
    if subs_shp:
        print(f"  Subcatchments from shapefile: {len(subs_shp)} written to [SUBCATCHMENTS]")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Build SWMM inp from UFIM shapefiles")
    parser.add_argument("sample", choices=["sample1", "sample2", "sample3"])
    parser.add_argument("--hours", type=float, default=SIM_HOURS)
    args = parser.parse_args()
    build_swmm_inp(args.sample, sim_hours=args.hours)


if __name__ == "__main__":
    main()
