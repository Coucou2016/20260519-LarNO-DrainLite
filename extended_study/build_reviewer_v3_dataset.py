#!/usr/bin/env python3
"""Build the full-domain matched DrainLite v3 dataset from accepted runs."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import networkx as nx
import numpy as np
from scipy.ndimage import uniform_filter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_reviewer_v3_network import sections
from analyze_reviewer_v3_physics import event_row


ROOT = Path(__file__).resolve().parents[1]
CASE = ROOT / "external_models" / "20260518-itzi-flood" / "test_cases" / "shenzhen_region1"
DEM_PATH = CASE / "input_data" / "geodata" / "region1_20m" / "dem.npy"
CELL_M = 20.0
H, W = 400, 560


def pixel(x_m: float, y_m: float) -> tuple[int, int]:
    return int((H * CELL_M - y_m) / CELL_M), int(x_m / CELL_M)


def line_pixels(start: tuple[int, int], end: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    count = max(abs(end[0] - start[0]), abs(end[1] - start[1])) + 1
    rows = np.rint(np.linspace(start[0], end[0], count)).astype(int)
    cols = np.rint(np.linspace(start[1], end[1], count)).astype(int)
    valid = (rows >= 0) & (rows < H) & (cols >= 0) & (cols < W)
    return rows[valid], cols[valid]


def static_features(network: Path) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    data = sections(network)
    dem = np.load(DEM_PATH).astype(np.float32)
    active = dem < 49.9
    coordinates = {row[0]: (float(row[1]), float(row[2])) for row in data["COORDINATES"]}
    junctions = {
        row[0]: {"invert": float(row[1]), "cover": float(row[2])}
        for row in data["JUNCTIONS"]
    }
    outfalls = {row[0]: float(row[1]) for row in data["OUTFALLS"]}
    inverts = {node: values["invert"] for node, values in junctions.items()}
    inverts.update(outfalls)
    diameters = {row[0]: float(row[2]) for row in data["XSECTIONS"]}

    arrays = {
        "dem": dem,
        "active_mask": active,
        "drain_inlet_mask": np.zeros((H, W), dtype=np.float32),
        "pipe_mask": np.zeros((H, W), dtype=np.float32),
        "pipe_diameter": np.zeros((H, W), dtype=np.float32),
        "pipe_slope": np.zeros((H, W), dtype=np.float32),
        "pipe_capacity": np.zeros((H, W), dtype=np.float32),
        "pipe_cover_depth": np.zeros((H, W), dtype=np.float32),
        "drain_inlet_count": np.zeros((H, W), dtype=np.float32),
        "pipe_segment_count": np.zeros((H, W), dtype=np.float32),
        "network_degree": np.zeros((H, W), dtype=np.float32),
    }
    graph = nx.DiGraph()
    graph.add_nodes_from(junctions)
    graph.add_nodes_from(outfalls)
    conduit_records = []
    for row in data["CONDUITS"]:
        link, upstream, downstream = row[0], row[1], row[2]
        length = float(row[3])
        roughness = float(row[4])
        slope = (inverts[upstream] + float(row[5]) - inverts[downstream] - float(row[6])) / length
        diameter = diameters[link]
        area = np.pi * diameter * diameter / 4.0
        hydraulic_radius = diameter / 4.0
        capacity = area * hydraulic_radius ** (2.0 / 3.0) * np.sqrt(max(slope, 0.0)) / roughness
        graph.add_edge(upstream, downstream, link=link)
        conduit_records.append((link, upstream, downstream, diameter, slope, capacity))

    for node, values in junctions.items():
        if node not in coordinates:
            continue
        row, col = pixel(*coordinates[node])
        if 0 <= row < H and 0 <= col < W and active[row, col]:
            arrays["drain_inlet_mask"][row, col] = 1.0
            arrays["drain_inlet_count"][row, col] += 1.0
            arrays["pipe_cover_depth"][row, col] = values["cover"]
            arrays["network_degree"][row, col] = graph.in_degree(node) + graph.out_degree(node)

    slope_sum = np.zeros((H, W), dtype=np.float64)
    slope_count = np.zeros((H, W), dtype=np.float64)
    cover_sum = np.zeros((H, W), dtype=np.float64)
    cover_count = np.zeros((H, W), dtype=np.float64)
    for _, upstream, downstream, diameter, slope, capacity in conduit_records:
        if upstream not in coordinates or downstream not in coordinates:
            continue
        rows, cols = line_pixels(pixel(*coordinates[upstream]), pixel(*coordinates[downstream]))
        valid = active[rows, cols]
        rows, cols = rows[valid], cols[valid]
        arrays["pipe_mask"][rows, cols] = 1.0
        arrays["pipe_segment_count"][rows, cols] += 1.0
        arrays["pipe_diameter"][rows, cols] = np.maximum(arrays["pipe_diameter"][rows, cols], diameter)
        arrays["pipe_capacity"][rows, cols] += capacity
        slope_sum[rows, cols] += slope
        slope_count[rows, cols] += 1.0
        if upstream in junctions and downstream in junctions:
            mean_cover = 0.5 * (junctions[upstream]["cover"] + junctions[downstream]["cover"])
            cover_sum[rows, cols] += mean_cover
            cover_count[rows, cols] += 1.0
    mask = slope_count > 0
    arrays["pipe_slope"][mask] = (slope_sum[mask] / slope_count[mask]).astype(np.float32)
    cover_mask = cover_count > 0
    arrays["pipe_cover_depth"][cover_mask] = (cover_sum[cover_mask] / cover_count[cover_mask]).astype(np.float32)
    # Preserve exact manhole cover depths at node cells after line averaging.
    for node, values in junctions.items():
        if node in coordinates:
            row, col = pixel(*coordinates[node])
            if 0 <= row < H and 0 <= col < W and active[row, col]:
                arrays["pipe_cover_depth"][row, col] = values["cover"]

    for size, label in [(3, "3x3"), (7, "7x7")]:
        arrays[f"pipe_density_{label}"] = uniform_filter(arrays["pipe_mask"], size=size, mode="constant")
    arrays["inlet_density_7x7"] = uniform_filter(arrays["drain_inlet_mask"], size=7, mode="constant")
    arrays["capacity_density_7x7"] = uniform_filter(arrays["pipe_capacity"], size=7, mode="constant")

    audit = {
        "network": str(network.resolve()),
        "shape": [H, W],
        "cell_size_m": CELL_M,
        "active_cells": int(active.sum()),
        "inlet_cells": int(np.count_nonzero(arrays["drain_inlet_mask"])),
        "pipe_cells": int(np.count_nonzero(arrays["pipe_mask"])),
        "feature_names": sorted(arrays),
        "excluded_primary_features": [
            "outfall_mask: synthetic receiving interfaces have no surface coordinates",
            "distance_to_outfall: excluded to avoid implicit location encoding",
        ],
    }
    return arrays, audit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("physics_output", type=Path)
    parser.add_argument("--network", type=Path, required=True)
    parser.add_argument("--dataset-name", default="region1_20m_drainage_v3_full")
    args = parser.parse_args()
    geo = ROOT / "LarNO-main" / "benchmark" / "urbanflood" / "geodata" / args.dataset_name
    flood = ROOT / "LarNO-main" / "benchmark" / "urbanflood" / "flood" / args.dataset_name
    geo.mkdir(parents=True, exist_ok=True)
    flood.mkdir(parents=True, exist_ok=True)

    arrays, audit = static_features(args.network)
    for name, values in arrays.items():
        np.save(geo / f"{name}.npy", values)
    (geo / "static_feature_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    shutil.copy2(args.network, geo / "conceptual_network.inp")
    for suffix in [".audit.json", ".independent_audit.json"]:
        source = args.network.with_suffix(suffix)
        if source.exists():
            shutil.copy2(source, geo / source.name)

    accepted = []
    rejected = []
    for source in sorted(args.physics_output.glob("event*")):
        if not (source / "metadata.json").exists():
            continue
        quality = event_row(source)
        if not quality["accepted"] or quality["domain"] != "full":
            rejected.append({"event": source.name, "reason": "physical acceptance or full-domain criterion failed", "quality": quality})
            continue
        target = flood / source.name
        target.mkdir(parents=True, exist_ok=True)
        copies = {
            "rainfall.npy": "rainfall.npy",
            "h_mike_ref.npy": "h_mike_ref.npy",
            "h_A_surface_n015.npy": "h_itzi_surface.npy",
            "h_B_surface_inlet_n012.npy": "h_itzi_surface_matched.npy",
            "residual_roughness_B_minus_A.npy": "residual_roughness.npy",
            "residual_drainage_C_minus_B.npy": "residual_drainage.npy",
            "metadata.json": "physics_metadata.json",
            "swmm_C.rpt": "swmm_report.rpt",
            "swmm_dynamic_states.npz": "swmm_dynamic_states.npz",
        }
        for source_name, target_name in list(copies.items()):
            source_path = source / source_name
            if source_path.exists():
                shutil.copy2(source_path, target / target_name)
        shutil.copy2(source / "h_C_itzi_swmm.npy", target / "h_itzi_swmm.npy")
        shutil.copy2(source / "h_C_itzi_swmm.npy", target / "h.npy")
        (target / "quality.json").write_text(json.dumps(quality, indent=2), encoding="utf-8")
        accepted.append(source.name)

    dataset_audit = {
        "dataset_name": args.dataset_name,
        "accepted_events": accepted,
        "rejected_events": rejected,
        "target_definition": "h_itzi_swmm - h_itzi_surface_matched (C-B)",
        "roughness_control": "h_itzi_surface_matched - h_itzi_surface (B-A)",
        "mike_role": "external full-domain plausibility reference; not a training label",
    }
    (flood / "dataset_audit.json").write_text(json.dumps(dataset_audit, indent=2), encoding="utf-8")
    print(json.dumps(dataset_audit, indent=2))
    return 0 if accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
