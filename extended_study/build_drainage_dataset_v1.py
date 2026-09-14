#!/usr/bin/env python3
"""Build the LarNO drainage-augmented benchmark dataset v1."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from heapq import heappop, heappush
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "LarNO-main" / "benchmark" / "urbanflood"
GEODATA20 = BENCH / "geodata" / "region1_20m"
FLOOD20 = BENCH / "flood" / "region1_20m"
NET_PATH = ROOT / "extended_study" / "output" / "osm_merged_network.npz"
TS_DIR = ROOT / "extended_study" / "output" / "itzi_drainage_timeseries"
SWMM_CSV = ROOT / "extended_study" / "output" / "swmm_network" / "swmm_summary_metrics.csv"
OUT_AUDIT = ROOT / "extended_study" / "output" / "drainage_dataset_v1"

LOCATION = "region1_20m_drainage_v1"
GEODATA_OUT = BENCH / "geodata" / LOCATION
FLOOD_OUT = BENCH / "flood" / LOCATION

FEATURE_NAMES = [
    "drain_inlet_mask",
    "drain_outfall_mask",
    "pipe_mask",
    "pipe_diameter",
    "pipe_slope",
    "pipe_capacity",
    "pipe_cover_depth",
    "distance_to_outfall",
]
DEFAULT_EVENTS = [
    "event1", "event20",
    "event65", "event66", "event67", "event68", "event69", "event70",
    "event71", "event72", "event73", "event74", "event75", "event76",
    "event77", "event78", "event80",
]
CELL = 20.0
Y0, Y1 = 80, 280
X0, X1 = 120, 400


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", nargs="*", default=DEFAULT_EVENTS)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in str(value))


def load_network() -> tuple[list[dict], list[dict]]:
    net = np.load(NET_PATH, allow_pickle=True)
    return [dict(n) for n in net["nodes"].tolist()], [dict(l) for l in net["links"].tolist()]


def full_pipe_capacity(diameter: float, slope: float, mannings_n: float) -> float:
    diameter = max(float(diameter), 0.05)
    slope = max(float(slope), 0.0005)
    mannings_n = max(float(mannings_n), 0.008)
    area = math.pi * diameter * diameter / 4.0
    hydraulic_radius = diameter / 4.0
    return (1.0 / mannings_n) * area * (hydraulic_radius ** (2.0 / 3.0)) * math.sqrt(slope)


def raster_line(row0: int, col0: int, row1: int, col1: int) -> list[tuple[int, int, float]]:
    steps = int(max(abs(row1 - row0), abs(col1 - col0))) + 1
    if steps <= 1:
        return [(row0, col0, 0.0)]
    out = []
    for i in range(steps + 1):
        frac = i / steps
        row = int(round(row0 + (row1 - row0) * frac))
        col = int(round(col0 + (col1 - col0) * frac))
        out.append((row, col, frac))
    return out


def graph_distances(nodes: list[dict], links: list[dict], outfall_ids: set[str]) -> dict[str, float]:
    graph: dict[str, list[tuple[str, float]]] = {}
    for link in links:
        a = safe_name(link["from_node"])
        b = safe_name(link["to_node"])
        length = float(link.get("length", CELL))
        graph.setdefault(a, []).append((b, length))
        graph.setdefault(b, []).append((a, length))
    dist = {safe_name(n["id"]): math.inf for n in nodes}
    heap = []
    for node_id in outfall_ids:
        dist[node_id] = 0.0
        heappush(heap, (0.0, node_id))
    while heap:
        d, node_id = heappop(heap)
        if d > dist[node_id]:
            continue
        for nbr, length in graph.get(node_id, []):
            nd = d + length
            if nd < dist.get(nbr, math.inf):
                dist[nbr] = nd
                heappush(heap, (nd, nbr))
    return dist


def rasterize_features() -> dict[str, np.ndarray]:
    dem = np.load(GEODATA20 / "dem.npy").astype(np.float32)
    height, width = dem.shape
    nodes, links = load_network()
    node_by_id = {safe_name(n["id"]): n for n in nodes}
    outfall_ids = {
        safe_name(n["id"]) for n in nodes
        if str(n.get("type", "")).lower() == "outfall"
    }
    if not outfall_ids:
        lowest = min(nodes, key=lambda n: float(n.get("invert", n.get("elevation", 0.0))))
        outfall_ids = {safe_name(lowest["id"])}
    network_dist = graph_distances(nodes, links, outfall_ids)

    features = {name: np.zeros((height, width), dtype=np.float32) for name in FEATURE_NAMES}
    outfall_points = []

    for node in nodes:
        row = int(round(float(node["row"])))
        col = int(round(float(node["col"])))
        if not (0 <= row < height and 0 <= col < width):
            continue
        node_id = safe_name(node["id"])
        if node_id in outfall_ids:
            features["drain_outfall_mask"][row, col] = 1.0
            outfall_points.append((row, col))
        else:
            features["drain_inlet_mask"][row, col] = 1.0
        dist = network_dist.get(node_id, math.inf)
        if math.isfinite(dist):
            current = features["distance_to_outfall"][row, col]
            features["distance_to_outfall"][row, col] = dist if current == 0 else min(current, dist)

    for link in links:
        a = node_by_id.get(safe_name(link["from_node"]))
        b = node_by_id.get(safe_name(link["to_node"]))
        if not a or not b:
            continue
        row0, col0 = int(round(float(a["row"]))), int(round(float(a["col"])))
        row1, col1 = int(round(float(b["row"]))), int(round(float(b["col"])))
        diameter = float(link.get("diameter", 0.6))
        slope = float(link.get("slope", 0.001))
        mannings_n = float(link.get("mannings_n", 0.013))
        capacity = full_pipe_capacity(diameter, slope, mannings_n)
        inv0 = float(a.get("invert", a.get("elevation", 0.0)))
        inv1 = float(b.get("invert", b.get("elevation", 0.0)))
        d0 = network_dist.get(safe_name(a["id"]), math.inf)
        d1 = network_dist.get(safe_name(b["id"]), math.inf)
        for row, col, frac in raster_line(row0, col0, row1, col1):
            if not (0 <= row < height and 0 <= col < width):
                continue
            if capacity >= features["pipe_capacity"][row, col]:
                invert = inv0 + (inv1 - inv0) * frac
                dist = min(d0, d1)
                if math.isfinite(d0) and math.isfinite(d1):
                    dist = d0 + (d1 - d0) * frac
                features["pipe_mask"][row, col] = 1.0
                features["pipe_diameter"][row, col] = diameter
                features["pipe_slope"][row, col] = max(slope, 0.0005)
                features["pipe_capacity"][row, col] = capacity
                features["pipe_cover_depth"][row, col] = max(float(dem[row, col]) - invert, 0.0)
                if math.isfinite(dist):
                    current = features["distance_to_outfall"][row, col]
                    features["distance_to_outfall"][row, col] = dist if current == 0 else min(current, dist)

    if outfall_points:
        rr, cc = np.indices((height, width))
        euclidean = np.full((height, width), np.inf, dtype=np.float32)
        for row, col in outfall_points:
            euclidean = np.minimum(euclidean, np.sqrt(((rr - row) * CELL) ** 2 + ((cc - col) * CELL) ** 2))
        missing = features["distance_to_outfall"] <= 0
        features["distance_to_outfall"][missing] = euclidean[missing]

    return features


def read_swmm_rows() -> dict[str, dict[str, str]]:
    if not SWMM_CSV.exists():
        return {}
    with SWMM_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        return {row["event"]: row for row in csv.DictReader(f)}


def write_one_row_csv(path: Path, row: dict[str, object]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def copy_event(event: str, swmm_rows: dict[str, dict[str, str]]) -> dict[str, object]:
    ts_event = TS_DIR / event
    surface_path = ts_event / "h_itzi_surface.npy"
    sink_path = ts_event / "h_itzi_sink.npy"
    if not surface_path.exists() or not sink_path.exists():
        raise FileNotFoundError(
            f"Missing ITZI time-series labels for {event}. "
            f"Run extended_study/run_itzi_drainage_timeseries.py first."
        )

    event_out = FLOOD_OUT / event
    event_out.mkdir(parents=True, exist_ok=True)
    rainfall = np.load(FLOOD20 / event / "rainfall.npy")[:, Y0:Y1, X0:X1].astype(np.float32)
    mike = np.load(FLOOD20 / event / "h.npy")[:, Y0:Y1, X0:X1].astype(np.float32)
    surface = np.load(surface_path).astype(np.float32)
    sink = np.load(sink_path).astype(np.float32)

    np.save(event_out / "rainfall.npy", rainfall)
    np.save(event_out / "h_mike_ref.npy", mike)
    np.save(event_out / "h_itzi_surface.npy", surface)
    np.save(event_out / "h_itzi_sink.npy", sink)
    np.save(event_out / "h.npy", sink)

    swmm_row = swmm_rows.get(event, {"event": event, "run_status": "missing"})
    write_one_row_csv(event_out / "swmm_metrics.csv", swmm_row)

    return {
        "event": event,
        "rainfall_shape": "x".join(map(str, rainfall.shape)),
        "h_shape": "x".join(map(str, sink.shape)),
        "mike_peak_m": float(np.nanmax(mike)),
        "itzi_surface_peak_m": float(np.nanmax(surface)),
        "itzi_sink_peak_m": float(np.nanmax(sink)),
        "sink_minus_surface_peak_m": float(np.nanmax(sink) - np.nanmax(surface)),
        "mean_sink_reduction_m": float(np.nanmean(surface - sink)),
        "swmm_status": swmm_row.get("run_status", swmm_row.get("status", "unknown")),
    }


def make_event_qa(event: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dem = np.load(GEODATA_OUT / "dem.npy")
    pipe = np.load(GEODATA_OUT / "pipe_mask.npy")
    inlet = np.load(GEODATA_OUT / "drain_inlet_mask.npy")
    rainfall = np.load(FLOOD_OUT / event / "rainfall.npy")
    mike = np.load(FLOOD_OUT / event / "h_mike_ref.npy")
    surface = np.load(FLOOD_OUT / event / "h_itzi_surface.npy")
    sink = np.load(FLOOD_OUT / event / "h_itzi_sink.npy")

    OUT_AUDIT.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.5), constrained_layout=True)
    ax = axes[0, 0]
    ax.imshow(np.where(dem >= 49.9, np.nan, dem), cmap="terrain", origin="upper")
    ax.imshow(np.ma.masked_where(pipe <= 0, pipe), cmap="Blues", alpha=0.55, origin="upper")
    rr, cc = np.where(inlet > 0)
    ax.scatter(cc, rr, s=3, c="#f39c12", alpha=0.7)
    ax.set_title("DEM + pipe/inlet rasters")

    plots = [
        (axes[0, 1], rainfall.sum(axis=0), "6 h rainfall (mm)", "viridis"),
        (axes[0, 2], np.max(mike, axis=0), "MIKE reference peak (m)", "Blues"),
        (axes[1, 0], np.max(surface, axis=0), "ITZI surface peak (m)", "Blues"),
        (axes[1, 1], np.max(sink, axis=0), "ITZI + sink peak (m)", "Blues"),
        (axes[1, 2], np.max(surface, axis=0) - np.max(sink, axis=0), "surface - sink peak (m)", "RdBu_r"),
    ]
    for ax, arr, title, cmap in plots:
        im = ax.imshow(arr, cmap=cmap, origin="upper")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"Drainage Dataset QA: {event}", fontsize=15, fontweight="bold")
    fig.savefig(OUT_AUDIT / f"{event}_qa.png", dpi=180)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    if args.overwrite and GEODATA_OUT.exists():
        shutil.rmtree(GEODATA_OUT)
    if args.overwrite and FLOOD_OUT.exists():
        shutil.rmtree(FLOOD_OUT)
    GEODATA_OUT.mkdir(parents=True, exist_ok=True)
    FLOOD_OUT.mkdir(parents=True, exist_ok=True)
    OUT_AUDIT.mkdir(parents=True, exist_ok=True)

    dem = np.load(GEODATA20 / "dem.npy").astype(np.float32)
    features = rasterize_features()

    full_dir = OUT_AUDIT / "full_20m_features"
    full_dir.mkdir(parents=True, exist_ok=True)
    for name, arr in features.items():
        np.save(full_dir / f"{name}.npy", arr.astype(np.float32))
        np.save(GEODATA_OUT / f"{name}.npy", arr[Y0:Y1, X0:X1].astype(np.float32))
    np.save(GEODATA_OUT / "dem.npy", dem[Y0:Y1, X0:X1].astype(np.float32))

    swmm_rows = read_swmm_rows()
    quality_rows = []
    for event in args.events:
        row = copy_event(event, swmm_rows)
        quality_rows.append(row)
        make_event_qa(event)
        print(f"{event}: {row}")

    with (OUT_AUDIT / "dataset_quality.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(quality_rows[0].keys()))
        writer.writeheader()
        writer.writerows(quality_rows)
    if swmm_rows:
        with (FLOOD_OUT / "swmm_metrics.csv").open("w", encoding="utf-8", newline="") as f:
            fieldnames = list(next(iter(swmm_rows.values())).keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for event in args.events:
                if event in swmm_rows:
                    writer.writerow(swmm_rows[event])

    metadata = {
        "location": LOCATION,
        "cell_size_m": CELL,
        "source_location": "region1_20m",
        "window": {"row_start": Y0, "row_stop": Y1, "col_start": X0, "col_stop": X1},
        "shape": [Y1 - Y0, X1 - X0],
        "events": args.events,
        "target_h": "h.npy is ITZI + conceptual road-aligned inlet-sink water depth",
        "drainage_features": FEATURE_NAMES,
        "swmm_role": "standalone 1D dynamic-wave event metrics only; not a two-way surface-sewer label",
        "event79": "excluded because local h.npy/rainfall.npy are corrupt or incomplete",
    }
    (GEODATA_OUT / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (OUT_AUDIT / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"\nWrote drainage dataset: {GEODATA_OUT} and {FLOOD_OUT}")
    print(f"Audit outputs: {OUT_AUDIT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
