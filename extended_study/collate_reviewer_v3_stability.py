#!/usr/bin/env python3
"""Collate diagnostic SWMM stability runs without treating them as labels."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np


def report_diagnostics(path: Path) -> tuple[int, int, str, float | None]:
    if not path.exists():
        return 0, 0, "", None
    text = path.read_text(encoding="utf-8", errors="replace")
    warnings = len(re.findall(r"^\s*WARNING\s+\d+:", text, flags=re.MULTILINE))
    errors = len(re.findall(r"^\s*ERROR\s+\d+:", text, flags=re.MULTILINE))
    block = re.search(
        r"Most Frequent Nonconverging Nodes\s*\n\s*\*+\s*\n(.*?)(?:\n\s*\n|Routing Time Step Summary)",
        text,
        flags=re.DOTALL,
    )
    nodes: list[tuple[str, float]] = []
    if block:
        nodes = [
            (node, float(percent))
            for node, percent in re.findall(
                r"Node\s+(\S+)\s+\(([0-9.]+)%\)", block.group(1)
            )
        ]
    return (
        warnings,
        errors,
        ";".join(f"{node}:{percent:.2f}%" for node, percent in nodes),
        nodes[0][1] if nodes else None,
    )


def outfall_type(path: str | None) -> str | None:
    if not path:
        return None
    inp = Path(path)
    if not inp.exists():
        return None
    section = None
    types = set()
    for raw in inp.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.split(";", 1)[0].strip()
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1].upper()
        elif section == "OUTFALLS" and line:
            fields = line.split()
            if len(fields) >= 3:
                types.add(fields[2].upper())
    return "+".join(sorted(types)) if types else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for root in args.inputs:
        for event_dir in sorted(root.glob("event*")):
            metadata_path = event_dir / "metadata.json"
            if not metadata_path.exists():
                continue
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            quality = metadata.get("swmm_quality", {})
            records = metadata.get("records", {}).get("C", {})
            actual_top = records.get("swmm_top_nonconverging_junctions", [])
            actual_counts = records.get("swmm_node_nonconverged_count", {})
            warning_count, error_count, nonconverging_nodes, top_node_pct = report_diagnostics(
                event_dir / "swmm_C.rpt"
            )
            depth_path = event_dir / "h_C_itzi_swmm.npy"
            depths = np.load(depth_path, mmap_mode="r") if depth_path.exists() else None
            rows.append(
                {
                    "run": root.name,
                    "event": event_dir.name,
                    "frames": int(depths.shape[0]) if depths is not None else 0,
                    "network": metadata.get("network"),
                    "outfall_type": outfall_type(metadata.get("network")),
                    "coupling_relaxation": metadata.get("coupling_relaxation", 0.8),
                    "coupling_damping": metadata.get("coupling_damping", 0.5),
                    "continuity_error_pct": quality.get("flow_routing_continuity_error_pct"),
                    "steps_not_converging_pct": quality.get("steps_not_converging_pct"),
                    "average_iterations": quality.get("average_iterations_per_step"),
                    "average_routing_step_s": quality.get("average_routing_step_s"),
                    "warning_count": warning_count,
                    "error_count": error_count,
                    "most_frequent_nonconverging_nodes": nonconverging_nodes,
                    "top_nonconverging_node_pct": top_node_pct,
                    "actual_top_nonconverging_junctions": json.dumps(actual_top),
                    "actual_top_junction_count": (
                        actual_top[0]["count"] if actual_top else 0
                    ),
                    "junctions_with_nonconverged_steps": sum(
                        int(value) > 0 for value in actual_counts.values()
                    ),
                    "final_surface_volume_m3": records.get("vol_m3", [None])[-1],
                    "net_exchange_m3": records.get("swmm_net_exchange_m3", [None])[-1],
                    "runtime_s": records.get("runtime_s"),
                    "finite": bool(np.isfinite(depths).all()) if depths is not None else False,
                }
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with args.output.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
