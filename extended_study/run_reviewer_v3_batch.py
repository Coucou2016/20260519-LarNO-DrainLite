#!/usr/bin/env python3
"""Run the accepted reviewer-v3 matched simulations with isolated SWMM files.

PySWMM writes its report and binary output beside the input file.  Each event
therefore receives a private INP copy before event-level parallel execution.
This avoids report races and leaves one complete command log per event.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "extended_study" / "run_reviewer_v3_matched_coupling.py"
DEFAULT_EVENTS = [
    "event1", "event20", "event65", "event66",
    "event67", "event68", "event69", "event70",
]


SCENARIO_FILES = {
    "A": ["h_A_surface_n015.npy"],
    "B": ["h_B_surface_inlet_n012.npy"],
    "C": ["h_C_itzi_swmm.npy", "swmm_C.rpt", "swmm_dynamic_states.npz"],
}


def complete(event_dir: Path, scenarios: list[str]) -> bool:
    required = ["rainfall.npy", "h_mike_ref.npy", "metadata.json"]
    for scenario in scenarios:
        required.extend(SCENARIO_FILES[scenario])
    if any(not (event_dir / name).exists() for name in required):
        return False
    try:
        array_names = ["rainfall.npy", "h_mike_ref.npy"]
        array_names.extend(
            name for scenario in scenarios for name in SCENARIO_FILES[scenario]
            if name.endswith(".npy")
        )
        shapes = {tuple(np.load(event_dir / name, mmap_mode="r").shape) for name in array_names}
        return shapes == {(72, 400, 560)}
    except Exception:
        return False


def run_event(args: argparse.Namespace, event: str) -> dict[str, object]:
    event_dir = args.output / event
    if args.resume and complete(event_dir, args.scenarios):
        return {"event": event, "status": "skipped_complete", "returncode": 0}
    runtime = args.output / "_network_runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    event_network = runtime / f"{event}.inp"
    for suffix in [".inp", ".rpt", ".out"]:
        candidate = event_network.with_suffix(suffix)
        if candidate.exists():
            candidate.unlink()
    shutil.copy2(args.network, event_network)
    command = [
        sys.executable,
        str(RUNNER),
        "--domain", "full",
        "--events", event,
        "--scenarios", *args.scenarios,
        "--network", str(event_network),
        "--output", str(args.output),
        "--infiltration-mmh", str(args.infiltration_mmh),
        "--inlet-manning", str(args.inlet_manning),
        "--building-rainfall-mode", args.building_rainfall_mode,
        "--coupling-relaxation", str(args.coupling_relaxation),
        "--coupling-damping", str(args.coupling_damping),
    ]
    log_path = args.output / "logs" / f"{event}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.run(
            command,
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    result = {
        "event": event,
        "status": "completed" if process.returncode == 0 and complete(event_dir, args.scenarios) else "failed",
        "returncode": process.returncode,
        "command": command,
        "log": str(log_path),
    }
    if event_network.with_suffix(".rpt").exists():
        shutil.copy2(event_network.with_suffix(".rpt"), event_dir / "swmm_C.rpt")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--events", nargs="+", default=DEFAULT_EVENTS)
    parser.add_argument("--scenarios", nargs="+", choices=["A", "B", "C"], default=["A", "B", "C"])
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--infiltration-mmh", type=float, default=1.0)
    parser.add_argument("--inlet-manning", type=float, default=0.012)
    parser.add_argument(
        "--building-rainfall-mode",
        choices=["global_redistribute", "nearest_redistribute", "exclude"],
        default="global_redistribute",
    )
    parser.add_argument("--coupling-relaxation", type=float, default=0.8)
    parser.add_argument("--coupling-damping", type=float, default=0.5)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.network = args.network.resolve()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    if not args.network.exists():
        raise FileNotFoundError(args.network)
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_event, args, event): event for event in args.events}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps(result), flush=True)
    results.sort(key=lambda row: args.events.index(str(row["event"])))
    (args.output / "batch_manifest.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    return 0 if all(row["returncode"] == 0 for row in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
