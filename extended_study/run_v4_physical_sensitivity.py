#!/usr/bin/env python3
"""Run paired B/C event68 sensitivities with the accepted conceptual network."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BATCH = ROOT / "extended_study" / "run_reviewer_v3_batch.py"
NETWORK = ROOT / "extended_study" / "output" / "reviewer_major_revision_v3" / "network" / "swmm_bidirectional_normal_step05.inp"
OUTPUT = ROOT / "extended_study" / "output" / "reviewer_major_revision_v4" / "physical_sensitivity_event68"
CASES = {
    "loss_0mmh": {"infiltration": 0.0, "manning": 0.012, "rain": "global_redistribute"},
    "loss_2mmh": {"infiltration": 2.0, "manning": 0.012, "rain": "global_redistribute"},
    "building_nearest": {"infiltration": 1.0, "manning": 0.012, "rain": "nearest_redistribute"},
    "building_exclude": {"infiltration": 1.0, "manning": 0.012, "rain": "exclude"},
    "inlet_manning_0p015": {"infiltration": 1.0, "manning": 0.015, "rain": "global_redistribute"},
}


def run_case(name: str, specification: dict[str, object], resume: bool) -> dict[str, object]:
    output = OUTPUT / name
    command = [
        sys.executable, str(BATCH), "--network", str(NETWORK), "--output", str(output),
        "--events", "event68", "--scenarios", "B", "C", "--workers", "1",
        "--infiltration-mmh", str(specification["infiltration"]),
        "--inlet-manning", str(specification["manning"]),
        "--building-rainfall-mode", str(specification["rain"]),
    ]
    if resume:
        command.append("--resume")
    log = OUTPUT / "logs" / f"{name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as handle:
        process = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, text=True, check=False)
    return {"case": name, "returncode": process.returncode, "command": command, "log": str(log)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--cases", nargs="+", choices=sorted(CASES), default=list(CASES))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_case, name, CASES[name], args.resume): name for name in args.cases}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps(result), flush=True)
    results.sort(key=lambda row: args.cases.index(str(row["case"])))
    (OUTPUT / "sensitivity_run_manifest.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return 0 if all(row["returncode"] == 0 for row in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
