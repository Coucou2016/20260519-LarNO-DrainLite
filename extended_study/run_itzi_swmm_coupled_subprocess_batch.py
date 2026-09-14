#!/usr/bin/env python3
"""Run coupled prototypes in separate Python processes.

PySWMM/EPA-SWMM cannot reliably complete multiple Simulation objects within
one Python interpreter. This wrapper launches each event as an independent
Python process and then runs the analysis step in another process.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "extended_study" / "output" / "itzi_swmm_coupled_prototype"
PROTOTYPE = ROOT / "extended_study" / "run_itzi_swmm_coupled_prototype.py"
ANALYZE = ROOT / "extended_study" / "analyze_itzi_swmm_coupled_prototype.py"
DEFAULT_EVENTS = [
    "event1", "event20",
    "event65", "event66", "event67", "event68", "event69", "event70",
    "event71", "event72", "event73", "event74", "event75", "event76",
    "event77", "event78", "event80",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", nargs="*", default=DEFAULT_EVENTS)
    parser.add_argument("--hours", type=float, default=6.0)
    parser.add_argument("--node-limit", type=int, default=0)
    parser.add_argument("--dtmax", type=float, default=1.0)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def complete(event: str, frames: int) -> bool:
    path = OUT / event / "h_coupled_prototype.npy"
    if not path.exists():
        return False
    try:
        arr = np.load(path, mmap_mode="r")
        return arr.shape == (frames, 200, 280)
    except Exception:
        return False


def write_status(rows: list[dict[str, object]]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "subprocess_batch_status.csv"
    fieldnames = ["event", "status", "elapsed_wall_s", "message", "returncode"]
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    frames = max(1, int(round(args.hours * 12)))
    rows: list[dict[str, object]] = []
    write_status(rows)
    for event in args.events:
        start = time.time()
        row = {"event": event, "status": "started", "elapsed_wall_s": "", "message": "", "returncode": ""}
        rows.append(row)
        write_status(rows)
        print(f"\n=== {event}: subprocess coupled prototype ===", flush=True)
        try:
            if args.skip_existing and complete(event, frames):
                print(f"{event}: existing complete output; analyzing only", flush=True)
                run = subprocess.run([sys.executable, str(ANALYZE), "--events", event], cwd=str(ROOT))
            else:
                run = subprocess.run([
                    sys.executable,
                    "-u",
                    str(PROTOTYPE),
                    "--event",
                    event,
                    "--hours",
                    str(args.hours),
                    "--node-limit",
                    str(args.node_limit),
                    "--dtmax",
                    str(args.dtmax),
                ], cwd=str(ROOT))
                if run.returncode == 0:
                    run = subprocess.run([sys.executable, str(ANALYZE), "--events", event], cwd=str(ROOT))
            row["returncode"] = run.returncode
            row["elapsed_wall_s"] = round(time.time() - start, 3)
            if run.returncode == 0 and complete(event, frames):
                row["status"] = "ok"
                row["message"] = "complete"
                print(f"{event}: ok", flush=True)
            else:
                row["status"] = "failed"
                row["message"] = f"returncode={run.returncode}; complete={complete(event, frames)}"
                print(f"{event}: failed {row['message']}", flush=True)
        except Exception as exc:
            row["status"] = "failed"
            row["elapsed_wall_s"] = round(time.time() - start, 3)
            row["message"] = repr(exc)
            row["returncode"] = ""
            print(f"{event}: failed {exc!r}", flush=True)
        write_status(rows)

    subprocess.run([sys.executable, str(ANALYZE), "--all"], cwd=str(ROOT))
    return 0 if all(r["status"] == "ok" for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
