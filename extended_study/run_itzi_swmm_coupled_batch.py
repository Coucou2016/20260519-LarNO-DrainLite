#!/usr/bin/env python3
"""Run ITZI + PySWMM coupled prototypes for multiple events."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

from run_itzi_swmm_coupled_prototype import OUT_ROOT, run_coupled_event
from analyze_itzi_swmm_coupled_prototype import OUT as ANALYSIS_OUT, analyze_event


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


def is_complete(event: str, expected_frames: int) -> bool:
    path = OUT_ROOT / event / "h_coupled_prototype.npy"
    if not path.exists():
        return False
    try:
        arr = np.load(path, mmap_mode="r")
        return arr.shape == (expected_frames, 200, 280)
    except Exception:
        return False


def write_status(rows: list[dict[str, object]]) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    out = OUT_ROOT / "batch_status.csv"
    fieldnames = [
        "event", "status", "hours", "frames", "elapsed_wall_s",
        "message", "h_path", "summary_path",
    ]
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    expected_frames = max(1, int(round(args.hours * 12)))
    rows: list[dict[str, object]] = []
    write_status(rows)

    for event in args.events:
        started = time.time()
        row = {
            "event": event,
            "status": "started",
            "hours": args.hours,
            "frames": expected_frames,
            "elapsed_wall_s": "",
            "message": "",
            "h_path": str(OUT_ROOT / event / "h_coupled_prototype.npy"),
            "summary_path": str(OUT_ROOT / event / "prototype_summary.json"),
        }
        rows.append(row)
        write_status(rows)
        print(f"\n=== {event}: coupled prototype, {args.hours:g} h ===", flush=True)
        try:
            if args.skip_existing and is_complete(event, expected_frames):
                summary = analyze_event(event)
                row["status"] = "skipped_existing"
                row["elapsed_wall_s"] = 0.0
                row["message"] = f"Existing complete output reused; coupled_nodes={summary.get('coupled_nodes')}"
                print(f"{event}: skipped existing complete output", flush=True)
            else:
                meta = run_coupled_event(event, args.hours, args.node_limit, args.dtmax)
                summary = analyze_event(event)
                row["status"] = "ok"
                row["elapsed_wall_s"] = round(time.time() - started, 3)
                row["message"] = (
                    f"surface_to_pipe={meta.get('surface_to_pipe_m3'):.3f}; "
                    f"pipe_to_surface={meta.get('pipe_to_surface_m3'):.3f}; "
                    f"coupled_nodes={meta.get('coupled_nodes')}"
                )
                print(f"{event}: ok", flush=True)
        except Exception as exc:
            row["status"] = "failed"
            row["elapsed_wall_s"] = round(time.time() - started, 3)
            row["message"] = repr(exc)
            print(f"{event}: failed: {exc!r}", flush=True)
        write_status(rows)

    # Regenerate cross-event summary for all completed event folders.
    try:
        from analyze_itzi_swmm_coupled_prototype import main as analyze_main
        sys.argv = ["analyze_itzi_swmm_coupled_prototype.py", "--all"]
        analyze_main()
    except Exception as exc:
        print(f"aggregate analysis failed: {exc!r}", flush=True)
    return 0 if all(r["status"] in {"ok", "skipped_existing"} for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
