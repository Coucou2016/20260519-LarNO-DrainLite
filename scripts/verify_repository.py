#!/usr/bin/env python3
"""Verify that canonical Git LFS objects exist, then run the final audit."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASET = (
    ROOT
    / "LarNO-main"
    / "benchmark"
    / "urbanflood"
    / "flood"
    / "region1_20m_drainage_v3_full"
)
EXPECTED = [
    DATASET / "event1" / "h_itzi_swmm.npy",
    ROOT
    / "extended_study"
    / "output"
    / "reviewer_major_revision_v3"
    / "drainlite_hybrid_v3"
    / "predictions"
    / "event68"
    / "h_clim_all_static.npy",
]


def is_lfs_pointer(path: Path) -> bool:
    if not path.exists() or path.stat().st_size > 1024:
        return False
    return path.read_bytes().startswith(b"version https://git-lfs.github.com/spec/v1")


def main() -> int:
    missing = [str(path.relative_to(ROOT)) for path in EXPECTED if not path.exists()]
    pointers = [str(path.relative_to(ROOT)) for path in EXPECTED if path.exists() and is_lfs_pointer(path)]
    if missing or pointers:
        print(json.dumps({"status": "FAIL", "missing": missing, "unresolved_lfs": pointers}, indent=2))
        print("Run `git lfs pull` and retry.", file=sys.stderr)
        return 2

    command = [sys.executable, str(ROOT / "extended_study" / "final_reviewer_v3_acceptance.py")]
    completed = subprocess.run(command, cwd=ROOT, check=False)
    acceptance = (
        ROOT
        / "extended_study"
        / "output"
        / "reviewer_major_revision_v3"
        / "submission_package_v3"
        / "final_acceptance.json"
    )
    result = json.loads(acceptance.read_text(encoding="utf-8")) if acceptance.exists() else {}
    print(
        json.dumps(
            {
                "status": result.get("status", "UNKNOWN"),
                "checks": f"{result.get('checks_passed', 0)}/{result.get('checks_total', 0)}",
            },
            indent=2,
        )
    )
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())

