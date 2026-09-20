#!/usr/bin/env python3
"""Run the complete repository verifier without modifying any artifact."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CHECKS = [
    ROOT / "scripts" / "verify_manifests.py",
    ROOT / "scripts" / "verify_git_evidence.py",
    ROOT / "scripts" / "verify_lfs.py",
    ROOT / "scripts" / "verify_network.py",
    ROOT / "extended_study" / "final_reviewer_v3_acceptance.py",
    ROOT / "extended_study" / "final_reviewer_v4_acceptance.py",
]


def main() -> int:
    results = []
    for script in CHECKS:
        print(f"[verify] {script.relative_to(ROOT)}", flush=True)
        completed = subprocess.run(
            [sys.executable, str(script)], cwd=ROOT, check=False
        )
        results.append(
            {
                "script": str(script.relative_to(ROOT)),
                "returncode": completed.returncode,
            }
        )
    status = "PASS" if all(row["returncode"] == 0 for row in results) else "FAIL"
    print(json.dumps({"status": status, "mode": "read-only", "checks": results}, indent=2))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
