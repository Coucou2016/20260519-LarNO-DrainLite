#!/usr/bin/env python3
"""Explicitly regenerate acceptance evidence and committed hashes."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    commands = [
        [
            sys.executable,
            str(ROOT / "extended_study" / "final_reviewer_v3_acceptance.py"),
            "--regenerate-evidence",
        ],
        [
            sys.executable,
            str(ROOT / "extended_study" / "final_reviewer_v4_acceptance.py"),
            "--write-json",
        ],
        [sys.executable, str(ROOT / "scripts" / "generate_v4_package_manifest.py")],
        [sys.executable, str(ROOT / "scripts" / "generate_data_manifest.py")],
    ]
    for command in commands:
        completed = subprocess.run(command, cwd=ROOT, check=False)
        if completed.returncode:
            return completed.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
