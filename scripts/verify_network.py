#!/usr/bin/env python3
"""Parse the committed SWMM input and verify topology without writing files."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "extended_study"))

from audit_reviewer_v3_network import audit  # noqa: E402


INPUT = (
    ROOT
    / "LarNO-main"
    / "benchmark"
    / "urbanflood"
    / "geodata"
    / "region1_20m_drainage_v3_full"
    / "conceptual_network.inp"
)


def main() -> int:
    result = audit(INPUT)
    failed = [name for name, passed in result["acceptance"].items() if not passed]
    summary = {
        "status": "PASS" if not failed else "FAIL",
        "input": str(INPUT.relative_to(ROOT)),
        "junctions": result["junctions"],
        "conduits": result["conduits"],
        "outfalls": result["outfalls"],
        "acceptance": result["acceptance"],
        "failures": failed,
    }
    print(json.dumps(summary, indent=2))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
