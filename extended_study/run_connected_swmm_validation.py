"""Run standalone validation for the connected SWMM input and parse continuity."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pyswmm


ROOT = Path(__file__).resolve().parents[1]
INP = ROOT / "external_models" / "20260518-itzi-flood" / "test_cases" / "shenzhen_region1" / "input_data" / "networks" / "swmm_connected_sub.inp"
RPT = INP.with_suffix(".rpt")
OUT = ROOT / "extended_study" / "output" / "connected_swmm_network"


def parse_report(path: Path) -> dict:
    text = path.read_text(errors="ignore") if path.exists() else ""
    m = re.search(r"Continuity Error \(%\)\s+\.*\s+([-+0-9.]+)", text)
    nonconv = re.search(r"% of Steps Not Converging\s+:\s+([-+0-9.]+)", text)
    return {
        "report_exists": path.exists(),
        "continuity_error_pct": float(m.group(1)) if m else None,
        "nonconverging_steps_pct": float(nonconv.group(1)) if nonconv else None,
        "report_path": str(path),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sim = pyswmm.Simulation(str(INP))
    try:
        for _ in sim:
            pass
    finally:
        try:
            sim.close()
        except Exception:
            pass
    result = parse_report(RPT)
    (OUT / "standalone_swmm_validation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
