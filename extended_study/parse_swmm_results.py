#!/usr/bin/env python3
"""Parse SWMM report files into compact CSV metrics for the final report."""

from __future__ import annotations

import csv
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SWMM_DIR = ROOT / "extended_study" / "output" / "swmm_network"


def discover_events() -> list[str]:
    status = SWMM_DIR / "swmm_build_status.csv"
    if status.exists():
        with status.open("r", encoding="utf-8-sig", newline="") as f:
            events = [row["event"] for row in csv.DictReader(f) if row.get("event")]
        if events:
            return events
    return sorted(
        {p.name.split("_osm_dynamic_wave", 1)[0] for p in SWMM_DIR.glob("*_osm_dynamic_wave.rpt")},
        key=lambda name: (len(name), name),
    )


def number_after(line: str, label: str) -> list[float] | None:
    if label not in line:
        return None
    vals = re.findall(r"[-+]?\d+(?:\.\d+)?", line.split(label, 1)[1])
    return [float(v) for v in vals]


def parse_report(event: str) -> dict[str, object]:
    rpt = SWMM_DIR / f"{event}_osm_dynamic_wave.rpt"
    row: dict[str, object] = {
        "event": event,
        "rpt_exists": rpt.exists(),
        "run_status": "parsed" if rpt.exists() else "missing",
        "precip_mm": "",
        "surface_runoff_m3": "",
        "surface_storage_m3": "",
        "runoff_continuity_error_pct": "",
        "routing_inflow_m3": "",
        "external_outflow_m3": "",
        "flooding_loss_m3": "",
        "routing_final_storage_m3": "",
        "routing_continuity_error_pct": "",
        "outfall_avg_flow_cms": "",
        "outfall_max_flow_cms": "",
        "outfall_volume_m3": "",
        "flooded_node_count": 0,
        "node_flooding_volume_m3": 0.0,
        "max_node_flooding_rate_cms": 0.0,
    }
    if not rpt.exists():
        return row

    text = rpt.read_text(encoding="utf-8", errors="ignore").splitlines()

    in_flood_table = False
    for line in text:
        vals = number_after(line, "Total Precipitation")
        if vals and len(vals) >= 2:
            row["precip_mm"] = vals[-1]
        vals = number_after(line, "Surface Runoff")
        if vals and len(vals) >= 1:
            row["surface_runoff_m3"] = vals[0] * 10000.0
        vals = number_after(line, "Final Storage")
        if vals and len(vals) >= 1 and row["surface_storage_m3"] == "":
            row["surface_storage_m3"] = vals[0] * 10000.0
        vals = number_after(line, "Continuity Error (%)")
        if vals:
            if row["runoff_continuity_error_pct"] == "":
                row["runoff_continuity_error_pct"] = vals[0]
            else:
                row["routing_continuity_error_pct"] = vals[0]
        vals = number_after(line, "Wet Weather Inflow")
        if vals and len(vals) >= 1:
            row["routing_inflow_m3"] = vals[0] * 10000.0
        vals = number_after(line, "External Outflow")
        if vals and len(vals) >= 1:
            row["external_outflow_m3"] = vals[0] * 10000.0
        vals = number_after(line, "Flooding Loss")
        if vals and len(vals) >= 1:
            row["flooding_loss_m3"] = vals[0] * 10000.0
        vals = number_after(line, "Final Stored Volume")
        if vals and len(vals) >= 1:
            row["routing_final_storage_m3"] = vals[0] * 10000.0

        if "Node Flooding Summary" in line:
            in_flood_table = True
            continue
        if in_flood_table and "Outfall Loading Summary" in line:
            in_flood_table = False
        if in_flood_table:
            m = re.match(r"\s*(N\d+)\s+([-+]?\d+(?:\.\d+)?)\s+([-+]?\d+(?:\.\d+)?).+?\s+([-+]?\d+(?:\.\d+)?)\s+([-+]?\d+(?:\.\d+)?)\s*$", line)
            if m:
                row["flooded_node_count"] = int(row["flooded_node_count"]) + 1
                row["max_node_flooding_rate_cms"] = max(float(row["max_node_flooding_rate_cms"]), float(m.group(3)))
                # Total flood volume is reported in 10^6 litres = 1000 m3.
                row["node_flooding_volume_m3"] = float(row["node_flooding_volume_m3"]) + float(m.group(4)) * 1000.0

        m = re.match(r"\s*System\s+[-+]?\d+(?:\.\d+)?\s+([-+]?\d+(?:\.\d+)?)\s+([-+]?\d+(?:\.\d+)?)\s+([-+]?\d+(?:\.\d+)?)", line)
        if m:
            row["outfall_avg_flow_cms"] = float(m.group(1))
            row["outfall_max_flow_cms"] = float(m.group(2))
            row["outfall_volume_m3"] = float(m.group(3)) * 1000.0

    return row


def main() -> int:
    events = discover_events()
    if not events:
        raise SystemExit(f"No SWMM reports found in {SWMM_DIR}")
    rows = [parse_report(event) for event in events]
    out = SWMM_DIR / "swmm_summary_metrics.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
