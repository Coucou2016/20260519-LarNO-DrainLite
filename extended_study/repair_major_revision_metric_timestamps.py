#!/usr/bin/env python3
"""Convert saved major-revision peak timestamps to interval-end convention."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "extended_study" / "output" / "larno_drainlite_major_revision"
STEP_H = 5.0 / 60.0


def update_csv(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
        fields = list(rows[0]) if rows else []
    changed = 0
    for row in rows:
        for field in ["peak_time_pred_h", "peak_time_target_h"]:
            if row.get(field, "") != "":
                row[field] = repr(float(row[field]) + STEP_H)
                changed += 1
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return changed


def main() -> int:
    marker = OUT / "metrics" / "timestamp_convention.json"
    if marker.exists():
        metadata = json.loads(marker.read_text(encoding="utf-8"))
        if metadata.get("convention") == "interval_end_5min":
            print("Peak timestamps already use interval-end convention.")
            return 0
    changed = 0
    for name in ["event_metrics.csv", "event_metrics_partial.csv"]:
        changed += update_csv(OUT / "metrics" / name)
    metadata = {
        "convention": "interval_end_5min",
        "first_frame_h": STEP_H,
        "last_frame_h": 6.0,
        "corrected_fields": ["peak_time_pred_h", "peak_time_target_h"],
        "unchanged_fields": ["peak_time_error_h", "abs_peak_time_error_h"],
        "changed_values": changed,
        "reason": "ITZI h_series frames are appended at t >= 300, 600, ..., 21600 s.",
    }
    marker.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
