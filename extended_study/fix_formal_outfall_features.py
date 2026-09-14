#!/usr/bin/env python3
"""Fix outfall raster features for the formal Shenzhen sub-domain.

The SWMM model has an explicit outfall in the input file, but the previous
static drainage rasters had an all-zero outfall mask and a distance field whose
minimum was not zero.  This script derives outfall cells directly from the
SWMM [COORDINATES] section and rewrites:

- drain_outfall_mask.npy
- distance_to_outfall.npy

Only files inside the current LarNO workspace are modified.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
GEO_DIR = ROOT / "LarNO-main" / "benchmark" / "urbanflood" / "geodata" / "region1_20m_drainage_v1"
INP = (
    ROOT
    / "external_models"
    / "20260518-itzi-flood"
    / "test_cases"
    / "shenzhen_region1"
    / "input_data"
    / "networks"
    / "swmm_coupled_sub.inp"
)
OUT_DIR = ROOT / "extended_study" / "output" / "coupling_difference_audit"

CELL = 20.0
H_FULL = 400
W_FULL = 560
ROW_OFF = 80
COL_OFF = 120


def parse_sections(path: Path) -> dict[str, list[list[str]]]:
    sections: dict[str, list[list[str]]] = {}
    current = ""
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith(";"):
            continue
        if line.startswith("[") and line.endswith("]"):
            current = line.strip("[]").upper()
            sections.setdefault(current, [])
            continue
        if current:
            sections[current].append(line.split())
    return sections


def coor_to_pixel(x_m: float, y_m: float) -> tuple[int, int]:
    col = int(x_m / CELL) - COL_OFF
    row = int((H_FULL * CELL - y_m) / CELL) - ROW_OFF
    return row, col


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dem = np.load(GEO_DIR / "dem.npy").astype(np.float32)
    h, w = dem.shape
    sections = parse_sections(INP)
    coords = {row[0]: (float(row[1]), float(row[2])) for row in sections.get("COORDINATES", []) if len(row) >= 3}
    outfall_ids = [row[0] for row in sections.get("OUTFALLS", []) if row]

    mask = np.zeros((h, w), dtype=np.float32)
    mapped_rows: list[dict[str, object]] = []
    for outfall_id in outfall_ids:
        xy = coords.get(outfall_id)
        if xy is None:
            mapped_rows.append({"outfall": outfall_id, "status": "missing coordinate"})
            continue
        row, col = coor_to_pixel(*xy)
        inside = 0 <= row < h and 0 <= col < w
        if inside:
            mask[row, col] = 1.0
        mapped_rows.append(
            {
                "outfall": outfall_id,
                "x_m": xy[0],
                "y_m": xy[1],
                "row": row,
                "col": col,
                "inside_subdomain": inside,
                "dem_m": float(dem[row, col]) if inside else math.nan,
            }
        )

    if not np.any(mask):
        raise RuntimeError(f"No outfall coordinates from {INP} mapped inside the sub-domain.")

    yy, xx = np.indices(mask.shape, dtype=np.float32)
    out_y, out_x = np.where(mask > 0)
    dist = np.full(mask.shape, np.inf, dtype=np.float32)
    for y0, x0 in zip(out_y, out_x):
        dist = np.minimum(dist, np.sqrt((yy - y0) ** 2 + (xx - x0) ** 2) * CELL)
    dist = dist.astype(np.float32)

    np.save(GEO_DIR / "drain_outfall_mask.npy", mask)
    np.save(GEO_DIR / "distance_to_outfall.npy", dist)

    with (OUT_DIR / "outfall_feature_fix.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(mapped_rows[0].keys()))
        writer.writeheader()
        writer.writerows(mapped_rows)
    meta = {
        "source_inp": str(INP.relative_to(ROOT)),
        "updated_files": [
            str((GEO_DIR / "drain_outfall_mask.npy").relative_to(ROOT)),
            str((GEO_DIR / "distance_to_outfall.npy").relative_to(ROOT)),
        ],
        "outfall_cells": int(mask.sum()),
        "distance_min_m": float(dist.min()),
        "distance_max_m": float(dist.max()),
        "mapped_outfalls": mapped_rows,
    }
    (OUT_DIR / "outfall_feature_fix.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
