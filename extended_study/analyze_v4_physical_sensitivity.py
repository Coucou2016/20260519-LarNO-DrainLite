#!/usr/bin/env python3
"""Summarise paired event68 B/C physical-sensitivity calculations."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
V3 = ROOT / "extended_study" / "output" / "reviewer_major_revision_v3"
BASELINE = V3 / "formal_matched_full" / "event68"
ROOT_OUTPUT = ROOT / "extended_study" / "output" / "reviewer_major_revision_v4" / "physical_sensitivity_event68"
GEO = ROOT / "LarNO-main" / "benchmark" / "urbanflood" / "geodata" / "region1_20m_drainage_v3_full"
ALTERNATIVES = ["loss_0mmh", "loss_2mmh", "building_nearest", "building_exclude", "inlet_manning_0p015"]
CELL_AREA_M2 = 400.0


def metrics(name: str, directory: Path, baseline: bool = False) -> dict[str, object]:
    b_name = "h_B_surface_inlet_n012.npy"
    c_name = "h_C_itzi_swmm.npy"
    b = np.load(directory / b_name, mmap_mode="r")
    c = np.load(directory / c_name, mmap_mode="r")
    mike = np.load(directory / "h_mike_ref.npy", mmap_mode="r")
    active = np.load(GEO / "active_mask.npy").astype(bool)
    residual = np.asarray(c[:, active]) - np.asarray(b[:, active])
    b_values = np.asarray(b[:, active])
    c_values = np.asarray(c[:, active])
    mike_values = np.asarray(mike[:, active])
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    quality = metadata.get("swmm_quality", {})
    b_volume = b_values.sum(axis=1, dtype=np.float64) * CELL_AREA_M2
    c_volume = c_values.sum(axis=1, dtype=np.float64) * CELL_AREA_M2
    return {
        "case": name,
        "baseline": baseline,
        "infiltration_mmh": metadata.get("infiltration_mmh"),
        "inlet_manning": metadata.get("matched_inlet_manning"),
        "building_rainfall_mode": metadata.get("building_rainfall_mode"),
        "drainage_effect_mae_mm": float(np.mean(np.abs(residual)) * 1000.0),
        "drainage_signed_mean_mm": float(np.mean(residual) * 1000.0),
        "final_surface_reduction_m3": float(b_volume[-1] - c_volume[-1]),
        "B_vs_MIKE_mae_mm": float(np.mean(np.abs(b_values - mike_values)) * 1000.0),
        "C_vs_MIKE_mae_mm": float(np.mean(np.abs(c_values - mike_values)) * 1000.0),
        "B_final_volume_m3": float(b_volume[-1]),
        "C_final_volume_m3": float(c_volume[-1]),
        "routing_continuity_error_pct": quality.get("flow_routing_continuity_error_pct"),
        "nonconverging_steps_pct": quality.get("steps_not_converging_pct"),
        "array_shape": "x".join(map(str, b.shape)),
        "finite": bool(np.isfinite(b_values).all() and np.isfinite(c_values).all()),
    }


def main() -> int:
    rows = [metrics("baseline", BASELINE, True)]
    rows.extend(metrics(name, ROOT_OUTPUT / name / "event68") for name in ALTERNATIVES)
    output = ROOT_OUTPUT / "physical_sensitivity_metrics.csv"
    fields = list(rows[0])
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
