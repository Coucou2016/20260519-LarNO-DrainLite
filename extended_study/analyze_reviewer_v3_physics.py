#!/usr/bin/env python3
"""Compute hydraulic, numerical and mass-balance checks for matched labels."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEM = ROOT / "external_models" / "20260518-itzi-flood" / "test_cases" / "shenzhen_region1" / "input_data" / "geodata" / "region1_20m" / "dem.npy"
CELL_AREA = 400.0


def csi(pred: np.ndarray, target: np.ndarray, threshold: float, active: np.ndarray) -> float:
    p = (pred > threshold) & active
    t = (target > threshold) & active
    denominator = np.count_nonzero(p | t)
    return float(np.count_nonzero(p & t) / denominator) if denominator else 1.0


def comparison(pred: np.ndarray, target: np.ndarray, active: np.ndarray) -> dict[str, float]:
    mask = np.broadcast_to(active, pred.shape)
    error = pred[mask] - target[mask]
    pred_peak_map = pred.max(axis=0)
    target_peak_map = target.max(axis=0)
    pred_volume = pred[:, active].sum(axis=1, dtype=np.float64) * CELL_AREA
    target_volume = target[:, active].sum(axis=1, dtype=np.float64) * CELL_AREA
    return {
        "mae_mm": float(np.mean(np.abs(error)) * 1000.0),
        "rmse_mm": float(np.sqrt(np.mean(error * error)) * 1000.0),
        "bias_mm": float(np.mean(error) * 1000.0),
        "csi_0p03": csi(pred, target, 0.03, active),
        "csi_0p15": csi(pred, target, 0.15, active),
        "peak_map_mae_mm": float(np.mean(np.abs(pred_peak_map[active] - target_peak_map[active])) * 1000.0),
        "global_peak_abs_error_mm": float(abs(pred.max() - target.max()) * 1000.0),
        "final_volume_abs_error_m3": float(abs(pred_volume[-1] - target_volume[-1])),
    }


def report_volumes(path: Path) -> dict[str, float | None]:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8", errors="replace")

    def value(label: str, last: bool = False):
        matches = re.findall(rf"{re.escape(label)}\s+\.{{2,}}\s+([-+0-9.]+)", text)
        if not matches:
            return None
        return float(matches[-1] if last else matches[0]) * 10000.0

    return {
        "external_inflow_m3": value("External Inflow"),
        "external_outflow_m3": value("External Outflow"),
        "flooding_loss_m3": value("Flooding Loss"),
        "initial_stored_m3": value("Initial Stored Volume"),
        "final_stored_m3": value("Final Stored Volume"),
        "swmm_warning_count": len(
            re.findall(r"^\s*WARNING\s+\d+:", text, flags=re.MULTILINE)
        ),
        "swmm_error_count": len(
            re.findall(r"^\s*ERROR\s+\d+:", text, flags=re.MULTILINE)
        ),
    }


def event_row(event_dir: Path) -> dict[str, object]:
    metadata = json.loads((event_dir / "metadata.json").read_text(encoding="utf-8"))
    rain = np.load(event_dir / "rainfall.npy")
    mike = np.load(event_dir / "h_mike_ref.npy")
    a = np.load(event_dir / "h_A_surface_n015.npy")
    b = np.load(event_dir / "h_B_surface_inlet_n012.npy")
    c = np.load(event_dir / "h_C_itzi_swmm.npy")
    domain = metadata["domain"]
    dem = np.load(DEM)
    if domain == "sub":
        dem = dem[80:280, 120:400]
    active = dem < 49.9
    records = metadata["records"]
    if metadata.get("building_rainfall_mode") == "exclude":
        rain_volume = float(rain[:, active].sum(dtype=np.float64) * CELL_AREA / 1000.0)
    else:
        rain_volume = float(rain.sum(dtype=np.float64) * CELL_AREA / 1000.0)

    row: dict[str, object] = {
        "event": metadata["event"],
        "domain": domain,
        "frames": int(c.shape[0]),
        "height": int(c.shape[1]),
        "width": int(c.shape[2]),
        "nonfinite_count": int(
            sum((~np.isfinite(values)).sum() for values in [rain, mike, a, b, c])
        ),
        "rain_volume_m3": rain_volume,
        "roughness_effect_mae_mm": float(np.mean(np.abs((b - a)[:, active])) * 1000.0),
        "drainage_effect_mae_mm": float(np.mean(np.abs((c - b)[:, active])) * 1000.0),
        "drainage_signed_mean_mm": float(np.mean((c - b)[:, active]) * 1000.0),
        "surface_final_volume_m3": float(a[-1, active].sum(dtype=np.float64) * CELL_AREA),
        "matched_control_final_volume_m3": float(b[-1, active].sum(dtype=np.float64) * CELL_AREA),
        "coupled_final_volume_m3": float(c[-1, active].sum(dtype=np.float64) * CELL_AREA),
        "surface_runtime_s": records["A"]["runtime_s"],
        "matched_control_runtime_s": records["B"]["runtime_s"],
        "coupled_runtime_s": records["C"]["runtime_s"],
        **{f"swmm_{key}": value for key, value in metadata["swmm_quality"].items()},
    }
    for label, values in [("A_vs_C", a), ("B_vs_C", b), ("A_vs_MIKE", a), ("B_vs_MIKE", b), ("C_vs_MIKE", c)]:
        target = c if label.endswith("vs_C") else mike
        row.update({f"{label}_{key}": value for key, value in comparison(values, target, active).items()})

    for scenario, values in [("A", a), ("B", b), ("C", c)]:
        rec = records[scenario]
        infiltration = float(rec["infiltrated_m3"][-1])
        exchange = float(rec.get("surface_held_exchange_m3", [0.0])[-1])
        final_volume = float(values[-1, active].sum(dtype=np.float64) * CELL_AREA)
        error = rain_volume - infiltration - exchange - final_volume
        row[f"{scenario}_infiltration_m3"] = infiltration
        row[f"{scenario}_surface_exchange_m3"] = exchange
        row[f"{scenario}_surface_mass_error_m3"] = error
        row[f"{scenario}_surface_mass_error_pct_rain"] = 100.0 * error / rain_volume

    swmm = report_volumes(event_dir / "swmm_C.rpt")
    row.update(swmm)
    if swmm and all(swmm.get(key) is not None for key in ["external_inflow_m3", "external_outflow_m3", "flooding_loss_m3", "initial_stored_m3", "final_stored_m3"]):
        closure = (
            swmm["external_inflow_m3"] + swmm["initial_stored_m3"]
            - swmm["external_outflow_m3"] - swmm["flooding_loss_m3"] - swmm["final_stored_m3"]
        )
        row["swmm_recomputed_mass_error_m3"] = closure
        row["swmm_recomputed_mass_error_pct_inflow"] = 100.0 * closure / max(swmm["external_inflow_m3"], 1e-9)
        row["swmm_flooding_loss_pct_rain"] = (
            100.0 * swmm["flooding_loss_m3"] / max(rain_volume, 1e-9)
        )
        combined_error = (
            rain_volume
            - float(row["C_infiltration_m3"])
            - float(row["coupled_final_volume_m3"])
            - swmm["external_outflow_m3"]
            - swmm["flooding_loss_m3"]
            - swmm["final_stored_m3"]
            + swmm["initial_stored_m3"]
        )
        row["combined_mass_error_m3"] = combined_error
        row["combined_mass_error_pct_rain"] = 100.0 * combined_error / rain_volume

    row["accepted"] = bool(
        row["frames"] == 72
        and row["nonfinite_count"] == 0
        and abs(float(row["C_surface_mass_error_pct_rain"])) <= 0.5
        and abs(float(row.get("swmm_flow_routing_continuity_error_pct", 999.0))) <= 2.0
        and float(row.get("swmm_steps_not_converging_pct", 999.0)) <= 2.0
        and int(row.get("swmm_warning_count", 999)) == 0
        and int(row.get("swmm_error_count", 999)) == 0
        and float(row.get("swmm_flooding_loss_pct_rain", 999.0)) <= 0.1
        and abs(float(row.get("combined_mass_error_pct_rain", 999.0))) <= 0.5
    )
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = [event_row(path) for path in sorted(args.input.glob("event*")) if (path / "metadata.json").exists()]
    output = args.output or args.input / "physics_quality.csv"
    if rows:
        with output.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(rows, indent=2))
    return 0 if rows and all(row["accepted"] for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
