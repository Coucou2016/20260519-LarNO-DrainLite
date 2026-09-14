#!/usr/bin/env python3
"""Build the infiltration-calibrated connected ITZI-SWMM LarNO-style dataset."""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from build_connected_swmm_drainage_dataset import (
    CELL_AREA_M2,
    INP,
    SRC_FLOOD,
    parse_swmm_network,
    write_event_metrics,
)


ROOT = Path(__file__).resolve().parents[1]
EVENTS = ["event1", "event20", "event65", "event66", "event67", "event68", "event69", "event70"]
RUN_DIR = ROOT / "extended_study" / "output" / "connected_itzi_swmm_inf_2mmh"
OUT_GEO = ROOT / "LarNO-main" / "benchmark" / "urbanflood" / "geodata" / "region1_20m_connected_swmm_v2_infiltration"
OUT_FLOOD = ROOT / "LarNO-main" / "benchmark" / "urbanflood" / "flood" / "region1_20m_connected_swmm_v2_infiltration"
OUT_AUDIT = ROOT / "extended_study" / "output" / "connected_swmm_dataset_v2_infiltration"
FIG = OUT_AUDIT / "figures"
INFILTRATION_MMH = 2.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default=str(RUN_DIR))
    parser.add_argument("--dataset-name", default="region1_20m_connected_swmm_v2_infiltration")
    parser.add_argument("--audit-output-name", default="connected_swmm_dataset_v2_infiltration")
    parser.add_argument("--infiltration-mmh", type=float, default=2.0)
    return parser.parse_args()


def configure_paths(args: argparse.Namespace) -> None:
    global RUN_DIR, OUT_GEO, OUT_FLOOD, OUT_AUDIT, FIG, INFILTRATION_MMH
    RUN_DIR = Path(args.run_dir)
    OUT_GEO = ROOT / "LarNO-main" / "benchmark" / "urbanflood" / "geodata" / args.dataset_name
    OUT_FLOOD = ROOT / "LarNO-main" / "benchmark" / "urbanflood" / "flood" / args.dataset_name
    OUT_AUDIT = ROOT / "extended_study" / "output" / args.audit_output_name
    FIG = OUT_AUDIT / "figures"
    INFILTRATION_MMH = float(args.infiltration_mmh)


def active_mask(dem: np.ndarray) -> np.ndarray:
    return np.isfinite(dem) & (dem < 49.9)


def parse_rpt(path: Path) -> dict[str, float]:
    text = path.read_text(encoding="utf-8", errors="ignore") if path.exists() else ""

    def grab(pattern: str) -> float:
        m = re.search(pattern, text, flags=re.IGNORECASE)
        return float(m.group(1)) if m else float("nan")

    return {
        "external_inflow_m3": grab(r"External Inflow\s+\.*\s+[-+\d.]+\s+([-+\d.]+)") * 1000.0,
        "external_outflow_m3": grab(r"External Outflow\s+\.*\s+[-+\d.]+\s+([-+\d.]+)") * 1000.0,
        "flooding_loss_m3": grab(r"Flooding Loss\s+\.*\s+[-+\d.]+\s+([-+\d.]+)") * 1000.0,
        "final_stored_m3": grab(r"Final Stored Volume\s+\.*\s+[-+\d.]+\s+([-+\d.]+)") * 1000.0,
        "continuity_error_pct": grab(r"Continuity Error \(%\)\s+\.*\s+([-+\d.]+)"),
        "nonconverging_steps_pct": grab(r"% of Steps Not Converging\s+:\s+([-+\d.]+)"),
    }


def quality_status(continuity_error_pct: float) -> tuple[str, bool]:
    abs_err = abs(float(continuity_error_pct))
    if abs_err <= 2.0:
        return "accepted", True
    if abs_err <= 8.0:
        return "warning", True
    return "excluded", False


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_event(event: str, dem: np.ndarray) -> dict[str, object]:
    src_npz = RUN_DIR / event / f"{event}_connected_itzi_swmm.npz"
    if not src_npz.exists():
        raise FileNotFoundError(f"Missing connected ITZI-SWMM result: {src_npz}")
    out_dir = OUT_FLOOD / event
    out_dir.mkdir(parents=True, exist_ok=True)
    z = np.load(src_npz, allow_pickle=True)
    surface = z["h_surf"].astype(np.float32)
    connected = z["h_swmm_connected"].astype(np.float32)
    residual = (connected - surface).astype(np.float32)
    rainfall = np.load(SRC_FLOOD / event / "rainfall.npy").astype(np.float32)
    mike = np.load(SRC_FLOOD / event / "h_mike_ref.npy").astype(np.float32)
    mask = active_mask(dem)
    rec_surf = z["rec_surf"].item()
    rec_swmm = z["rec_swmm"].item()
    rpt_metrics = parse_rpt(RUN_DIR / f"{event}_swmm_connected.rpt")
    status, use_label = quality_status(rpt_metrics["continuity_error_pct"])

    np.save(out_dir / "rainfall.npy", rainfall)
    np.save(out_dir / "h_mike_ref.npy", mike)
    np.save(out_dir / "h_itzi_surface.npy", surface)
    np.save(out_dir / "h_itzi_swmm_connected.npy", connected)
    np.save(out_dir / "h_connected_residual.npy", residual)
    np.save(out_dir / "h.npy", connected)

    event_metrics = {
        "event": event,
        "infiltration_mmh": INFILTRATION_MMH,
        "shape": "x".join(str(v) for v in connected.shape),
        "surface_peak_m": float(surface.max()),
        "connected_peak_m": float(connected.max()),
        "peak_reduction_mm": float((surface.max() - connected.max()) * 1000.0),
        "mean_abs_residual_mm": float(np.mean(np.abs(residual)) * 1000.0),
        "final_volume_surface_m3": float(surface[-1, mask].sum() * CELL_AREA_M2),
        "final_volume_connected_m3": float(connected[-1, mask].sum() * CELL_AREA_M2),
        "final_volume_reduction_m3": float((surface[-1, mask].sum() - connected[-1, mask].sum()) * CELL_AREA_M2),
        "surface_infiltrated_m3": float(rec_surf.get("infiltrated_m3", [np.nan])[-1]),
        "connected_infiltrated_m3": float(rec_swmm.get("infiltrated_m3", [np.nan])[-1]),
        "connected_drained_m3": float(rec_swmm.get("drained_m3", [np.nan])[-1]),
        "quality_status": status,
        "use_as_training_label": bool(use_label),
        **rpt_metrics,
    }
    write_event_metrics(out_dir / "swmm_connected_metrics.csv", event_metrics)
    return event_metrics


def make_quality_figure(rows: list[dict[str, object]]) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    events = [str(r["event"]) for r in rows]
    continuity = [float(r["continuity_error_pct"]) for r in rows]
    nonconv = [float(r["nonconverging_steps_pct"]) for r in rows]
    colors = [
        "#15803d" if r["quality_status"] == "accepted" else "#ca8a04" if r["quality_status"] == "warning" else "#dc2626"
        for r in rows
    ]
    x = np.arange(len(events))
    fig, ax1 = plt.subplots(figsize=(11.8, 5.6), constrained_layout=True)
    ax1.bar(x, continuity, color=colors, alpha=0.82, label="Continuity error")
    ax1.axhline(2.0, color="#15803d", lw=1.0, ls="--")
    ax1.axhline(-2.0, color="#15803d", lw=1.0, ls="--")
    ax1.axhline(8.0, color="#dc2626", lw=1.0, ls=":")
    ax1.axhline(-8.0, color="#dc2626", lw=1.0, ls=":")
    ax1.set_ylabel("SWMM continuity error (%)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(events, rotation=25)
    ax1.grid(axis="y", alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(x, nonconv, color="#1f2937", marker="o", lw=1.3, label="Steps not converging")
    ax2.set_ylabel("Nonconverging steps (%)")
    ax1.set_title("v2-infiltration ITZI-SWMM label quality grading")
    fig.savefig(FIG / "v2_infiltration_label_quality_grading.png", dpi=180)
    plt.close(fig)


def make_mike_summary_figure(rows: list[dict[str, object]]) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    events = []
    surface_mae = []
    connected_mae = []
    for row in rows:
        event = str(row["event"])
        d = OUT_FLOOD / event
        mike = np.load(d / "h_mike_ref.npy").astype(np.float32)
        surface = np.load(d / "h_itzi_surface.npy").astype(np.float32)
        connected = np.load(d / "h_itzi_swmm_connected.npy").astype(np.float32)
        events.append(event)
        surface_mae.append(float(np.mean(np.abs(surface - mike)) * 1000.0))
        connected_mae.append(float(np.mean(np.abs(connected - mike)) * 1000.0))
    x = np.arange(len(events))
    width = 0.36
    fig, ax = plt.subplots(figsize=(11.8, 5.4), constrained_layout=True)
    ax.bar(x - width / 2, surface_mae, width=width, label="ITZI surface-only", color="#2563eb")
    ax.bar(x + width / 2, connected_mae, width=width, label="ITZI-SWMM + infiltration", color="#dc2626")
    ax.set_xticks(x)
    ax.set_xticklabels(events, rotation=25)
    ax.set_ylabel("MAE to MIKE reference (mm)")
    ax.set_title("External MIKE comparison for v2-infiltration labels")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.savefig(FIG / "v2_infiltration_mike_mae.png", dpi=180)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    configure_paths(args)
    OUT_GEO.mkdir(parents=True, exist_ok=True)
    OUT_FLOOD.mkdir(parents=True, exist_ok=True)
    OUT_AUDIT.mkdir(parents=True, exist_ok=True)
    geo = parse_swmm_network((200, 280))
    for key in [
        "dem",
        "drain_inlet_mask",
        "drain_inlet_count",
        "drain_outfall_mask",
        "pipe_mask",
        "pipe_segment_count",
        "pipe_diameter",
        "pipe_slope",
        "pipe_capacity",
        "pipe_cover_depth",
        "distance_to_outfall",
    ]:
        np.save(OUT_GEO / f"{key}.npy", geo[key])

    event_rows = [build_event(event, geo["dem"]) for event in EVENTS]
    quality_counts: dict[str, int] = {}
    for row in event_rows:
        quality_counts[str(row["quality_status"])] = quality_counts.get(str(row["quality_status"]), 0) + 1
    summary_rows = [
        {"quality_status": status, "n_events": count, "share_pct": count / len(event_rows) * 100.0}
        for status, count in sorted(quality_counts.items())
    ]
    write_csv(OUT_AUDIT / "v2_infiltration_event_summary.csv", event_rows)
    write_csv(OUT_AUDIT / "v2_infiltration_label_quality_grading.csv", event_rows)
    write_csv(OUT_AUDIT / "v2_infiltration_label_quality_summary.csv", summary_rows)
    write_csv(OUT_FLOOD / "connected_swmm_metrics_rebuilt.csv", event_rows)
    write_csv(OUT_FLOOD / "v2_infiltration_label_quality_grading.csv", event_rows)
    make_quality_figure(event_rows)
    make_mike_summary_figure(event_rows)

    metadata = {
        "location": args.dataset_name,
        "cell_size_m": 20.0,
        "shape": [200, 280],
        "time_steps": 72,
        "events": EVENTS,
        "infiltration_mmh": INFILTRATION_MMH,
        "target_h": f"h.npy is ITZI native SurfaceFlow + SWMM DYNWAVE coupled water depth with {INFILTRATION_MMH:g} mm/h effective active-cell infiltration.",
        "surface_baseline": "h_itzi_surface.npy is ITZI dynamic surface-only water depth with the same rainfall timing and infiltration setting.",
        "connected_label": "h_itzi_swmm_connected.npy is native ITZI DrainageSimulation + SWMM DYNWAVE connected network water depth.",
        "residual": "h_connected_residual.npy = h_itzi_swmm_connected - h_itzi_surface, in m.",
        "quality_grading": {
            "accepted": "abs(SWMM continuity error) <= 2%",
            "warning": "2% < abs(SWMM continuity error) <= 8%",
            "excluded": "abs(SWMM continuity error) > 8%; keep for diagnosis, avoid overclaiming as formal calibrated label",
            "counts": quality_counts,
        },
        "static_features": [
            "drain_inlet_mask",
            "drain_inlet_count",
            "drain_outfall_mask",
            "pipe_mask",
            "pipe_segment_count",
            "pipe_diameter",
            "pipe_slope",
            "pipe_capacity",
            "pipe_cover_depth",
            "distance_to_outfall",
        ],
        "source_run_dir": str(RUN_DIR),
        "source_workflow": str(ROOT / "extended_study" / "ITZI_SWMM_FIXED_WORKFLOW.md"),
        "network_audit": geo["audit"],
        "boundary_statement": "This is not a surveyed real municipal network. It is the fixed road-aligned conceptual connected SWMM main network coupled through native ITZI drainage APIs.",
    }
    (OUT_GEO / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_FLOOD / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_AUDIT / "v2_infiltration_dataset_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if INP.exists():
        shutil.copy2(INP, OUT_AUDIT / INP.name)
    print(
        json.dumps(
            {
                "dataset": metadata["location"],
                "events": EVENTS,
                "quality_counts": quality_counts,
                "geodata": str(OUT_GEO),
                "flood": str(OUT_FLOOD),
                "audit": str(OUT_AUDIT),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
