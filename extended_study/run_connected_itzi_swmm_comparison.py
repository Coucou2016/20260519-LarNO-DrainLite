"""Run ITZI-SWMM with the repaired connected SWMM network and compare to surface-only."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import re
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(r"E:\Projects\20260519-LarNO")
CASE = ROOT / "external_models" / "20260518-itzi-flood" / "test_cases" / "shenzhen_region1"
RUNNER = CASE / "run_full_domain_coupled.py"
CONNECTED_INP = CASE / "input_data" / "networks" / "swmm_connected_sub.inp"
CONNECTED_RPT = CONNECTED_INP.with_suffix(".rpt")
OUT_BASE = ROOT / "extended_study" / "output" / "connected_itzi_swmm"
OUT = OUT_BASE
FIG = OUT / "figures"

EVENTS = ["event1", "event20", "event65", "event66", "event67", "event68", "event69", "event70"]
CELL_AREA = 20.0 * 20.0


def load_runner():
    spec = importlib.util.spec_from_file_location("connected_runner", RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {RUNNER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["connected_runner"] = module
    spec.loader.exec_module(module)
    return module


def parse_swmm_report(path: Path) -> dict:
    text = path.read_text(errors="ignore") if path.exists() else ""
    patterns = {
        "continuity_error_pct": r"Continuity Error \(%\)\s+\.*\s+([-+0-9.]+)",
        "external_inflow_m3": r"External Inflow \.*\s+[-+0-9.]+\s+([-+0-9.]+)",
        "external_outflow_m3": r"External Outflow \.*\s+[-+0-9.]+\s+([-+0-9.]+)",
        "flooding_loss_m3": r"Flooding Loss \.*\s+[-+0-9.]+\s+([-+0-9.]+)",
        "final_stored_m3": r"Final Stored Volume \.*\s+[-+0-9.]+\s+([-+0-9.]+)",
        "nonconverging_steps_pct": r"% of Steps Not Converging\s+:\s+([-+0-9.]+)",
    }
    out = {"report_exists": path.exists(), "report_path": str(path)}
    for key, pat in patterns.items():
        m = re.search(pat, text)
        out[key] = float(m.group(1)) if m else None
    # Report table second numeric volume column is 10^6 liters, equal to 1000 m3.
    for key in ["external_inflow_m3", "external_outflow_m3", "flooding_loss_m3", "final_stored_m3"]:
        if out[key] is not None:
            out[key] *= 1000.0
    return out


def volume(h: np.ndarray) -> np.ndarray:
    return h.sum(axis=(1, 2)) * CELL_AREA


def peak(h: np.ndarray) -> np.ndarray:
    return h.max(axis=(1, 2))


def flooded_area(h: np.ndarray) -> np.ndarray:
    return (h > 0.03).sum(axis=(1, 2)) * CELL_AREA / 1_000_000.0


def setup_style() -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.unicode_minus": False, "figure.dpi": 160})


def set_output_dir(infiltration_mmh: float) -> None:
    global OUT, FIG
    if infiltration_mmh > 0:
        safe_rate = f"{infiltration_mmh:g}".replace(".", "p")
        OUT = ROOT / "extended_study" / "output" / f"connected_itzi_swmm_inf_{safe_rate}mmh"
    else:
        OUT = OUT_BASE
    FIG = OUT / "figures"


def plot_event(event: str, h_ref: np.ndarray, h_s: np.ndarray, h_w: np.ndarray) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    t = np.arange(1, h_s.shape[0] + 1) * 5.0 / 60.0

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.2), constrained_layout=True)
    ax = axes[0, 0]
    ax.plot(t, peak(h_ref), color="#111827", label="MIKE reference", lw=1.7)
    ax.plot(t, peak(h_s), color="#2563eb", label="ITZI surface-only", lw=1.7)
    ax.plot(t, peak(h_w), color="#dc2626", label="ITZI-SWMM connected", lw=1.6)
    ax.set_title("Maximum water depth")
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("m")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.plot(t, volume(h_ref) / 1000.0, color="#111827", label="MIKE reference", lw=1.7)
    ax.plot(t, volume(h_s) / 1000.0, color="#2563eb", label="ITZI surface-only", lw=1.7)
    ax.plot(t, volume(h_w) / 1000.0, color="#dc2626", label="ITZI-SWMM connected", lw=1.6)
    ax.set_title("Surface water volume")
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("10^3 m3")
    ax.grid(alpha=0.25)

    ax = axes[1, 0]
    ax.plot(t, flooded_area(h_ref), color="#111827", label="MIKE reference", lw=1.7)
    ax.plot(t, flooded_area(h_s), color="#2563eb", label="ITZI surface-only", lw=1.7)
    ax.plot(t, flooded_area(h_w), color="#dc2626", label="ITZI-SWMM connected", lw=1.6)
    ax.set_title("Flooded area above 0.03 m")
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("km2")
    ax.grid(alpha=0.25)

    ax = axes[1, 1]
    ax.plot(t, (peak(h_s) - peak(h_w)) * 1000.0, color="#b91c1c", lw=1.7, label="Peak reduction")
    ax2 = ax.twinx()
    ax2.plot(t, (volume(h_s) - volume(h_w)) / 1000.0, color="#f97316", lw=1.5, ls=":", label="Volume reduction")
    ax.set_title("Connected SWMM effect")
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Peak reduction (mm)")
    ax2.set_ylabel("Volume reduction (10^3 m3)")
    ax.grid(alpha=0.25)
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, fontsize=8)

    fig.suptitle(f"{event}: repaired connected ITZI-SWMM versus surface-only", fontsize=14, weight="bold")
    fig.savefig(FIG / f"{event}_connected_itzi_swmm_timeseries.png", bbox_inches="tight")
    plt.close(fig)

    ref_peak = h_ref.max(axis=0)
    surf_peak = h_s.max(axis=0)
    swmm_peak = h_w.max(axis=0)
    diff = surf_peak - swmm_peak
    vmax = float(max(ref_peak.max(), surf_peak.max(), swmm_peak.max()))
    lim = max(0.05, float(np.max(np.abs(diff))))
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.8), constrained_layout=True)
    specs = [
        ("MIKE reference peak", ref_peak, "viridis", 0, vmax),
        ("ITZI surface-only peak", surf_peak, "viridis", 0, vmax),
        ("Connected ITZI-SWMM peak", swmm_peak, "viridis", 0, vmax),
        ("Surface - connected SWMM", diff, "coolwarm", -lim, lim),
    ]
    for ax, (title, arr, cmap, vmin, vmax_i) in zip(axes, specs):
        im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax_i)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cb.set_label("m")
    fig.suptitle(f"{event}: repaired connected ITZI-SWMM peak-depth maps", fontsize=14, weight="bold")
    fig.savefig(FIG / f"{event}_connected_itzi_swmm_spatial.png", bbox_inches="tight")
    plt.close(fig)


def run_events(events: list[str], infiltration_mmh: float = 0.0) -> list[dict]:
    runner = load_runner()
    dem, bldg, slc, full_shape = runner.load_domain("sub")
    h_full, w_full = full_shape
    row_off = slc[0].start
    col_off = slc[1].start
    geo_kw = dict(H_full=h_full, W_full=w_full, row_off=row_off, col_off=col_off)

    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    rows = []

    for event in events:
        event_dir = Path(runner.FLOOD_DIR) / event
        rainfall_full = np.load(event_dir / "rainfall.npy")
        h_ref_full = np.load(event_dir / "h.npy")
        rainfall = rainfall_full[(slice(None),) + slc]
        h_ref = h_ref_full[(slice(None),) + slc].astype(np.float32)

        print(f"\n=== {event} surface-only ===", flush=True)
        rec_s, h_s_final, h_s = runner.run_simulation(
            f"{event} connected surface",
            dem.copy(),
            bldg.copy(),
            rainfall,
            swmm_inp=None,
            save_timeseries=True,
            infiltration_mmh=infiltration_mmh,
            **geo_kw,
        )
        print(f"\n=== {event} connected ITZI-SWMM ===", flush=True)
        rec_w, h_w_final, h_w = runner.run_simulation(
            f"{event} connected swmm",
            dem.copy(),
            bldg.copy(),
            rainfall,
            swmm_inp=str(CONNECTED_INP),
            save_timeseries=True,
            infiltration_mmh=infiltration_mmh,
            **geo_kw,
        )

        if CONNECTED_RPT.exists():
            event_rpt = OUT / f"{event}_swmm_connected.rpt"
            shutil.copy2(CONNECTED_RPT, event_rpt)
        else:
            event_rpt = CONNECTED_RPT
        report = parse_swmm_report(event_rpt)

        event_out = OUT / event
        event_out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            event_out / f"{event}_connected_itzi_swmm.npz",
            h_ref=h_ref,
            h_surf=h_s,
            h_swmm_connected=h_w,
            h_surf_final=h_s_final,
            h_swmm_connected_final=h_w_final,
            rec_surf=rec_s,
            rec_swmm=rec_w,
            swmm_report=report,
        )
        plot_event(event, h_ref, h_s, h_w)

        diff = h_s - h_w
        row = {
            "event": event,
            "surface_peak_m": float(peak(h_s).max()),
            "connected_swmm_peak_m": float(peak(h_w).max()),
            "peak_reduction_mm": float((peak(h_s).max() - peak(h_w).max()) * 1000.0),
            "mean_abs_diff_mm": float(np.mean(np.abs(diff)) * 1000.0),
            "max_abs_diff_m": float(np.max(np.abs(diff))),
            "final_volume_reduction_m3": float(volume(h_s)[-1] - volume(h_w)[-1]),
            "max_volume_reduction_m3": float(np.max(volume(h_s) - volume(h_w))),
            "final_flooded_area_reduction_km2": float(flooded_area(h_s)[-1] - flooded_area(h_w)[-1]),
            "infiltration_mmh": float(infiltration_mmh),
            "surface_infiltrated_m3": float(rec_s["infiltrated_m3"][-1]) if rec_s.get("infiltrated_m3") else 0.0,
            "connected_infiltrated_m3": float(rec_w["infiltrated_m3"][-1]) if rec_w.get("infiltrated_m3") else 0.0,
            **report,
        }
        rows.append(row)
        print(json.dumps(row, indent=2), flush=True)

    return rows


def write_rows(rows: list[dict]) -> None:
    if not rows:
        return
    csv_path = OUT / "connected_itzi_swmm_metrics.csv"
    json_path = OUT / "connected_itzi_swmm_metrics.json"
    merged: dict[str, dict] = {}
    if csv_path.exists():
        with csv_path.open("r", newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                if row.get("event"):
                    merged[row["event"]] = dict(row)
    for row in rows:
        merged[row["event"]] = row
    ordered = [merged[event] for event in EVENTS if event in merged]
    extra = [row for key, row in merged.items() if key not in EVENTS]
    ordered.extend(sorted(extra, key=lambda row: row.get("event", "")))
    fieldnames: list[str] = []
    for row in ordered:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(ordered)
    json_path.write_text(json.dumps(ordered, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", nargs="+", default=["event1"])
    parser.add_argument("--infiltration-mmh", type=float, default=0.0)
    args = parser.parse_args()
    setup_style()
    set_output_dir(args.infiltration_mmh)
    rows = run_events(args.events, infiltration_mmh=args.infiltration_mmh)
    write_rows(rows)
    print(f"Wrote connected ITZI-SWMM comparison to {OUT}")


if __name__ == "__main__":
    main()
