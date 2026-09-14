#!/usr/bin/env python3
"""Import copied formal ITZI-SWMM coupled labels into the LarNO drainage dataset.

The source model is the copied project under external_models/20260518-itzi-flood.
This script never reads from or writes to the original E:/Projects/20260518-itzi-flood
path.  It keeps the previous conceptual sink labels and adds official coupled
labels under explicit filenames.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = (
    ROOT
    / "external_models"
    / "20260518-itzi-flood"
    / "test_cases"
    / "shenzhen_region1"
    / "output"
    / "full_domain_swmm"
)
SOURCE_CASE_DIR = (
    ROOT
    / "external_models"
    / "20260518-itzi-flood"
    / "test_cases"
    / "shenzhen_region1"
)
DATASET_DIR = (
    ROOT
    / "LarNO-main"
    / "benchmark"
    / "urbanflood"
    / "flood"
    / "region1_20m_drainage_v1"
)
GEODATA_DIR = (
    ROOT
    / "LarNO-main"
    / "benchmark"
    / "urbanflood"
    / "geodata"
    / "region1_20m_drainage_v1"
)
OUT_DIR = ROOT / "extended_study" / "output" / "official_itzi_swmm_import"
EVENTS = ["event1", "event20", "event65", "event66", "event67", "event68", "event69", "event70"]
EXPECTED_SHAPE = (72, 200, 280)
CELL_AREA_M2 = 20.0 * 20.0


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def csi(pred: np.ndarray, ref: np.ndarray, threshold: float) -> float:
    p = pred > threshold
    r = ref > threshold
    tp = int(np.logical_and(p, r).sum())
    fp = int(np.logical_and(p, ~r).sum())
    fn = int(np.logical_and(~p, r).sum())
    denom = tp + fp + fn
    return float(tp / denom) if denom else 1.0


def np_stats(arr: np.ndarray) -> dict[str, float]:
    return {
        "min": float(np.nanmin(arr)),
        "max": float(np.nanmax(arr)),
        "mean": float(np.nanmean(arr)),
    }


def count_swmm(inp_path: Path) -> dict[str, int | str]:
    counts: dict[str, int | str] = {
        "routing": "",
        "junctions": 0,
        "outfalls": 0,
        "conduits": 0,
        "xsections": 0,
        "coordinates": 0,
    }
    section = None
    mapping = {
        "[JUNCTIONS]": "junctions",
        "[OUTFALLS]": "outfalls",
        "[CONDUITS]": "conduits",
        "[XSECTIONS]": "xsections",
        "[COORDINATES]": "coordinates",
    }
    for raw in inp_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if line.startswith("FLOW_ROUTING"):
            counts["routing"] = line.split()[-1]
        if line in mapping:
            section = mapping[line]
            continue
        if line.startswith("["):
            section = None
            continue
        if section and line and not line.startswith(";"):
            counts[section] = int(counts[section]) + 1
    return counts


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dem = np.load(GEODATA_DIR / "dem.npy", mmap_mode="r")
    active = np.asarray(dem) < 49.9
    rows: list[dict[str, object]] = []
    summary_obj: dict[str, object] = {}

    for event in EVENTS:
        source_npz = SOURCE_DIR / f"{event}_coupled.npz"
        event_dir = DATASET_DIR / event
        if not source_npz.exists():
            raise FileNotFoundError(source_npz)
        if not event_dir.exists():
            raise FileNotFoundError(event_dir)

        z = np.load(source_npz, allow_pickle=True)
        h_surf_series = np.asarray(z["h_surf_series"], dtype=np.float32)
        h_swmm_series = np.asarray(z["h_swmm_series"], dtype=np.float32)
        h_mike = np.load(event_dir / "h_mike_ref.npy").astype(np.float32)
        h_sink = np.load(event_dir / "h_itzi_sink.npy").astype(np.float32)

        if h_surf_series.shape != EXPECTED_SHAPE:
            raise ValueError(f"{event} h_surf_series shape {h_surf_series.shape}, expected {EXPECTED_SHAPE}")
        if h_swmm_series.shape != EXPECTED_SHAPE:
            raise ValueError(f"{event} h_swmm_series shape {h_swmm_series.shape}, expected {EXPECTED_SHAPE}")
        if not np.isfinite(h_swmm_series).all():
            raise ValueError(f"{event} h_swmm_series contains NaN/Inf")

        reduction = np.maximum(h_surf_series - h_swmm_series, 0.0).astype(np.float32)

        np.save(event_dir / "h_itzi_swmm_official.npy", h_swmm_series)
        np.save(event_dir / "h_itzi_swmm_official_surface.npy", h_surf_series)
        np.save(event_dir / "h_itzi_swmm_official_reduction.npy", reduction)

        peak_mike = np.max(h_mike, axis=0)
        peak_surface = np.max(h_surf_series, axis=0)
        peak_swmm = np.max(h_swmm_series, axis=0)
        peak_sink = np.max(h_sink, axis=0)
        final_mike = h_mike[-1]
        final_surface = h_surf_series[-1]
        final_swmm = h_swmm_series[-1]
        final_sink = h_sink[-1]
        final_surcharge = np.maximum(final_swmm - final_surface, 0.0)
        rec_swmm = z["rec_swmm"].item()
        rec_surface = z["rec_surf"].item()

        row = {
            "event": event,
            "frames": int(h_swmm_series.shape[0]),
            "height": int(h_swmm_series.shape[1]),
            "width": int(h_swmm_series.shape[2]),
            "mike_peak_max_m": float(np.max(peak_mike)),
            "surface_peak_max_m": float(np.max(peak_surface)),
            "official_swmm_peak_max_m": float(np.max(peak_swmm)),
            "conceptual_sink_peak_max_m": float(np.max(peak_sink)),
            "surface_vs_mike_mae_m": mae(h_surf_series, h_mike),
            "official_swmm_vs_mike_mae_m": mae(h_swmm_series, h_mike),
            "conceptual_sink_vs_mike_mae_m": mae(h_sink, h_mike),
            "surface_peak_vs_mike_mae_m": mae(peak_surface, peak_mike),
            "official_swmm_peak_vs_mike_mae_m": mae(peak_swmm, peak_mike),
            "conceptual_sink_peak_vs_mike_mae_m": mae(peak_sink, peak_mike),
            "official_swmm_final_vs_mike_mae_m": mae(final_swmm, final_mike),
            "official_swmm_final_vs_mike_rmse_m": rmse(final_swmm, final_mike),
            "official_swmm_peak_csi_003": csi(peak_swmm, peak_mike, 0.03),
            "official_swmm_peak_csi_015": csi(peak_swmm, peak_mike, 0.15),
            "official_swmm_final_volume_m3": float(np.sum(final_swmm[active]) * CELL_AREA_M2),
            "surface_final_volume_m3": float(np.sum(final_surface[active]) * CELL_AREA_M2),
            "conceptual_sink_final_volume_m3": float(np.sum(final_sink[active]) * CELL_AREA_M2),
            "official_swmm_positive_reduction_volume_m3": float(np.sum(reduction[-1][active]) * CELL_AREA_M2),
            "official_swmm_local_surcharge_volume_m3": float(np.sum(final_surcharge[active]) * CELL_AREA_M2),
            "official_swmm_local_surcharge_cells": int(np.sum(final_surcharge > 1e-6)),
            "official_swmm_drained_m3": float(rec_swmm["drained_m3"][-1]) if rec_swmm["drained_m3"] else 0.0,
            "surface_recorded_final_volume_m3": float(rec_surface["vol_m3"][-1]) if rec_surface["vol_m3"] else 0.0,
        }
        rows.append(row)
        summary_obj[event] = row

    summary_npz = SOURCE_DIR / "summary_all_events_timeseries.npz"
    np.savez(summary_npz, summary=summary_obj)

    csv_path = OUT_DIR / "official_itzi_swmm_event_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    manifest = {
        "source_project_copy": str(SOURCE_CASE_DIR),
        "source_result_dir": str(SOURCE_DIR),
        "dataset_dir": str(DATASET_DIR),
        "events": EVENTS,
        "expected_shape": EXPECTED_SHAPE,
        "new_dataset_files": [
            "h_itzi_swmm_official.npy",
            "h_itzi_swmm_official_surface.npy",
            "h_itzi_swmm_official_reduction.npy",
        ],
        "old_labels_preserved": ["h.npy", "h_itzi_sink.npy", "h_itzi_surface.npy", "h_mike_ref.npy"],
        "swmm_network": count_swmm(SOURCE_CASE_DIR / "input_data" / "networks" / "swmm_coupled_sub.inp"),
        "notes": [
            "Official ITZI-SWMM labels are imported from the copied calibrated project only.",
            "Conceptual sink labels are preserved and must not be described as full ITZI-SWMM coupling.",
            "h.npy is not overwritten; downstream training must choose the target explicitly.",
        ],
    }
    manifest_path = OUT_DIR / "official_itzi_swmm_import_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    md_lines = [
        "# Official ITZI-SWMM Import Audit",
        "",
        "This audit supersedes the earlier prototype-only interpretation. The formal",
        "coupled labels now come from the copied calibrated project:",
        "",
        f"- Source copy: `{SOURCE_CASE_DIR}`",
        f"- Imported dataset: `{DATASET_DIR}`",
        "- Original user project path was not modified.",
        "",
        "## Added dataset files",
        "",
        "- `h_itzi_swmm_official.npy`: formal ITZI-SWMM coupled water-depth series.",
        "- `h_itzi_swmm_official_surface.npy`: matching formal surface-only series.",
        "- `h_itzi_swmm_official_reduction.npy`: nonnegative surface minus coupled reduction field.",
        "",
        "## SWMM network",
        "",
        f"- Routing: {manifest['swmm_network']['routing']}",
        f"- Junctions: {manifest['swmm_network']['junctions']}",
        f"- Outfalls: {manifest['swmm_network']['outfalls']}",
        f"- Conduits: {manifest['swmm_network']['conduits']}",
        "",
        "## Event metric table",
        "",
        "| Event | MIKE peak m | Surface peak m | Official SWMM peak m | SWMM drained m3 | Local surcharge cells | SWMM vs MIKE MAE m |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        md_lines.append(
            f"| {row['event']} | {row['mike_peak_max_m']:.3f} | "
            f"{row['surface_peak_max_m']:.3f} | {row['official_swmm_peak_max_m']:.3f} | "
            f"{row['official_swmm_drained_m3']:.0f} | "
            f"{row['official_swmm_local_surcharge_cells']} | "
            f"{row['official_swmm_vs_mike_mae_m']:.4f} |"
        )
    md_lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The official coupled model produces a measurable but modest drainage effect in",
            "this sub-region. The conceptual inlet-sink label generally removes more water",
            "than the formal ITZI-SWMM coupled result and must be treated as a separate",
            "scenario, not as the formal coupled reference.",
            "",
            "Because the formal model is bidirectionally coupled, the coupled water depth",
            "is not required to be lower than the surface-only result at every cell. Local",
            "surcharge/backwater cells are recorded separately. A strict monotone reduction",
            "constraint is valid only for the conceptual residual/sink model.",
            "",
        ]
    )
    (OUT_DIR / "OFFICIAL_ITZI_SWMM_IMPORT_AUDIT.md").write_text("\n".join(md_lines), encoding="utf-8")

    print(f"Imported {len(rows)} events")
    print(f"Metrics: {csv_path}")
    print(f"Manifest: {manifest_path}")
    print(f"Summary npz: {summary_npz}")


if __name__ == "__main__":
    main()
