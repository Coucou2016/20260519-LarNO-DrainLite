#!/usr/bin/env python3
"""Validate arrays, metrics, embedded documents and evidence hashes."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path

import numpy as np
from bs4 import BeautifulSoup
from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "extended_study" / "output" / "larno_drainlite_major_revision"
GEO = ROOT / "LarNO-main" / "benchmark" / "urbanflood" / "geodata" / "region1_20m_connected_swmm_v2_inf1mmh"
FLOOD = ROOT / "LarNO-main" / "benchmark" / "urbanflood" / "flood" / "region1_20m_connected_swmm_v2_inf1mmh"
EVENTS = ["event1", "event67", "event68", "event69", "event70"]
STATIC = [
    "drain_inlet_mask.npy",
    "drain_outfall_mask.npy",
    "pipe_mask.npy",
    "pipe_diameter.npy",
    "pipe_slope.npy",
    "pipe_capacity.npy",
    "pipe_cover_depth.npy",
    "distance_to_outfall.npy",
    "drain_inlet_count.npy",
    "pipe_segment_count.npy",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str, checks: list[str]) -> None:
    if not condition:
        raise AssertionError(message)
    checks.append(message)


def main() -> int:
    checks: list[str] = []
    for name in STATIC:
        array = np.load(GEO / name, mmap_mode="r")
        require(array.shape == (200, 280), f"{name}: shape 200 x 280", checks)
        require(np.isfinite(array).all(), f"{name}: finite", checks)

    for event in EVENTS:
        arrays = {}
        for name in ["rainfall.npy", "h_itzi_surface.npy", "h_itzi_swmm_connected.npy", "h_mike_ref.npy"]:
            arrays[name] = np.load(FLOOD / event / name, mmap_mode="r")
            require(arrays[name].shape == (72, 200, 280), f"{event}/{name}: shape 72 x 200 x 280", checks)
            require(np.isfinite(arrays[name]).all(), f"{event}/{name}: finite", checks)
        surface = arrays["h_itzi_surface.npy"]
        for name in ["h_no_xy_all_static.npy", "h_no_xy_all_static_count.npy", "h_sink_only_no_xy_all_static_count.npy"]:
            pred = np.load(OUT / "predictions" / event / name, mmap_mode="r")
            require(pred.shape == (72, 200, 280), f"{event}/{name}: full prediction shape", checks)
            require(np.isfinite(pred).all(), f"{event}/{name}: finite", checks)
            require(float(pred.min()) >= -1e-7, f"{event}/{name}: nonnegative", checks)
            if name.startswith("h_sink_only"):
                require(float(np.max(pred - surface)) <= 1e-6, f"{event}/{name}: does not exceed surface-only", checks)

    with (OUT / "metrics" / "event_metrics.csv").open(encoding="utf-8-sig", newline="") as handle:
        event_rows = list(csv.DictReader(handle))
    with (OUT / "metrics" / "ablation_summary.csv").open(encoding="utf-8-sig", newline="") as handle:
        summary_rows = list(csv.DictReader(handle))
    require(len(event_rows) == 55, "event_metrics.csv: 11 configurations x 5 held events", checks)
    require(len(summary_rows) == 11, "ablation_summary.csv: 11 configurations", checks)
    event_metric_lookup = {(row["event"], row["model"]): row for row in event_rows}
    summary_lookup = {row["model"]: row for row in summary_rows}
    active = np.isfinite(np.load(GEO / "dem.npy")) & (np.load(GEO / "dem.npy") < 49.9)
    verified_predictions = {
        "surface_only": None,
        "base_no_xy": "h_base_no_xy.npy",
        "no_xy_all_static": "h_no_xy_all_static.npy",
        "no_xy_all_static_count": "h_no_xy_all_static_count.npy",
        "sink_only_no_xy_all_static_count": "h_sink_only_no_xy_all_static_count.npy",
    }
    recomputed: dict[str, list[dict[str, float]]] = {name: [] for name in verified_predictions}
    for event in EVENTS:
        target = np.load(FLOOD / event / "h_itzi_swmm_connected.npy", mmap_mode="r")
        surface = np.load(FLOOD / event / "h_itzi_surface.npy", mmap_mode="r")
        target_values = np.asarray(target[:, active])
        for model, filename in verified_predictions.items():
            prediction = surface if filename is None else np.load(OUT / "predictions" / event / filename, mmap_mode="r")
            prediction_values = np.asarray(prediction[:, active])
            # Match the experiment runner's float32 metric arithmetic exactly;
            # converting to float64 changes RMSE at approximately 1e-9 m.
            difference = prediction_values - target_values
            calculated = {
                "mae_m": float(np.mean(np.abs(difference))),
                "rmse_m": float(np.sqrt(np.mean(np.square(difference)))),
            }
            for threshold, key in [(0.03, "csi_0p03"), (0.15, "csi_0p15")]:
                predicted_wet = prediction_values >= threshold
                target_wet = target_values >= threshold
                hits = np.count_nonzero(predicted_wet & target_wet)
                union = np.count_nonzero(predicted_wet | target_wet)
                calculated[key] = float(hits / union) if union else 1.0
            reported = event_metric_lookup[(event, model)]
            for key, value in calculated.items():
                require(abs(value - float(reported[key])) < 1e-9, f"{event}/{model}: {key} recomputed from NPY", checks)
            recomputed[model].append(calculated)
    for model, event_values in recomputed.items():
        summary = summary_lookup[model]
        for event_key, summary_key, scale in [
            ("mae_m", "mae_mm", 1000.0),
            ("rmse_m", "rmse_mm", 1000.0),
            ("csi_0p03", "csi_0p03", 1.0),
            ("csi_0p15", "csi_0p15", 1.0),
        ]:
            value = float(np.mean([row[event_key] for row in event_values])) * scale
            require(abs(value - float(summary[summary_key])) < 1e-8, f"{model}: {summary_key} macro-average recomputed", checks)
    timestamp_path = OUT / "metrics" / "timestamp_convention.json"
    require(timestamp_path.exists(), "timestamp convention metadata exists", checks)
    timestamp = json.loads(timestamp_path.read_text(encoding="utf-8"))
    require(timestamp["convention"] == "interval_end_5min", "timestamp convention: five-minute interval end", checks)
    require(abs(float(timestamp["first_frame_h"]) - 1.0 / 12.0) < 1e-12, "timestamp convention: first frame at 0.0833 h", checks)
    require(abs(float(timestamp["last_frame_h"]) - 6.0) < 1e-12, "timestamp convention: last frame at 6 h", checks)
    for row in event_rows:
        predicted = float(row["peak_time_pred_h"])
        target = float(row["peak_time_target_h"])
        difference = float(row["peak_time_error_h"])
        absolute = float(row["abs_peak_time_error_h"])
        require(1.0 / 12.0 - 1e-12 <= predicted <= 6.0 + 1e-12, f"{row['event']}/{row['model']}: predicted peak uses interval-end time", checks)
        require(1.0 / 12.0 - 1e-12 <= target <= 6.0 + 1e-12, f"{row['event']}/{row['model']}: target peak uses interval-end time", checks)
        require(abs((predicted - target) - difference) < 1e-9, f"{row['event']}/{row['model']}: signed peak-time difference consistent", checks)
        require(abs(abs(difference) - absolute) < 1e-9, f"{row['event']}/{row['model']}: absolute peak-time difference consistent", checks)

    for stem in ["manuscript_major_revision", "report"]:
        markdown_text = (OUT / f"{stem}.md").read_text(encoding="utf-8")
        figure_numbers = [int(value) for value in re.findall(r"\*\*(?:Figure|图) (\d+)\.", markdown_text)]
        require(figure_numbers == list(range(1, 12)), f"{stem}.md: figures numbered 1-11 in first-use order", checks)
        html_path = OUT / f"{stem}.html"
        raw_html = html_path.read_text(encoding="utf-8")
        require(raw_html.lstrip().lower().startswith("<!doctype html>"), f"{stem}.html: complete HTML document", checks)
        soup = BeautifulSoup(raw_html, "html.parser")
        images = soup.find_all("img")
        require(len(images) == 11, f"{stem}.html: 11 figures", checks)
        require(all(str(image.get("src", "")).startswith("data:image/") for image in images), f"{stem}.html: all images Base64 embedded", checks)
        require("publication_figures/" not in raw_html, f"{stem}.html: no local image path", checks)

    manifest = json.loads((OUT / "evidence_manifest.json").read_text(encoding="utf-8"))
    require(len(manifest["records"]) >= 40, "evidence manifest: expanded source/result coverage", checks)
    for record in manifest["records"]:
        path = ROOT / record["path"]
        require(path.exists(), f"manifest file exists: {record['path']}", checks)
        require(path.stat().st_size == record["bytes"], f"manifest byte count: {record['path']}", checks)
        require(sha256(path) == record["sha256"], f"manifest SHA-256: {record['path']}", checks)

    pdf_pages = {}
    for stem in ["manuscript_major_revision", "report", "scientific_integrity_audit"]:
        reader = PdfReader(OUT / f"{stem}.pdf")
        pdf_pages[stem] = len(reader.pages)
        require(len(reader.pages) > 0, f"{stem}.pdf: readable PDF", checks)
        text = "".join((page.extract_text() or "") for page in reader.pages)
        require(len(text) > 1000, f"{stem}.pdf: extractable text", checks)

    report = {
        "status": "passed",
        "formal_events": EVENTS,
        "checks_passed": len(checks),
        "pdf_pages": pdf_pages,
        "key_invariants": {
            "array_shape": [72, 200, 280],
            "static_shape": [200, 280],
            "html_figures_embedded": 11,
            "manifest_records": len(manifest["records"]),
        },
    }
    (OUT / "validation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
