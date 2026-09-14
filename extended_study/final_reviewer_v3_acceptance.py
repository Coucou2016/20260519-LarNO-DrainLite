#!/usr/bin/env python3
"""Independently verify reviewer-v3 arrays, metrics, and final artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from bs4 import BeautifulSoup
from PIL import Image
from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]
REV = ROOT / "extended_study" / "output" / "reviewer_major_revision_v3"
PKG = REV / "submission_package_v3"
DATASET = "region1_20m_drainage_v3_full"
FLOOD = ROOT / "LarNO-main" / "benchmark" / "urbanflood" / "flood" / DATASET
GEO = ROOT / "LarNO-main" / "benchmark" / "urbanflood" / "geodata" / DATASET
HYBRID = REV / "drainlite_hybrid_v3"
EVENTS = ["event1", "event20", "event65", "event66", "event67", "event68", "event69", "event70"]
SHAPE = (72, 400, 560)
STATIC_SHAPE = (400, 560)
CELL_AREA_M2 = 400.0


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def csi(pred: np.ndarray, target: np.ndarray, threshold: float) -> float:
    p, t = pred >= threshold, target >= threshold
    tp = np.logical_and(p, t).sum(dtype=np.int64)
    union = tp + np.logical_and(p, ~t).sum(dtype=np.int64) + np.logical_and(~p, t).sum(dtype=np.int64)
    return 1.0 if union == 0 else float(tp / union)


def main() -> int:
    checks: list[dict[str, object]] = []
    failures: list[str] = []
    active = np.load(GEO / "active_mask.npy").astype(bool)
    checks.append({"check": "active_cell_count", "value": int(active.sum()), "expected": 105527, "pass": int(active.sum()) == 105527})
    static_files = sorted(GEO.glob("*.npy"))
    for path in static_files:
        values = np.load(path, mmap_mode="r")
        ok = values.shape == STATIC_SHAPE and np.isfinite(values).all()
        checks.append({"check": f"static:{path.name}", "shape": list(values.shape), "finite": bool(np.isfinite(values).all()), "pass": bool(ok)})
        if not ok:
            failures.append(f"Static feature failed: {path}")

    physics_rows = {row["event"]: row for row in read_csv(REV / "formal_matched_full" / "physics_quality.csv")}
    metric_rows = {(row["event"], row["model"], row["reference"]): row for row in read_csv(HYBRID / "metrics" / "event_metrics.csv")}
    recomputed_mae = []
    for event in EVENTS:
        directory = FLOOD / event
        required = ["rainfall.npy", "h_itzi_surface.npy", "h_itzi_surface_matched.npy", "h_itzi_swmm.npy", "h_mike_ref.npy", "h.npy"]
        loaded = {}
        for name in required:
            path = directory / name
            values = np.load(path, mmap_mode="r")
            ok = values.shape == SHAPE and np.isfinite(values).all()
            checks.append({"check": f"event_array:{event}:{name}", "shape": list(values.shape), "finite": bool(np.isfinite(values).all()), "pass": bool(ok)})
            if not ok:
                failures.append(f"Event array failed: {path}")
            loaded[name] = values
        target_alias_equal = np.array_equal(loaded["h.npy"], loaded["h_itzi_swmm.npy"])
        checks.append({"check": f"target_alias:{event}", "pass": bool(target_alias_equal)})
        if not target_alias_equal:
            failures.append(f"h.npy differs from h_itzi_swmm.npy for {event}")

        b = np.asarray(loaded["h_itzi_surface_matched.npy"][:, active])
        c = np.asarray(loaded["h_itzi_swmm.npy"][:, active])
        drainage_mae = float(np.mean(np.abs(c - b)) * 1000.0)
        signed = float(np.mean(c - b) * 1000.0)
        source = physics_rows[event]
        physics_ok = abs(drainage_mae - float(source["drainage_effect_mae_mm"])) < 2e-5 and abs(signed - float(source["drainage_signed_mean_mm"])) < 2e-5
        checks.append({"check": f"physical_metric_recompute:{event}", "mae_mm": drainage_mae, "signed_mm": signed, "pass": physics_ok})
        if not physics_ok:
            failures.append(f"Physical metric mismatch for {event}")

        pred_path = HYBRID / "predictions" / event / "h_clim_all_static.npy"
        pred = np.load(pred_path, mmap_mode="r")
        nonnegative = float(pred.min()) >= 0.0
        p = np.asarray(pred[:, active])
        err = p - c
        mae = float(np.mean(np.abs(err)) * 1000.0)
        rmse = float(np.sqrt(np.mean(err * err)) * 1000.0)
        csi03, csi15 = csi(p, c, 0.03), csi(p, c, 0.15)
        final_error = float(abs((p[-1].sum(dtype=np.float64) - c[-1].sum(dtype=np.float64)) * CELL_AREA_M2))
        row = metric_rows[(event, "clim_all_static", "coupled_label")]
        metrics_ok = (
            pred.shape == SHAPE and nonnegative
            and abs(mae - float(row["mae_mm"])) < 2e-5
            and abs(rmse - float(row["rmse_mm"])) < 2e-5
            and abs(csi03 - float(row["csi_0p03"])) < 1e-10
            and abs(csi15 - float(row["csi_0p15"])) < 1e-10
            and abs(final_error - float(row["final_volume_abs_error_m3"])) < 0.1
        )
        checks.append({"check": f"hybrid_metric_recompute:{event}", "mae_mm": mae, "rmse_mm": rmse, "csi_0p03": csi03, "csi_0p15": csi15, "final_volume_error_m3": final_error, "nonnegative": nonnegative, "pass": bool(metrics_ok)})
        if not metrics_ok:
            failures.append(f"Hybrid metric mismatch for {event}")
        recomputed_mae.append(mae)

    summary = {row["model"]: row for row in read_csv(HYBRID / "metrics" / "summary_coupled.csv")}
    macro = float(np.mean(recomputed_mae))
    summary_ok = abs(macro - float(summary["clim_all_static"]["mae_mm"])) < 2e-5
    checks.append({"check": "hybrid_macro_mae_recompute", "mae_mm": macro, "pass": summary_ok})
    if not summary_ok:
        failures.append("Hybrid summary macro MAE mismatch")

    network = json.loads((REV / "network" / "swmm_bidirectional_normal_step05.independent_audit.json").read_text(encoding="utf-8"))
    network_ok = all(network["acceptance"].values())
    checks.append({"check": "network_acceptance", "criteria": network["acceptance"], "pass": network_ok})
    if not network_ok:
        failures.append("Network acceptance criterion failed")

    for stem, expected_images in [("manuscript", 11), ("report", 11), ("scientific_integrity_audit", 0)]:
        html = PKG / f"{stem}.html"
        text = html.read_text(encoding="utf-8")
        soup = BeautifulSoup(text, "html.parser")
        images = soup.find_all("img")
        html_ok = text.lstrip().lower().startswith("<!doctype html>") and len(images) == expected_images and all(image.get("src", "").startswith("data:image/") for image in images) and not soup.find_all("link")
        checks.append({"check": f"standalone_html:{stem}", "images": len(images), "bytes": html.stat().st_size, "pass": bool(html_ok)})
        if not html_ok:
            failures.append(f"Standalone HTML failed: {stem}")
        pdf = PKG / f"{stem}.pdf"
        reader = PdfReader(str(pdf))
        replacement = sum((page.extract_text() or "").count("\ufffd") for page in reader.pages)
        pdf_ok = len(reader.pages) > 0 and replacement == 0
        checks.append({"check": f"pdf:{stem}", "pages": len(reader.pages), "bytes": pdf.stat().st_size, "replacement_chars": replacement, "pass": pdf_ok})
        if not pdf_ok:
            failures.append(f"PDF failed: {stem}")

    for index in range(1, 12):
        matches = list((PKG / "figures").glob(f"fig{index:02d}_*.png"))
        ok = len(matches) == 1
        dimensions = None
        if ok:
            with Image.open(matches[0]) as image:
                dimensions = list(image.size)
                ok = image.width >= 1800 and image.height >= 850
        checks.append({"check": f"figure_png:{index:02d}", "count": len(matches), "dimensions": dimensions, "pass": bool(ok)})
        if not ok:
            failures.append(f"Figure {index:02d} failed")

    result = {
        "status": "PASS" if not failures else "FAIL",
        "events": EVENTS,
        "checks_total": len(checks),
        "checks_passed": sum(bool(item["pass"]) for item in checks),
        "failures": failures,
        "checks": checks,
    }
    (PKG / "final_acceptance.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    files = sorted(path for path in PKG.rglob("*") if path.is_file() and path.name != "sha256_manifest.csv")
    with (PKG / "sha256_manifest.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "bytes", "sha256"])
        writer.writeheader()
        for path in files:
            writer.writerow({"path": str(path.relative_to(PKG)), "bytes": path.stat().st_size, "sha256": sha256(path)})
    print(json.dumps({key: result[key] for key in ["status", "checks_total", "checks_passed", "failures"]}, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
