#!/usr/bin/env python3
"""Read-only acceptance checks for the fixed-network V4 publication package."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from bs4 import BeautifulSoup
from PIL import Image
from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]
V4 = ROOT / "extended_study" / "output" / "reviewer_major_revision_v4"
EXP = V4 / "final_hybrid_controls"
PACKAGE = V4 / "submission_package_v4"
FLOOD = ROOT / "LarNO-main" / "benchmark" / "urbanflood" / "flood" / "region1_20m_drainage_v3_full"
GEO = ROOT / "LarNO-main" / "benchmark" / "urbanflood" / "geodata" / "region1_20m_drainage_v3_full"
EVENTS = ["event1", "event20", "event65", "event66", "event67", "event68", "event69", "event70"]
SEEDS = [1907, 2718, 3141, 5772, 8119]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-json", action="store_true", help="Write a new acceptance record after verification.")
    return parser.parse_args()


def rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    args = parse_args()
    checks: list[dict[str, object]] = []

    def check(name: str, condition: bool, detail: object = "") -> None:
        checks.append({"name": name, "passed": bool(condition), "detail": str(detail)})

    metric_paths = {
        name: EXP / "metrics" / name
        for name in [
            "event_metrics_all_seeds.csv",
            "seed_macro_metrics.csv",
            "model_summary_mean_sd.csv",
            "runtime_end_to_end.csv",
            "runtime_uncached_canonical.csv",
            "training_runtime.csv",
            "conditional_metrics_canonical_seed.csv",
            "final_hybrid_network_controls.csv",
            "clipping_sensitivity.csv",
            "event_selection_inventory.csv",
        ]
    }
    for name, path in metric_paths.items():
        check(f"metric file: {name}", path.exists() and path.stat().st_size > 100, path)
    if not all(path.exists() for path in metric_paths.values()):
        print(json.dumps({"status": "FAIL", "checks": checks}, indent=2))
        return 1

    event_metrics = rows(metric_paths["event_metrics_all_seeds.csv"])
    runtime = rows(metric_paths["runtime_uncached_canonical.csv"])
    check("uncached runtime: eight events", {row["test_event"] for row in runtime} == set(EVENTS))
    for row in runtime:
        check(f"{row['test_event']}: neighbourhood included in timing", float(row["neighbourhood_seconds"]) > 0 and row["uses_cached_neighbourhoods"] == "False")
        check(f"{row['test_event']}: benchmark preserves prediction", float(row["maximum_difference_from_saved_m"]) <= 1e-6)
    sensitivity = rows(V4 / "physical_sensitivity_event68/physical_sensitivity_metrics.csv")
    check("paired physical sensitivity: baseline and five alternatives", len(sensitivity) == 6)
    for row in sensitivity:
        check(f"sensitivity {row['case']}: finite complete arrays", row["finite"] == "True" and row["array_shape"] == "72x400x560")
        check(f"sensitivity {row['case']}: routing continuity", abs(float(row["routing_continuity_error_pct"])) <= 2)
    controls = rows(metric_paths["final_hybrid_network_controls.csv"])
    clipping = rows(metric_paths["clipping_sensitivity.csv"])
    conditional = rows(metric_paths["conditional_metrics_canonical_seed.csv"])
    inventory = rows(metric_paths["event_selection_inventory.csv"])
    metadata = json.loads((EXP / "experiment_metadata.json").read_text(encoding="utf-8"))

    check("metadata events", metadata["events"] == EVENTS, metadata["events"])
    check("metadata seeds", metadata["seeds"] == SEEDS, metadata["seeds"])
    check("primary residual unclipped", metadata["primary_residual_clipping"] == "none")
    check("MIKE excluded from fitting", "never used" in metadata["mike_role"])
    check("whole-event outer validation", metadata["outer_validation"] == "leave one complete rainfall event out")
    check("fixed pre-specified complexity", "fixed max_iter" in metadata["complexity_selection"])
    check("mask-aware neighbourhood", "mask-aware" in metadata["surface_depth_neighbourhood"])

    for seed in SEEDS:
        for model in ["hybrid_dynamic", "hybrid_all"]:
            subset = [row for row in event_metrics if int(row["seed"]) == seed and row["model"] == model and row["reference"] == "coupled_label"]
            check(f"{model} seed {seed}: eight held events", {row["event"] for row in subset} == set(EVENTS), len(subset))
    for model in ["surface_matched", "prior_only", "dynamic_no_prior", "hybrid_mask", "hybrid_hydraulic"]:
        subset = [row for row in event_metrics if int(row["seed"]) == 1907 and row["model"] == model and row["reference"] == "coupled_label"]
        check(f"canonical {model}: eight held events", {row["event"] for row in subset} == set(EVENTS), len(subset))

    active = np.load(GEO / "active_mask.npy").astype(bool)
    check("active grid count", int(active.sum()) == 105527, int(active.sum()))
    for event in EVENTS:
        pred_path = EXP / "predictions" / event / "h_hybrid_all.npy"
        check(f"{event}: canonical prediction exists", pred_path.exists(), pred_path)
        if not pred_path.exists():
            continue
        pred = np.load(pred_path, mmap_mode="r")
        coupled = np.load(FLOOD / event / "h_itzi_swmm.npy", mmap_mode="r")
        check(f"{event}: prediction shape", pred.shape == (72, 400, 560), pred.shape)
        check(f"{event}: finite prediction", bool(np.isfinite(pred).all()))
        check(f"{event}: non-negative depth", float(np.min(pred)) >= 0.0, float(np.min(pred)))
        recalculated = float(np.mean(np.abs(np.asarray(pred[:, active]) - np.asarray(coupled[:, active]))) * 1000.0)
        recorded = next(float(row["mae_mm"]) for row in event_metrics if row["event"] == event and int(row["seed"]) == 1907 and row["model"] == "hybrid_all" and row["reference"] == "coupled_label")
        check(f"{event}: prediction/CSV MAE", abs(recalculated - recorded) < 1e-4, f"{recalculated:.8f} vs {recorded:.8f}")

    expected_controls = len(EVENTS) * (1 + 16 + 10 + 5)
    check("final-hybrid control count", len(controls) == expected_controls, len(controls))
    for event in EVENTS:
        names = [row["control"] for row in controls if row["event"] == event]
        check(f"{event}: all displacement controls", sum(name.startswith("shift_") for name in names) == 16)
        check(f"{event}: block controls", names.count("block_shuffle") == 10)
        check(f"{event}: permutation controls", names.count("group_permutation") == 5)
        check(f"{event}: zero-field control", names.count("zero_static_network_fields") == 1)

    conditions = {row["condition"] for row in conditional}
    for name in ["all_active", "target_wet_0p03m", "effect_abs_over_5mm", "net_drainage_over_5mm", "positive_residual_over_5mm", "near_inlet_0_20m", "near_pipe_100_infm"]:
        check(f"conditional metric: {name}", name in conditions)
    check("conditional rows canonical only", {int(row.get("seed", "1907")) for row in conditional if row.get("seed")} in ({1907}, set()))

    for row in runtime:
        component_sum = sum(float(row[name]) for name in ["feature_seconds", "assembly_seconds", "model_seconds", "postprocess_seconds"])
        total = float(row["total_correction_seconds"])
        check("runtime contains all components", total + 1e-6 >= component_sum, f"{total:.4f} >= {component_sum:.4f}")
        expected_total = float(row["surface_runtime_s"]) + total
        check("surface plus correction arithmetic", abs(float(row["surface_plus_correction_seconds"]) - expected_total) < 1e-6)
    check("unclipped diagnostic present per event", len([row for row in clipping if row["clip_mm"] == "none"]) == len(EVENTS))
    check("500 mm clipping sensitivity per event", len([row for row in clipping if row["clip_mm"] in {"500", "500.0"}]) == len(EVENTS))
    check("1200 mm clipping sensitivity per event", len([row for row in clipping if row["clip_mm"] in {"1200", "1200.0"}]) == len(EVENTS))

    check("event inventory includes paired events", all(any(row["event"] == event and row["used_in_paired_v3"].lower() == "true" for row in inventory) for event in EVENTS))
    check("event79 recorded unreadable", any(row["event"] == "event79" and row["public_arrays_valid"].lower() == "false" for row in inventory))

    figure_stems = [
        "fig01_workflow", "fig02_network_audit", "fig03_physical_quality",
        "fig04_eight_event_hydrographs", "fig05_event68_physical_maps",
        "fig06_fixed_network_residual_maps", "fig07_skill_and_sampling_seeds",
        "fig08_final_hybrid_network_controls", "fig09_conditioned_performance",
        "fig10_mike_metric_tradeoff", "fig11_runtime_clipping_event_selection",
        "fig12_larno_reproduction",
        "fig13_physical_sensitivity",
    ]
    for stem in figure_stems:
        for suffix in ["png", "pdf", "svg"]:
            path = PACKAGE / "figures" / f"{stem}.{suffix}"
            check(f"figure artifact: {stem}.{suffix}", path.exists() and path.stat().st_size > 5000, path)
        png = PACKAGE / "figures" / f"{stem}.png"
        if png.exists():
            with Image.open(png) as image:
                check(f"figure resolution: {stem}", image.width >= 1800 and image.height >= 900, image.size)

    for stem in ["manuscript", "report", "scientific_integrity_audit"]:
        for suffix in ["md", "html", "pdf"]:
            path = PACKAGE / f"{stem}.{suffix}"
            check(f"document artifact: {stem}.{suffix}", path.exists() and path.stat().st_size > 1000, path)
        html_path = PACKAGE / f"{stem}.html"
        if html_path.exists():
            soup = BeautifulSoup(html_path.read_text(encoding="utf-8"), "html.parser")
            sources = [image.get("src", "") for image in soup.find_all("img")]
            check(f"{stem}: all images embedded", (bool(sources) or stem == "scientific_integrity_audit") and all(source.startswith("data:image/") for source in sources), len(sources))
            check(f"{stem}: no external stylesheet", not soup.find_all("link", rel="stylesheet"))
        pdf_path = PACKAGE / f"{stem}.pdf"
        if pdf_path.exists():
            check(f"{stem}: readable PDF", len(PdfReader(str(pdf_path)).pages) >= 2)

    failed = [item for item in checks if not item["passed"]]
    result = {"status": "PASS" if not failed else "FAIL", "passed": len(checks) - len(failed), "total": len(checks), "failed": failed, "checks": checks}
    if args.write_json:
        (PACKAGE / "v4_acceptance.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: result[key] for key in ["status", "passed", "total", "failed"]}, indent=2))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
