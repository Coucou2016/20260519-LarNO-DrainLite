#!/usr/bin/env python3
"""Compare full-event 1 mm/h and 2 mm/h infiltration result packages."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from publication_plot_style import add_panel_labels, configure_publication_style


configure_publication_style()


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "extended_study" / "output" / "infiltration_model_selection"
FIG = OUT / "figures"

RUNS = [
    {
        "label": "1 mm/h effective infiltration",
        "rate_mmh": 1.0,
        "dataset": "region1_20m_connected_swmm_v2_inf1mmh",
        "package": ROOT / "extended_study" / "output" / "larno_drainlite_v2_inf1mmh_package",
        "cv": ROOT / "extended_study" / "output" / "drainlite_connected_cross_validation_v2_inf1mmh",
    },
    {
        "label": "2 mm/h effective infiltration",
        "rate_mmh": 2.0,
        "dataset": "region1_20m_connected_swmm_v2_infiltration",
        "package": ROOT / "extended_study" / "output" / "larno_drainlite_v2_infiltration_package",
        "cv": ROOT / "extended_study" / "output" / "drainlite_connected_cross_validation_v2_infiltration",
    },
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def as_float(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, "nan"))
    except Exception:
        return float("nan")


def find(rows: list[dict[str, str]], key: str, value: str) -> dict[str, str]:
    return next((r for r in rows if r.get(key) == value), {})


def summarize_run(run: dict[str, object]) -> dict[str, object]:
    package = Path(run["package"])
    cv = Path(run["cv"])
    quality = read_csv(package / "metrics" / "v2_label_quality_summary.csv")
    mike = read_csv(package / "metrics" / "v2_mike_reference_summary.csv")
    cv_summary = read_csv(cv / "metrics" / "connected_residual_cv_summary.csv")
    all_static = find(cv_summary, "model", "all_static")
    surface_cv = find(cv_summary, "model", "surface_only")
    mike_surface = find(mike, "model", "ITZI surface-only")
    mike_connected = find(mike, "model", "ITZI-SWMM connected")
    mike_drainlite = find(mike, "model", "DrainLite all_static CV")
    q = {r["quality_status"]: int(float(r["n_events"])) for r in quality}
    return {
        "label": run["label"],
        "rate_mmh": run["rate_mmh"],
        "dataset": run["dataset"],
        "accepted_events": q.get("accepted", 0),
        "warning_events": q.get("warning", 0),
        "excluded_events": q.get("excluded", 0),
        "cv_surface_mae_to_label_mm": as_float(surface_cv, "mae_mm"),
        "cv_all_static_mae_to_label_mm": as_float(all_static, "mae_mm"),
        "cv_all_static_improvement_vs_surface_pct": as_float(all_static, "mae_improvement_vs_surface_pct"),
        "mike_surface_mae_mm": as_float(mike_surface, "mae_mm"),
        "mike_connected_mae_mm": as_float(mike_connected, "mae_mm"),
        "mike_drainlite_mae_mm": as_float(mike_drainlite, "mae_mm"),
        "mike_surface_peak_error_mm": as_float(mike_surface, "peak_depth_error_mm"),
        "mike_connected_peak_error_mm": as_float(mike_connected, "peak_depth_error_mm"),
        "mike_drainlite_peak_error_mm": as_float(mike_drainlite, "peak_depth_error_mm"),
        "mike_surface_final_volume_error_m3": as_float(mike_surface, "final_volume_error_m3"),
        "mike_connected_final_volume_error_m3": as_float(mike_connected, "final_volume_error_m3"),
        "mike_drainlite_final_volume_error_m3": as_float(mike_drainlite, "final_volume_error_m3"),
    }


def plot_selection(rows: list[dict[str, object]]) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    labels = [f"{r['rate_mmh']:g} mm/h" for r in rows]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.55), constrained_layout=True)
    axes[0].bar(x, [float(r["cv_all_static_mae_to_label_mm"]) for r in rows], color="#2563eb")
    axes[0].set_title("DrainLite LOEO MAE to label")
    axes[0].set_ylabel("mm")
    axes[1].bar(x, [float(r["mike_drainlite_peak_error_mm"]) for r in rows], color="#f97316")
    axes[1].axhline(0, color="#111827", lw=1)
    axes[1].set_title("DrainLite peak-depth bias to MIKE")
    axes[1].set_ylabel("mm")
    axes[2].bar(x, [float(r["mike_drainlite_final_volume_error_m3"]) / 1000.0 for r in rows], color="#22c55e")
    axes[2].axhline(0, color="#111827", lw=1)
    axes[2].set_title("DrainLite final-volume bias to MIKE")
    axes[2].set_ylabel("10^3 m3")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.grid(axis="y", alpha=0.25)
    add_panel_labels(axes, x=-0.18, y=1.04)
    fig.savefig(FIG / "infiltration_final_selection.png")
    plt.close(fig)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [summarize_run(run) for run in RUNS]
    selected = rows[0]
    note = {
        "selected_rate_mmh": 1.0,
        "selected_dataset": selected["dataset"],
        "reason": [
            "Both 1 mm/h and 2 mm/h have the same event-count quality split under the continuity-error grading.",
            "The 1 mm/h run is less aggressive relative to MIKE: DrainLite peak-depth bias is close to zero on average, whereas 2 mm/h underpredicts peak depth much more strongly.",
            "The 1 mm/h run still preserves a strong DrainLite learning signal, reducing leave-one-event-out MAE to the ITZI-SWMM label by about half relative to surface-only.",
            "The 2 mm/h run is retained as sensitivity evidence but is not fixed as the main label source because it over-depresses peak depth and final water volume in the eight-event MIKE check.",
        ],
    }
    write_csv(OUT / "infiltration_final_selection.csv", rows)
    plot_selection(rows)
    (OUT / "infiltration_final_selection.json").write_text(
        json.dumps({"selection": note, "runs": rows}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"selection": note, "rows": rows}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
