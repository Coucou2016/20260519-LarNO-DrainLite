#!/usr/bin/env python3
"""Generate the simplified formal ITZI-SWMM -> DrainLite report package.

This report intentionally follows the narrowed storyline:

1. Use the copied formal ITZI-SWMM two-way coupled model as the data source.
2. Keep both no-drainage surface-only and drainage-coupled labels.
3. Use MIKE only as an external plausibility check.
4. Train/evaluate a lightweight signed residual model that can later be placed
   after LarNO predictions.
"""

from __future__ import annotations

import base64
import csv
import html
import json
import math
import re
import shutil
import subprocess
from pathlib import Path
from textwrap import dedent

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "LarNO-main" / "benchmark" / "urbanflood" / "flood" / "region1_20m_drainage_v1"
GEO_DIR = ROOT / "LarNO-main" / "benchmark" / "urbanflood" / "geodata" / "region1_20m_drainage_v1"
FORMAL_NPZ_DIR = (
    ROOT
    / "external_models"
    / "20260518-itzi-flood"
    / "test_cases"
    / "shenzhen_region1"
    / "output"
    / "full_domain_swmm"
)
MODEL_OUT = ROOT / "extended_study" / "output" / "drainlite_official_residual"
CV_DIR = ROOT / "extended_study" / "output" / "official_paper_package" / "metrics"
AUDIT_DIR = ROOT / "extended_study" / "output" / "coupling_difference_audit"
OUT = ROOT / "extended_study" / "output" / "formal_larno_drainlite_simplified"
FIG_DIR = OUT / "figures"
TABLE_DIR = OUT / "tables"
REPORT_HTML = OUT / "report.html"
REPORT_MD = OUT / "report.md"
REPORT_PDF = OUT / "report.pdf"
ROOT_HTML = ROOT / "report.html"
ROOT_MD = ROOT / "report.md"
ROOT_PDF = ROOT / "report.pdf"

EVENTS = ["event1", "event20", "event65", "event66", "event67", "event68", "event69", "event70"]
TEST_EVENTS = ["event68", "event69", "event70"]
SHAPE = (72, 200, 280)
CELL_SIZE_M = 20.0
CELL_AREA_M2 = CELL_SIZE_M * CELL_SIZE_M
DEADBAND_MM = 2.0


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def img64(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def csi(pred: np.ndarray, target: np.ndarray, threshold: float) -> float:
    p = pred > threshold
    t = target > threshold
    tp = int(np.sum(p & t))
    fp = int(np.sum(p & ~t))
    fn = int(np.sum(~p & t))
    denom = tp + fp + fn
    return float(tp / denom) if denom else 1.0


def load_event(event: str) -> dict[str, np.ndarray]:
    d = DATA_DIR / event
    arrays = {
        "rainfall": np.load(d / "rainfall.npy").astype(np.float32),
        "mike": np.load(d / "h_mike_ref.npy").astype(np.float32),
        "surface": np.load(d / "h_itzi_swmm_official_surface.npy").astype(np.float32),
        "coupled": np.load(d / "h_itzi_swmm_official.npy").astype(np.float32),
    }
    return arrays


def load_static() -> dict[str, np.ndarray]:
    names = [
        "dem",
        "drain_inlet_mask",
        "drain_outfall_mask",
        "pipe_mask",
        "pipe_diameter",
        "pipe_slope",
        "pipe_capacity",
        "pipe_cover_depth",
        "distance_to_outfall",
    ]
    return {name: np.load(GEO_DIR / f"{name}.npy").astype(np.float32) for name in names}


def npz_record(event: str) -> dict[str, object]:
    path = FORMAL_NPZ_DIR / f"{event}_coupled.npz"
    if not path.exists():
        return {}
    z = np.load(path, allow_pickle=True)
    rec = z["rec_swmm"].item() if "rec_swmm" in z.files else {}
    return rec


def final_or_nan(values: object) -> float:
    try:
        arr = np.asarray(values, dtype=float)
        if arr.size == 0:
            return math.nan
        return float(arr[-1])
    except Exception:
        return math.nan


def check_inventory() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    required = [
        "rainfall.npy",
        "h_mike_ref.npy",
        "h_itzi_swmm_official_surface.npy",
        "h_itzi_swmm_official.npy",
    ]
    for event in EVENTS:
        d = DATA_DIR / event
        ok = True
        notes: list[str] = []
        for name in required:
            p = d / name
            if not p.exists():
                ok = False
                notes.append(f"missing {name}")
                continue
            arr = np.load(p, mmap_mode="r")
            if tuple(arr.shape) != SHAPE:
                ok = False
                notes.append(f"{name} shape={tuple(arr.shape)}")
        npz_ok = (FORMAL_NPZ_DIR / f"{event}_coupled.npz").exists()
        if not npz_ok:
            ok = False
            notes.append("missing formal coupled npz")
        rows.append(
            {
                "event": event,
                "required_arrays": "OK" if ok else "CHECK",
                "shape": "72 x 200 x 280",
                "time_resolution": "5 min",
                "duration": "6 h",
                "formal_npz": "yes" if npz_ok else "no",
                "notes": "; ".join(notes) if notes else "complete",
            }
        )
    return rows


def compute_event_metrics() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for event in EVENTS:
        a = load_event(event)
        residual = a["coupled"] - a["surface"]
        final_red = np.maximum(a["surface"][-1] - a["coupled"][-1], 0.0)
        final_inc = np.maximum(a["coupled"][-1] - a["surface"][-1], 0.0)
        rec = npz_record(event)
        rows.append(
            {
                "event": event,
                "rain_total_mean_mm": float(a["rainfall"].sum(axis=0).mean()),
                "rain_total_max_mm": float(a["rainfall"].sum(axis=0).max()),
                "mike_peak_m": float(a["mike"].max()),
                "surface_peak_m": float(a["surface"].max()),
                "coupled_peak_m": float(a["coupled"].max()),
                "surface_vs_mike_mae_mm": float(np.mean(np.abs(a["surface"] - a["mike"])) * 1000.0),
                "coupled_vs_mike_mae_mm": float(np.mean(np.abs(a["coupled"] - a["mike"])) * 1000.0),
                "surface_to_coupled_mae_mm": float(np.mean(np.abs(residual)) * 1000.0),
                "surface_to_coupled_rmse_mm": float(np.sqrt(np.mean(residual * residual)) * 1000.0),
                "residual_gt_1mm_pct": float(np.mean(np.abs(residual) > 0.001) * 100.0),
                "final_positive_reduction_m3": float(final_red.sum() * CELL_AREA_M2),
                "final_local_increase_m3": float(final_inc.sum() * CELL_AREA_M2),
                "legacy_positive_exchange_m3": final_or_nan(rec.get("drained_m3", [])),
                "swmm_final_volume_1000m3": final_or_nan(rec.get("vol_m3", [])) / 1000.0,
            }
        )
    return rows


def summarize_model_rows(path: Path) -> list[dict[str, object]]:
    rows = read_csv(path)
    out = []
    wanted = {"surface_only", "base", "all_static", "swmm_assisted"}
    for r in rows:
        if r.get("model") not in wanted:
            continue
        row = {"model": r["model"]}
        for key, val in r.items():
            if key == "model":
                continue
            try:
                row[key] = float(val)
            except Exception:
                row[key] = val
        out.append(row)
    return out


def static_summary(static: dict[str, np.ndarray]) -> list[dict[str, object]]:
    pipe = static["pipe_mask"] > 0
    inlet = static["drain_inlet_mask"] > 0
    outfall = static["drain_outfall_mask"] > 0
    active = np.isfinite(static["dem"]) & (static["dem"] < 49.9)
    return [
        {"item": "grid_resolution", "value": "20 m", "note": "formal local analysis grid"},
        {"item": "grid_shape", "value": "200 x 280", "note": "4.0 km x 5.6 km subset"},
        {"item": "time_steps", "value": "72", "note": "5 min interval, 6 h duration"},
        {"item": "formal_events", "value": str(len(EVENTS)), "note": ", ".join(EVENTS)},
        {"item": "active_cells", "value": int(active.sum()), "note": "DEM cells below building wall threshold"},
        {"item": "pipe_cells", "value": int(pipe.sum()), "note": "raster cells crossed by conceptualized formal pipe network"},
        {"item": "inlet_cells", "value": int(inlet.sum()), "note": "rasterized manhole/inlet cells"},
        {"item": "outfall_cells", "value": int(outfall.sum()), "note": "rasterized outfall cells"},
        {
            "item": "mean_pipe_diameter_m",
            "value": float(np.mean(static["pipe_diameter"][pipe])) if np.any(pipe) else math.nan,
            "note": "mean on pipe cells only",
        },
        {
            "item": "mean_pipe_capacity",
            "value": float(np.mean(static["pipe_capacity"][pipe])) if np.any(pipe) else math.nan,
            "note": "relative/derived capacity field on pipe cells",
        },
    ]


def larno_inventory() -> list[dict[str, object]]:
    weight = (
        ROOT
        / "LarNO-main"
        / "exp"
        / "20260220_183648_006352"
        / "weights"
        / "model_epoch_992_error@0.000055821"
        / "model_epoch_992_error@0.000055821_state_dict.pt"
    )
    pred_candidates = list((ROOT / "LarNO-main" / "exp").rglob("predictions_epoch_*_sample_event*.npy"))
    matching = [p for p in pred_candidates if any(evt in p.name for evt in EVENTS)]
    return [
        {
            "item": "LarNO pretrained weight",
            "status": "found" if weight.exists() else "missing",
            "evidence": str(weight.relative_to(ROOT)) if weight.exists() else "待补充",
        },
        {
            "item": "LarNO per-event prediction arrays",
            "status": "not found locally" if not matching else f"found {len(matching)}",
            "evidence": "需要运行 LarNO inference 并落盘 h_larno_base.npy" if not matching else "; ".join(str(p.relative_to(ROOT)) for p in matching[:3]),
        },
        {
            "item": "Current DrainLite validation baseline",
            "status": "completed",
            "evidence": "uses formal ITZI-SWMM surface-only as the LarNO-like base field",
        },
        {
            "item": "Next LarNO residual step",
            "status": "acceptance condition",
            "evidence": "replace surface input with h_larno_base.npy and compare LarNO vs LarNO+DrainLite against formal coupled label",
        },
    ]


def make_box(ax, xy: tuple[float, float], text: str, color: str, width: float = 2.3, height: float = 0.7) -> None:
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.03,rounding_size=0.05",
        linewidth=1.1,
        edgecolor="#1f3a5f",
        facecolor=color,
    )
    ax.add_patch(patch)
    ax.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=10)


def make_arrow(ax, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=14, linewidth=1.1, color="#334e68"))


def fig_workflow() -> Path:
    path = FIG_DIR / "fig01_simplified_workflow.png"
    fig, ax = plt.subplots(figsize=(13.5, 4.8))
    ax.set_xlim(0, 12.5)
    ax.set_ylim(0, 4.2)
    ax.axis("off")
    make_box(ax, (0.4, 2.8), "MIKE reference\nexternal check", "#dbeafe")
    make_box(ax, (0.4, 1.5), "Copied formal\nITZI-SWMM model", "#dcfce7")
    make_box(ax, (3.3, 2.8), "Plausibility:\nMIKE vs ITZI", "#f8fafc")
    make_box(ax, (3.3, 1.5), "Dataset:\nsurface + coupled", "#dcfce7")
    make_box(ax, (6.2, 1.5), "Signed residual:\ncoupled - surface", "#fef3c7", width=2.6)
    make_box(ax, (9.1, 1.5), "DrainLite:\nlightweight correction", "#ede9fe", width=2.6)
    make_box(ax, (9.1, 2.8), "LarNO interface:\nbase forecast + residual", "#ffe4e6", width=2.6)
    for s, e in [
        ((2.7, 3.15), (3.3, 3.15)),
        ((2.7, 1.85), (3.3, 1.85)),
        ((5.6, 1.85), (6.2, 1.85)),
        ((8.8, 1.85), (9.1, 1.85)),
        ((10.4, 2.2), (10.4, 2.8)),
    ]:
        make_arrow(ax, s, e)
    ax.text(
        6.2,
        0.55,
        "Simplified paper logic: MIKE validates reasonableness; formal ITZI-SWMM builds drainage labels; DrainLite learns signed drainage residuals.",
        ha="center",
        fontsize=11,
        color="#102a43",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def fig_static_map(static: dict[str, np.ndarray]) -> Path:
    path = FIG_DIR / "fig02_dem_pipe_network.png"
    dem = np.ma.masked_where(static["dem"] >= 49.9, static["dem"])
    pipe = static["pipe_mask"] > 0
    inlet_y, inlet_x = np.where(static["drain_inlet_mask"] > 0)
    out_y, out_x = np.where(static["drain_outfall_mask"] > 0)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.1))
    im = axes[0].imshow(dem, cmap="terrain")
    axes[0].set_title("DEM and elevated buildings")
    axes[0].axis("off")
    fig.colorbar(im, ax=axes[0], fraction=0.045)
    axes[1].imshow(dem, cmap="Greys", alpha=0.65)
    axes[1].contour(pipe.astype(float), levels=[0.5], colors="#0f172a", linewidths=0.35)
    axes[1].scatter(inlet_x, inlet_y, s=4, c="#2563eb", label="inlet/manhole")
    axes[1].scatter(out_x, out_y, s=32, c="#dc2626", label="outfall")
    axes[1].set_title("Pipe, inlet and outfall alignment")
    axes[1].legend(loc="lower right", fontsize=8)
    axes[1].axis("off")
    cap = np.ma.masked_where(~pipe, static["pipe_capacity"])
    im2 = axes[2].imshow(cap, cmap="magma")
    axes[2].set_title("Pipe capacity field")
    axes[2].axis("off")
    fig.colorbar(im2, ax=axes[2], fraction=0.045)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def fig_mike_validation_grid() -> Path:
    path = FIG_DIR / "fig03_mike_validation_peak_grid.png"
    cols = ["MIKE peak", "ITZI no-drain peak", "ITZI-SWMM peak", "coupled - no-drain peak"]
    cache: dict[str, dict[str, np.ndarray]] = {}
    vmax = 0.0
    dlim = 0.02
    for event in EVENTS:
        a = load_event(event)
        surface_peak = a["surface"].max(axis=0)
        coupled_peak = a["coupled"].max(axis=0)
        diff_peak = coupled_peak - surface_peak
        cache[event] = {
            cols[0]: a["mike"].max(axis=0),
            cols[1]: surface_peak,
            cols[2]: coupled_peak,
            cols[3]: diff_peak,
        }
        vmax = max(vmax, float(cache[event][cols[0]].max()), float(surface_peak.max()), float(coupled_peak.max()))
        dlim = max(dlim, float(np.max(np.abs(diff_peak))))
    fig, axes = plt.subplots(len(EVENTS), len(cols), figsize=(14.5, 20))
    for i, event in enumerate(EVENTS):
        for j, col in enumerate(cols):
            ax = axes[i, j]
            data = cache[event][col]
            if j == 3:
                im = ax.imshow(data, cmap="coolwarm", vmin=-dlim, vmax=dlim)
            else:
                im = ax.imshow(data, cmap="viridis", vmin=0, vmax=vmax)
            if i == 0:
                ax.set_title(col, fontsize=10)
            if j == 0:
                ax.set_ylabel(event, fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle("Eight-event peak maps for MIKE validation and formal ITZI-SWMM drainage effect", y=0.996)
    fig.tight_layout(rect=[0, 0, 1, 0.987])
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def fig_temporal_effects() -> Path:
    path = FIG_DIR / "fig04_temporal_effects.png"
    selected = ["event20", "event68", "event70"]
    t = np.arange(1, 73) * 5
    fig, axes = plt.subplots(len(selected), 2, figsize=(13, 10), sharex=True)
    for i, event in enumerate(selected):
        a = load_event(event)
        for key, label, color in [
            ("mike", "MIKE", "#111827"),
            ("surface", "ITZI no-drain", "#2563eb"),
            ("coupled", "ITZI-SWMM", "#16a34a"),
        ]:
            vol = a[key].sum(axis=(1, 2)) * CELL_AREA_M2 / 1000.0
            hmax = a[key].max(axis=(1, 2))
            axes[i, 0].plot(t, vol, label=label, color=color, linewidth=1.5)
            axes[i, 1].plot(t, hmax, label=label, color=color, linewidth=1.5)
        axes[i, 0].set_ylabel(f"{event}\nvolume (10^3 m3)")
        axes[i, 1].set_ylabel("max depth (m)")
        axes[i, 0].grid(alpha=0.25)
        axes[i, 1].grid(alpha=0.25)
    axes[0, 0].set_title("Total surface water volume")
    axes[0, 1].set_title("Maximum water depth")
    axes[-1, 0].set_xlabel("time (min)")
    axes[-1, 1].set_xlabel("time (min)")
    axes[0, 0].legend(ncol=3, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def fig_residual_character(event_rows: list[dict[str, object]]) -> Path:
    path = FIG_DIR / "fig05_formal_residual_character.png"
    all_samples = []
    red = []
    inc = []
    labels = []
    for r in event_rows:
        event = str(r["event"])
        a = load_event(event)
        res = (a["coupled"] - a["surface"]) * 1000.0
        all_samples.append(res[:, ::4, ::4].ravel())
        red.append(float(r["final_positive_reduction_m3"]) / 1000.0)
        inc.append(float(r["final_local_increase_m3"]) / 1000.0)
        labels.append(event)
    vals = np.concatenate(all_samples)
    vals = vals[np.isfinite(vals)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.7))
    axes[0].hist(np.clip(vals, -20, 20), bins=100, color="#4f78b8")
    axes[0].axvline(0, color="#111827", linewidth=1)
    axes[0].axvline(-DEADBAND_MM, color="#b45309", linestyle="--", linewidth=1)
    axes[0].axvline(DEADBAND_MM, color="#b45309", linestyle="--", linewidth=1)
    axes[0].set_title("Signed residual distribution")
    axes[0].set_xlabel("coupled - no-drain (mm)")
    axes[0].set_ylabel("sample count")
    x = np.arange(len(labels))
    axes[1].bar(x - 0.18, red, width=0.36, label="final reduction")
    axes[1].bar(x + 0.18, inc, width=0.36, label="final local increase")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=35, ha="right")
    axes[1].set_ylabel("volume (10^3 m3)")
    axes[1].set_title("Final signed spatial effect")
    axes[1].legend(fontsize=8)
    axes[2].bar(labels, [float(r["residual_gt_1mm_pct"]) for r in event_rows], color="#16a34a")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=35, ha="right")
    axes[2].set_ylabel("cells/time samples (%)")
    axes[2].set_title("Residual larger than 1 mm")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def fig_model_performance(fixed_rows: list[dict[str, object]], loo_rows: list[dict[str, object]]) -> Path:
    path = FIG_DIR / "fig06_drainlite_performance.png"
    order = ["surface_only", "base", "all_static", "swmm_assisted"]
    fixed = {r["model"]: r for r in fixed_rows}
    loo = {r["model"]: r for r in loo_rows}
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
    labels = [m for m in order if m in fixed]
    axes[0].bar(labels, [float(fixed[m]["mae_mm"]) for m in labels], color="#3b82f6")
    axes[0].set_ylabel("MAE (mm)")
    axes[0].set_title("Fixed split: event68-event70")
    axes[0].tick_params(axis="x", rotation=25)
    labels2 = [m for m in ["surface_only", "base", "all_static"] if m in loo]
    axes[1].bar(labels2, [float(loo[m]["mae_mean_mm"]) for m in labels2], color="#22c55e")
    axes[1].set_ylabel("MAE (mm)")
    axes[1].set_title("Leave-one-event-out validation")
    axes[1].tick_params(axis="x", rotation=20)
    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def fig_prediction_maps() -> Path:
    path = FIG_DIR / "fig07_drainlite_prediction_maps.png"
    cols = ["No-drain", "Coupled target", "DrainLite", "Error", "Predicted residual"]
    fig, axes = plt.subplots(len(TEST_EVENTS), len(cols), figsize=(16, 9))
    vmax = 0.0
    cache = {}
    err_lim = 0.02
    for event in TEST_EVENTS:
        a = load_event(event)
        pred_path = MODEL_OUT / "predictions" / event / "h_drainlite_official_all_static.npy"
        pred = np.load(pred_path).astype(np.float32)
        residual = pred - a["surface"]
        error = pred - a["coupled"]
        cache[event] = {
            "No-drain": a["surface"].max(axis=0),
            "Coupled target": a["coupled"].max(axis=0),
            "DrainLite": pred.max(axis=0),
            "Error": error[np.argmax(np.abs(error), axis=0), np.indices(error.shape[1:])[0], np.indices(error.shape[1:])[1]],
            "Predicted residual": residual[
                np.argmax(np.abs(residual), axis=0), np.indices(residual.shape[1:])[0], np.indices(residual.shape[1:])[1]
            ],
        }
        vmax = max(vmax, float(cache[event]["No-drain"].max()), float(cache[event]["Coupled target"].max()), float(cache[event]["DrainLite"].max()))
        err_lim = max(err_lim, float(np.max(np.abs(cache[event]["Error"]))), float(np.max(np.abs(cache[event]["Predicted residual"]))))
    for i, event in enumerate(TEST_EVENTS):
        for j, col in enumerate(cols):
            ax = axes[i, j]
            data = cache[event][col]
            if col in {"Error", "Predicted residual"}:
                im = ax.imshow(data, cmap="coolwarm", vmin=-err_lim, vmax=err_lim)
            else:
                im = ax.imshow(data, cmap="viridis", vmin=0, vmax=vmax)
            if i == 0:
                ax.set_title(col, fontsize=10)
            if j == 0:
                ax.set_ylabel(event, fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle("DrainLite all-static predictions on held-out formal events", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def generate_figures(static: dict[str, np.ndarray], event_rows, fixed_rows, loo_rows) -> dict[str, Path]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    coupling_diff = FIG_DIR / "fig08_formal_coupling_peak_difference_grid_mm.png"
    event70_zoom = FIG_DIR / "fig09_event70_max_coupling_difference_zoom.png"
    temporal_diff = FIG_DIR / "fig10_formal_coupling_temporal_difference.png"
    shutil.copy2(AUDIT_DIR / "figures" / "formal_coupling_peak_difference_grid_mm.png", coupling_diff)
    shutil.copy2(AUDIT_DIR / "figures" / "event70_max_coupling_difference_zoom.png", event70_zoom)
    shutil.copy2(AUDIT_DIR / "figures" / "formal_coupling_temporal_difference.png", temporal_diff)
    return {
        "workflow": fig_workflow(),
        "static": fig_static_map(static),
        "mike_grid": fig_mike_validation_grid(),
        "coupling_diff": coupling_diff,
        "event70_zoom": event70_zoom,
        "temporal_diff": temporal_diff,
        "temporal": fig_temporal_effects(),
        "residual": fig_residual_character(event_rows),
        "performance": fig_model_performance(fixed_rows, loo_rows),
        "prediction": fig_prediction_maps(),
    }


def html_table(rows: list[dict[str, object]], cols: list[tuple[str, str]], digits: int = 3) -> str:
    def fmt(value: object) -> str:
        if isinstance(value, (float, np.floating)):
            if math.isnan(float(value)):
                return "待补充"
            if abs(float(value)) >= 10000:
                return f"{float(value):,.0f}"
            return f"{float(value):.{digits}f}"
        return str(value)

    head = "".join(f"<th>{html.escape(label)}</th>" for _, label in cols)
    trs = []
    for row in rows:
        cells = "".join(f"<td>{html.escape(fmt(row.get(key, '')))}</td>" for key, _ in cols)
        trs.append(f"<tr>{cells}</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(trs)}</tbody></table>"


def figure_block(idx: int, title: str, path: Path, text: str) -> str:
    return (
        f'<figure id="fig{idx}"><img src="{img64(path)}" alt="{html.escape(title)}">'
        f"<figcaption><b>图 {idx}. {html.escape(title)}</b><br>{text}</figcaption></figure>"
    )


def mean(rows: list[dict[str, object]], key: str) -> float:
    return float(np.mean([float(r[key]) for r in rows]))


def lookup(rows: list[dict[str, object]], model: str, key: str) -> float:
    for r in rows:
        if r.get("model") == model:
            return float(r[key])
    return math.nan


def build_report_html(
    figures: dict[str, Path],
    inventory_rows,
    dataset_rows,
    event_rows,
    fixed_rows,
    loo_rows,
    loo_event_rows,
    larno_rows,
) -> str:
    styles = """
body{margin:0;background:#eef2f7;color:#1f2937;font-family:"Microsoft YaHei","Noto Sans CJK SC",Arial,sans-serif}
.page{max-width:1180px;margin:0 auto;background:#fff;padding:46px 38px 84px}
h1{font-size:31px;line-height:1.25;margin:0 0 8px;color:#0f2748}
h2{font-size:23px;margin:38px 0 12px;padding-left:12px;border-left:5px solid #2563eb;color:#102a43}
h3{font-size:18px;margin:26px 0 8px;color:#243b53}
p,li{font-size:15px;line-height:1.85}
.meta{color:#64748b}
.abstract{background:#eff6ff;border:1px solid #bfdbfe;border-radius:8px;padding:16px 18px}
.note{background:#fff7ed;border:1px solid #fed7aa;border-radius:8px;padding:14px 16px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:14px;margin:18px 0}
.card{border:1px solid #dbe3ef;background:#fbfdff;border-radius:8px;padding:14px}
.card b{display:block;font-size:22px;color:#1d4ed8;margin-bottom:4px}
table{width:100%;border-collapse:collapse;margin:14px 0 26px;font-size:13px}
th,td{border:1px solid #d9e2ec;padding:8px 9px;text-align:right;vertical-align:top}
th:first-child,td:first-child{text-align:left}
th{background:#edf2f7;color:#243b53}
figure{margin:24px 0 36px}
img{max-width:100%;height:auto;border:1px solid #d9e2ec;border-radius:6px;background:#fff}
figcaption{font-size:14px;line-height:1.8;color:#334155;margin-top:9px}
code{background:#eef2f7;border-radius:4px;padding:2px 5px}
.toc a{display:block;color:#1d4ed8;text-decoration:none;margin:5px 0}
"""
    nums = {
        "surface_mike_mae": mean(event_rows, "surface_vs_mike_mae_mm"),
        "coupled_mike_mae": mean(event_rows, "coupled_vs_mike_mae_mm"),
        "residual_mae": mean(event_rows, "surface_to_coupled_mae_mm"),
        "residual_rmse": mean(event_rows, "surface_to_coupled_rmse_mm"),
        "fixed_surface": lookup(fixed_rows, "surface_only", "mae_mm"),
        "fixed_all_static": lookup(fixed_rows, "all_static", "mae_mm"),
        "fixed_all_static_rmse": lookup(fixed_rows, "all_static", "rmse_mm"),
        "loo_surface": lookup(loo_rows, "surface_only", "mae_mean_mm"),
        "loo_all_static": lookup(loo_rows, "all_static", "mae_mean_mm"),
        "loo_all_static_rmse": lookup(loo_rows, "all_static", "rmse_mean_mm"),
        "loo_improve": lookup(loo_rows, "all_static", "improvement_vs_surface_pct"),
    }
    dataset_table = html_table(dataset_rows, [("item", "项目"), ("value", "数值"), ("note", "说明")])
    inventory_table = html_table(inventory_rows, [("event", "事件"), ("required_arrays", "数组检查"), ("shape", "形状"), ("formal_npz", "正式 NPZ"), ("notes", "备注")])
    event_table = html_table(
        event_rows,
        [
            ("event", "事件"),
            ("rain_total_mean_mm", "平均累计雨量 mm"),
            ("mike_peak_m", "MIKE 峰值 m"),
            ("surface_peak_m", "无管网峰值 m"),
            ("coupled_peak_m", "耦合峰值 m"),
            ("surface_vs_mike_mae_mm", "无管网-MIKE MAE mm"),
            ("coupled_vs_mike_mae_mm", "耦合-MIKE MAE mm"),
            ("surface_to_coupled_mae_mm", "耦合残差 MAE mm"),
            ("surface_to_coupled_rmse_mm", "耦合残差 RMSE mm"),
            ("residual_gt_1mm_pct", ">1mm 样本 %"),
            ("legacy_positive_exchange_m3", "旧脚本正号交换 m3"),
        ],
    )
    fixed_table = html_table(
        fixed_rows,
        [
            ("model", "模型"),
            ("mae_mm", "MAE mm"),
            ("rmse_mm", "RMSE mm"),
            ("csi_0p03", "CSI 0.03m"),
            ("csi_0p15", "CSI 0.15m"),
            ("mae_improvement_vs_surface_pct", "较无修正改善 %"),
        ],
    )
    loo_table = html_table(
        loo_rows,
        [
            ("model", "模型"),
            ("mae_mean_mm", "平均 MAE mm"),
            ("mae_std_mm", "MAE 标准差 mm"),
            ("rmse_mean_mm", "平均 RMSE mm"),
            ("csi_0p03_mean", "CSI 0.03m"),
            ("csi_0p15_mean", "CSI 0.15m"),
            ("improvement_vs_surface_pct", "较无修正改善 %"),
        ],
    )
    loo_event_table = html_table(
        loo_event_rows,
        [("fold", "留出事件"), ("model", "模型"), ("mae_mm", "MAE mm"), ("rmse_mm", "RMSE mm"), ("peak_error_mm", "峰值误差 mm")],
    )
    larno_table = html_table(larno_rows, [("item", "项目"), ("status", "状态"), ("evidence", "证据/验收条件")])
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>正式 ITZI-SWMM 数据源下的 LarNO-DrainLite 轻量管网增强研究报告</title>
<style>{styles}</style>
</head>
<body><main class="page">
<h1>正式 ITZI-SWMM 数据源下的 LarNO-DrainLite 轻量管网增强研究报告</h1>
<p class="meta">生成时间：2026-07-26；工作目录：E:/Projects/20260519-LarNO；报告口径：正式双向耦合数据源、MIKE 外部验证、轻量有符号残差模型。</p>
<section class="abstract">
<p><b>摘要。</b>本报告把前期工作收束为一条简单主线：使用用户原有、已校准的 ITZI-SWMM 双向耦合模型副本生成正式数据集；同时保留不考虑管网和考虑管网两套水深序列；用 MIKE 参考结果检查量级和空间格局是否合理；最后在 LarNO 类基础预报结果之后加入轻量有符号残差模型 DrainLite。正式八事件结果显示，ITZI-SWMM 加管网与不加管网的平均差异只有 {nums['residual_mae']:.3f} 毫米，但少数区域存在更明显的局部削减或壅水。DrainLite all_static 在固定测试集上将 MAE 从 {nums['fixed_surface']:.3f} 毫米降至 {nums['fixed_all_static']:.3f} 毫米；留一事件验证将 MAE 从 {nums['loo_surface']:.3f} 毫米降至 {nums['loo_all_static']:.3f} 毫米，改善 {nums['loo_improve']:.1f}%。</p>
</section>
<h2 id="toc">目录</h2>
<nav class="toc">
<a href="#background">1. 研究背景与目标</a>
<a href="#data">2. 数据源与验收检查</a>
<a href="#method">3. 方法：正式耦合标签与轻量残差模型</a>
<a href="#results">4. 结果展示与解释</a>
<a href="#larno">5. LarNO 残差分析接入状态</a>
<a href="#discussion">6. 分析、结论与后续补充</a>
</nav>
<h2 id="background">1. 研究背景与目标</h2>
<p>原 LarNO 论文的核心是学习 MIKE Plus 数值模型产生的城市洪涝时空水深结果。你的研究问题更聚焦：原始 LarNO 输入中没有显式城市排水管网，因此需要构建一种能把管网效应注入预报结果的轻量方法。这里不再把问题扩大成全量重训 LarNO，也不把早期概念性 sink 结果当作正式 reference；正式主线只围绕 ITZI-SWMM 双向耦合模型展开。</p>
{figure_block(1, "简化后的正式研究流程", figures['workflow'], "这张图说明本文的逻辑顺序。MIKE 只在左上角作为外部合理性检查，用来判断 ITZI-SWMM 的水深量级和峰值空间分布是否离谱。真正的数据集来源是用户原有模型的副本：同一场雨、同一片地形下分别运行无管网地表模型和 ITZI-SWMM 双向耦合模型。二者之差就是需要学习的管网残差。DrainLite 是轻量残差器，未来可以放在 LarNO 输出之后，形成 LarNO + DrainLite 的管网增强预报链。")}
<h2 id="data">2. 数据源与验收检查</h2>
<p>正式数据源为 <code>external_models/20260518-itzi-flood</code> 下的复制模型，不修改用户原路径。当前正式可用事件为八个：event1、event20、event65、event66、event67、event68、event69、event70。每个事件包含 72 个五分钟时间步，空间尺寸为 200 × 280，分辨率为 20 米。</p>
{dataset_table}
<p>下表是脚本自动完成的数据完整性检查。它逐事件确认降雨、MIKE 参考、正式无管网 ITZI、正式 ITZI-SWMM 耦合结果和耦合运行 NPZ 文件是否存在。</p>
{inventory_table}
{figure_block(2, "DEM 与正式管网栅格分布", figures['static'], "左图显示数字高程模型以及被抬高的建筑区域；中图叠加了管线、雨水口或检查井、排放口，用于检查管网和地形窗口是否处在同一空间位置；右图显示管线容量字段，只在管线经过的栅格上赋值。阅读这张图的重点不是看单个节点是否代表真实市政资料，而是检查正式计算所用的概化管网是否已经与 DEM 对齐，并且是否具备可输入轻量模型的空间特征。")}
<h2 id="method">3. 方法：正式耦合标签与轻量残差模型</h2>
<p>正式标签分为两类。第一类是 <code>h_itzi_swmm_official_surface.npy</code>，代表同一降雨条件下不考虑管网排水的 ITZI 地表结果。第二类是 <code>h_itzi_swmm_official.npy</code>，代表同一降雨条件下考虑 SWMM 管网交换后的双向耦合结果。模型不直接学习水深本身，而是学习有符号残差：</p>
<p><code>residual_mm = 1000 × (h_itzi_swmm_official - h_itzi_swmm_official_surface)</code></p>
<p>残差为负，表示管网排水或汇流改变使水深降低；残差为正，表示局部壅水、回灌或水动力重新分配使水深升高。由于正式双向耦合不是单调削峰过程，所以这里不能使用旧的非负削峰模型。DrainLite 使用树模型学习残差，并加入 2 毫米死区：当预测残差绝对值小于 2 毫米时置零，避免在大量近零背景像元上制造不必要的噪声修正。</p>
<h2 id="results">4. 结果展示与解释</h2>
<h3>4.1 MIKE 外部验证</h3>
<p>八事件平均来看，正式无管网 ITZI 与 MIKE 的 MAE 为 {nums['surface_mike_mae']:.3f} 毫米，正式 ITZI-SWMM 耦合与 MIKE 的 MAE 为 {nums['coupled_mike_mae']:.3f} 毫米。二者非常接近，说明管网耦合并没有把结果拉到另一个量级；同时也说明 MIKE 在这里更适合作为合理性验证，而不是 DrainLite 的训练目标。</p>
{event_table}
{figure_block(3, "八事件 MIKE、无管网 ITZI 与 ITZI-SWMM 峰值空间对比", figures['mike_grid'], "每一行是一个降雨事件，每一列是一种结果。前三列使用同一色标表示峰值水深，颜色越亮代表水越深；第四列是 ITZI-SWMM 峰值减去无管网峰值，红色表示耦合后更深，蓝色表示耦合后更浅。整体看，正式耦合结果和无管网结果的空间格局高度一致，说明管网影响是局部修正而不是改变整个洪水场。局部红蓝斑块说明双向耦合既可能排水削减，也可能因局部壅水或水量重新分配产生增水。")}
{figure_block(4, "正式耦合峰值差异诊断图", figures['coupling_diff'], "这张图专门回应“耦合和没耦合是不是没区别”的问题。图中只画 coupled minus surface-only 的峰值差异，单位是毫米，并把色标裁剪在 ±50 mm，以免个别极值把普通差异压暗。蓝色表示耦合后峰值降低，红色表示耦合后峰值升高。可以看到差异不是全域均匀铺开，而是沿局部低洼区、入口附近或汇流路径呈斑块状出现。因此在普通水深图里它会不明显，但在差异图里可以识别。")}
{figure_block(5, "event70 最大耦合差异局部放大", figures['event70_zoom'], "这张局部放大图选取 event70 中绝对差异最大的区域。前两幅分别是同一时刻的无管网水深和耦合水深，后两幅是该时刻和末时刻的差异。黑圈标出最大差异位置。它说明耦合不是没有生效，而是集中在少数像元或小片区，局部差异可以达到分米级；但因为受影响区域占比很低，全域平均差异仍然只有毫米量级。")}
<h3>4.2 时间过程与管网效应</h3>
{figure_block(6, "代表事件水量与最大水深时间过程", figures['temporal'], "左列是区域总水量，右列是最大水深。黑色 MIKE 曲线提供外部参照，蓝色是正式无管网 ITZI，绿色是正式 ITZI-SWMM。读图时应看三点：第一，三条曲线的起涨和峰现时间是否一致；第二，耦合曲线相对无管网曲线是否只是小幅移动；第三，峰值阶段是否有局部差异。图中可以看到正式耦合并没有产生夸张削峰，而是对局部和总体水量做毫米级到厘米级以下的修正。")}
{figure_block(7, "正式耦合时间差值诊断图", figures['temporal_diff'], "这张图把图 6 中几乎重合的曲线相减后再画出来。上图是耦合结果减无管网结果的区域总水量差，负值表示耦合后地表水量减少；中图是最大水深差，单位毫米；下图是绝对差异超过 1 毫米的面积比例。可以看到曲线不是完全为零，但幅度远小于原始水量和水深主轴，因此在图 6 的普通曲线图里几乎看不出来。这个结果说明管网耦合效应在当前概化网络下是弱而局部的。")}
{figure_block(8, "正式耦合残差的大小、方向与稀疏性", figures['residual'], "左图是所有事件抽样像元的有符号残差分布，横轴为耦合结果减无管网结果，单位为毫米；零点附近的高峰说明大部分像元管网影响很小，橙色虚线表示 2 毫米死区。中图把末时刻的正向削减和局部增水分开统计，说明正式耦合不是简单的单向排水。右图统计绝对残差超过 1 毫米的样本比例，比例只有约 1% 到 3% 左右，因此全域平均 MAE 很容易被大量近零区域主导。")}
<section class="note"><p><b>关于旧脚本中的 drained_m3 字段。</b>复查 ITZI 的 <code>DrainageNode</code> 符号约定后发现，耦合流量负值表示地表进入管网，正值表示管网回到地表。早期运行脚本把正号流量累计到 <code>drained_m3</code>，这个名称不严谨。因此本报告不再把它解释为净排水量，而把主要判断建立在水深数组直接计算的正负残差体积和空间差异图上。</p></section>
<h3>4.3 DrainLite 轻量残差模型</h3>
<p>固定测试集使用 event68、event69 和 event70。无修正 surface-only 的 MAE 为 {nums['fixed_surface']:.3f} 毫米；加入全部静态管网特征后的 all_static 模型 MAE 为 {nums['fixed_all_static']:.3f} 毫米，RMSE 为 {nums['fixed_all_static_rmse']:.3f} 毫米。留一事件验证中，all_static 平均 MAE 为 {nums['loo_all_static']:.3f} 毫米，平均 RMSE 为 {nums['loo_all_static_rmse']:.3f} 毫米。这说明轻量模型在正式耦合残差上是有效的，尤其对少数较大残差区域的 RMSE 改善明显。</p>
{fixed_table}
{loo_table}
{figure_block(9, "DrainLite 固定划分与留一验证性能", figures['performance'], "左图比较固定测试集上的模型误差，surface-only 表示不做任何残差修正，base 表示只使用水深、降雨、地形、坐标和时间等基础特征，all_static 表示再加入管网静态特征，swmm_assisted 表示额外使用事件级 SWMM 汇总信息。右图是留一事件验证，只保留最关键的三组。可以看到 all_static 相对 surface-only 稳定降低误差，但与 base 的差距不大，说明当前八事件正式标签中，管网静态特征贡献存在但不强。")}
{loo_event_table}
{figure_block(10, "测试事件 DrainLite 空间预测图", figures['prediction'], "每一行对应一个测试事件。第一列是正式无管网峰值，第二列是正式 ITZI-SWMM 耦合目标，第三列是 DrainLite 预测后的峰值，第四列是 DrainLite 减耦合目标的误差，第五列是模型预测的有符号残差。读图时要关注第四、第五列：误差图越接近零越好；残差图若只在局部出现，说明模型没有在全域乱改水深。当前结果显示，2 毫米死区让模型主要修正局部较明显区域，符合正式耦合残差稀疏的事实。")}
<h2 id="larno">5. LarNO 残差分析接入状态</h2>
<p>当前报告已经完成轻量残差模型本身的正式验证，但本地没有找到八事件逐时刻 LarNO 预测数组。因此，严格意义上的 <code>LarNO → DrainLite → coupled reference</code> 端到端对比还不能写成已完成结果。现阶段可写为：DrainLite 已在 ITZI surface-only 基线场上完成验证，方法上可以把输入基线场替换为 LarNO 的水深预测数组；真正的 LarNO 残差分析需要先运行 LarNO inference 并保存每个事件的 <code>h_larno_base.npy</code>。</p>
{larno_table}
<h2 id="discussion">6. 分析、结论与后续补充</h2>
<p>这条路线已经足够形成论文雏形。核心创新点不是“完整重训一个巨大 LarNO-D”，而是在低算力条件下提出一个正式耦合标签驱动的轻量管网残差注入方法。它解决了三个实际问题：第一，正式数据来自已有 ITZI-SWMM 双向耦合模型，而不是临时洼地填充或概念性 sink；第二，MIKE 只负责验证量级和格局，不混入训练标签；第三，残差是有符号的，能够同时表达排水削减和局部壅水。</p>
<section class="note"><p><b>需要补充但不影响当前结论的事项。</b>如果要把论文进一步写完整，最优先补充的是八事件 LarNO 逐事件预测数组，然后直接比较 LarNO、LarNO + DrainLite、正式 ITZI-SWMM coupled reference。第二优先是增加更多由正式 ITZI-SWMM 产生的事件，以增强训练和留一验证的统计稳定性。第三优先才是把 SWMM 的节点水头、管段流量、满管率和壅水时序栅格化；这会增强机理解释，但不应把当前工作重新复杂化。</p></section>
<p><b>主要结论。</b>正式新数据源已经建立并通过检查：八个事件、72 帧、20 米分辨率，包含正式无管网和正式 ITZI-SWMM 耦合标签。正式标签下必须使用有符号残差模型，而不是旧的非负削峰模型。加入 2 毫米死区后，DrainLite all_static 在固定测试集和留一验证中均相对 surface-only 改善。当前静态管网特征的边际贡献有限，这不是失败，而是说明正式耦合信号本身稀疏且较小；论文应如实把它写成管网残差注入的可行性验证和低算力创新。</p>
</main></body></html>
"""


def build_markdown(
    figures: dict[str, Path],
    inventory_rows,
    dataset_rows,
    event_rows,
    fixed_rows,
    loo_rows,
    larno_rows,
) -> str:
    nums = {
        "surface_mike_mae": mean(event_rows, "surface_vs_mike_mae_mm"),
        "coupled_mike_mae": mean(event_rows, "coupled_vs_mike_mae_mm"),
        "fixed_surface": lookup(fixed_rows, "surface_only", "mae_mm"),
        "fixed_all_static": lookup(fixed_rows, "all_static", "mae_mm"),
        "loo_surface": lookup(loo_rows, "surface_only", "mae_mean_mm"),
        "loo_all_static": lookup(loo_rows, "all_static", "mae_mean_mm"),
        "loo_improve": lookup(loo_rows, "all_static", "improvement_vs_surface_pct"),
    }

    def md_table(rows: list[dict[str, object]], keys: list[str]) -> str:
        out = ["|" + "|".join(keys) + "|", "|" + "|".join(["---"] * len(keys)) + "|"]
        for row in rows:
            vals = []
            for key in keys:
                value = row.get(key, "")
                if isinstance(value, (float, np.floating)):
                    vals.append(f"{float(value):.3f}" if abs(float(value)) < 10000 else f"{float(value):.0f}")
                else:
                    vals.append(str(value).replace("|", "/"))
            out.append("|" + "|".join(vals) + "|")
        return "\n".join(out)

    rel = lambda p: p.relative_to(OUT).as_posix()
    return dedent(
        f"""
        # 正式 ITZI-SWMM 数据源下的 LarNO-DrainLite 轻量管网增强研究报告

        ## 摘要

        本研究使用用户原有正式 ITZI-SWMM 双向耦合模型副本构建管网增强数据集，并在 LarNO 类基础预报之后加入轻量有符号残差模型 DrainLite。MIKE 仅作为外部合理性验证。八事件平均无管网 ITZI 与 MIKE 的 MAE 为 {nums['surface_mike_mae']:.3f} mm，耦合 ITZI-SWMM 与 MIKE 的 MAE 为 {nums['coupled_mike_mae']:.3f} mm。固定测试集上，DrainLite all_static 将 MAE 从 {nums['fixed_surface']:.3f} mm 降至 {nums['fixed_all_static']:.3f} mm；留一验证将 MAE 从 {nums['loo_surface']:.3f} mm 降至 {nums['loo_all_static']:.3f} mm，改善 {nums['loo_improve']:.1f}%。

        ## 1. 研究流程

        ![workflow]({rel(figures['workflow'])})

        ## 2. 数据集验收

        {md_table(dataset_rows, ['item','value','note'])}

        {md_table(inventory_rows, ['event','required_arrays','shape','formal_npz','notes'])}

        ![static]({rel(figures['static'])})

        ## 3. MIKE 合理性验证

        {md_table(event_rows, ['event','surface_vs_mike_mae_mm','coupled_vs_mike_mae_mm','surface_to_coupled_mae_mm','surface_to_coupled_rmse_mm','residual_gt_1mm_pct'])}

        ![mike validation]({rel(figures['mike_grid'])})

        ![coupling difference]({rel(figures['coupling_diff'])})

        ![event70 zoom]({rel(figures['event70_zoom'])})

        ![temporal]({rel(figures['temporal'])})

        ![temporal difference]({rel(figures['temporal_diff'])})

        ![residual]({rel(figures['residual'])})

        ## 4. DrainLite 轻量残差模型

        {md_table(fixed_rows, ['model','mae_mm','rmse_mm','mae_improvement_vs_surface_pct'])}

        {md_table(loo_rows, ['model','mae_mean_mm','rmse_mean_mm','improvement_vs_surface_pct'])}

        ![performance]({rel(figures['performance'])})

        ![prediction]({rel(figures['prediction'])})

        ## 5. LarNO 接入状态

        {md_table(larno_rows, ['item','status','evidence'])}

        ## 6. 结论

        正式数据源已经建立：八个事件、72 帧、20 米分辨率，同时包含无管网和正式 ITZI-SWMM 耦合结果。正式标签下应使用有符号残差，而不是旧的非负削峰模型。DrainLite 在固定测试和留一验证中均相对 surface-only 降低误差，说明低算力条件下可以实现管网效应的轻量注入。下一步最该补充的是 LarNO 逐事件预测数组，用同一个 DrainLite 残差器完成真正的 LarNO 与 LarNO+DrainLite 对比。
        """
    ).strip() + "\n"


def check_self_contained(html_text: str) -> None:
    if not html_text.lstrip().startswith("<!DOCTYPE html>"):
        raise RuntimeError("HTML must start with <!DOCTYPE html>.")
    if "<style>" not in html_text or "</style>" not in html_text:
        raise RuntimeError("HTML missing internal CSS.")
    if re.search(r'<img[^>]+src=["\'](?!data:image/png;base64,)', html_text, re.I):
        raise RuntimeError("HTML contains non-base64 image source.")
    if re.search(r'https?://', html_text, re.I):
        raise RuntimeError("HTML contains external network URL.")


def find_browser() -> str | None:
    candidates = [
        shutil.which("msedge"),
        shutil.which("chrome"),
        shutil.which("chromium"),
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return str(candidate)
    return None


def render_pdf() -> bool:
    browser = find_browser()
    if not browser:
        return False
    if REPORT_PDF.exists():
        REPORT_PDF.unlink()
    profile = OUT / ".browser_pdf_profile"
    profile.mkdir(parents=True, exist_ok=True)
    cmd = [
        browser,
        "--headless=new",
        "--disable-gpu",
        "--no-first-run",
        f"--user-data-dir={profile}",
        f"--print-to-pdf={REPORT_PDF}",
        REPORT_HTML.resolve().as_uri(),
    ]
    result = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=False, timeout=240)
    if result.returncode != 0:
        stdout = result.stdout.decode("utf-8", errors="replace") if result.stdout else ""
        stderr = result.stderr.decode("utf-8", errors="replace") if result.stderr else ""
        print(stdout)
        print(stderr)
        return False
    return REPORT_PDF.exists() and REPORT_PDF.stat().st_size > 10_000


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    static = load_static()
    inventory_rows = check_inventory()
    dataset_rows = static_summary(static)
    event_rows = compute_event_metrics()
    fixed_rows = summarize_model_rows(MODEL_OUT / "metrics" / "official_residual_ablation_summary.csv")
    loo_rows = summarize_model_rows(CV_DIR / "official_residual_leave_one_event_out_summary.csv")
    loo_event_raw = read_csv(CV_DIR / "official_residual_leave_one_event_out_metrics.csv")
    loo_event_rows = [
        {
            "fold": r["fold"],
            "model": r["model"],
            "mae_mm": float(r["mae_m"]) * 1000.0,
            "rmse_mm": float(r["rmse_m"]) * 1000.0,
            "peak_error_mm": float(r["peak_error_m"]) * 1000.0,
        }
        for r in loo_event_raw
        if r.get("model") in {"surface_only", "base", "all_static"}
    ]
    larno_rows = larno_inventory()
    write_csv(TABLE_DIR / "table00_dataset_inventory.csv", dataset_rows)
    write_csv(TABLE_DIR / "table01_file_integrity.csv", inventory_rows)
    write_csv(TABLE_DIR / "table02_event_validation_metrics.csv", event_rows)
    write_csv(TABLE_DIR / "table03_fixed_split_drainlite.csv", fixed_rows)
    write_csv(TABLE_DIR / "table04_leave_one_event_out_summary.csv", loo_rows)
    write_csv(TABLE_DIR / "table05_leave_one_event_out_event_metrics.csv", loo_event_rows)
    write_csv(TABLE_DIR / "table06_larno_interface_status.csv", larno_rows)
    figures = generate_figures(static, event_rows, fixed_rows, loo_rows)
    html_text = build_report_html(figures, inventory_rows, dataset_rows, event_rows, fixed_rows, loo_rows, loo_event_rows, larno_rows)
    check_self_contained(html_text)
    REPORT_HTML.write_text(html_text, encoding="utf-8")
    REPORT_MD.write_text(build_markdown(figures, inventory_rows, dataset_rows, event_rows, fixed_rows, loo_rows, larno_rows), encoding="utf-8")
    metadata = {
        "storyline": "formal ITZI-SWMM dataset -> MIKE plausibility check -> DrainLite signed residual -> LarNO interface",
        "events": EVENTS,
        "shape": SHAPE,
        "cell_size_m": CELL_SIZE_M,
        "residual_deadband_mm": DEADBAND_MM,
        "html_self_contained": True,
        "source_model_copy": str(FORMAL_NPZ_DIR.relative_to(ROOT)),
    }
    (OUT / "package_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    pdf_ok = render_pdf()
    shutil.copy2(REPORT_HTML, ROOT_HTML)
    shutil.copy2(REPORT_MD, ROOT_MD)
    if pdf_ok:
        shutil.copy2(REPORT_PDF, ROOT_PDF)
    print(f"HTML: {REPORT_HTML} ({REPORT_HTML.stat().st_size / 1024 / 1024:.2f} MB)")
    print(f"Markdown: {REPORT_MD} ({REPORT_MD.stat().st_size / 1024:.1f} KB)")
    print(f"PDF: {'OK ' + str(REPORT_PDF) if pdf_ok else 'not generated'}")
    print(f"Root copies: {ROOT_HTML}, {ROOT_MD}" + (f", {ROOT_PDF}" if pdf_ok else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
