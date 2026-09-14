#!/usr/bin/env python3
"""Generate the formal ITZI-SWMM based paper package and standalone report."""

from __future__ import annotations

import base64
import csv
import html
import json
import math
import re
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
IMPORT_DIR = ROOT / "extended_study" / "output" / "official_itzi_swmm_import"
OFFICIAL_MODEL_DIR = ROOT / "extended_study" / "output" / "drainlite_official_residual"
CV_DIR = ROOT / "extended_study" / "output" / "official_paper_package" / "metrics"
OUT = ROOT / "extended_study" / "output" / "official_paper_package"
FIG_DIR = OUT / "figures"
TABLE_DIR = OUT / "tables"
EVENTS = ["event1", "event20", "event65", "event66", "event67", "event68", "event69", "event70"]
CELL = 20.0
CELL_AREA = CELL * CELL


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def f(row: dict[str, str], key: str, default: float = math.nan) -> float:
    try:
        return float(row[key])
    except Exception:
        return default


def save_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def img64(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def inline_md(text: str) -> str:
    escaped = html.escape(text)
    return re.sub(r"`([^`]+)`", r"<code>\1</code>", escaped)


def simple_markdown_to_html(markdown_text: str, base_dir: Path) -> str:
    """Small markdown subset converter for the generated paper draft."""
    parts: list[str] = []
    in_list = False
    for raw in markdown_text.splitlines():
        line = raw.rstrip()
        if not line:
            if in_list:
                parts.append("</ul>")
                in_list = False
            continue
        img_match = re.match(r"!\[([^\]]*)\]\(([^)]+)\)", line)
        if img_match:
            if in_list:
                parts.append("</ul>")
                in_list = False
            alt, src = img_match.groups()
            path = base_dir / src
            parts.append(
                f'<figure><img src="{img64(path)}" alt="{html.escape(alt)}">'
                f"<figcaption>{html.escape(alt)}</figcaption></figure>"
            )
            continue
        if line.startswith("### "):
            if in_list:
                parts.append("</ul>")
                in_list = False
            parts.append(f"<h3>{inline_md(line[4:])}</h3>")
        elif line.startswith("## "):
            if in_list:
                parts.append("</ul>")
                in_list = False
            parts.append(f"<h2>{inline_md(line[3:])}</h2>")
        elif line.startswith("# "):
            if in_list:
                parts.append("</ul>")
                in_list = False
            parts.append(f"<h1>{inline_md(line[2:])}</h1>")
        elif line.startswith("- "):
            if not in_list:
                parts.append("<ul>")
                in_list = True
            parts.append(f"<li>{inline_md(line[2:])}</li>")
        else:
            if in_list:
                parts.append("</ul>")
                in_list = False
            parts.append(f"<p>{inline_md(line)}</p>")
    if in_list:
        parts.append("</ul>")
    return "\n".join(parts)


def html_table(rows: list[dict[str, object]], cols: list[tuple[str, str]], digits: int = 3) -> str:
    thead = "".join(f"<th>{html.escape(label)}</th>" for _, label in cols)
    trs = []
    for row in rows:
        cells = []
        for key, _ in cols:
            val = row.get(key, "")
            if isinstance(val, float):
                text = f"{val:.{digits}f}" if abs(val) < 10000 else f"{val:,.0f}"
            else:
                text = str(val)
            cells.append(f"<td>{html.escape(text)}</td>")
        trs.append("<tr>" + "".join(cells) + "</tr>")
    return "<table><thead><tr>" + thead + "</tr></thead><tbody>" + "".join(trs) + "</tbody></table>"


def load_event_arrays(event: str) -> dict[str, np.ndarray]:
    d = DATA_DIR / event
    return {
        "mike": np.load(d / "h_mike_ref.npy").astype(np.float32),
        "surface": np.load(d / "h_itzi_swmm_official_surface.npy").astype(np.float32),
        "official": np.load(d / "h_itzi_swmm_official.npy").astype(np.float32),
        "sink": np.load(d / "h_itzi_sink.npy").astype(np.float32),
        "rainfall": np.load(d / "rainfall.npy").astype(np.float32),
    }


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


def metric_tables() -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    import_rows_raw = read_csv(IMPORT_DIR / "official_itzi_swmm_event_metrics.csv")
    single_raw = read_csv(OFFICIAL_MODEL_DIR / "metrics" / "official_residual_ablation_summary.csv")
    loo_raw = read_csv(CV_DIR / "official_residual_leave_one_event_out_summary.csv")
    loo_event_raw = read_csv(CV_DIR / "official_residual_leave_one_event_out_metrics.csv")

    event_rows: list[dict[str, object]] = []
    for r in import_rows_raw:
        event_rows.append(
            {
                "event": r["event"],
                "mike_peak_m": f(r, "mike_peak_max_m"),
                "surface_peak_m": f(r, "surface_peak_max_m"),
                "official_peak_m": f(r, "official_swmm_peak_max_m"),
                "sink_peak_m": f(r, "conceptual_sink_peak_max_m"),
                "surface_mae_m": f(r, "surface_vs_mike_mae_m"),
                "official_mae_m": f(r, "official_swmm_vs_mike_mae_m"),
                "sink_mae_m": f(r, "conceptual_sink_vs_mike_mae_m"),
                "drained_m3": f(r, "official_swmm_drained_m3"),
                "positive_reduction_m3": f(r, "official_swmm_positive_reduction_volume_m3"),
                "local_surcharge_m3": f(r, "official_swmm_local_surcharge_volume_m3"),
                "surcharge_cells": f(r, "official_swmm_local_surcharge_cells"),
            }
        )

    single_rows = [
        {
            "model": r["model"],
            "mae_mm": f(r, "mae_mm"),
            "rmse_mm": f(r, "rmse_mm"),
            "csi_003": f(r, "csi_0p03"),
            "csi_015": f(r, "csi_0p15"),
            "improve_surface_pct": f(r, "mae_improvement_vs_surface_pct"),
            "improve_base_pct": f(r, "mae_improvement_vs_base_pct"),
        }
        for r in single_raw
    ]

    loo_rows = [
        {
            "model": r["model"],
            "mae_mean_mm": f(r, "mae_mean_mm"),
            "mae_std_mm": f(r, "mae_std_mm"),
            "rmse_mean_mm": f(r, "rmse_mean_mm"),
            "csi_003": f(r, "csi_0p03_mean"),
            "csi_015": f(r, "csi_0p15_mean"),
            "peak_error_mean_mm": f(r, "peak_error_mean_mm"),
            "improve_surface_pct": f(r, "improvement_vs_surface_pct"),
            "improve_base_pct": f(r, "improvement_vs_base_pct"),
        }
        for r in loo_raw
    ]

    loo_event_rows = [
        {
            "fold": r["fold"],
            "model": r["model"],
            "mae_mm": f(r, "mae_m") * 1000.0,
            "rmse_mm": f(r, "rmse_m") * 1000.0,
            "peak_error_mm": f(r, "peak_error_m") * 1000.0,
        }
        for r in loo_event_raw
    ]
    return event_rows, single_rows, loo_rows, loo_event_rows


def make_box(ax, xy: tuple[float, float], text: str, color: str, width: float = 2.3, height: float = 0.75) -> None:
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.03,rounding_size=0.06",
        linewidth=1.1,
        edgecolor="#243b53",
        facecolor=color,
    )
    ax.add_patch(patch)
    ax.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=10)


def make_arrow(ax, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=15, linewidth=1.1, color="#334e68"))


def fig_workflow() -> Path:
    path = FIG_DIR / "fig01_workflow.png"
    fig, ax = plt.subplots(figsize=(13, 5.2))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    make_box(ax, (0.3, 3.6), "LarNO / MIKE\nbenchmark", "#d9e8ff")
    make_box(ax, (0.3, 2.2), "Official copied\nITZI-SWMM model", "#e2f7dc")
    make_box(ax, (0.3, 0.8), "Road-aligned\nconceptual sink", "#fff1cc")
    make_box(ax, (3.4, 3.6), "Rainfall + DEM\n20 m, 72 frames", "#f0f4f8")
    make_box(ax, (3.4, 2.2), "h_itzi_swmm_official\nsigned coupling label", "#e2f7dc", width=2.8)
    make_box(ax, (3.4, 0.8), "h_itzi_sink\ncontrolled reduction label", "#fff1cc", width=2.8)
    make_box(ax, (7.0, 2.8), "DrainLite-OfficialResidual\nsigned residual learning", "#e6e6ff", width=3.1)
    make_box(ax, (7.0, 1.5), "DrainLite-Sink\nmonotone reduction baseline", "#fbe0e6", width=3.1)
    make_box(ax, (10.4, 2.15), "Paper results:\nvalidation, maps, limits", "#edf2f7", width=1.4, height=1.3)
    for s, e in [
        ((2.6, 4.0), (3.4, 4.0)),
        ((2.6, 2.6), (3.4, 2.6)),
        ((2.6, 1.2), (3.4, 1.2)),
        ((6.2, 2.6), (7.0, 3.1)),
        ((6.2, 1.2), (7.0, 1.8)),
        ((10.1, 3.1), (10.4, 2.8)),
        ((10.1, 1.8), (10.4, 2.5)),
    ]:
        make_arrow(ax, s, e)
    ax.text(
        6.0,
        4.55,
        "Formal storyline: the official ITZI-SWMM label is the main drainage-aware reference;\n"
        "the conceptual sink branch is retained only as a controlled comparison.",
        ha="center",
        va="center",
        fontsize=11,
        color="#102a43",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def fig_static_map(static: dict[str, np.ndarray]) -> Path:
    path = FIG_DIR / "fig02_dem_network.png"
    dem = np.ma.masked_where(static["dem"] >= 49.9, static["dem"])
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    im0 = axes[0].imshow(dem, cmap="terrain")
    axes[0].set_title("DEM / building-elevated terrain")
    axes[0].axis("off")
    fig.colorbar(im0, ax=axes[0], fraction=0.046)
    axes[1].imshow(dem, cmap="Greys", alpha=0.65)
    axes[1].contour(static["pipe_mask"], levels=[0.5], colors="#1f2937", linewidths=0.4)
    y, x = np.where(static["drain_inlet_mask"] > 0)
    axes[1].scatter(x, y, s=3, c="#1261a0", label="inlet")
    yo, xo = np.where(static["drain_outfall_mask"] > 0)
    axes[1].scatter(xo, yo, s=30, c="#b42318", label="outfall")
    axes[1].legend(loc="lower right", fontsize=8)
    axes[1].set_title("Formal SWMM network alignment")
    axes[1].axis("off")
    im2 = axes[2].imshow(static["distance_to_outfall"], cmap="magma")
    axes[2].set_title("Distance to outfall")
    axes[2].axis("off")
    fig.colorbar(im2, ax=axes[2], fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def fig_peak_grid() -> Path:
    path = FIG_DIR / "fig03_eight_event_peak_grid.png"
    nrow, ncol = len(EVENTS), 5
    peaks = {}
    vmax = 0.0
    for evt in EVENTS:
        arr = load_event_arrays(evt)
        peaks[evt] = {
            "MIKE": arr["mike"].max(axis=0),
            "Surface": arr["surface"].max(axis=0),
            "Official": arr["official"].max(axis=0),
            "Sink": arr["sink"].max(axis=0),
            "Official-Surface": arr["official"].max(axis=0) - arr["surface"].max(axis=0),
        }
        vmax = max(vmax, float(peaks[evt]["MIKE"].max()), float(peaks[evt]["Surface"].max()), float(peaks[evt]["Official"].max()), float(peaks[evt]["Sink"].max()))
    fig, axes = plt.subplots(nrow, ncol, figsize=(15, 20))
    cols = ["MIKE", "Surface", "Official", "Sink", "Official-Surface"]
    for i, evt in enumerate(EVENTS):
        for j, col in enumerate(cols):
            ax = axes[i, j]
            data = peaks[evt][col]
            if col == "Official-Surface":
                lim = max(0.12, float(np.nanmax(np.abs(data))))
                im = ax.imshow(data, cmap="coolwarm", vmin=-lim, vmax=lim)
            else:
                im = ax.imshow(data, cmap="viridis", vmin=0, vmax=vmax)
            if i == 0:
                ax.set_title(col, fontsize=10)
            if j == 0:
                ax.set_ylabel(evt, fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle("Eight-event peak depth comparison: MIKE, surface-only, formal ITZI-SWMM and conceptual sink", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def fig_volume_bars(event_rows: list[dict[str, object]]) -> Path:
    path = FIG_DIR / "fig04_official_volume_balance.png"
    labels = [str(r["event"]) for r in event_rows]
    x = np.arange(len(labels))
    drained = np.array([float(r["drained_m3"]) for r in event_rows]) / 1000.0
    red = np.array([float(r["positive_reduction_m3"]) for r in event_rows]) / 1000.0
    sur = np.array([float(r["local_surcharge_m3"]) for r in event_rows]) / 1000.0
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - 0.25, drained, width=0.25, label="SWMM node drained")
    ax.bar(x, red, width=0.25, label="Net positive final reduction")
    ax.bar(x + 0.25, sur, width=0.25, label="Local final surcharge")
    ax.set_ylabel("Volume (10^3 m3)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.legend()
    ax.set_title("Formal ITZI-SWMM exchange and final spatial effects")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def fig_temporal_curves() -> Path:
    path = FIG_DIR / "fig05_temporal_hydrographs.png"
    selected = ["event20", "event65", "event70"]
    fig, axes = plt.subplots(len(selected), 2, figsize=(13, 10), sharex=True)
    t = np.arange(1, 73) * 5
    for i, evt in enumerate(selected):
        arr = load_event_arrays(evt)
        for name, color in [("mike", "#111827"), ("surface", "#1f77b4"), ("official", "#2ca02c"), ("sink", "#ff7f0e")]:
            vol = arr[name].sum(axis=(1, 2)) * CELL_AREA / 1000.0
            hmax = arr[name].max(axis=(1, 2))
            axes[i, 0].plot(t, vol, label=name, color=color, linewidth=1.5)
            axes[i, 1].plot(t, hmax, label=name, color=color, linewidth=1.5)
        axes[i, 0].set_ylabel(f"{evt}\nVolume (10^3 m3)")
        axes[i, 1].set_ylabel("Max depth (m)")
        axes[i, 0].grid(alpha=0.25)
        axes[i, 1].grid(alpha=0.25)
    axes[0, 0].set_title("Event water volume")
    axes[0, 1].set_title("Event maximum water depth")
    axes[-1, 0].set_xlabel("Time (min)")
    axes[-1, 1].set_xlabel("Time (min)")
    axes[0, 0].legend(ncol=4, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def fig_residual_distribution() -> Path:
    path = FIG_DIR / "fig06_signed_residual_distribution.png"
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    all_res = []
    final_red = []
    final_sur = []
    peak_change = []
    for evt in EVENTS:
        arr = load_event_arrays(evt)
        res = arr["official"] - arr["surface"]
        sample = res[:, ::5, ::5].ravel() * 1000.0
        all_res.append(sample[np.isfinite(sample)])
        final_red.append(float(np.maximum(arr["surface"][-1] - arr["official"][-1], 0).sum() * CELL_AREA))
        final_sur.append(float(np.maximum(arr["official"][-1] - arr["surface"][-1], 0).sum() * CELL_AREA))
        peak_change.append(float(arr["official"].max() - arr["surface"].max()))
    all_vals = np.concatenate(all_res)
    axes[0, 0].hist(np.clip(all_vals, -20, 20), bins=80, color="#5577aa")
    axes[0, 0].axvline(0, color="#111827", linewidth=1)
    axes[0, 0].set_title("Signed residual distribution")
    axes[0, 0].set_xlabel("Official - surface (mm)")
    axes[0, 0].set_ylabel("Sample count")
    x = np.arange(len(EVENTS))
    axes[0, 1].bar(x - 0.18, np.array(final_red) / 1000, width=0.36, label="reduction")
    axes[0, 1].bar(x + 0.18, np.array(final_sur) / 1000, width=0.36, label="surcharge")
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(EVENTS, rotation=35, ha="right")
    axes[0, 1].set_ylabel("Final volume (10^3 m3)")
    axes[0, 1].legend()
    axes[0, 1].set_title("Final reduction and local surcharge")
    axes[1, 0].bar(EVENTS, np.array(peak_change) * 1000, color="#aa5577")
    axes[1, 0].axhline(0, color="#111827", linewidth=1)
    axes[1, 0].set_xticks(np.arange(len(EVENTS)))
    axes[1, 0].set_xticklabels(EVENTS, rotation=35, ha="right")
    axes[1, 0].set_ylabel("Peak change (mm)")
    axes[1, 0].set_title("Peak official minus surface")
    evt = "event70"
    arr = load_event_arrays(evt)
    data = arr["official"][-1] - arr["surface"][-1]
    lim = max(0.02, float(np.nanmax(np.abs(data))))
    im = axes[1, 1].imshow(data, cmap="coolwarm", vmin=-lim, vmax=lim)
    axes[1, 1].set_title(f"{evt} final signed residual map")
    axes[1, 1].axis("off")
    fig.colorbar(im, ax=axes[1, 1], fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def fig_model_ablation(single_rows: list[dict[str, object]], loo_rows: list[dict[str, object]]) -> Path:
    path = FIG_DIR / "fig07_model_ablation.png"
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    order = ["surface_only", "base", "mask_only", "hydraulic_only", "all_static", "swmm_assisted", "conceptual_sink_label"]
    s_map = {r["model"]: r for r in single_rows}
    labels = [m for m in order if m in s_map]
    axes[0].bar(labels, [float(s_map[m]["mae_mm"]) for m in labels], color="#4477aa")
    axes[0].set_ylabel("MAE (mm)")
    axes[0].set_title("Single split: event68-70 test")
    axes[0].tick_params(axis="x", rotation=35)
    l_map = {r["model"]: r for r in loo_rows}
    labels2 = [m for m in ["surface_only", "base", "all_static"] if m in l_map]
    means = np.array([float(l_map[m]["mae_mean_mm"]) for m in labels2])
    stds = np.array([float(l_map[m]["mae_std_mm"]) for m in labels2])
    axes[1].bar(labels2, means, yerr=stds, capsize=4, color="#66aa77")
    axes[1].set_ylabel("MAE (mm)")
    axes[1].set_title("Leave-one-event-out validation")
    axes[1].tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def fig_eventwise_loo(loo_event_rows: list[dict[str, object]]) -> Path:
    path = FIG_DIR / "fig08_eventwise_loo.png"
    models = ["surface_only", "base", "all_static"]
    data = {m: [] for m in models}
    for evt in EVENTS:
        for m in models:
            row = next(r for r in loo_event_rows if r["fold"] == evt and r["model"] == m)
            data[m].append(float(row["mae_mm"]))
    x = np.arange(len(EVENTS))
    fig, ax = plt.subplots(figsize=(12, 5))
    width = 0.25
    for i, m in enumerate(models):
        ax.bar(x + (i - 1) * width, data[m], width=width, label=m)
    ax.set_xticks(x)
    ax.set_xticklabels(EVENTS, rotation=35, ha="right")
    ax.set_ylabel("MAE (mm)")
    ax.set_title("Event-wise formal label error under leave-one-event-out validation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def fig_prediction_maps() -> Path:
    path = FIG_DIR / "fig09_official_prediction_maps.png"
    rows = ["event68", "event69", "event70"]
    cols = ["MIKE", "Surface", "Official", "Predicted", "Error", "Residual"]
    fig, axes = plt.subplots(len(rows), len(cols), figsize=(18, 9))
    vmax = 0.0
    data_cache = {}
    for evt in rows:
        arr = load_event_arrays(evt)
        pred = np.load(OFFICIAL_MODEL_DIR / "predictions" / evt / "h_drainlite_official_base.npy").astype(np.float32)
        residual = pred.max(axis=0) - arr["surface"].max(axis=0)
        data_cache[evt] = {
            "MIKE": arr["mike"].max(axis=0),
            "Surface": arr["surface"].max(axis=0),
            "Official": arr["official"].max(axis=0),
            "Predicted": pred.max(axis=0),
            "Error": pred.max(axis=0) - arr["official"].max(axis=0),
            "Residual": residual,
        }
        vmax = max(vmax, *(float(data_cache[evt][c].max()) for c in ["MIKE", "Surface", "Official", "Predicted"]))
    for i, evt in enumerate(rows):
        for j, col in enumerate(cols):
            ax = axes[i, j]
            data = data_cache[evt][col]
            if col in ["Error", "Residual"]:
                lim = max(0.08, float(np.nanmax(np.abs(data))))
                im = ax.imshow(data, cmap="coolwarm", vmin=-lim, vmax=lim)
            else:
                im = ax.imshow(data, cmap="viridis", vmin=0, vmax=vmax)
            if i == 0:
                ax.set_title(col, fontsize=10)
            if j == 0:
                ax.set_ylabel(evt, fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle("DrainLite-OfficialResidual spatial predictions on held-out events", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def write_tables(event_rows, single_rows, loo_rows, loo_event_rows) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    save_csv(TABLE_DIR / "table01_official_event_metrics.csv", event_rows)
    save_csv(TABLE_DIR / "table02_single_split_model_summary.csv", single_rows)
    save_csv(TABLE_DIR / "table03_leave_one_event_out_summary.csv", loo_rows)
    save_csv(TABLE_DIR / "table04_leave_one_event_out_event_metrics.csv", loo_event_rows)
    dataset_rows = [
        {"item": "study_window", "value": "200 x 280 cells", "note": "20 m grid, 4.0 km x 5.6 km"},
        {"item": "time_steps", "value": "72", "note": "5 min interval, 6 h duration"},
        {"item": "formal_events", "value": "8", "note": ", ".join(EVENTS)},
        {"item": "formal_label", "value": "h_itzi_swmm_official.npy", "note": "formal copied ITZI-SWMM coupled output"},
        {"item": "conceptual_label", "value": "h_itzi_sink.npy", "note": "road-inlet sink comparison only"},
        {"item": "SWMM_sub_network", "value": "82 junctions, 1 outfall, 29 conduits", "note": "DYNWAVE routing"},
    ]
    save_csv(TABLE_DIR / "table00_dataset_inventory.csv", dataset_rows)


def make_figures(event_rows, single_rows, loo_rows, loo_event_rows) -> dict[str, Path]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    static = load_static()
    figures = {
        "workflow": fig_workflow(),
        "static": fig_static_map(static),
        "peak_grid": fig_peak_grid(),
        "volume": fig_volume_bars(event_rows),
        "temporal": fig_temporal_curves(),
        "residual": fig_residual_distribution(),
        "ablation": fig_model_ablation(single_rows, loo_rows),
        "eventwise": fig_eventwise_loo(loo_event_rows),
        "prediction": fig_prediction_maps(),
    }
    return figures


def key_numbers(event_rows, single_rows, loo_rows) -> dict[str, float | str]:
    def mean(key):
        return float(np.mean([float(r[key]) for r in event_rows]))

    single_map = {r["model"]: r for r in single_rows}
    loo_map = {r["model"]: r for r in loo_rows}
    return {
        "mean_official_mae_mm": mean("official_mae_m") * 1000,
        "mean_surface_mae_mm": mean("surface_mae_m") * 1000,
        "mean_sink_mae_mm": mean("sink_mae_m") * 1000,
        "mean_drained_m3": mean("drained_m3"),
        "mean_reduction_m3": mean("positive_reduction_m3"),
        "mean_surcharge_m3": mean("local_surcharge_m3"),
        "single_best_model": min(single_rows, key=lambda r: float(r["mae_mm"]))["model"],
        "single_best_mae_mm": min(float(r["mae_mm"]) for r in single_rows),
        "single_surface_mae_mm": float(single_map["surface_only"]["mae_mm"]),
        "loo_base_mae_mm": float(loo_map["base"]["mae_mean_mm"]),
        "loo_all_static_mae_mm": float(loo_map["all_static"]["mae_mean_mm"]),
        "loo_surface_mae_mm": float(loo_map["surface_only"]["mae_mean_mm"]),
        "loo_base_improve_pct": float(loo_map["base"]["improve_surface_pct"]),
        "loo_all_static_improve_pct": float(loo_map["all_static"]["improve_surface_pct"]),
    }


def paper_markdown(figures: dict[str, Path], event_rows, single_rows, loo_rows, nums: dict[str, float | str]) -> str:
    rel = lambda p: p.relative_to(OUT).as_posix()
    return dedent(
        f"""
        # DrainLite-OfficialResidual: a lightweight signed-residual approach for incorporating formal ITZI-SWMM drainage effects into urban flood forecasting

        ## Highlights

        - A formal ITZI-SWMM coupled label set was constructed for eight Shenzhen Region1 storm events at 20 m and 5 min resolution.
        - The previous conceptual road-inlet sink labels are retained only as controlled comparison labels, not as formal coupled references.
        - A signed residual model, DrainLite-OfficialResidual, learns both drainage reduction and local surcharge/backwater effects.
        - Leave-one-event-out validation reduces mean absolute error from {nums['loo_surface_mae_mm']:.3f} mm for surface-only to {nums['loo_base_mae_mm']:.3f} mm for the best lightweight residual model.
        - Static drainage features produced limited extra benefit over local surface, rainfall, terrain, time and coordinate features, indicating that the formal coupled signal is modest in the selected sub-region.

        ## Abstract

        Urban pluvial flood prediction depends on rainfall, terrain, buildings and drainage-system exchange. Recent neural-operator flood models such as LarNO learn high-resolution hydrodynamic mappings from numerical references, but the released benchmark inputs do not explicitly encode a user-defined drainage network for lightweight local experimentation. This study rebuilds the drainage-aware learning problem around a formal copied ITZI-SWMM coupled model for the Shenzhen Region1 subset. The formal model output is imported as 72-frame water-depth sequences for eight storm events and is distinguished from an earlier conceptual road-inlet sink label. Because bidirectional ITZI-SWMM coupling can reduce water depth in some cells and increase it in others through surcharge or backwater, we formulate a signed residual task: predicting the difference between the coupled and surface-only ITZI outputs. A CPU-friendly histogram-gradient-boosting model is trained as DrainLite-OfficialResidual. In leave-one-event-out validation, the best residual model reduces mean absolute error from {nums['loo_surface_mae_mm']:.3f} mm to {nums['loo_base_mae_mm']:.3f} mm. However, adding all static drainage features does not materially outperform the base residual model ({nums['loo_all_static_mae_mm']:.3f} mm), suggesting that the selected formal coupled label has weak spatial drainage signal relative to local water depth, rainfall, terrain and time features. The result supports a cautious innovation claim: formal drainage-aware residual learning is feasible on local hardware, but stronger pipe-network benefit will require larger event sets, stronger exchange labels, or time-varying hydraulic features from SWMM.

        ## 1. Introduction

        The LarNO paper demonstrates that a latent autoregressive neural operator can forecast urban flood depths efficiently at large scale. Its reference target is MIKE Plus output, while the released benchmark inputs emphasize rainfall and terrain fields. The present study asks a narrower applied question: can drainage-network influence be injected into such a forecasting workflow under limited local computing resources?

        The earlier lightweight branch used a conceptual sink target, where road-aligned inlets removed surface water and therefore imposed a monotone reduction constraint. That branch is useful as a controlled drainage-prior experiment, but it is not equivalent to formal two-way ITZI-SWMM coupling. The present paper therefore reorganizes the work around the copied calibrated ITZI-SWMM model and uses its outputs as the formal drainage-aware label.

        ## 2. Data and methods

        ### 2.1 Study data

        The working subset covers 200 x 280 cells at 20 m resolution, corresponding to 4.0 km x 5.6 km. Each event contains 72 time steps at 5 min spacing. The formal events are event1, event20 and event65 to event70. For each event, the dataset contains MIKE reference depth, ITZI surface-only depth, formal ITZI-SWMM coupled depth and conceptual sink depth.

        ![Workflow]({rel(figures['workflow'])})

        ### 2.2 Formal ITZI-SWMM label

        The formal label is `h_itzi_swmm_official.npy`, imported from the copied calibrated model under `external_models/20260518-itzi-flood`. The SWMM sub-network uses DYNWAVE routing with 82 junctions, one outfall and 29 conduits. The original user path was not modified.

        ![DEM and network]({rel(figures['static'])})

        ### 2.3 Signed residual learning

        The model target is:

        `residual_mm = 1000 * (h_itzi_swmm_official - h_itzi_surface)`

        Negative values represent drainage-induced reduction, while positive values represent local surcharge/backwater or coupled redistribution. The prediction is:

        `h_pred = max(h_itzi_surface + residual_mm / 1000, 0)`

        This differs from the previous sink-based DrainLite model, which predicted only nonnegative reduction.

        ## 3. Results

        ### 3.1 Formal coupled labels versus MIKE, surface-only and sink labels

        The formal ITZI-SWMM label is close to the surface-only ITZI result because the formal SWMM exchange in this selected sub-region is modest. Across the eight formal events, the mean official-versus-MIKE MAE is {nums['mean_official_mae_mm']:.3f} mm, compared with {nums['mean_surface_mae_mm']:.3f} mm for surface-only and {nums['mean_sink_mae_mm']:.3f} mm for the conceptual sink label.

        ![Eight event peak maps]({rel(figures['peak_grid'])})

        ![Volume balance]({rel(figures['volume'])})

        ### 3.2 Time evolution

        The time series show that formal ITZI-SWMM changes the surface-only hydrograph slightly, whereas the conceptual sink label can remove substantially more water. This distinction is important: a model trained on the sink label cannot be treated as a model trained on formal coupled physics.

        ![Temporal curves]({rel(figures['temporal'])})

        ### 3.3 Signed residual characteristics

        The signed residual distribution is strongly centered near zero. Final maps nevertheless include both positive reduction and local surcharge. This is why the official-label model must allow signed correction rather than enforcing monotone water-depth reduction.

        ![Signed residual]({rel(figures['residual'])})

        ### 3.4 Lightweight model performance

        In the fixed split using event68 to event70 as tests, the best model is {nums['single_best_model']} with MAE {nums['single_best_mae_mm']:.3f} mm. In leave-one-event-out validation, base features reduce MAE by {nums['loo_base_improve_pct']:.1f}% relative to surface-only, from {nums['loo_surface_mae_mm']:.3f} mm to {nums['loo_base_mae_mm']:.3f} mm. The all-static drainage-feature model gives {nums['loo_all_static_mae_mm']:.3f} mm, nearly identical to base.

        ![Model ablation]({rel(figures['ablation'])})

        ![Event-wise validation]({rel(figures['eventwise'])})

        ![Prediction maps]({rel(figures['prediction'])})

        ## 4. Discussion

        The formal-label results are more conservative than the conceptual sink results. This is scientifically useful: it prevents overstating the pipe-network effect. The present formal SWMM network drains measurable water at nodes, but the final two-dimensional water-depth change is small because drainage, local redistribution, and surcharge act together. Therefore, the primary paper claim should not be that static pipe features alone dramatically improve prediction. A stronger and more defensible claim is that formal drainage-aware residual learning is technically feasible on local hardware, and that signed residual learning correctly represents the bidirectional nature of ITZI-SWMM exchange.

        The limited marginal improvement from static drainage features has three likely causes. First, only eight formal events are available. Second, the formal coupled signal is very small relative to the baseline water-depth field. Third, static masks and pipe parameters do not include time-varying hydraulic states such as node head, pipe flow, storage, surcharge and outfall boundary effects.

        ## 5. Conclusions

        1. The formal drainage-aware data source is now the copied calibrated ITZI-SWMM model, not the earlier prototype.
        2. Eight events have complete 72-frame formal coupled labels.
        3. Signed residual learning is the correct lightweight formulation for formal ITZI-SWMM labels.
        4. The lightweight residual model improves over surface-only on formal labels, but static drainage features provide limited extra gain in the current subset.
        5. Future improvement should prioritize more formal coupled events and time-varying SWMM hydraulic features.

        ## Data and code availability

        All generated artifacts in this local study are stored under `extended_study/output/official_paper_package`, `extended_study/output/official_itzi_swmm_import`, and `extended_study/output/drainlite_official_residual`.
        """
    ).strip() + "\n"


def figure_block(idx: int, title: str, path: Path, explanation: str) -> str:
    return f"""
<figure id="fig-{idx}">
<img src="{img64(path)}" alt="{html.escape(title)}">
<figcaption><b>图 {idx} {html.escape(title)}</b><br>{explanation}</figcaption>
</figure>
"""


def report_html(figures: dict[str, Path], event_rows, single_rows, loo_rows, loo_event_rows, nums: dict[str, float | str]) -> str:
    styles = """
body{margin:0;background:#f3f6fa;color:#1f2933;font-family:"Microsoft YaHei","Noto Sans CJK SC",Arial,sans-serif}
.page{max-width:1200px;margin:0 auto;background:#fff;padding:44px 34px 80px}
h1{font-size:32px;margin:0 0 8px;color:#102a43}
h2{margin-top:38px;padding-left:12px;border-left:5px solid #2454a6;color:#102a43;font-size:23px}
h3{margin-top:24px;color:#243b53}
p,li{font-size:15px;line-height:1.8}
.meta{color:#627d98}
.abstract{background:#eef5ff;border:1px solid #c7ddff;border-radius:8px;padding:16px 18px}
.warn{background:#fff7ed;border:1px solid #fed7aa;border-radius:8px;padding:14px 16px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:14px;margin:20px 0}
.card{border:1px solid #d9e2ec;background:#fbfdff;border-radius:8px;padding:14px}
.card b{display:block;font-size:22px;color:#173b76;margin-bottom:5px}
table{width:100%;border-collapse:collapse;margin:14px 0 24px;font-size:13px}
th,td{border:1px solid #d9e2ec;padding:8px 9px;text-align:right}
th:first-child,td:first-child{text-align:left}
th{background:#edf2f7;color:#243b53}
figure{margin:24px 0 34px}
img{max-width:100%;height:auto;border:1px solid #d8dee9;border-radius:6px;background:white}
figcaption{font-size:14px;line-height:1.75;color:#334e68;margin-top:9px}
code{background:#edf2f7;border-radius:4px;padding:2px 5px}
.toc a{display:block;color:#2454a6;text-decoration:none;margin:5px 0}
"""
    event_table = html_table(event_rows, [
        ("event", "Event"),
        ("mike_peak_m", "MIKE peak m"),
        ("surface_peak_m", "Surface peak m"),
        ("official_peak_m", "Official SWMM peak m"),
        ("sink_peak_m", "Sink peak m"),
        ("official_mae_m", "Official MAE vs MIKE m"),
        ("drained_m3", "SWMM drained m3"),
        ("local_surcharge_m3", "Local surcharge m3"),
    ], digits=4)
    inventory_rows = [
        {"item": "study_window", "value": "200 x 280 cells", "note": "20 m grid, 4.0 km x 5.6 km"},
        {"item": "time_steps", "value": "72", "note": "5 min interval, 6 h duration"},
        {"item": "formal_events", "value": "8", "note": ", ".join(EVENTS)},
        {"item": "formal_label", "value": "h_itzi_swmm_official.npy", "note": "formal copied ITZI-SWMM coupled output"},
        {"item": "conceptual_label", "value": "h_itzi_sink.npy", "note": "road-inlet sink comparison only"},
        {"item": "SWMM_sub_network", "value": "82 junctions, 1 outfall, 29 conduits", "note": "DYNWAVE routing"},
    ]
    inventory_table = html_table(inventory_rows, [
        ("item", "Item"),
        ("value", "Value"),
        ("note", "Note"),
    ], digits=3)
    single_table = html_table(single_rows, [
        ("model", "Model"),
        ("mae_mm", "MAE mm"),
        ("rmse_mm", "RMSE mm"),
        ("csi_003", "CSI 0.03 m"),
        ("csi_015", "CSI 0.15 m"),
        ("improve_surface_pct", "Improve vs surface %"),
    ], digits=3)
    loo_table = html_table(loo_rows, [
        ("model", "Model"),
        ("mae_mean_mm", "LOEO MAE mean mm"),
        ("mae_std_mm", "MAE std mm"),
        ("rmse_mean_mm", "RMSE mean mm"),
        ("csi_003", "CSI 0.03 m"),
        ("csi_015", "CSI 0.15 m"),
        ("improve_surface_pct", "Improve vs surface %"),
    ], digits=3)
    loo_event_table = html_table(loo_event_rows, [
        ("fold", "Held-out event"),
        ("model", "Model"),
        ("mae_mm", "MAE mm"),
        ("rmse_mm", "RMSE mm"),
        ("peak_error_mm", "Peak error mm"),
    ], digits=3)
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head><meta charset="utf-8"><title>正式 ITZI-SWMM 数据源下的 LarNO-DrainLite 研究报告</title><style>{styles}</style></head>
<body><main class="page">
<h1>正式 ITZI-SWMM 数据源下的 LarNO-DrainLite 城市排水管网增强研究报告</h1>
<p class="meta">生成路径：<code>{OUT}</code>。主数据源：复制后的正式 ITZI-SWMM 工程 <code>external_models/20260518-itzi-flood</code>。原始工程 <code>E:/Projects/20260518-itzi-flood</code> 未被修改。</p>

<section class="abstract">
<h2>摘要</h2>
<p>本报告把前期工作重新组织到正式 ITZI-SWMM 耦合数据源上。核心变化是：不再把早期 ITZI-PySWMM 原型或独立 SWMM 指标作为正式 reference，而是使用从用户完整耦合模型副本导入的 <code>h_itzi_swmm_official.npy</code>。由于正式 ITZI-SWMM 是双向耦合，水深变化既可能表现为排水削减，也可能表现为局部壅水或回灌。因此，本报告采用有符号残差学习，而不是旧的单调削峰学习。留一事件交叉验证显示，surface-only 平均绝对误差为 {nums['loo_surface_mae_mm']:.3f} 毫米，最佳轻量残差模型为 {nums['loo_base_mae_mm']:.3f} 毫米，误差下降 {nums['loo_base_improve_pct']:.1f}%。但 all_static 管网静态特征模型为 {nums['loo_all_static_mae_mm']:.3f} 毫米，未明显优于 base，说明当前正式耦合信号较弱，管网静态特征的边际贡献有限。</p>
</section>

<h2>目录</h2>
<nav class="toc">
<a href="#background">1. 研究背景与口径修正</a>
<a href="#data">2. 数据与方法</a>
<a href="#labels">3. 正式标签统计</a>
<a href="#model">4. 轻量模型结果</a>
<a href="#discussion">5. 分析与讨论</a>
<a href="#conclusion">6. 结论与后续工作</a>
</nav>

<h2 id="background">1. 研究背景与口径修正</h2>
<p>原 LarNO 论文的主线是使用 MIKE Plus 参考结果训练神经算子，从降雨、地形等输入预测城市洪涝水深。本研究的创新问题是：如果要把城市排水管网引入这个预报链条，应该如何构造数据、标签和轻量模型？前期曾经使用概念性道路入口 sink 结果训练非负削峰模型，这条线能够说明道路对齐排水先验有作用，但它不是正式 ITZI-SWMM 耦合。</p>
<p>现在主线已经切换为正式数据源：使用你原完整耦合模型的副本，生成并导入八个事件的正式 ITZI-SWMM 72 帧水深标签。旧 sink 结果保留为对照，用于解释“概念性入口模型”和“正式双向耦合模型”的区别。</p>
{figure_block(1, "正式研究流程与数据口径", figures['workflow'], "这张图是本报告最重要的口径图。左侧展示三个来源：LarNO/MIKE benchmark、正式复制的 ITZI-SWMM 模型、以及概念性道路入口 sink。中间说明正式标签文件和概念标签文件分开保存。右侧说明新模型采用有符号残差学习，旧 sink 模型只作为对照。读图时要特别注意：正式 ITZI-SWMM 标签是主线，sink 标签不再被称为正式管网 reference。")}

<h2 id="data">2. 数据与方法</h2>
<p>研究区域为深圳 Region1 的 20 米分辨率子窗口，尺寸为 200 × 280，代表约 4.0 km × 5.6 km。每个事件包含 72 个五分钟时刻，总时长 6 小时。正式事件包括 event1、event20、event65 至 event70。新增正式标签包括 <code>h_itzi_swmm_official.npy</code>、<code>h_itzi_swmm_official_surface.npy</code> 和 <code>h_itzi_swmm_official_reduction.npy</code>。</p>
<h3>表 0 数据集与模型输入清单</h3>
{inventory_table}
{figure_block(2, "DEM、管网和排放口距离", figures['static'], "左图是地形和建筑抬高后的数字高程模型，用于说明水流为什么会在低洼和边界区域集中。中图把管线、雨水口和排放口叠加到地形上，用于检查管网是否沿道路和低洼通道合理布置。右图是到排放口距离，它不是水动力方程本身，而是一个静态空间先验：距离越近，理论上越可能较快受排放条件影响。")}

<h3>模型定义</h3>
<p>正式轻量模型命名为 <code>DrainLite-OfficialResidual</code>。目标变量为 <code>1000 × (h_itzi_swmm_official - h_itzi_surface)</code>，单位为毫米。这个残差是有符号的：负值表示正式耦合后水深降低，正值表示管网局部回灌、壅水或耦合改变汇流造成的水深升高。预测水深为 <code>max(h_surface + residual / 1000, 0)</code>。</p>

<h2 id="labels">3. 正式标签统计</h2>
<p>表 1 汇总八个正式事件。可以看到，正式 ITZI-SWMM 峰值通常只比 surface-only 略低，平均 SWMM 节点排水量约 {nums['mean_drained_m3']:,.0f} 立方米，但最终水深图上的正向削减体积约 {nums['mean_reduction_m3']:,.0f} 立方米，同时还存在约 {nums['mean_surcharge_m3']:,.0f} 立方米的局部回灌或水位抬升。这说明管网交换不能简单等同于全域扣水。</p>
<h3>表 1 正式事件与标签指标</h3>
{event_table}
{figure_block(3, "八事件峰值水深空间对比", figures['peak_grid'], "每一行是一个降雨事件，每一列是一类结果。MIKE 是原 benchmark 外部参考，Surface 是正式脚本中的地表-only 结果，Official 是正式 ITZI-SWMM 耦合结果，Sink 是概念性入口排水对照，最后一列显示 Official 与 Surface 的差值。读图时应先看积水区域是否大体一致，再看 Official-Surface 的红蓝分布。蓝色表示正式耦合后降低，红色表示局部升高。")}
{figure_block(4, "正式耦合排水量、削减量和局部回灌量", figures['volume'], "这张柱状图解释了为什么 SWMM drained 不等于最终净削减。蓝色是 SWMM 节点累计接收的水量，橙色是最终图上 surface-only 高于 official 的体积，绿色是 official 高于 surface-only 的局部回灌体积。由于二维地表水会继续汇流，管网也可能改变局部水位，三者不会一一相等。")}
{figure_block(5, "代表事件水量与峰值时间过程", figures['temporal'], "左列展示随时间变化的总水量，右列展示随时间变化的最大水深。MIKE、surface、official 和 sink 四条曲线的间距说明不同 reference 的物理含义。正式 official 曲线通常贴近 surface，表示当前正式耦合改变量较小；sink 曲线偏低，表示概念性入口模型削减更强。")}
{figure_block(6, "正式 ITZI-SWMM 有符号残差分布", figures['residual'], "左上直方图显示绝大多数残差接近 0，说明正式耦合对大部分像元影响很小。右上展示最终时刻削减与回灌体积，左下展示峰值水深变化，右下给出 event70 的最终残差空间图。蓝色位置表示排水降低水深，红色位置表示局部抬升。这个图支撑了本研究从非负削峰改为有符号残差的必要性。")}

<h2 id="model">4. 轻量模型结果</h2>
<p>固定划分实验使用 event1、event20、event65、event66、event67 训练，event68、event69、event70 测试。留一事件交叉验证进一步把八个事件逐一留出测试。两个实验共同说明：轻量残差模型能学习正式耦合相对 surface-only 的微小修正，但管网静态特征相对 base 特征没有稳定显著增益。</p>
<h3>表 2 固定划分模型结果</h3>
{single_table}
<h3>表 3 留一事件交叉验证结果</h3>
{loo_table}
<h3>表 4 逐事件留一交叉验证明细</h3>
{loo_event_table}
{figure_block(7, "正式标签轻量模型消融", figures['ablation'], "左图是固定测试集结果，右图是留一事件交叉验证结果。Surface-only 是不做任何残差修正的基线。Base 只使用水深、降雨、累计降雨、地形、坡度、行列坐标和时间等基础特征。All_static 加入全部静态管网特征。图中可以看出 base 和 all_static 都优于 surface-only，但二者非常接近，说明当前正式标签中可学习的主要是局部状态相关残差。")}
{figure_block(8, "逐事件留一验证误差", figures['eventwise'], "每个事件作为一次完全未见测试事件。蓝色 surface-only 普遍更高，base 和 all_static 普遍降低误差。event67 到 event70 的误差相对较高，说明强事件和较大空间积水范围对轻量模型更难。")}
{figure_block(9, "测试事件正式残差模型空间预测", figures['prediction'], "每一行对应一个测试事件。MIKE、Surface、Official 和 Predicted 展示峰值水深空间图；Error 展示模型相对正式 official 标签的误差；Residual 展示模型预测的有符号修正。读图时应重点看 Error 是否集中在局部峰值区域，以及 Residual 是否与低洼汇流和管网影响区域相符。")}

<h2 id="discussion">5. 分析与讨论</h2>
<p>本次正式重做后的结论比旧 sink 版本更克制，也更适合论文写作。旧 sink 版本显示管网先验能带来显著削峰，但那是概念性入口模型；正式 ITZI-SWMM 标签中，节点排水存在，但对最终二维水深的净影响较小，并且伴随局部回灌。因此，论文不能写成“加入管网后所有位置都削峰明显”，而应写成“正式双向耦合下的管网效应可以通过有符号残差学习注入预报结果”。</p>
<p>为什么 all_static 没有明显优于 base？第一，正式事件只有八个，样本的事件多样性不足。第二，正式 coupled 与 surface 的差异本身在毫米级，很多像元几乎没有管网信号。第三，静态管网特征缺少 SWMM 的动态状态，例如节点水头、管段流量、满管程度、壅水持续时间和排放口边界过程。因此，管网特征不是无意义，而是当前输入还不够表达双向耦合的动态机理。</p>
<section class="warn"><p><b>论文写作边界。</b>正式论文应避免三种表述：一是把旧 ITZI-PySWMM 原型写成正式 reference；二是把概念性 sink 结果写成完整耦合结果；三是把正式 ITZI-SWMM 简化成单调削峰过程。建议把创新点写为：正式 ITZI-SWMM 标签驱动的有符号排水残差学习，在低算力条件下为 LarNO 类模型提供管网效应注入路径。</p></section>

<h2 id="conclusion">6. 结论与后续工作</h2>
<ul>
<li>正式新数据源已经建立：八个事件、72 帧、20 米分辨率、正式 ITZI-SWMM 耦合标签。</li>
<li>正式标签下应使用有符号残差模型，而不是旧的非负削峰模型。</li>
<li>轻量模型相对 surface-only 稳定改善，留一验证 MAE 从 {nums['loo_surface_mae_mm']:.3f} 毫米降至 {nums['loo_base_mae_mm']:.3f} 毫米。</li>
<li>静态管网特征在当前正式标签上边际贡献有限，这是重要结果，不应回避。</li>
<li>下一步若要增强论文创新性，应优先补充更多正式耦合事件，或把 SWMM 动态水头、流量、满管率、节点壅水等时变变量栅格化后加入模型。</li>
</ul>
</main></body></html>"""


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    event_rows, single_rows, loo_rows, loo_event_rows = metric_tables()
    write_tables(event_rows, single_rows, loo_rows, loo_event_rows)
    figures = make_figures(event_rows, single_rows, loo_rows, loo_event_rows)
    nums = key_numbers(event_rows, single_rows, loo_rows)
    paper_md = paper_markdown(figures, event_rows, single_rows, loo_rows, nums)
    (OUT / "official_drainlite_paper.md").write_text(
        paper_md,
        encoding="utf-8",
    )
    paper_css = """
body{margin:0;background:#f8fafc;color:#1f2933;font-family:Georgia,"Times New Roman","Microsoft YaHei",serif}
.paper{max-width:980px;margin:0 auto;background:#fff;padding:46px 52px 80px}
h1{font-size:30px;line-height:1.25;margin:0 0 18px;color:#111827}
h2{font-size:22px;margin-top:34px;border-bottom:1px solid #d9e2ec;padding-bottom:6px;color:#102a43}
h3{font-size:18px;margin-top:24px;color:#243b53}
p,li{font-size:15px;line-height:1.75}
ul{padding-left:22px}
figure{margin:22px 0 32px}
img{max-width:100%;height:auto;border:1px solid #d8dee9;border-radius:4px}
figcaption{font-size:13px;color:#52616b;line-height:1.6;margin-top:8px}
code{background:#edf2f7;border-radius:4px;padding:2px 5px;font-family:Consolas,monospace}
"""
    paper_html = (
        "<!DOCTYPE html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<title>DrainLite-OfficialResidual manuscript</title>"
        f"<style>{paper_css}</style></head><body><main class=\"paper\">"
        + simple_markdown_to_html(paper_md, OUT)
        + "</main></body></html>"
    )
    (OUT / "official_drainlite_paper.html").write_text(paper_html, encoding="utf-8")
    (OUT / "official_drainlite_report.html").write_text(
        report_html(figures, event_rows, single_rows, loo_rows, loo_event_rows, nums),
        encoding="utf-8",
    )
    (OUT / "official_drainlite_report.md").write_text(
        "# 正式 ITZI-SWMM 数据源下的 LarNO-DrainLite 研究报告\n\n"
        f"完整自包含 HTML 报告见：`{OUT / 'official_drainlite_report.html'}`\n\n"
        "论文草稿见：`official_drainlite_paper.md`\n",
        encoding="utf-8",
    )
    summary = {"key_numbers": nums, "figures": {k: str(v) for k, v in figures.items()}}
    (OUT / "official_paper_package_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote official paper package to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
