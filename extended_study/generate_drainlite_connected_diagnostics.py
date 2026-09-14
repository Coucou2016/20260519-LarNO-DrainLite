#!/usr/bin/env python3
"""Generate diagnostic figures for DrainLite connected residual validation."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from publication_plot_style import MODEL_COLORS, add_panel_labels, configure_publication_style, model_label


configure_publication_style()


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "LarNO-main" / "benchmark" / "urbanflood" / "flood" / "region1_20m_connected_swmm_v1"
GEO = ROOT / "LarNO-main" / "benchmark" / "urbanflood" / "geodata" / "region1_20m_connected_swmm_v1"
MAIN_OUT = ROOT / "extended_study" / "output" / "drainlite_connected_residual"
CV_OUT = ROOT / "extended_study" / "output" / "drainlite_connected_cross_validation"
FIG = MAIN_OUT / "figures"
CELL_AREA_M2 = 20.0 * 20.0
EVENTS = ["event1", "event20", "event65", "event66", "event67", "event68", "event69", "event70"]
MODELS = ["surface_only", "base", "all_static", "swmm_assisted"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", default="region1_20m_connected_swmm_v2_inf1mmh")
    parser.add_argument("--main-output-name", default="drainlite_connected_residual_v2_inf1mmh")
    parser.add_argument("--cv-output-name", default="drainlite_connected_cross_validation_v2_inf1mmh")
    return parser.parse_args()


def configure_paths(dataset_name: str, main_output_name: str, cv_output_name: str) -> None:
    global DATA, GEO, MAIN_OUT, CV_OUT, FIG
    DATA = ROOT / "LarNO-main" / "benchmark" / "urbanflood" / "flood" / dataset_name
    GEO = ROOT / "LarNO-main" / "benchmark" / "urbanflood" / "geodata" / dataset_name
    MAIN_OUT = ROOT / "extended_study" / "output" / main_output_name
    CV_OUT = ROOT / "extended_study" / "output" / cv_output_name
    FIG = MAIN_OUT / "figures"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def as_float(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, "nan"))
    except Exception:
        return float("nan")


def load_cv_pred(event: str, model: str) -> np.ndarray:
    if model == "surface_only":
        return np.load(DATA / event / "h_itzi_surface.npy").astype(np.float32)
    return np.load(CV_OUT / "predictions" / event / f"h_drainlite_connected_cv_{model}.npy").astype(np.float32)


def plot_cv_summary() -> None:
    rows = read_csv(CV_OUT / "metrics" / "connected_residual_cv_summary.csv")
    rows = [r for r in rows if r["model"] in MODELS]
    rows.sort(key=lambda r: MODELS.index(r["model"]))
    labels = [r["model"] for r in rows]
    mae = [as_float(r, "mae_mm") for r in rows]
    rmse = [as_float(r, "rmse_mm") for r in rows]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7.1, 3.4))
    ax.bar(x - 0.18, mae, width=0.36, label="MAE", color="#0072B2")
    ax.bar(x + 0.18, rmse, width=0.36, label="RMSE", color="#E69F00")
    ax.set_xticks(x)
    ax.set_xticklabels([model_label(label) for label in labels], rotation=12, ha="right")
    ax.set_ylabel("Error (mm)")
    ax.set_title("Leave-one-event-out validation")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "cv_summary_mae_rmse.png")
    plt.close(fig)


def plot_cv_event_mae() -> None:
    rows = read_csv(CV_OUT / "metrics" / "connected_residual_cv_event_metrics.csv")
    by: dict[tuple[str, str], float] = {}
    for row in rows:
        if row["model"] in MODELS:
            by[(row["event"], row["model"])] = as_float(row, "mae_m") * 1000.0
    x = np.arange(len(EVENTS))
    width = 0.2
    fig, ax = plt.subplots(figsize=(7.1, 3.6))
    colors = MODEL_COLORS
    for i, model in enumerate(MODELS):
        values = [by.get((event, model), np.nan) for event in EVENTS]
        ax.bar(x + (i - 1.5) * width, values, width=width, label=model_label(model), color=colors[model])
    ax.set_xticks(x)
    ax.set_xticklabels(EVENTS, rotation=20)
    ax.set_ylabel("MAE to ITZI-SWMM label (mm)")
    ax.set_title("Leave-one-event-out validation: event-wise MAE")
    ax.legend(ncol=2)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "cv_event_mae_by_model.png")
    plt.close(fig)


def plot_cv_volume_reduction() -> None:
    target = []
    pred = []
    for event in EVENTS:
        surface = np.load(DATA / event / "h_itzi_surface.npy").astype(np.float32)
        connected = np.load(DATA / event / "h_itzi_swmm_connected.npy").astype(np.float32)
        all_static = load_cv_pred(event, "all_static")
        active = np.load(GEO / "dem.npy").astype(np.float32) < 49.9
        target.append(float(np.sum(surface[-1, active] - connected[-1, active]) * CELL_AREA_M2) / 1000.0)
        pred.append(float(np.sum(surface[-1, active] - all_static[-1, active]) * CELL_AREA_M2) / 1000.0)
    x = np.arange(len(EVENTS))
    fig, ax = plt.subplots(figsize=(7.1, 3.6))
    ax.bar(x - 0.18, target, width=0.36, label="ITZI-SWMM label", color="#102a43")
    ax.bar(x + 0.18, pred, width=0.36, label="DrainLite all_static CV", color="#2f80ed")
    ax.set_xticks(x)
    ax.set_xticklabels(EVENTS, rotation=20)
    ax.set_ylabel(r"Net surface-volume change ($10^3$ m$^3$)")
    ax.set_title("Final drainage-induced net surface-volume change")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "cv_volume_reduction_capture.png")
    plt.close(fig)


def plot_event_timeseries(event: str) -> None:
    mike = np.load(DATA / event / "h_mike_ref.npy").astype(np.float32)
    surface = np.load(DATA / event / "h_itzi_surface.npy").astype(np.float32)
    connected = np.load(DATA / event / "h_itzi_swmm_connected.npy").astype(np.float32)
    pred = load_cv_pred(event, "all_static")
    active = np.load(GEO / "dem.npy").astype(np.float32) < 49.9
    t = np.arange(1, surface.shape[0] + 1) * 5.0 / 60.0
    series = {
        "MIKE reference": mike,
        "ITZI surface-only": surface,
        "ITZI-SWMM connected": connected,
        "DrainLite all_static CV": pred,
    }
    colors = MODEL_COLORS
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.55), sharex=True)
    for label, arr in series.items():
        values = arr[:, active]
        axes[0].plot(t, values.max(axis=1), label=label, color=colors[label], linewidth=1.8)
        axes[1].plot(t, values.sum(axis=1) * CELL_AREA_M2 / 1000.0, label=label, color=colors[label], linewidth=1.8)
        axes[2].plot(t, (values >= 0.03).sum(axis=1) * CELL_AREA_M2 / 1_000_000.0, label=label, color=colors[label], linewidth=1.8)
    axes[0].set_ylabel("Max depth (m)")
    axes[1].set_ylabel(r"Water volume ($10^3$ m$^3$)")
    axes[2].set_ylabel(r"Flooded area $>0.03$ m (km$^2$)")
    for ax in axes:
        ax.set_xlabel("Time (h)")
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=6.8, loc="best")
    add_panel_labels(axes, x=-0.17, y=1.03)
    fig.suptitle(f"{event}: hydrodynamic response")
    fig.tight_layout()
    fig.savefig(FIG / f"{event}_cv_timeseries.png")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    configure_paths(args.dataset_name, args.main_output_name, args.cv_output_name)
    FIG.mkdir(parents=True, exist_ok=True)
    plot_cv_summary()
    plot_cv_event_mae()
    plot_cv_volume_reduction()
    for event in EVENTS:
        plot_event_timeseries(event)
    print(FIG)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
