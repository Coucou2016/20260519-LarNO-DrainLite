#!/usr/bin/env python3
"""Generate enhanced result figures and tables for the DrainLite manuscript.

The script only reads existing DrainLite predictions and benchmark arrays. It
does not retrain models or modify source labels.
"""

from __future__ import annotations

import importlib.util
import json
import math
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import psutil
except Exception:  # pragma: no cover - optional runtime dependency
    psutil = None

try:
    from scipy.ndimage import distance_transform_edt
except Exception:  # pragma: no cover - optional runtime dependency
    distance_transform_edt = None


ROOT = Path(__file__).resolve().parents[1]
DL = ROOT / "extended_study" / "output" / "drainlite_residual"
PRED = DL / "predictions"
METRICS = DL / "metrics"
MODELS = DL / "models"
FLOOD = ROOT / "LarNO-main" / "benchmark" / "urbanflood" / "flood" / "region1_20m_drainage_v1"
GEO = ROOT / "LarNO-main" / "benchmark" / "urbanflood" / "geodata" / "region1_20m_drainage_v1"
PAPER = ROOT / "paper_draft" / "drainlite"
FIG = PAPER / "figures"
TAB = PAPER / "tables"
CELL_SIZE_M = 20.0
CELL_AREA_M2 = CELL_SIZE_M * CELL_SIZE_M
TEST_EVENTS = ["event75", "event76", "event77", "event78", "event80"]
T_MIN = np.arange(72) * 5


def ensure_dirs() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    METRICS.mkdir(parents=True, exist_ok=True)


def csi(pred: np.ndarray, target: np.ndarray, threshold: float) -> float:
    pred_wet = pred >= threshold
    target_wet = target >= threshold
    tp = np.logical_and(pred_wet, target_wet).sum()
    fp = np.logical_and(pred_wet, ~target_wet).sum()
    fn = np.logical_and(~pred_wet, target_wet).sum()
    denom = tp + fp + fn
    return float(tp / denom) if denom else 1.0


def load_event(event: str) -> dict[str, np.ndarray]:
    event_dir = FLOOD / event
    pred_dir = PRED / event
    return {
        "rainfall": np.load(event_dir / "rainfall.npy").astype(np.float32),
        "mike": np.load(event_dir / "h_mike_ref.npy").astype(np.float32),
        "surface": np.load(event_dir / "h_itzi_surface.npy").astype(np.float32),
        "sink": np.load(event_dir / "h_itzi_sink.npy").astype(np.float32),
        "pred": np.load(pred_dir / "h_drainlite_all_static.npy").astype(np.float32),
        "reduction_pred_m": (np.load(pred_dir / "reduction_pred_all_static_mm.npy").astype(np.float32) / 1000.0),
    }


def peak_map(arr: np.ndarray) -> np.ndarray:
    return np.nanmax(arr, axis=0)


def make_time_series_results() -> pd.DataFrame:
    rows: list[dict[str, float | str | int]] = []
    for event in TEST_EVENTS:
        a = load_event(event)
        true_red = np.maximum(a["surface"] - a["sink"], 0.0)
        pred_red = a["reduction_pred_m"]
        for t in range(a["sink"].shape[0]):
            surface_t = a["surface"][t]
            sink_t = a["sink"][t]
            pred_t = a["pred"][t]
            true_red_t = true_red[t]
            pred_red_t = pred_red[t]
            rows.append(
                {
                    "event": event,
                    "time_min": int(t * 5),
                    "mae_surface_mm": float(np.mean(np.abs(surface_t - sink_t)) * 1000.0),
                    "mae_drainlite_mm": float(np.mean(np.abs(pred_t - sink_t)) * 1000.0),
                    "csi03_surface": csi(surface_t, sink_t, 0.03),
                    "csi03_drainlite": csi(pred_t, sink_t, 0.03),
                    "csi15_surface": csi(surface_t, sink_t, 0.15),
                    "csi15_drainlite": csi(pred_t, sink_t, 0.15),
                    "reduction_mae_surface_mm": float(np.mean(np.abs(true_red_t)) * 1000.0),
                    "reduction_mae_drainlite_mm": float(np.mean(np.abs(pred_red_t - true_red_t)) * 1000.0),
                    "volume_error_surface_m3": float((surface_t - sink_t).sum() * CELL_AREA_M2),
                    "volume_error_drainlite_m3": float((pred_t - sink_t).sum() * CELL_AREA_M2),
                    "true_reduction_volume_m3": float(true_red_t.sum() * CELL_AREA_M2),
                    "pred_reduction_volume_m3": float(pred_red_t.sum() * CELL_AREA_M2),
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(TAB / "table5_time_series_metrics.csv", index=False)
    df.to_csv(METRICS / "enhanced_time_series_metrics.csv", index=False)
    return df


def plot_mean_std(ax, df: pd.DataFrame, x: str, y: str, label: str, color: str) -> None:
    g = df.groupby(x)[y].agg(["mean", "std"]).reset_index()
    xx = g[x].to_numpy(dtype=float)
    mean = g["mean"].to_numpy(dtype=float)
    std = g["std"].fillna(0).to_numpy(dtype=float)
    ax.plot(xx, mean, lw=1.8, label=label, color=color)
    ax.fill_between(xx, mean - std, mean + std, color=color, alpha=0.18, linewidth=0)


def make_time_series_figure(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.2), sharex=True)
    axes = axes.ravel()
    plot_mean_std(axes[0], df, "time_min", "mae_surface_mm", "ITZI surface-only", "#8c564b")
    plot_mean_std(axes[0], df, "time_min", "mae_drainlite_mm", "DrainLite", "#1f77b4")
    axes[0].set_title("MAE to ITZI + sink")
    axes[0].set_ylabel("MAE (mm)")

    plot_mean_std(axes[1], df, "time_min", "csi03_surface", "ITZI surface-only", "#8c564b")
    plot_mean_std(axes[1], df, "time_min", "csi03_drainlite", "DrainLite", "#1f77b4")
    axes[1].set_title("CSI at 0.03 m")
    axes[1].set_ylabel("CSI")
    axes[1].set_ylim(0, 1.02)

    plot_mean_std(axes[2], df, "time_min", "reduction_mae_surface_mm", "No correction", "#8c564b")
    plot_mean_std(axes[2], df, "time_min", "reduction_mae_drainlite_mm", "DrainLite", "#1f77b4")
    axes[2].set_title("Reduction-signal MAE")
    axes[2].set_ylabel("MAE (mm)")

    plot_mean_std(axes[3], df, "time_min", "volume_error_surface_m3", "ITZI surface-only", "#8c564b")
    plot_mean_std(axes[3], df, "time_min", "volume_error_drainlite_m3", "DrainLite", "#1f77b4")
    axes[3].axhline(0, color="black", lw=0.8, alpha=0.5)
    axes[3].set_title("Signed water-volume error")
    axes[3].set_ylabel("m3")

    for ax in axes:
        ax.grid(alpha=0.25)
        ax.set_xlabel("Forecast lead time (min)")
        ax.legend(frameon=False, fontsize=9)
    fig.suptitle("Six-hour spatiotemporal performance on five held-out events", fontsize=14, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(FIG / "fig08_spatiotemporal_curves.png", dpi=220)
    plt.close(fig)


def make_spatial_composite() -> None:
    data = {event: load_event(event) for event in TEST_EVENTS}
    depth_values = []
    red_values = []
    err_values = []
    for event, a in data.items():
        for name in ["mike", "surface", "sink", "pred"]:
            depth_values.append(peak_map(a[name]))
        red_values.append(peak_map(a["reduction_pred_m"]))
        err_values.append(peak_map(a["pred"]) - peak_map(a["sink"]))
    depth_vmax = float(np.nanpercentile(np.concatenate([v.ravel() for v in depth_values]), 99.5))
    red_vmax = float(np.nanpercentile(np.concatenate([v.ravel() for v in red_values]), 99.5))
    err_abs = float(np.nanpercentile(np.abs(np.concatenate([v.ravel() for v in err_values])), 99.0))
    err_abs = max(err_abs, 0.02)

    columns = [
        ("MIKE ref", "mike", "depth"),
        ("ITZI surface", "surface", "depth"),
        ("ITZI + sink", "sink", "depth"),
        ("DrainLite", "pred", "depth"),
        ("DrainLite - sink", "error", "error"),
        ("Pred. reduction", "reduction", "reduction"),
    ]
    fig, axes = plt.subplots(len(TEST_EVENTS), len(columns), figsize=(15.5, 12.5))
    for i, event in enumerate(TEST_EVENTS):
        a = data[event]
        maps = {
            "mike": peak_map(a["mike"]),
            "surface": peak_map(a["surface"]),
            "sink": peak_map(a["sink"]),
            "pred": peak_map(a["pred"]),
            "error": peak_map(a["pred"]) - peak_map(a["sink"]),
            "reduction": peak_map(a["reduction_pred_m"]),
        }
        for j, (title, key, kind) in enumerate(columns):
            ax = axes[i, j]
            if kind == "error":
                im = ax.imshow(maps[key], cmap="RdBu_r", vmin=-err_abs, vmax=err_abs)
            elif kind == "reduction":
                im = ax.imshow(maps[key], cmap="viridis", vmin=0, vmax=red_vmax)
            else:
                im = ax.imshow(maps[key], cmap="Blues", vmin=0, vmax=depth_vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(title, fontsize=10)
            if j == 0:
                ax.set_ylabel(event, fontsize=10, rotation=0, labelpad=28, va="center")
            if i == len(TEST_EVENTS) - 1:
                cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
                cb.ax.tick_params(labelsize=7)
    fig.suptitle("Peak-depth spatial comparison across all held-out events", fontsize=14, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    fig.savefig(FIG / "fig09_all_events_spatial_comparison.png", dpi=220)
    plt.close(fig)


def make_peak_error_analysis() -> pd.DataFrame:
    rows = []
    for event in TEST_EVENTS:
        a = load_event(event)
        sink_peak = peak_map(a["sink"])
        pred_peak = peak_map(a["pred"])
        peak_error_m = float(pred_peak.max() - sink_peak.max())
        sink_idx = np.unravel_index(int(np.nanargmax(sink_peak)), sink_peak.shape)
        pred_idx = np.unravel_index(int(np.nanargmax(pred_peak)), pred_peak.shape)
        loc_shift_m = float(math.hypot(pred_idx[0] - sink_idx[0], pred_idx[1] - sink_idx[1]) * CELL_SIZE_M)
        wet_vals = sink_peak[sink_peak > 0]
        top_thr = float(np.nanpercentile(wet_vals, 99.0)) if wet_vals.size else np.inf
        masks = {
            "top1": sink_peak >= top_thr,
            "gt_0p15": sink_peak >= 0.15,
            "gt_0p30": sink_peak >= 0.30,
        }
        def masked_mae(mask: np.ndarray) -> float:
            if not np.any(mask):
                return float("nan")
            return float(np.mean(np.abs(pred_peak[mask] - sink_peak[mask])) * 1000.0)

        rows.append(
            {
                "event": event,
                "sink_peak_m": float(sink_peak.max()),
                "drainlite_peak_m": float(pred_peak.max()),
                "peak_error_mm": peak_error_m * 1000.0,
                "peak_location_shift_m": loc_shift_m,
                "top1pct_deep_cell_mae_mm": masked_mae(masks["top1"]),
                "mae_gt_0p15_mm": masked_mae(masks["gt_0p15"]),
                "mae_gt_0p30_mm": masked_mae(masks["gt_0p30"]),
                "n_cells_gt_0p15": int(masks["gt_0p15"].sum()),
                "n_cells_gt_0p30": int(masks["gt_0p30"].sum()),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(TAB / "table6_peak_error_analysis.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.0))
    axes = axes.ravel()
    x = np.arange(len(df))
    axes[0].bar(x, df["peak_error_mm"], color="#cc6677")
    axes[0].axhline(0, color="black", lw=0.8)
    axes[0].set_title("Peak-depth error")
    axes[0].set_ylabel("mm")
    axes[1].bar(x, df["peak_location_shift_m"], color="#4477aa")
    axes[1].set_title("Peak-location shift")
    axes[1].set_ylabel("m")
    axes[2].bar(x, df["top1pct_deep_cell_mae_mm"], color="#228833")
    axes[2].set_title("Top 1% deep-cell MAE")
    axes[2].set_ylabel("mm")
    width = 0.38
    axes[3].bar(x - width / 2, df["mae_gt_0p15_mm"], width, label=">0.15 m", color="#66c2a5")
    axes[3].bar(x + width / 2, df["mae_gt_0p30_mm"], width, label=">0.30 m", color="#fc8d62")
    axes[3].set_title("High-depth-cell MAE")
    axes[3].set_ylabel("mm")
    axes[3].legend(frameon=False)
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(df["event"], rotation=35, ha="right")
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Peak-depth and high-risk-cell error analysis", fontsize=14, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(FIG / "fig10_peak_error_analysis.png", dpi=220)
    plt.close(fig)
    return df


def pipe_proximity_mask() -> np.ndarray:
    pipe = np.load(GEO / "pipe_mask.npy").astype(bool)
    if distance_transform_edt is None:
        return pipe
    dist_m = distance_transform_edt(~pipe) * CELL_SIZE_M
    return dist_m <= 60.0


def make_physical_consistency(ts: pd.DataFrame) -> pd.DataFrame:
    near_pipe = pipe_proximity_mask()
    rows = []
    ratio_rows = []
    for event in TEST_EVENTS:
        a = load_event(event)
        true_red = np.maximum(a["surface"] - a["sink"], 0.0)
        pred_red = a["reduction_pred_m"]
        pred = a["pred"]
        surface = a["surface"]
        violation_high = int((pred > surface + 1e-6).sum())
        violation_low = int((pred < -1e-9).sum())
        true_vol = true_red.sum(axis=(1, 2)) * CELL_AREA_M2
        pred_vol = pred_red.sum(axis=(1, 2)) * CELL_AREA_M2
        min_valid_true_vol = max(float(np.nanmax(true_vol)) * 0.01, 1e-6)
        valid = true_vol > min_valid_true_vol
        ratio = np.full_like(true_vol, np.nan, dtype=np.float64)
        ratio[valid] = pred_vol[valid] / true_vol[valid] * 100.0
        for t, value in enumerate(ratio):
            ratio_rows.append({"event": event, "time_min": int(t * 5), "capture_ratio_pct": float(value)})
        near_true = float(true_red[:, near_pipe].sum() * CELL_AREA_M2)
        near_pred = float(pred_red[:, near_pipe].sum() * CELL_AREA_M2)
        total_true = float(true_red.sum() * CELL_AREA_M2)
        total_pred = float(pred_red.sum() * CELL_AREA_M2)
        rows.append(
            {
                "event": event,
                "mean_true_reduction_volume_m3": float(np.mean(true_vol)),
                "mean_pred_reduction_volume_m3": float(np.mean(pred_vol)),
                "final_true_reduction_volume_m3": float(true_vol[-1]),
                "final_pred_reduction_volume_m3": float(pred_vol[-1]),
                "mean_capture_ratio_pct": float(np.nanmean(ratio)),
                "volume_weighted_capture_ratio_pct": float(pred_vol.sum() / true_vol.sum() * 100.0) if true_vol.sum() else np.nan,
                "monotonic_high_violation_cells": violation_high,
                "negative_depth_violation_cells": violation_low,
                "true_reduction_near_pipe_share_pct": near_true / total_true * 100.0 if total_true else np.nan,
                "pred_reduction_near_pipe_share_pct": near_pred / total_pred * 100.0 if total_pred else np.nan,
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(TAB / "table7_drainage_physical_consistency.csv", index=False)
    pd.DataFrame(ratio_rows).to_csv(METRICS / "enhanced_capture_ratio_timeseries.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.2))
    axes = axes.ravel()
    plot_mean_std(axes[0], ts, "time_min", "true_reduction_volume_m3", "ITZI + sink target", "#228833")
    plot_mean_std(axes[0], ts, "time_min", "pred_reduction_volume_m3", "DrainLite", "#1f77b4")
    axes[0].set_title("Reduction volume over time")
    axes[0].set_ylabel("m3")

    ratio_df = pd.DataFrame(ratio_rows)
    plot_mean_std(axes[1], ratio_df, "time_min", "capture_ratio_pct", "Predicted / target", "#aa3377")
    axes[1].axhline(100, color="black", lw=0.8, ls="--")
    axes[1].set_title("Reduction capture ratio")
    axes[1].set_ylabel("%")

    x = np.arange(len(summary))
    width = 0.38
    axes[2].bar(x - width / 2, summary["final_true_reduction_volume_m3"], width, label="Target", color="#228833")
    axes[2].bar(x + width / 2, summary["final_pred_reduction_volume_m3"], width, label="DrainLite", color="#1f77b4")
    axes[2].set_title("Final-step reduction volume")
    axes[2].set_ylabel("m3")
    axes[2].legend(frameon=False)

    axes[3].bar(x - width / 2, summary["true_reduction_near_pipe_share_pct"], width, label="Target", color="#228833")
    axes[3].bar(x + width / 2, summary["pred_reduction_near_pipe_share_pct"], width, label="DrainLite", color="#1f77b4")
    axes[3].set_title("Time-integrated reduction near pipes")
    axes[3].set_ylabel("Share within 60 m of pipe (%)")
    axes[3].legend(frameon=False)
    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
        if ax in (axes[2], axes[3]):
            ax.set_xticks(x)
            ax.set_xticklabels(summary["event"], rotation=35, ha="right")
        else:
            ax.set_xlabel("Forecast lead time (min)")
    fig.suptitle("Drainage-reduction physical consistency", fontsize=14, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(FIG / "fig11_drainage_physical_consistency.png", dpi=220)
    plt.close(fig)
    return summary


def make_event_severity_analysis() -> pd.DataFrame:
    event_metrics = pd.read_csv(METRICS / "drainlite_event_metrics.csv")
    all_static = event_metrics[(event_metrics["target"] == "itzi_sink") & (event_metrics["model"] == "all_static")]
    swmm = pd.read_csv(FLOOD / "swmm_metrics.csv")
    rows = []
    for event in TEST_EVENTS:
        a = load_event(event)
        surface_peak = peak_map(a["surface"])
        row = {
            "event": event,
            "mean_cumulative_rainfall_mm": float(a["rainfall"].sum(axis=0).mean()),
            "max_5min_rainfall_mm": float(a["rainfall"].max()),
            "surface_peak_depth_m": float(surface_peak.max()),
            "surface_peak_volume_m3": float(surface_peak.sum() * CELL_AREA_M2),
        }
        metric = all_static[all_static["event"] == event].iloc[0].to_dict()
        row.update(
            {
                "drainlite_mae_mm": float(metric["mae_m"] * 1000.0),
                "drainlite_peak_error_mm": float(metric["peak_error_m"] * 1000.0),
                "drainlite_reduction_mae_mm": float(metric["reduction_mae_m"] * 1000.0),
            }
        )
        sw = swmm[swmm["event"] == event].iloc[0].to_dict()
        row.update(
            {
                "swmm_routing_inflow_m3": float(sw["routing_inflow_m3"]),
                "swmm_outfall_volume_m3": float(sw["outfall_volume_m3"]),
                "swmm_node_flooding_volume_m3": float(sw["node_flooding_volume_m3"]),
            }
        )
        rows.append(row)
    df = pd.DataFrame(rows)
    df["severity_group"] = pd.qcut(df["mean_cumulative_rainfall_mm"], q=2, labels=["lower", "higher"])
    df.to_csv(TAB / "table8_event_severity_analysis.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.0))
    axes = axes.ravel()
    scatter_specs = [
        ("mean_cumulative_rainfall_mm", "drainlite_mae_mm", "Cumulative rainfall (mm)", "MAE (mm)"),
        ("surface_peak_volume_m3", "drainlite_mae_mm", "Surface peak volume (m3)", "MAE (mm)"),
        ("swmm_routing_inflow_m3", "drainlite_reduction_mae_mm", "SWMM routing inflow (m3)", "Reduction MAE (mm)"),
        ("mean_cumulative_rainfall_mm", "drainlite_peak_error_mm", "Cumulative rainfall (mm)", "Peak error (mm)"),
    ]
    for ax, (xcol, ycol, xlabel, ylabel) in zip(axes, scatter_specs):
        ax.scatter(df[xcol], df[ycol], s=70, color="#4477aa")
        for _, r in df.iterrows():
            ax.annotate(r["event"].replace("event", "e"), (r[xcol], r[ycol]), xytext=(4, 4), textcoords="offset points", fontsize=8)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
    fig.suptitle("Event-severity stratification of DrainLite errors", fontsize=14, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(FIG / "fig12_event_severity_analysis.png", dpi=220)
    plt.close(fig)
    return df


def make_mike_difference_figure() -> None:
    data = {event: load_event(event) for event in TEST_EVENTS}
    errs = []
    for a in data.values():
        mike_peak = peak_map(a["mike"])
        errs.extend([
            peak_map(a["surface"]) - mike_peak,
            peak_map(a["sink"]) - mike_peak,
            peak_map(a["pred"]) - mike_peak,
        ])
    err_abs = max(float(np.nanpercentile(np.abs(np.concatenate([e.ravel() for e in errs])), 99.0)), 0.03)
    columns = [("ITZI surface - MIKE", "surface"), ("ITZI + sink - MIKE", "sink"), ("DrainLite - MIKE", "pred")]
    fig, axes = plt.subplots(len(TEST_EVENTS), len(columns), figsize=(10.5, 12.5))
    for i, event in enumerate(TEST_EVENTS):
        a = data[event]
        mike_peak = peak_map(a["mike"])
        maps = {
            "surface": peak_map(a["surface"]) - mike_peak,
            "sink": peak_map(a["sink"]) - mike_peak,
            "pred": peak_map(a["pred"]) - mike_peak,
        }
        for j, (title, key) in enumerate(columns):
            ax = axes[i, j]
            im = ax.imshow(maps[key], cmap="RdBu_r", vmin=-err_abs, vmax=err_abs)
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(title, fontsize=10)
            if j == 0:
                ax.set_ylabel(event, fontsize=10, rotation=0, labelpad=28, va="center")
            if i == len(TEST_EVENTS) - 1:
                cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
                cb.ax.tick_params(labelsize=7)
    fig.suptitle("External MIKE-reference peak-depth error maps", fontsize=14, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    fig.savefig(FIG / "fig13_mike_difference_maps.png", dpi=220)
    plt.close(fig)


def make_swmm_correlation_analysis() -> pd.DataFrame:
    swmm = pd.read_csv(FLOOD / "swmm_metrics.csv")
    event_metrics = pd.read_csv(METRICS / "drainlite_event_metrics.csv")
    all_static = event_metrics[(event_metrics["target"] == "itzi_sink") & (event_metrics["model"] == "all_static")].copy()
    all_static["mae_mm"] = all_static["mae_m"] * 1000.0
    all_static["reduction_mae_mm"] = all_static["reduction_mae_m"] * 1000.0
    all_static["peak_error_mm"] = all_static["peak_error_m"] * 1000.0
    df = all_static.merge(swmm, on="event", how="left")
    xcols = [
        "routing_inflow_m3",
        "outfall_volume_m3",
        "routing_final_storage_m3",
        "routing_continuity_error_pct",
        "node_flooding_volume_m3",
    ]
    ycols = ["mae_mm", "reduction_mae_mm", "peak_error_mm"]
    rows = []
    for x in xcols:
        for y in ycols:
            rows.append({"swmm_metric": x, "drainlite_metric": y, "pearson_r": float(df[[x, y]].corr().iloc[0, 1])})
    corr = pd.DataFrame(rows)
    corr.to_csv(TAB / "table9_swmm_error_correlation.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.0))
    axes = axes.ravel()
    specs = [
        ("routing_inflow_m3", "mae_mm", "Routing inflow (m3)", "MAE (mm)"),
        ("node_flooding_volume_m3", "mae_mm", "Node flooding volume (m3)", "MAE (mm)"),
        ("outfall_volume_m3", "reduction_mae_mm", "Outfall volume (m3)", "Reduction MAE (mm)"),
        ("routing_continuity_error_pct", "mae_mm", "Routing continuity error (%)", "MAE (mm)"),
    ]
    for ax, (x, y, xlabel, ylabel) in zip(axes, specs):
        ax.scatter(df[x], df[y], s=70, color="#117733")
        r = df[[x, y]].corr().iloc[0, 1]
        ax.set_title(f"r = {r:.2f}")
        for _, row in df.iterrows():
            ax.annotate(row["event"].replace("event", "e"), (row[x], row[y]), xytext=(4, 4), textcoords="offset points", fontsize=8)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
    fig.suptitle("Event-level SWMM descriptors versus DrainLite errors", fontsize=14, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(FIG / "fig14_swmm_error_correlation.png", dpi=220)
    plt.close(fig)
    return corr


def benchmark_inference() -> pd.DataFrame:
    meta = json.loads((DL / "drainlite_run_metadata.json").read_text(encoding="utf-8"))
    rows = []
    for name in ["base", "mask_only", "hydraulic_only", "all_static", "swmm_assisted"]:
        path = MODELS / f"{name}.joblib"
        rows.append(
            {
                "item": f"{name}_model_file_size_mb",
                "value": f"{path.stat().st_size / (1024 * 1024):.3f}",
                "unit": "MB",
                "note": "Serialized joblib model size",
            }
        )
    rows.extend(
        [
            {"item": "training_rows", "value": str(len(meta["train_events"]) * 72 * meta["pixels_per_step"]), "unit": "rows", "note": "Derived from event count, time steps and sampled pixels per step"},
            {"item": "pixels_per_event_inference", "value": str(72 * 200 * 280), "unit": "cell-time rows", "note": "Dense prediction for one six-hour event"},
            {"item": "gpu_required", "value": "No", "unit": "-", "note": "HistGradientBoostingRegressor CPU inference"},
            {"item": "reference_larno_training_time", "value": "1.8", "unit": "days", "note": "Reported in the LarNO paper for 8 x NVIDIA A800 GPUs; not measured in this local DrainLite run"},
            {"item": "reference_larno_trt_inference_time", "value": "34", "unit": "s/event", "note": "Reported in the LarNO paper for TensorRT inference on NVIDIA RTX 4090"},
            {"item": "reference_mike_plus_runtime", "value": "31961", "unit": "s/event", "note": "Reported in the LarNO paper for the traditional MIKE+ reference model"},
            {"item": "reference_larno_speedup_vs_mike", "value": "940", "unit": "x", "note": "Reported in the LarNO paper; included only as context for computational scale"},
        ]
    )

    infer_seconds = np.nan
    rss_delta_mb = np.nan
    training_benchmark: dict[str, float | int | str] = {}
    try:
        spec = importlib.util.spec_from_file_location("train_drainlite_residual", ROOT / "extended_study" / "train_drainlite_residual.py")
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            payload = joblib.load(MODELS / "all_static.joblib")
            model = payload["model"] if isinstance(payload, dict) else payload
            static = module.load_static()
            swmm = module.read_swmm()
            arrays = module.load_event("event75")
            proc = psutil.Process() if psutil else None
            before = proc.memory_info().rss / (1024 * 1024) if proc else np.nan
            start = time.perf_counter()
            pred, red = module.predict_event(
                "event75",
                "all_static",
                model,
                module.FEATURE_GROUPS["all_static"],
                arrays,
                static,
                swmm,
                120000,
            )
            infer_seconds = time.perf_counter() - start
            after = proc.memory_info().rss / (1024 * 1024) if proc else np.nan
            rss_delta_mb = after - before if proc else np.nan
            assert pred.shape == (72, 200, 280)
            assert red.shape == (72, 200, 280)

            cache = METRICS / "training_runtime_benchmark.json"
            if cache.exists():
                training_benchmark = json.loads(cache.read_text(encoding="utf-8"))
            else:
                proc = psutil.Process() if psutil else None
                before_train = proc.memory_info().rss / (1024 * 1024) if proc else np.nan
                sample_start = time.perf_counter()
                x_all, y, weights = module.build_training_samples(
                    meta["train_events"],
                    static,
                    swmm,
                    int(meta["pixels_per_step"]),
                    int(meta["random_state"]),
                )
                sample_seconds = time.perf_counter() - sample_start
                col_index = {name: i for i, name in enumerate(module.SUPERSET_FEATURES)}
                features = module.FEATURE_GROUPS["all_static"]
                cols = [col_index[name] for name in features]
                fit_model = module.HistGradientBoostingRegressor(
                    loss="squared_error",
                    learning_rate=float(meta["learning_rate"]),
                    max_iter=int(meta["max_iter"]),
                    max_leaf_nodes=int(meta["max_leaf_nodes"]),
                    l2_regularization=float(meta["l2_regularization"]),
                    early_stopping=True,
                    validation_fraction=0.15,
                    random_state=int(meta["random_state"]),
                )
                fit_start = time.perf_counter()
                fit_model.fit(x_all[:, cols], y, sample_weight=weights)
                fit_seconds = time.perf_counter() - fit_start
                after_train = proc.memory_info().rss / (1024 * 1024) if proc else np.nan
                training_benchmark = {
                    "sample_seconds": sample_seconds,
                    "fit_seconds": fit_seconds,
                    "total_seconds": sample_seconds + fit_seconds,
                    "rss_delta_mb": after_train - before_train if proc else "not measured",
                    "rows": int(x_all.shape[0]),
                    "features": int(len(cols)),
                    "model": "all_static",
                    "note": "One-off benchmark fit only; existing saved model was not overwritten.",
                }
                cache.write_text(json.dumps(training_benchmark, indent=2), encoding="utf-8")
    except Exception as exc:  # keep paper generation robust
        rows.append({"item": "event75_inference_benchmark_error", "value": str(exc), "unit": "-", "note": "Benchmark failed"})

    rows.extend(
        [
            {"item": "all_static_event75_dense_inference_time", "value": f"{infer_seconds:.3f}" if np.isfinite(infer_seconds) else "not measured", "unit": "s", "note": "Measured by rerunning dense prediction for event75"},
            {"item": "all_static_event75_rss_delta", "value": f"{rss_delta_mb:.3f}" if np.isfinite(rss_delta_mb) else "not measured", "unit": "MB", "note": "Process RSS change during event75 dense inference; not a peak-RAM profiler"},
        ]
    )
    if training_benchmark:
        rows.extend(
            [
                {"item": "all_static_training_sample_build_time", "value": f"{float(training_benchmark['sample_seconds']):.3f}", "unit": "s", "note": "One-off benchmark; sampled training table construction"},
                {"item": "all_static_training_fit_time", "value": f"{float(training_benchmark['fit_seconds']):.3f}", "unit": "s", "note": "One-off benchmark; all_static fit only"},
                {"item": "all_static_training_total_time", "value": f"{float(training_benchmark['total_seconds']):.3f}", "unit": "s", "note": str(training_benchmark.get("note", ""))},
                {"item": "all_static_training_rss_delta", "value": f"{float(training_benchmark['rss_delta_mb']):.3f}" if isinstance(training_benchmark.get("rss_delta_mb"), (int, float)) else str(training_benchmark.get("rss_delta_mb", "not measured")), "unit": "MB", "note": "Process RSS change during one-off training benchmark; not a peak-RAM profiler"},
            ]
        )
    else:
        rows.append({"item": "all_static_training_total_time", "value": "not measured", "unit": "s", "note": "Training benchmark unavailable"})
    df = pd.DataFrame(rows)
    df.to_csv(TAB / "table10_lightweight_runtime.csv", index=False)
    return df


def main() -> int:
    ensure_dirs()
    ts = make_time_series_results()
    make_time_series_figure(ts)
    make_spatial_composite()
    make_peak_error_analysis()
    make_physical_consistency(ts)
    make_event_severity_analysis()
    make_mike_difference_figure()
    make_swmm_correlation_analysis()
    benchmark_inference()
    print(f"Wrote enhanced DrainLite result assets to {PAPER}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
