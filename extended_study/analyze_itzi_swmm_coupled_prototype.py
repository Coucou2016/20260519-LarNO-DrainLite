#!/usr/bin/env python3
"""Analyze ITZI + PySWMM coupling prototype outputs."""

from __future__ import annotations

import csv
import json
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "LarNO-main" / "benchmark" / "urbanflood" / "flood" / "region1_20m_drainage_v1"
OUT = ROOT / "extended_study" / "output" / "itzi_swmm_coupled_prototype"
CELL_AREA = 20.0 * 20.0


def csi(pred: np.ndarray, ref: np.ndarray, threshold: float) -> float:
    pred_wet = pred >= threshold
    ref_wet = ref >= threshold
    tp = float(np.logical_and(pred_wet, ref_wet).sum())
    fp = float(np.logical_and(pred_wet, ~ref_wet).sum())
    fn = float(np.logical_and(~pred_wet, ref_wet).sum())
    denom = tp + fp + fn
    return 1.0 if denom == 0 else tp / denom


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", nargs="*", default=["event75"])
    parser.add_argument("--all", action="store_true", help="Analyze every event folder with h_coupled_prototype.npy.")
    return parser.parse_args()


def metrics(pred: np.ndarray, ref: np.ndarray, name: str) -> dict[str, float | str]:
    diff = pred - ref
    return {
        "comparison": name,
        "frames": pred.shape[0],
        "mae_mm": float(np.mean(np.abs(diff)) * 1000.0),
        "rmse_mm": float(np.sqrt(np.mean(diff**2)) * 1000.0),
        "csi_0p03": csi(pred, ref, 0.03),
        "csi_0p15": csi(pred, ref, 0.15),
        "peak_pred_m": float(np.max(pred)),
        "peak_ref_m": float(np.max(ref)),
        "peak_error_mm": float((np.max(pred) - np.max(ref)) * 1000.0),
        "final_volume_pred_m3": float(np.sum(pred[-1]) * CELL_AREA),
        "final_volume_ref_m3": float(np.sum(ref[-1]) * CELL_AREA),
        "final_volume_error_m3": float((np.sum(pred[-1]) - np.sum(ref[-1])) * CELL_AREA),
    }


def exchange_audit(out_dir: Path, frames: int) -> dict[str, float | int | str]:
    path = out_dir / "exchange_log.csv"
    if not path.exists():
        return {
            "exchange_log_rows": 0,
            "exchange_log_status": "missing",
            "surface_to_pipe_integrated_m3": float("nan"),
            "pipe_to_surface_integrated_m3": float("nan"),
            "net_pipe_to_surface_m3": float("nan"),
            "mean_surface_to_pipe_cms": float("nan"),
            "mean_pipe_to_surface_cms": float("nan"),
            "max_surface_to_pipe_cms": float("nan"),
            "max_pipe_to_surface_cms": float("nan"),
            "mean_coupled_nodes_per_step": float("nan"),
        }
    rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    if not rows:
        return {
            "exchange_log_rows": 0,
            "exchange_log_status": "empty",
            "surface_to_pipe_integrated_m3": 0.0,
            "pipe_to_surface_integrated_m3": 0.0,
            "net_pipe_to_surface_m3": 0.0,
            "mean_surface_to_pipe_cms": 0.0,
            "mean_pipe_to_surface_cms": 0.0,
            "max_surface_to_pipe_cms": 0.0,
            "max_pipe_to_surface_cms": 0.0,
            "mean_coupled_nodes_per_step": 0.0,
        }
    dt = np.array([float(r["dt_drain_s"]) for r in rows], dtype=np.float64)
    q_in = np.array([float(r["surface_to_pipe_cms"]) for r in rows], dtype=np.float64)
    q_out = np.array([float(r["pipe_to_surface_cms"]) for r in rows], dtype=np.float64)
    nodes = np.array([float(r["coupled_nodes"]) for r in rows], dtype=np.float64)
    surface_to_pipe = float(np.sum(q_in * dt))
    pipe_to_surface = float(np.sum(q_out * dt))
    expected_min_rows = max(frames * 5, 1)
    return {
        "exchange_log_rows": len(rows),
        "exchange_log_status": "ok" if len(rows) >= expected_min_rows else "sparse",
        "surface_to_pipe_integrated_m3": surface_to_pipe,
        "pipe_to_surface_integrated_m3": pipe_to_surface,
        "net_pipe_to_surface_m3": pipe_to_surface - surface_to_pipe,
        "mean_surface_to_pipe_cms": float(np.mean(q_in)),
        "mean_pipe_to_surface_cms": float(np.mean(q_out)),
        "max_surface_to_pipe_cms": float(np.max(q_in)),
        "max_pipe_to_surface_cms": float(np.max(q_out)),
        "mean_coupled_nodes_per_step": float(np.mean(nodes)),
    }


def analyze_event(event: str) -> dict[str, object]:
    out_dir = OUT / event
    h_coupled = np.load(out_dir / "h_coupled_prototype.npy")
    n = h_coupled.shape[0]
    h_surface = np.load(DATA / event / "h_itzi_surface.npy")[:n]
    h_sink = np.load(DATA / event / "h_itzi_sink.npy")[:n]
    h_mike = np.load(DATA / event / "h_mike_ref.npy")[:n]
    meta = json.loads((out_dir / "metadata.json").read_text(encoding="utf-8"))

    rows = [
        metrics(h_coupled, h_surface, "coupled_vs_itzi_surface"),
        metrics(h_coupled, h_sink, "coupled_vs_itzi_sink"),
        metrics(h_coupled, h_mike, "coupled_vs_mike_ref"),
        metrics(h_sink, h_surface, "sink_vs_itzi_surface"),
        metrics(h_surface, h_mike, "surface_vs_mike_ref"),
        metrics(h_sink, h_mike, "sink_vs_mike_ref"),
    ]
    with (out_dir / "prototype_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    exchange = exchange_audit(out_dir, n)
    final_surface_delta_m3 = float((np.sum(h_coupled[-1]) - np.sum(h_surface[-1])) * CELL_AREA)
    final_sink_delta_m3 = float((np.sum(h_coupled[-1]) - np.sum(h_sink[-1])) * CELL_AREA)
    summary = {
        **meta,
        **exchange,
        "surface_peak_over_run_m": float(np.max(h_surface)),
        "sink_peak_over_run_m": float(np.max(h_sink)),
        "mike_peak_over_run_m": float(np.max(h_mike)),
        "coupled_peak_over_run_m": float(np.max(h_coupled)),
        "surface_final_volume_m3": float(np.sum(h_surface[-1]) * CELL_AREA),
        "sink_final_volume_m3": float(np.sum(h_sink[-1]) * CELL_AREA),
        "mike_final_volume_m3": float(np.sum(h_mike[-1]) * CELL_AREA),
        "coupled_final_volume_m3": float(np.sum(h_coupled[-1]) * CELL_AREA),
        "final_volume_delta_vs_surface_m3": final_surface_delta_m3,
        "final_volume_delta_vs_sink_m3": final_sink_delta_m3,
        "exchange_net_vs_final_surface_delta_m3": float(exchange["net_pipe_to_surface_m3"] - final_surface_delta_m3),
        "exchange_net_vs_final_sink_delta_m3": float(exchange["net_pipe_to_surface_m3"] - final_sink_delta_m3),
        "metrics": rows,
        "prototype_interpretation": (
            "This is an online exchange prototype using ITZI surface flow and a "
            "network-only SWMM dynamic-wave branch. The audit includes exchange "
            "volumes and comparisons with ITZI surface-only, ITZI + sink and "
            "MIKE reference. It remains a prototype until all events pass water "
            "balance, node-exchange and reference-comparison checks."
        ),
    }
    (out_dir / "prototype_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    flood_cmap = LinearSegmentedColormap.from_list(
        "flood",
        ["#f7fbff", "#c6dbef", "#6baed6", "#2171b5", "#08306b", "#041838"],
    )
    final_idx = -1
    maps = [
        ("ITZI surface", h_surface[final_idx], "depth"),
        ("ITZI + sink", h_sink[final_idx], "depth"),
        ("MIKE reference", h_mike[final_idx], "depth"),
        ("ITZI + SWMM prototype", h_coupled[final_idx], "depth"),
        ("prototype - surface", h_coupled[final_idx] - h_surface[final_idx], "diff"),
        ("prototype - sink", h_coupled[final_idx] - h_sink[final_idx], "diff"),
        ("prototype - MIKE", h_coupled[final_idx] - h_mike[final_idx], "diff"),
    ]
    fig = plt.figure(figsize=(21, 9))
    gs = fig.add_gridspec(2, 4)
    vmax = max(
        float(np.max(h_surface[final_idx])),
        float(np.max(h_sink[final_idx])),
        float(np.max(h_mike[final_idx])),
        float(np.max(h_coupled[final_idx])),
        0.01,
    )
    diff_lim = max(
        float(np.max(np.abs(h_coupled[final_idx] - h_surface[final_idx]))),
        float(np.max(np.abs(h_coupled[final_idx] - h_sink[final_idx]))),
        float(np.max(np.abs(h_coupled[final_idx] - h_mike[final_idx]))),
        0.001,
    )
    for i, (title, arr, kind) in enumerate(maps):
        ax = fig.add_subplot(gs[i // 4, i % 4])
        if kind == "depth":
            im = ax.imshow(arr, cmap=flood_cmap, vmin=0, vmax=vmax)
            label = "Depth (m)"
        else:
            im = ax.imshow(arr, cmap="RdBu_r", vmin=-diff_lim, vmax=diff_lim)
            label = "Difference (m)"
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, shrink=0.75, label=label)

    ax = fig.add_subplot(gs[1, 3])
    times = np.arange(1, n + 1) * 5.0 / 60.0
    ax.plot(times, [np.sum(x) * CELL_AREA for x in h_surface], label="surface", color="#d73027")
    ax.plot(times, [np.sum(x) * CELL_AREA for x in h_sink], label="sink", color="#4575b4")
    ax.plot(times, [np.sum(x) * CELL_AREA for x in h_coupled], label="coupled prototype", color="#1a9850")
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Surface water volume (m3)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title("Surface-water volume")

    fig.suptitle(f"{event} ITZI-SWMM Coupled Prototype: diagnostic", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_dir / "prototype_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote {out_dir / 'prototype_metrics.csv'}")
    print(f"Wrote {out_dir / 'prototype_summary.json'}")
    print(f"Wrote {out_dir / 'prototype_comparison.png'}")
    return summary


def make_aggregate_figure(rows: list[dict[str, object]]) -> None:
    events = [str(r["event"]) for r in rows]
    x = np.arange(len(events))
    width = 0.25
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    surface_to_pipe = np.array([float(r["surface_to_pipe_m3"]) for r in rows]) / 1000.0
    pipe_to_surface = np.array([float(r["pipe_to_surface_m3"]) for r in rows]) / 1000.0
    axes[0].bar(x - width / 2, surface_to_pipe, width, label="surface to pipe", color="#4575b4")
    axes[0].bar(x + width / 2, pipe_to_surface, width, label="pipe to surface", color="#d73027")
    axes[0].set_title("Bidirectional exchange")
    axes[0].set_ylabel(r"Volume ($10^3$ m$^3$)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(events, rotation=30, ha="right")
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", alpha=0.25)

    surface_vol = np.array([float(r["surface_final_volume_m3"]) for r in rows]) / 1000.0
    sink_vol = np.array([float(r["sink_final_volume_m3"]) for r in rows]) / 1000.0
    coupled_vol = np.array([float(r["coupled_final_volume_m3"]) for r in rows]) / 1000.0
    axes[1].bar(x - width, surface_vol, width, label="surface-only", color="#d73027")
    axes[1].bar(x, sink_vol, width, label="ITZI + sink", color="#4575b4")
    axes[1].bar(x + width, coupled_vol, width, label="ITZI-SWMM prototype", color="#1a9850")
    axes[1].set_title("Final surface-water volume")
    axes[1].set_ylabel(r"Volume ($10^3$ m$^3$)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(events, rotation=30, ha="right")
    axes[1].legend(fontsize=8)
    axes[1].grid(axis="y", alpha=0.25)

    mae_surface = np.array([float(r["coupled_vs_surface_mae_mm"]) for r in rows])
    mae_sink = np.array([float(r["coupled_vs_sink_mae_mm"]) for r in rows])
    axes[2].bar(x - width / 2, mae_surface, width, label="vs surface-only", color="#9467bd")
    axes[2].bar(x + width / 2, mae_sink, width, label="vs ITZI + sink", color="#8c564b")
    axes[2].set_title("Prototype spatial MAE")
    axes[2].set_ylabel("MAE (mm)")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(events, rotation=30, ha="right")
    axes[2].legend(fontsize=8)
    axes[2].grid(axis="y", alpha=0.25)

    fig.suptitle("Five held-out event ITZI-SWMM coupled prototype diagnostics", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT / "prototype_summary_all_events.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    if args.all:
        events = sorted(
            [p.name for p in OUT.iterdir() if (p / "h_coupled_prototype.npy").exists()],
            key=lambda x: (len(x), x),
        )
    else:
        events = args.events
    if not events:
        raise SystemExit(f"No prototype events found in {OUT}")

    summaries = [analyze_event(event) for event in events]
    rows: list[dict[str, object]] = []
    for summary in summaries:
        event = str(summary["event"])
        metric_lookup = {row["comparison"]: row for row in summary["metrics"]}
        c_vs_s = metric_lookup["coupled_vs_itzi_surface"]
        c_vs_sink = metric_lookup["coupled_vs_itzi_sink"]
        c_vs_mike = metric_lookup["coupled_vs_mike_ref"]
        rows.append({
            "event": event,
            "frames": summary["frames"],
            "coupled_nodes": summary["coupled_nodes"],
            "swmm_nodes": summary["swmm_nodes"],
            "swmm_links": summary["swmm_links"],
            "exchange_log_rows": summary.get("exchange_log_rows", ""),
            "exchange_log_status": summary.get("exchange_log_status", ""),
            "surface_to_pipe_m3": summary["surface_to_pipe_m3"],
            "pipe_to_surface_m3": summary["pipe_to_surface_m3"],
            "exchange_surface_to_pipe_integrated_m3": summary.get("surface_to_pipe_integrated_m3", ""),
            "exchange_pipe_to_surface_integrated_m3": summary.get("pipe_to_surface_integrated_m3", ""),
            "exchange_net_pipe_to_surface_m3": summary.get("net_pipe_to_surface_m3", ""),
            "coupled_final_volume_m3": summary["coupled_final_volume_m3"],
            "surface_final_volume_m3": summary["surface_final_volume_m3"],
            "sink_final_volume_m3": summary["sink_final_volume_m3"],
            "mike_final_volume_m3": summary["mike_final_volume_m3"],
            "coupled_vs_surface_mae_mm": c_vs_s["mae_mm"],
            "coupled_vs_surface_csi_0p03": c_vs_s["csi_0p03"],
            "coupled_vs_surface_csi_0p15": c_vs_s["csi_0p15"],
            "coupled_vs_surface_peak_error_mm": c_vs_s["peak_error_mm"],
            "coupled_vs_surface_final_volume_error_m3": c_vs_s["final_volume_error_m3"],
            "coupled_vs_sink_mae_mm": c_vs_sink["mae_mm"],
            "coupled_vs_sink_csi_0p03": c_vs_sink["csi_0p03"],
            "coupled_vs_sink_csi_0p15": c_vs_sink["csi_0p15"],
            "coupled_vs_sink_peak_error_mm": c_vs_sink["peak_error_mm"],
            "coupled_vs_sink_final_volume_error_m3": c_vs_sink["final_volume_error_m3"],
            "coupled_vs_mike_mae_mm": c_vs_mike["mae_mm"],
            "coupled_vs_mike_csi_0p03": c_vs_mike["csi_0p03"],
            "coupled_vs_mike_csi_0p15": c_vs_mike["csi_0p15"],
            "coupled_vs_mike_peak_error_mm": c_vs_mike["peak_error_mm"],
            "coupled_vs_mike_final_volume_error_m3": c_vs_mike["final_volume_error_m3"],
            "exchange_net_vs_final_surface_delta_m3": summary.get("exchange_net_vs_final_surface_delta_m3", ""),
            "exchange_net_vs_final_sink_delta_m3": summary.get("exchange_net_vs_final_sink_delta_m3", ""),
        })
    out = OUT / "prototype_summary_all_events.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    make_aggregate_figure(rows)
    print(f"Wrote {out}")
    print(f"Wrote {OUT / 'prototype_summary_all_events.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
