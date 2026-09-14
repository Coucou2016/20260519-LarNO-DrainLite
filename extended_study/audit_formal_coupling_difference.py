#!/usr/bin/env python3
"""Audit whether formal ITZI-SWMM coupled outputs differ from surface-only."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "LarNO-main" / "benchmark" / "urbanflood" / "flood" / "region1_20m_drainage_v1"
NPZ_DIR = (
    ROOT
    / "external_models"
    / "20260518-itzi-flood"
    / "test_cases"
    / "shenzhen_region1"
    / "output"
    / "full_domain_swmm"
)
OUT = ROOT / "extended_study" / "output" / "coupling_difference_audit"
FIG_DIR = OUT / "figures"
EVENTS = ["event1", "event20", "event65", "event66", "event67", "event68", "event69", "event70"]
CELL_AREA = 400.0


def load(event: str) -> dict[str, np.ndarray]:
    d = DATA_DIR / event
    return {
        "mike": np.load(d / "h_mike_ref.npy").astype(np.float64),
        "surface": np.load(d / "h_itzi_swmm_official_surface.npy").astype(np.float64),
        "coupled": np.load(d / "h_itzi_swmm_official.npy").astype(np.float64),
    }


def raw_npz_equal(event: str) -> bool:
    z = np.load(NPZ_DIR / f"{event}_coupled.npz", allow_pickle=True)
    return bool(np.array_equal(z["h_surf_series"], z["h_swmm_series"]))


def metrics() -> list[dict[str, object]]:
    rows = []
    for event in EVENTS:
        a = load(event)
        r = a["coupled"] - a["surface"]
        abs_mm = np.abs(r).ravel() * 1000.0
        final = r[-1] * CELL_AREA
        rows.append(
            {
                "event": event,
                "raw_surface_equals_coupled": raw_npz_equal(event),
                "mean_abs_diff_mm": float(abs_mm.mean()),
                "rmse_diff_mm": float(np.sqrt(np.mean((r * 1000.0) ** 2))),
                "p50_abs_diff_mm": float(np.quantile(abs_mm, 0.50)),
                "p90_abs_diff_mm": float(np.quantile(abs_mm, 0.90)),
                "p95_abs_diff_mm": float(np.quantile(abs_mm, 0.95)),
                "p99_abs_diff_mm": float(np.quantile(abs_mm, 0.99)),
                "p999_abs_diff_mm": float(np.quantile(abs_mm, 0.999)),
                "max_abs_diff_mm": float(abs_mm.max()),
                "gt_0p1mm_pct": float(np.mean(abs_mm > 0.1) * 100.0),
                "gt_1mm_pct": float(np.mean(abs_mm > 1.0) * 100.0),
                "gt_10mm_pct": float(np.mean(abs_mm > 10.0) * 100.0),
                "final_reduction_m3": float(np.minimum(final, 0).sum()),
                "final_local_increase_m3": float(np.maximum(final, 0).sum()),
                "final_net_change_m3": float(final.sum()),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def fig_peak_diff_grid() -> Path:
    path = FIG_DIR / "formal_coupling_peak_difference_grid_mm.png"
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.5))
    for ax, event in zip(axes.flat, EVENTS):
        a = load(event)
        diff = (a["coupled"].max(axis=0) - a["surface"].max(axis=0)) * 1000.0
        im = ax.imshow(np.clip(diff, -50, 50), cmap="coolwarm", vmin=-50, vmax=50)
        ax.set_title(event)
        ax.axis("off")
    fig.suptitle("Formal ITZI-SWMM peak difference: coupled minus surface-only (mm, clipped to +/-50)")
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_zoom_event(event: str = "event70") -> Path:
    path = FIG_DIR / f"{event}_max_coupling_difference_zoom.png"
    a = load(event)
    r = a["coupled"] - a["surface"]
    t, row, col = np.unravel_index(np.argmax(np.abs(r)), r.shape)
    pad = 24
    r0, r1 = max(0, row - pad), min(r.shape[1], row + pad + 1)
    c0, c1 = max(0, col - pad), min(r.shape[2], col + pad + 1)
    panels = [
        ("surface-only depth", a["surface"][t, r0:r1, c0:c1], "viridis", None),
        ("coupled depth", a["coupled"][t, r0:r1, c0:c1], "viridis", None),
        ("coupled - surface (mm)", r[t, r0:r1, c0:c1] * 1000.0, "coolwarm", 100),
        ("final coupled - surface (mm)", r[-1, r0:r1, c0:c1] * 1000.0, "coolwarm", 100),
    ]
    vmax = max(float(panels[0][1].max()), float(panels[1][1].max()))
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.1))
    for ax, (title, data, cmap, lim) in zip(axes, panels):
        if lim is None:
            im = ax.imshow(data, cmap=cmap, vmin=0, vmax=vmax)
        else:
            im = ax.imshow(np.clip(data, -lim, lim), cmap=cmap, vmin=-lim, vmax=lim)
        ax.scatter([col - c0], [row - r0], s=32, facecolors="none", edgecolors="black", linewidths=1.1)
        ax.set_title(title)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.suptitle(f"{event}: local zoom around max absolute difference, t={(t + 1) * 5} min")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def fig_temporal_difference() -> Path:
    path = FIG_DIR / "formal_coupling_temporal_difference.png"
    t_min = np.arange(1, 73) * 5
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    for event in EVENTS:
        a = load(event)
        diff = a["coupled"] - a["surface"]
        vol = diff.sum(axis=(1, 2)) * CELL_AREA
        hmax_diff = (a["coupled"].max(axis=(1, 2)) - a["surface"].max(axis=(1, 2))) * 1000.0
        affected_1mm = (np.abs(diff) > 0.001).mean(axis=(1, 2)) * 100.0
        axes[0].plot(t_min, vol, linewidth=1.3, label=event)
        axes[1].plot(t_min, hmax_diff, linewidth=1.3, label=event)
        axes[2].plot(t_min, affected_1mm, linewidth=1.3, label=event)
    axes[0].axhline(0, color="#111827", linewidth=0.8)
    axes[1].axhline(0, color="#111827", linewidth=0.8)
    axes[0].set_ylabel("Coupled - surface\nvolume (m3)")
    axes[1].set_ylabel("Coupled - surface\nmax depth (mm)")
    axes[2].set_ylabel("|difference| > 1 mm\narea (%)")
    axes[2].set_xlabel("Time (min)")
    axes[0].set_title("Temporal diagnostics of formal ITZI-SWMM coupling effect")
    for ax in axes:
        ax.grid(alpha=0.25)
    axes[0].legend(ncol=4, fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = metrics()
    write_csv(OUT / "formal_coupling_difference_metrics.csv", rows)
    figs = [fig_peak_diff_grid(), fig_zoom_event("event68"), fig_zoom_event("event70"), fig_temporal_difference()]
    summary = {
        "all_raw_arrays_equal": all(bool(r["raw_surface_equals_coupled"]) for r in rows),
        "mean_abs_diff_mm_mean": float(np.mean([float(r["mean_abs_diff_mm"]) for r in rows])),
        "rmse_diff_mm_mean": float(np.mean([float(r["rmse_diff_mm"]) for r in rows])),
        "gt_1mm_pct_mean": float(np.mean([float(r["gt_1mm_pct"]) for r in rows])),
        "max_abs_diff_mm_max": float(max(float(r["max_abs_diff_mm"]) for r in rows)),
        "figures": [str(p.relative_to(ROOT)) for p in figs],
        "interpretation": (
            "The coupled and surface-only arrays are not identical. Differences are sparse: "
            "most cells are near zero, while local cells can differ by decimeter scale."
        ),
    }
    (OUT / "formal_coupling_difference_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
