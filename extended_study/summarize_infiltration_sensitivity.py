#!/usr/bin/env python3
"""Summarize event68 infiltration-rate sensitivity for ITZI-SWMM labels."""

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
DATA = ROOT / "LarNO-main" / "benchmark" / "urbanflood" / "flood" / "region1_20m_connected_swmm_v1"
GEO = ROOT / "LarNO-main" / "benchmark" / "urbanflood" / "geodata" / "region1_20m_connected_swmm_v1"
RUN_BASE = ROOT / "extended_study" / "output"
OUT = RUN_BASE / "infiltration_sensitivity"
FIG = OUT / "figures"
EVENT = "event68"
RATES = [1, 2, 3, 4, 5]
CELL_AREA_M2 = 20.0 * 20.0
DT_H = 5.0 / 60.0


def active_mask() -> np.ndarray:
    dem = np.load(GEO / "dem.npy").astype(np.float32)
    return dem < 49.9


def max_depth(h: np.ndarray) -> np.ndarray:
    return h.max(axis=(1, 2))


def volume(h: np.ndarray, active: np.ndarray) -> np.ndarray:
    return h[:, active].sum(axis=1) * CELL_AREA_M2


def flooded_area(h: np.ndarray, active: np.ndarray, threshold: float = 0.03) -> np.ndarray:
    return (h[:, active] >= threshold).sum(axis=1) * CELL_AREA_M2 / 1_000_000.0


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def read_event_row(path: Path, event: str) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    matches = [row for row in rows if row.get("event", "").strip() == event]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one {event!r} row in {path}, found {len(matches)}. "
            "Sensitivity metadata must not be borrowed from another event."
        )
    return matches[0]


def to_float(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, "nan"))
    except ValueError:
        return float("nan")


def summarize_array(
    label: str,
    rate_mmh: float,
    scenario: str,
    h: np.ndarray,
    mike: np.ndarray,
    active: np.ndarray,
    swmm_row: dict[str, str] | None = None,
) -> dict[str, object]:
    hmax = max_depth(h)
    vol = volume(h, active)
    area = flooded_area(h, active)
    mike_peak_map = mike.max(axis=0)
    h_idx = int(np.argmax(hmax))
    v_idx = int(np.argmax(vol))
    a_idx = int(np.argmax(area))
    return {
        "label": label,
        "event": EVENT,
        "rate_mmh": rate_mmh,
        "scenario": scenario,
        "max_depth_peak_hour": float((h_idx + 1) * DT_H),
        "max_depth_peak_m": float(hmax[h_idx]),
        "max_depth_final_m": float(hmax[-1]),
        "max_depth_final_to_peak_ratio": float(hmax[-1] / hmax[h_idx]) if hmax[h_idx] else np.nan,
        "volume_peak_hour": float((v_idx + 1) * DT_H),
        "volume_peak_m3": float(vol[v_idx]),
        "volume_final_m3": float(vol[-1]),
        "volume_final_to_peak_ratio": float(vol[-1] / vol[v_idx]) if vol[v_idx] else np.nan,
        "flooded_area_peak_hour": float((a_idx + 1) * DT_H),
        "flooded_area_peak_km2": float(area[a_idx]),
        "flooded_area_final_km2": float(area[-1]),
        "mae_to_mike_mm": float(np.mean(np.abs(h - mike)) * 1000.0) if scenario != "mike_reference" else 0.0,
        "rmse_to_mike_mm": rmse(h, mike) * 1000.0 if scenario != "mike_reference" else 0.0,
        "peak_map_mae_to_mike_mm": float(np.mean(np.abs(h.max(axis=0) - mike_peak_map)) * 1000.0)
        if scenario != "mike_reference"
        else 0.0,
        "continuity_error_pct": to_float(swmm_row or {}, "continuity_error_pct"),
        "nonconverging_steps_pct": to_float(swmm_row or {}, "nonconverging_steps_pct"),
        "connected_drainage_m3": to_float(swmm_row or {}, "external_outflow_m3"),
        "surface_infiltrated_m3": to_float(swmm_row or {}, "surface_infiltrated_m3"),
        "connected_infiltrated_m3": to_float(swmm_row or {}, "connected_infiltrated_m3"),
    }


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


def choose_candidate(rows: list[dict[str, object]]) -> dict[str, object]:
    mike = next(r for r in rows if r["scenario"] == "mike_reference")
    connected = [r for r in rows if r["scenario"] == "connected"]
    eligible = []
    for r in connected:
        continuity = abs(float(r["continuity_error_pct"]))
        peak_ratio = float(r["max_depth_peak_m"]) / max(float(mike["max_depth_peak_m"]), 1e-6)
        if continuity <= 8.0 and 0.8 <= peak_ratio <= 1.2:
            eligible.append(r)
    if not eligible:
        eligible = connected

    def score(r: dict[str, object]) -> float:
        mike_peak = float(mike["max_depth_peak_m"])
        return (
            0.35 * abs(float(r["max_depth_peak_m"]) - mike_peak) / max(mike_peak, 1e-6)
            + 0.30
            * abs(float(r["volume_final_to_peak_ratio"]) - float(mike["volume_final_to_peak_ratio"]))
            + 0.20 * float(r["mae_to_mike_mm"]) / 100.0
            + 0.15 * abs(float(r["volume_peak_hour"]) - float(mike["volume_peak_hour"])) / 6.0
        )

    ranked = sorted((dict(r, selection_score=score(r)) for r in eligible), key=lambda r: r["selection_score"])
    selected = ranked[0]
    return {
        "selected_rate_mmh": float(selected["rate_mmh"]),
        "selected_label": selected["label"],
        "selected_scenario": selected["scenario"],
        "selection_score": float(selected["selection_score"]),
        "selection_rules": [
            "Primary calibration target is the active-cell surface-water volume hydrograph, because the single-cell maximum-depth curve is dominated by local boundary/depression cells.",
            "Eligible rates require absolute SWMM continuity error <= 8% and connected peak depth within 80-120% of MIKE peak depth.",
            "Within eligible rates, the score balances peak-depth magnitude, final/peak volume ratio, full-sequence MAE to MIKE, and volume peak timing.",
        ],
        "eligible_rates_mmh": [float(r["rate_mmh"]) for r in ranked],
    }


def plot_hydrographs(series: dict[str, np.ndarray], active: np.ndarray) -> None:
    t = np.arange(1, next(iter(series.values())).shape[0] + 1) * DT_H
    fig, axes = plt.subplots(2, 1, figsize=(7.1, 5.1), sharex=True, constrained_layout=True)
    colors = {
        "MIKE reference": "#111827",
        "No infiltration connected": "#ef4444",
        "1 mm/h connected": "#f97316",
        "2 mm/h connected": "#22c55e",
        "3 mm/h connected": "#06b6d4",
        "4 mm/h connected": "#3b82f6",
        "5 mm/h connected": "#7c3aed",
    }
    for name, h in series.items():
        axes[0].plot(t, max_depth(h), label=name, color=colors.get(name), lw=1.55)
        axes[1].plot(t, volume(h, active) / 1000.0, label=name, color=colors.get(name), lw=1.55)
    axes[0].set_ylabel("Maximum depth (m)")
    axes[0].set_title(f"{EVENT}: connected ITZI-SWMM maximum-depth sensitivity")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8, ncol=3)
    axes[1].set_xlabel("Time (h)")
    axes[1].set_ylabel("Active-cell volume (10^3 m3)")
    axes[1].set_title(f"{EVENT}: connected ITZI-SWMM active-cell water-volume sensitivity")
    axes[1].grid(alpha=0.25)
    add_panel_labels(axes, x=-0.09, y=1.02)
    fig.savefig(FIG / "event68_infiltration_sensitivity_hydrographs.png")
    plt.close(fig)


def plot_metrics(rows: list[dict[str, object]], candidate: dict[str, object]) -> None:
    connected = [r for r in rows if r["scenario"] == "connected"]
    connected.sort(key=lambda r: float(r["rate_mmh"]))
    rates = [float(r["rate_mmh"]) for r in connected]
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 5.1), constrained_layout=True)
    specs = [
        ("mae_to_mike_mm", "Full-sequence MAE to MIKE (mm)"),
        ("max_depth_peak_m", "Peak maximum depth (m)"),
        ("volume_final_to_peak_ratio", "Final / peak active-cell volume"),
        ("continuity_error_pct", "SWMM continuity error (%)"),
    ]
    for ax, (key, title) in zip(axes.ravel(), specs):
        vals = [float(r[key]) for r in connected]
        ax.plot(rates, vals, marker="o", color="#2563eb", lw=1.8)
        ax.axvline(float(candidate["selected_rate_mmh"]), color="#dc2626", lw=1.1, ls="--")
        ax.set_xlabel("Infiltration rate (mm/h)")
        ax.set_title(title)
        ax.grid(alpha=0.25)
    add_panel_labels(axes.ravel(), x=-0.14, y=1.03)
    fig.savefig(FIG / "event68_infiltration_sensitivity_metrics.png")
    plt.close(fig)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    active = active_mask()
    base_dir = DATA / EVENT
    mike = np.load(base_dir / "h_mike_ref.npy").astype(np.float32)
    rows = [
        summarize_array("MIKE reference", 0.0, "mike_reference", mike, mike, active),
        summarize_array(
            "No infiltration surface",
            0.0,
            "surface",
            np.load(base_dir / "h_itzi_surface.npy").astype(np.float32),
            mike,
            active,
        ),
        summarize_array(
            "No infiltration connected",
            0.0,
            "connected",
            np.load(base_dir / "h_itzi_swmm_connected.npy").astype(np.float32),
            mike,
            active,
        ),
    ]
    hydro_series = {
        "MIKE reference": mike,
        "No infiltration connected": np.load(base_dir / "h_itzi_swmm_connected.npy").astype(np.float32),
    }
    for rate in RATES:
        run_dir = RUN_BASE / f"connected_itzi_swmm_inf_{rate}mmh"
        npz_path = run_dir / EVENT / f"{EVENT}_connected_itzi_swmm.npz"
        if not npz_path.exists():
            raise FileNotFoundError(f"Missing sensitivity result: {npz_path}")
        z = np.load(npz_path)
        swmm_row = read_event_row(run_dir / "connected_itzi_swmm_metrics.csv", EVENT)
        rows.append(
            summarize_array(
                f"{rate} mm/h infiltration surface",
                float(rate),
                "surface",
                z["h_surf"].astype(np.float32),
                mike,
                active,
            )
        )
        connected = z["h_swmm_connected"].astype(np.float32)
        rows.append(
            summarize_array(
                f"{rate} mm/h infiltration connected",
                float(rate),
                "connected",
                connected,
                mike,
                active,
                swmm_row,
            )
        )
        hydro_series[f"{rate} mm/h connected"] = connected

    candidate = choose_candidate(rows)
    write_csv(OUT / "event68_infiltration_sensitivity_metrics.csv", rows)
    plot_hydrographs(hydro_series, active)
    plot_metrics(rows, candidate)
    note = {
        "event": EVENT,
        **candidate,
        "metrics_csv": str(OUT / "event68_infiltration_sensitivity_metrics.csv"),
        "figures": {
            "hydrographs": str(FIG / "event68_infiltration_sensitivity_hydrographs.png"),
            "metrics": str(FIG / "event68_infiltration_sensitivity_metrics.png"),
        },
        "interpretation": (
            "The selected rate is a pragmatic effective-loss setting for the current conceptual network, "
            "not a calibrated soil infiltration parameter. It is selected to improve recession behavior "
            "without over-depressing peak depth or violating the SWMM continuity threshold."
        ),
    }
    (OUT / "event68_infiltration_sensitivity_summary.json").write_text(
        json.dumps(note, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(note, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
