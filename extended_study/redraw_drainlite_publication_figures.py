#!/usr/bin/env python3
"""Redraw DrainLite figures from existing arrays and metrics without retraining."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import train_drainlite_connected_residual as drainlite
from publication_plot_style import configure_publication_style


configure_publication_style()
ROOT = Path(__file__).resolve().parents[1]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def redraw_feature_importance(output: Path, model: str) -> None:
    rows = read_csv(output / "metrics" / f"feature_importance_{model}.csv")
    rows.sort(key=lambda row: float(row["importance_mean"]), reverse=True)
    top = rows[:20]
    labels = [row["feature"].replace("_", " ") for row in reversed(top)]
    values = [float(row["importance_mean"]) for row in reversed(top)]
    errors = [float(row.get("importance_std", 0.0)) for row in reversed(top)]
    fig, ax = plt.subplots(figsize=(7.1, 4.6))
    ax.barh(labels, values, xerr=errors, color="#0072B2", error_kw={"linewidth": 0.6, "capsize": 1.5})
    ax.set_xlabel("Permutation importance (increase in MAE, mm)")
    title = "All static features" if model == "all_static" else "SWMM-assisted diagnostic"
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "figures" / f"feature_importance_{model}.png")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", default="region1_20m_connected_swmm_v2_inf1mmh")
    parser.add_argument("--output-name", default="drainlite_connected_residual_v2_inf1mmh")
    parser.add_argument("--events", nargs="+", default=["event68", "event69", "event70"])
    args = parser.parse_args()

    output = ROOT / "extended_study" / "output" / args.output_name
    drainlite.configure_paths(args.dataset_name, args.output_name)
    static = drainlite.load_static()
    drainlite.make_static_figures(static)
    records = []
    for event in args.events:
        arrays = drainlite.load_event(event)
        prediction_dir = output / "predictions" / event
        pred = np.load(prediction_dir / "h_drainlite_connected_all_static.npy").astype(np.float32)
        residual = np.load(prediction_dir / "signed_residual_pred_all_static_mm.npy").astype(np.float32)
        records.append((event, arrays, pred, residual))
    active = static["active_mask"]
    water_vmax = max(
        float(field.max(axis=0)[active].max())
        for _, arrays, pred, _ in records
        for field in [arrays["mike"], arrays["surface"], arrays["official"], pred]
    )
    error_values = np.concatenate(
        [np.abs((pred.max(axis=0) - arrays["official"].max(axis=0))[active]) for _, arrays, pred, _ in records]
    )
    residual_values = []
    for _, arrays, _, residual in records:
        peak_index = np.argmax(arrays["official"], axis=0)
        rr, cc = np.indices(peak_index.shape)
        residual_values.append(np.abs((residual[peak_index, rr, cc] / 1000.0)[active]))
    error_limit = max(0.02, float(np.percentile(error_values, 99.5)))
    residual_limit = max(0.02, float(np.percentile(np.concatenate(residual_values), 99.5)))
    for event, arrays, pred, residual in records:
        drainlite.make_peak_map(
            event,
            arrays,
            pred,
            residual,
            output / "figures" / f"{event}_connected_peak_maps.png",
            water_vmax=water_vmax,
            error_limit=error_limit,
            residual_limit=residual_limit,
        )
    for model in ["all_static", "swmm_assisted"]:
        redraw_feature_importance(output, model)
    print(output / "figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
