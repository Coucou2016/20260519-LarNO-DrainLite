#!/usr/bin/env python3
"""Generate publication figures from the verified experiment outputs."""

from __future__ import annotations

import csv
import shutil
from collections import defaultdict
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

import train_drainlite_connected_residual as connected
from publication_plot_style import add_panel_labels, configure_publication_style
from run_drainlite_major_revision_experiments import FEATURE_GROUPS, predict_event


configure_publication_style()

ROOT = Path(__file__).resolve().parents[1]
DATASET = "region1_20m_connected_swmm_v2_inf1mmh"
EXP = ROOT / "extended_study" / "output" / "larno_drainlite_major_revision"
FIG = EXP / "publication_figures"
FLOOD = ROOT / "LarNO-main" / "benchmark" / "urbanflood" / "flood" / DATASET
GEO = ROOT / "LarNO-main" / "benchmark" / "urbanflood" / "geodata" / DATASET
EVENTS = ["event1", "event67", "event68", "event69", "event70"]
DT_H = 5.0 / 60.0
CELL_AREA_M2 = 400.0


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_event(event: str) -> dict[str, np.ndarray]:
    directory = FLOOD / event
    return {
        "surface": np.load(directory / "h_itzi_surface.npy").astype(np.float32),
        "official": np.load(directory / "h_itzi_swmm_connected.npy").astype(np.float32),
        "mike": np.load(directory / "h_mike_ref.npy").astype(np.float32),
        "rainfall": np.load(directory / "rainfall.npy").astype(np.float32),
    }


def ensure_main_predictions() -> None:
    connected.configure_paths(DATASET, "larno_drainlite_major_revision")
    static = connected.load_static()
    swmm = connected.read_connected_swmm()
    for event in EVENTS:
        output = EXP / "predictions" / event / "h_no_xy_all_static.npy"
        if output.exists():
            continue
        arrays = connected.load_event(event)
        bundle = joblib.load(EXP / "models" / event / "no_xy_all_static.joblib")
        pred, residual = predict_event(
            event,
            bundle["model"],
            bundle["features"],
            arrays,
            static,
            swmm,
            60000,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        np.save(output, pred)
        np.save(output.with_name("residual_no_xy_all_static_mm.npy"), residual)


def active_mask() -> np.ndarray:
    dem = np.load(GEO / "dem.npy")
    return np.isfinite(dem) & (dem < 49.9)


def volume(h: np.ndarray, active: np.ndarray) -> np.ndarray:
    return h[:, active].sum(axis=1) * CELL_AREA_M2


def maximum_depth(h: np.ndarray, active: np.ndarray) -> np.ndarray:
    return h[:, active].max(axis=1)


def save(fig: plt.Figure, name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / name)
    plt.close(fig)


def figure1_workflow() -> None:
    fig, ax = plt.subplots(figsize=(7.1, 3.4), constrained_layout=True)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    boxes = [
        (0.02, 0.61, 0.16, 0.23, "Rainfall and\nterrain fields", "#dbeafe"),
        (0.22, 0.61, 0.16, 0.23, "Road-aligned\nconceptual network", "#e0f2fe"),
        (0.42, 0.61, 0.16, 0.23, "Paired Itzï\nsurface and coupled\nsimulations", "#dcfce7"),
        (0.62, 0.61, 0.16, 0.23, "Signed residual\nr = h$_c$ - h$_s$", "#fef3c7"),
        (0.82, 0.61, 0.16, 0.23, "DrainLite\ncorrected depth", "#fee2e2"),
        (0.42, 0.14, 0.16, 0.20, "MIKE water-depth\nreference", "#f3e8ff"),
    ]
    for x, y, w, h, text, color in boxes:
        patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.012", facecolor=color, edgecolor="#374151", linewidth=0.8)
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8)
    for start, end in [((0.18, 0.725), (0.22, 0.725)), ((0.38, 0.725), (0.42, 0.725)), ((0.58, 0.725), (0.62, 0.725)), ((0.78, 0.725), (0.82, 0.725))]:
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=10, color="#374151", lw=1.0))
    ax.add_patch(FancyArrowPatch((0.50, 0.34), (0.50, 0.61), arrowstyle="-|>", mutation_scale=10, color="#7c3aed", lw=1.0, linestyle="--"))
    ax.add_patch(FancyArrowPatch((0.58, 0.24), (0.90, 0.61), arrowstyle="-|>", mutation_scale=10, color="#7c3aed", lw=1.0, linestyle="--"))
    ax.text(0.5, 0.97, "Drainage-residual modelling framework", ha="center", va="top", fontsize=11, fontweight="bold")
    ax.text(0.5, 0.04, "Solid arrows: model-development pathway    Dashed arrows: MIKE-informed configuration and descriptive comparison", ha="center", va="bottom", fontsize=7.2)
    save(fig, "fig01_provenance_workflow.png")


def figure2_network() -> None:
    for source, target in [
        (EXP / "network_audit" / "aligned_network_dem_map.png", FIG / "fig02_aligned_network_dem.png"),
        (EXP / "network_audit" / "network_stability_audit.png", FIG / "fig03_swmm_stability.png"),
    ]:
        FIG.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def figure4_hydrographs() -> None:
    active = active_mask()
    fig, axes = plt.subplots(len(EVENTS), 2, figsize=(7.1, 8.8), sharex=True, constrained_layout=True)
    colors = {"MIKE": "#111827", "Itzï surface-only": "#2563eb", "ITZI-SWMM coupled": "#dc2626"}
    for row, event in enumerate(EVENTS):
        arrays = load_event(event)
        t = np.arange(1, arrays["surface"].shape[0] + 1) * DT_H
        series = {
            "MIKE": maximum_depth(arrays["mike"], active),
            "Itzï surface-only": maximum_depth(arrays["surface"], active),
            "ITZI-SWMM coupled": maximum_depth(arrays["official"], active),
        }
        for label, values in series.items():
            axes[row, 0].plot(t, values, color=colors[label], lw=1.0, label=label)
            peak_index = int(np.argmax(values))
            marker = ">" if peak_index == len(values) - 1 else "o"
            axes[row, 0].scatter(t[peak_index], values[peak_index], color=colors[label], marker=marker, s=18, zorder=3)
        axes[row, 0].set_ylabel(f"{event}\nDepth (m)")
        axes[row, 0].grid(alpha=0.2)
        for label, h in [("MIKE", arrays["mike"]), ("Itzï surface-only", arrays["surface"]), ("ITZI-SWMM coupled", arrays["official"])]:
            values = volume(h, active) / 1000.0
            axes[row, 1].plot(t, values, color=colors[label], lw=1.0)
            peak_index = int(np.argmax(values))
            marker = ">" if peak_index == len(values) - 1 else "o"
            axes[row, 1].scatter(t[peak_index], values[peak_index], color=colors[label], marker=marker, s=18, zorder=3)
        axes[row, 1].set_ylabel("Volume\n($10^3$ m$^3$)")
        axes[row, 1].grid(alpha=0.2)
    axes[0, 0].set_title("Domain maximum depth")
    axes[0, 1].set_title("Active-cell surface-water volume")
    axes[-1, 0].set_xlabel("Time (h)")
    axes[-1, 1].set_xlabel("Time (h)")
    axes[0, 0].legend(fontsize=7, ncol=3)
    add_panel_labels([axes[0, 0], axes[0, 1]], x=-0.11, y=1.03)
    save(fig, "fig04_five_event_hydrographs.png")


def figure5_effective_loss() -> None:
    active = active_mask()
    event = "event68"
    arrays = load_event(event)
    rates = [1, 2, 3, 4, 5]
    connected_depths = {}
    continuity = []
    for rate in rates:
        run = ROOT / "extended_study" / "output" / f"connected_itzi_swmm_inf_{rate}mmh"
        z = np.load(run / event / f"{event}_connected_itzi_swmm.npz", allow_pickle=True)
        connected_depths[rate] = z["h_swmm_connected"].astype(np.float32)
        rows = read_csv(run / "connected_itzi_swmm_metrics.csv")
        row = next(item for item in rows if item["event"] == event)
        continuity.append(float(row["continuity_error_pct"]))
    t = np.arange(1, 73) * DT_H
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.65), constrained_layout=True)
    axes[0].plot(t, maximum_depth(arrays["mike"], active), color="#111827", lw=1.3, label="MIKE")
    for rate, color in zip(rates, plt.get_cmap("viridis")(np.linspace(0.15, 0.9, len(rates)))):
        axes[0].plot(t, maximum_depth(connected_depths[rate], active), color=color, lw=1.0, label=f"{rate}")
        axes[1].plot(t, volume(connected_depths[rate], active) / 1000.0, color=color, lw=1.0)
    axes[0].set_title("Maximum depth")
    axes[0].set_ylabel("m")
    axes[0].legend(title="Effective loss\n(mm h$^{-1}$)", fontsize=6.3, title_fontsize=6.3, ncol=2)
    axes[1].plot(t, volume(arrays["mike"], active) / 1000.0, color="#111827", lw=1.3)
    axes[1].set_title("Active-cell volume")
    axes[1].set_ylabel("$10^3$ m$^3$")
    axes[2].plot(rates, continuity, marker="o", color="#2563eb")
    axes[2].axhline(0, color="#111827", lw=0.7)
    axes[2].axhline(2, color="#6b7280", ls="--", lw=0.7)
    axes[2].axhline(-2, color="#6b7280", ls="--", lw=0.7)
    axes[2].set_title("SWMM continuity error")
    axes[2].set_ylabel("%")
    for ax in axes:
        ax.set_xlabel("Time (h)" if ax is not axes[2] else "Effective loss (mm h$^{-1}$)")
        ax.grid(alpha=0.22)
    add_panel_labels(axes, x=-0.15, y=1.04)
    save(fig, "fig05_effective_loss_sensitivity.png")


def figure6_mike() -> None:
    rows = read_csv(EXP / "metrics" / "mike_configuration_screening_comparison.csv")
    models = ["ITZI surface-only", "ITZI-SWMM coupled", "base_no_xy", "no_xy_all_static_count", "all_static_xy"]
    display = ["Itzï surface-only", "ITZI-SWMM coupled", "DrainLite base", "DrainLite static + counts", "DrainLite static + coordinates"]
    colors = ["#2563eb", "#dc2626", "#6b7280", "#16a34a", "#f97316"]
    metrics = [
        ("mae_mm", "Full-sequence MAE (mm)"),
        ("global_peak_abs_error_mm", "Absolute global-peak error (mm)"),
        ("final_volume_abs_error_m3", "Absolute final-volume error ($10^3$ m$^3$)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.9), constrained_layout=True)
    for ax, (key, ylabel) in zip(axes, metrics):
        for index, (model, color) in enumerate(zip(models, colors)):
            values = np.array([float(row[key]) for row in rows if row["model"] == model])
            if key.endswith("m3"):
                values = values / 1000.0
            ax.bar(index, values.mean(), color=color, alpha=0.78)
            ax.scatter(np.full(values.size, index), values, color="#111827", s=10, zorder=3)
        ax.set_xticks(range(len(display)))
        ax.set_xticklabels(display, rotation=52, ha="right", fontsize=6.5)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.22)
    axes[0].set_title("Pixel-time discrepancy")
    axes[1].set_title("Peak magnitude")
    axes[2].set_title("Final stored water")
    add_panel_labels(axes, x=-0.15, y=1.03)
    fig.suptitle("Comparison with the MIKE water-depth reference", fontsize=10)
    save(fig, "fig06_mike_metric_dependent_comparison.png")


def figure7_event_spatial(event: str = "event68") -> None:
    arrays = load_event(event)
    pred = np.load(EXP / "predictions" / event / "h_no_xy_all_static.npy")
    active = active_mask()
    target_curve = maximum_depth(arrays["official"], active)
    time_index = int(np.argmax(target_curve))
    surface = arrays["surface"][time_index]
    target = arrays["official"][time_index]
    mike = arrays["mike"][time_index]
    prediction = pred[time_index]
    true_residual = target - surface
    pred_residual = prediction - surface
    error = pred_residual - true_residual
    vmax = float(np.quantile(np.concatenate([surface[active], target[active], prediction[active], mike[active]]), 0.999))
    rlim = float(np.quantile(np.abs(np.concatenate([true_residual[active], pred_residual[active], error[active]])), 0.995))
    fig, axes = plt.subplots(2, 4, figsize=(7.1, 4.1))
    panels = [
        ("Surface-only", surface, "viridis", 0, vmax),
        ("ITZI-SWMM target", target, "viridis", 0, vmax),
        ("DrainLite prediction", prediction, "viridis", 0, vmax),
        ("MIKE at same time", mike, "viridis", 0, vmax),
        ("True signed residual", true_residual, "coolwarm", -rlim, rlim),
        ("Predicted signed residual", pred_residual, "coolwarm", -rlim, rlim),
        ("Residual prediction error", error, "coolwarm", -rlim, rlim),
    ]
    top_image = None
    bottom_image = None
    for index, (ax, (title, data, cmap, vmin, vmax_i)) in enumerate(zip(axes.ravel(), panels)):
        im = ax.imshow(np.ma.array(data, mask=~active), cmap=cmap, vmin=vmin, vmax=vmax_i, aspect="auto")
        ax.set_title(title, fontsize=7.8)
        ax.set_xticks([])
        ax.set_yticks([])
        if index < 4:
            top_image = im
        else:
            bottom_image = im
    axes.ravel()[-1].axis("off")
    axes.ravel()[-1].text(0.02, 0.78, f"Event: {event}\nTarget-conditioned time: {(time_index + 1) * DT_H:.2f} h\nResidual = coupled depth - surface-only depth", va="top", fontsize=7.5)
    fig.subplots_adjust(left=0.025, right=0.94, top=0.93, bottom=0.05, wspace=0.18, hspace=0.28)
    fig.colorbar(top_image, ax=axes[0, :], location="right", fraction=0.018, pad=0.012, label="Water depth (m)")
    fig.colorbar(bottom_image, ax=axes[1, :3], location="bottom", fraction=0.045, pad=0.08, label="Residual (m)")
    add_panel_labels(axes.ravel()[:7], x=-0.07, y=1.02)
    save(fig, "fig07_event68_target_conditioned_spatial.png")


def figure8_residual_grid() -> None:
    active = active_mask()
    fig, axes = plt.subplots(len(EVENTS), 3, figsize=(7.1, 8.1), constrained_layout=True)
    for row, event in enumerate(EVENTS):
        arrays = load_event(event)
        pred = np.load(EXP / "predictions" / event / "h_no_xy_all_static.npy")
        time_index = int(np.argmax(maximum_depth(arrays["official"], active)))
        true = arrays["official"][time_index] - arrays["surface"][time_index]
        estimated = pred[time_index] - arrays["surface"][time_index]
        error = estimated - true
        limit = max(0.005, float(np.quantile(np.abs(np.concatenate([true[active], estimated[active], error[active]])), 0.995)))
        im = None
        for col, (data, title) in enumerate([(true, "True residual"), (estimated, "Predicted residual"), (error, "Residual error")]):
            im = axes[row, col].imshow(np.ma.array(data, mask=~active), cmap="coolwarm", vmin=-limit, vmax=limit)
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            if row == 0:
                axes[row, col].set_title(title)
            if col == 0:
                axes[row, col].set_ylabel(f"{event}\n{(time_index + 1) * DT_H:.2f} h")
        fig.colorbar(im, ax=axes[row, :], fraction=0.018, pad=0.012, label="Residual (m)")
    save(fig, "fig08_five_event_signed_residual_maps.png")


def figure9_ablation() -> None:
    rows = read_csv(EXP / "metrics" / "event_metrics.csv")
    selected = [
        ("surface_only", "Surface-only"),
        ("base_no_xy", "Base\nno XY"),
        ("base_xy", "Base\n+ XY"),
        ("no_xy_masks", "Masks\nno XY"),
        ("no_xy_hydraulic", "Hydraulic\nno XY"),
        ("no_xy_all_static", "All static\nno XY"),
        ("all_static_xy", "All static\n+ XY"),
    ]
    controls = [
        ("no_xy_all_static", "Aligned signed"),
        ("no_xy_all_static_shift", "Shifted 20 m"),
        ("no_xy_all_static_shuffle", "Shuffled"),
        ("sink_only_no_xy_all_static_count", "Sink-only"),
    ]
    lookup = defaultdict(list)
    event_lookup = defaultdict(dict)
    for row in rows:
        value = float(row["mae_m"]) * 1000.0
        lookup[row["model"]].append(value)
        event_lookup[row["model"]][row["event"]] = value
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.15), constrained_layout=True)
    for index, (model, label) in enumerate(selected):
        values = np.array(lookup[model])
        axes[0].bar(index, values.mean(), color="#2563eb", alpha=0.8)
        axes[0].scatter(np.full(values.size, index), values, color="#111827", s=11, zorder=3)
    axes[0].set_xticks(range(len(selected)))
    axes[0].set_xticklabels([label for _, label in selected], fontsize=7)
    axes[0].set_ylabel("MAE to coupled label (mm)")
    axes[0].set_title("Feature and coordinate ablation")
    axes[0].grid(axis="y", alpha=0.22)
    colors = ["#16a34a", "#f97316", "#7c3aed", "#6b7280"]
    for (model, label), color in zip(controls, colors):
        values = [event_lookup[model][event] for event in EVENTS]
        axes[1].plot(EVENTS, values, marker="o", lw=1.2, label=label, color=color)
    axes[1].set_ylabel("Event MAE (mm)")
    axes[1].set_title("Spatial controls and target form")
    axes[1].tick_params(axis="x", rotation=28)
    axes[1].grid(alpha=0.22)
    axes[1].legend(fontsize=6.8)
    add_panel_labels(axes, x=-0.13, y=1.03)
    save(fig, "fig09_ablation_event_dots.png")


def figure10_conditional() -> None:
    rows = read_csv(EXP / "metrics" / "conditional_metrics.csv")
    models = ["surface_only", "base_no_xy", "no_xy_all_static_count", "sink_only_no_xy_all_static_count"]
    model_labels = ["Surface-only", "Base, no XY", "Signed static, no XY", "Sink-only, no XY"]
    conditions = ["all_active", "target_severe_30mm", "residual_active_1mm", "negative_residual_1mm", "positive_residual_1mm", "network_0_20m", "network_over_140m"]
    labels = ["All", "Target >30 mm", "|Residual| >1 mm", "Residual <-1 mm", "Residual >1 mm", "0-20 m to network", ">140 m to network"]
    colors = ["#2563eb", "#6b7280", "#16a34a", "#f97316"]
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["model"], row["condition"])].append(float(row["mae_mm"]))
    fig, ax = plt.subplots(figsize=(7.1, 3.5), constrained_layout=True)
    x = np.arange(len(conditions))
    width = 0.2
    for idx, (model, color, model_label) in enumerate(zip(models, colors, model_labels)):
        means = [np.mean(grouped[(model, condition)]) for condition in conditions]
        ax.bar(x + (idx - 1.5) * width, means, width=width, label=model_label, color=color, alpha=0.82)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=34, ha="right", fontsize=7.5)
    ax.set_ylabel("Conditional MAE (mm)")
    ax.set_title("Conditional residual-model performance")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(fontsize=6.8, ncol=2)
    save(fig, "fig10_conditional_performance.png")


def figure11_importance() -> None:
    rows = read_csv(EXP / "metrics" / "cross_fold_permutation_importance.csv")
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["feature"]].append(float(row["importance_mae_mm"]))
    ranked = sorted(grouped, key=lambda feature: np.mean(grouped[feature]), reverse=True)[:15][::-1]
    labels = {
        "surface_depth_mm": "Surface depth",
        "dem_m": "Elevation",
        "surface_depth_3x3_mean_mm": "3 x 3 mean depth",
        "surface_depth_7x7_mean_mm": "7 x 7 mean depth",
        "distance_to_outfall": "Distance to outfall",
        "pipe_density_7x7": "7 x 7 pipe density",
        "pipe_density_3x3": "3 x 3 pipe density",
        "capacity_density_7x7": "7 x 7 capacity density",
        "drain_inlet_mask": "Inlet mask",
        "inlet_density_7x7": "7 x 7 inlet density",
        "cumsum_rainfall_mm": "Cumulative rainfall",
        "rainfall_mm": "Current rainfall",
        "pipe_cover_depth": "Pipe cover depth",
        "dem_slope": "Terrain slope",
        "time_norm": "Time within event",
    }
    means = [np.mean(grouped[feature]) for feature in ranked]
    stds = [np.std(grouped[feature], ddof=1) for feature in ranked]
    fig, ax = plt.subplots(figsize=(7.1, 4.15), constrained_layout=True)
    positions = np.arange(len(ranked))
    ax.barh(positions, means, xerr=stds, color="#2563eb", alpha=0.82, capsize=2)
    for position, feature in zip(positions, ranked):
        ax.scatter(grouped[feature], np.full(len(grouped[feature]), position), color="#111827", s=10, zorder=3)
    ax.set_yticks(positions)
    ax.set_yticklabels([labels.get(feature, feature.replace("_", " ")) for feature in ranked], fontsize=8)
    ax.set_xlabel("Held-event permutation increase in MAE (mm)")
    ax.set_title("Cross-fold permutation importance")
    ax.grid(axis="x", alpha=0.22)
    save(fig, "fig11_cross_fold_importance.png")


def main() -> int:
    FIG.mkdir(parents=True, exist_ok=True)
    ensure_main_predictions()
    figure1_workflow()
    figure2_network()
    figure4_hydrographs()
    figure5_effective_loss()
    figure6_mike()
    figure7_event_spatial()
    figure8_residual_grid()
    figure9_ablation()
    figure10_conditional()
    figure11_importance()
    print("\n".join(str(path) for path in sorted(FIG.glob("*.png"))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
