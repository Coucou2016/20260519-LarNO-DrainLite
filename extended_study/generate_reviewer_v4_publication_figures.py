#!/usr/bin/env python3
"""Generate journal figures for the fixed-network DrainLite major revision."""

from __future__ import annotations

import csv
import shutil
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

from publication_plot_style import add_panel_labels, configure_publication_style


configure_publication_style()
ROOT = Path(__file__).resolve().parents[1]
V3 = ROOT / "extended_study" / "output" / "reviewer_major_revision_v3"
V4 = ROOT / "extended_study" / "output" / "reviewer_major_revision_v4"
EXP = V4 / "final_hybrid_controls"
OUT = V4 / "submission_package_v4"
FIG = OUT / "figures"
FLOOD = ROOT / "LarNO-main" / "benchmark" / "urbanflood" / "flood" / "region1_20m_drainage_v3_full"
GEO = ROOT / "LarNO-main" / "benchmark" / "urbanflood" / "geodata" / "region1_20m_drainage_v3_full"
PHYS = V3 / "formal_matched_full" / "physics_quality.csv"
EVENTS = ["event1", "event20", "event65", "event66", "event67", "event68", "event69", "event70"]
CANONICAL_SEED = 1907
CELL_AREA_M2 = 400.0


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def save(fig: plt.Figure, name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(FIG / f"{name}.{suffix}", dpi=400 if suffix == "png" else None)
    plt.close(fig)


def copy_v3(old_name: str, new_name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    source = V3 / "submission_package_v3" / "figures"
    for suffix in ("png", "pdf", "svg"):
        shutil.copy2(source / f"{old_name}.{suffix}", FIG / f"{new_name}.{suffix}")


def macro(rows: list[dict[str, str]], model: str, reference: str, metric: str, seed: int = CANONICAL_SEED) -> float:
    values = [
        float(row[metric]) for row in rows
        if row["model"] == model and row["reference"] == reference and int(row["seed"]) == seed
    ]
    return float(np.mean(values))


def fig01_workflow() -> None:
    fig, ax = plt.subplots(figsize=(7.1, 4.15), constrained_layout=True)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    boxes = [
        (0.03, 0.72, 0.17, 0.14, "Rainfall, terrain\nand buildings", "#e8eef5"),
        (0.24, 0.72, 0.17, 0.14, "B: matched\nsurface-only Itzï", "#f1f1f1"),
        (0.45, 0.72, 0.17, 0.14, "C: native\nItzï-SWMM coupling", "#e1f2e7"),
        (0.66, 0.72, 0.14, 0.14, "Signed target\nr = C - B", "#fde8e7"),
        (0.83, 0.72, 0.14, 0.14, "Labels split\nby event", "#f9edda"),
        (0.04, 0.42, 0.24, 0.12, "Training-event prior\n(own event excluded)", "#eee8f7"),
        (0.04, 0.25, 0.24, 0.12, "Current B state,\nrainfall and terrain", "#e4eff8"),
        (0.04, 0.08, 0.24, 0.12, "Fixed-network masks\nand hydraulic fields", "#e8f3ee"),
        (0.40, 0.21, 0.20, 0.22, "DrainLite\nresidual estimator", "#e8eef5"),
        (0.72, 0.21, 0.23, 0.22, "Predicted residual r-hat\nh = max(B + r-hat, 0)", "#f9edda"),
        (0.73, 0.51, 0.21, 0.10, "Training: fit loss\nHeld event: score only", "#fde8e7"),
    ]
    for x, y, w, h, label, colour in boxes:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.007", facecolor=colour, edgecolor="#374151", lw=0.75))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=7.2)
    for x0, x1 in ((0.20, 0.24), (0.41, 0.45), (0.62, 0.66), (0.80, 0.83)):
        ax.add_patch(FancyArrowPatch((x0, 0.79), (x1, 0.79), arrowstyle="-|>", mutation_scale=9, lw=0.85))
    for y0, y1 in ((0.48, 0.38), (0.31, 0.32), (0.14, 0.26)):
        ax.add_patch(FancyArrowPatch((0.28, y0), (0.40, y1), arrowstyle="-|>", mutation_scale=9, lw=0.85))
    ax.add_patch(FancyArrowPatch((0.60, 0.32), (0.72, 0.32), arrowstyle="-|>", mutation_scale=9, lw=0.85))
    ax.add_patch(FancyArrowPatch((0.835, 0.72), (0.835, 0.61), arrowstyle="-|>", mutation_scale=9, lw=0.85, linestyle="--"))
    ax.add_patch(FancyArrowPatch((0.835, 0.43), (0.835, 0.51), arrowstyle="-|>", mutation_scale=9, lw=0.85, linestyle="--"))
    ax.text(0.03, 0.94, "Fixed-network residual emulation: whole-event separation", fontsize=9.5, fontweight="bold")
    ax.text(0.03, 0.62, "B and C share surface settings.\nLabels supervise fitting or score held events.", fontsize=7.2)
    ax.text(0.39, 0.10, "MIKE: external comparison only.\nPublic LarNO checkpoint: separate reproduction.\nInference uses the current B field.", fontsize=7.0)
    save(fig, "fig01_workflow")


def fig06_residual_maps() -> None:
    active = np.load(GEO / "active_mask.npy").astype(bool)
    selected = ["event1", "event20", "event68", "event70"]
    fig, axes = plt.subplots(4, 4, figsize=(7.1, 8.4), constrained_layout=True)
    for row, event in enumerate(selected):
        directory = FLOOD / event
        surface = np.load(directory / "h_itzi_surface_matched.npy", mmap_mode="r")
        coupled = np.load(directory / "h_itzi_swmm.npy", mmap_mode="r")
        pred = np.load(EXP / "predictions" / event / "h_hybrid_all.npy", mmap_mode="r")
        volume = np.asarray(coupled[:, active]).sum(axis=1, dtype=np.float64)
        peak_t = int(np.argmax(volume))
        fields = [
            np.asarray(surface[peak_t]),
            np.asarray(coupled[peak_t] - surface[peak_t]) * 1000.0,
            np.asarray(pred[peak_t] - surface[peak_t]) * 1000.0,
            np.asarray(pred[peak_t] - coupled[peak_t]) * 1000.0,
        ]
        depth_max = max(float(np.quantile(fields[0][active], 0.995)), 0.03)
        residual_bound = max(float(np.quantile(np.abs(np.concatenate([fields[1][active], fields[2][active], fields[3][active]])), 0.995)), 5.0)
        for col, field in enumerate(fields):
            shown = np.ma.masked_where(~active, field)
            if col == 0:
                im = axes[row, col].imshow(shown, origin="upper", cmap="Blues", vmin=0, vmax=depth_max)
                label = "Surface depth (m)"
            else:
                im = axes[row, col].imshow(shown, origin="upper", cmap="RdBu_r", vmin=-residual_bound, vmax=residual_bound)
                label = "Signed residual (mm)"
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            if row == 0:
                axes[row, col].set_title(["Input B", "True C-B", "Predicted residual", "Depth error"][col])
            if col == 0:
                axes[row, col].set_ylabel(f"{event}\n{(peak_t + 1) * 5 / 60:.2f} h")
            if col in (0, 3):
                cbar = fig.colorbar(im, ax=axes[row, col], fraction=0.045, pad=0.02)
                cbar.set_label(label, fontsize=6.5)
    add_panel_labels(axes.flat, x=-0.05, y=1.01)
    save(fig, "fig06_fixed_network_residual_maps")


def fig07_skill_and_seeds() -> None:
    rows = read_csv(EXP / "metrics" / "event_metrics_all_seeds.csv")
    summary = read_csv(EXP / "metrics" / "model_summary_mean_sd.csv")
    order = ["surface_matched", "prior_only", "dynamic_no_prior", "hybrid_dynamic", "hybrid_mask", "hybrid_hydraulic", "hybrid_all"]
    labels = ["Surface B", "ST prior", "Dynamic\n(no prior)", "Prior +\ndynamic", "+ masks", "+ hydraulics", "+ all network"]
    colours = ["#777777", "#8b5cf6", "#e69f00", "#56b4e9", "#5f9ed1", "#cc79a7", "#009e73"]
    lookup = {(row["reference"], row["model"]): row for row in summary}
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 3.25), constrained_layout=True)
    for ax, metric, ylabel in [(axes[0], "mae_mm", "MAE (mm)"), (axes[1], "rmse_mm", "RMSE (mm)"), (axes[2], "csi_0p15", "CSI at 0.15 m")]:
        means = [float(lookup[("coupled_label", name)][f"{metric}_mean"]) for name in order]
        errors = [float(lookup[("coupled_label", name)][f"{metric}_sd"]) for name in order]
        ax.bar(np.arange(len(order)), means, yerr=errors, color=colours, width=0.72, capsize=2, error_kw={"lw": 0.7})
        ax.set_xticks(np.arange(len(order)))
        ax.set_xticklabels(labels, rotation=42, ha="right", fontsize=6.2)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.22)
    for event in EVENTS:
        values = [
            float(next(row["mae_mm"] for row in rows if row["event"] == event and row["model"] == model and row["reference"] == "coupled_label" and int(row["seed"]) == CANONICAL_SEED))
            for model in ["prior_only", "hybrid_dynamic", "hybrid_all"]
        ]
        axes[0].plot([1, 3, 6], values, color="#333333", alpha=0.26, lw=0.55, marker="o", ms=2)
    axes[0].set_title("Whole-event absolute error")
    axes[1].set_title("Large-error sensitivity")
    axes[2].set_title("Inundation detection")
    add_panel_labels(axes, x=-0.17, y=1.04)
    save(fig, "fig07_skill_and_sampling_seeds")


def fig08_final_hybrid_controls() -> None:
    rows = read_csv(EXP / "metrics" / "final_hybrid_network_controls.csv")
    events = read_csv(EXP / "metrics" / "event_metrics_all_seeds.csv")
    summary = read_csv(EXP / "metrics" / "model_summary_mean_sd.csv")
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 5.25), constrained_layout=True)
    shifts: dict[int, list[float]] = defaultdict(list)
    control_groups: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        name = row["control"]
        delta = float(row["delta_mae_vs_aligned_mm"])
        if name.startswith("shift_"):
            shifts[int(name.split("_")[1].replace("m", ""))].append(delta)
        else:
            control_groups[name].append(delta)
    distances = sorted(shifts)
    means = np.array([np.mean(shifts[value]) for value in distances])
    sds = np.array([np.std(shifts[value], ddof=1) for value in distances])
    axes[0, 0].errorbar(distances, means, yerr=sds, marker="o", capsize=2, color="#0072b2")
    axes[0, 0].axhline(0, color="#444444", lw=0.7)
    axes[0, 0].set_xlabel("Network displacement (m)")
    axes[0, 0].set_ylabel("MAE penalty (mm)")
    axes[0, 0].set_title("Final-hybrid spatial displacement")
    names = ["zero_static_network_fields", "block_shuffle", "group_permutation"]
    labels = ["Zero fields", "Block shuffle", "Group permutation"]
    axes[0, 1].bar(labels, [np.mean(control_groups[name]) for name in names], yerr=[np.std(control_groups[name], ddof=1) if len(control_groups[name]) > 1 else 0 for name in names], color=["#999999", "#d55e00", "#cc79a7"], capsize=2)
    axes[0, 1].axhline(0, color="#444444", lw=0.7)
    axes[0, 1].tick_params(axis="x", rotation=24)
    axes[0, 1].set_ylabel("MAE penalty (mm)")
    axes[0, 1].set_title("Final-hybrid destructive controls")
    subgroup = ["hybrid_dynamic", "hybrid_mask", "hybrid_hydraulic", "hybrid_all"]
    subgroup_labels = ["Dynamic", "+ masks", "+ hydraulics", "+ all"]
    canonical = [macro(events, name, "coupled_label", "mae_mm") for name in subgroup]
    improvement = canonical[0] - np.asarray(canonical)
    axes[1, 0].bar(subgroup_labels, improvement, color=["#56b4e9", "#5f9ed1", "#cc79a7", "#009e73"])
    axes[1, 0].axhline(0, color="#444444", lw=0.7)
    axes[1, 0].set_ylabel("MAE improvement vs dynamic (mm)")
    axes[1, 0].set_title("Network-group ablation")
    summary_lookup = {(row["reference"], row["model"]): row for row in summary}
    delta_by_seed = []
    seeds = sorted({int(row["seed"]) for row in events})
    for seed in seeds:
        dyn = macro(events, "hybrid_dynamic", "coupled_label", "mae_mm", seed)
        all_network = macro(events, "hybrid_all", "coupled_label", "mae_mm", seed)
        delta_by_seed.append(dyn - all_network)
    axes[1, 1].bar([str(seed) for seed in seeds], delta_by_seed, color="#009e73")
    axes[1, 1].axhline(0, color="#444444", lw=0.7)
    axes[1, 1].set_xlabel("Spatial sampling seed")
    axes[1, 1].set_ylabel("Network increment (mm)")
    axes[1, 1].set_title("Seed-to-seed stability")
    for ax in axes.flat:
        ax.grid(axis="y", alpha=0.22)
    add_panel_labels(axes.flat, x=-0.14, y=1.04)
    save(fig, "fig08_final_hybrid_network_controls")


def fig09_conditional_metrics() -> None:
    rows = read_csv(EXP / "metrics" / "conditional_metrics_canonical_seed.csv")
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(row["model"], row["condition"])].append(float(row["mae_mm"]))
    models = ["surface_matched", "prior_only", "hybrid_dynamic", "hybrid_all"]
    labels = ["Surface B", "ST prior", "Prior + dynamic", "Full hybrid"]
    colours = ["#777777", "#8b5cf6", "#56b4e9", "#009e73"]
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 3.25), constrained_layout=True)
    condition_sets = [
        (["all_active", "target_wet_0p03m", "target_wet_0p15m"], ["All active", "Wet >0.03 m", "Wet >0.15 m"], "Hydraulic state"),
        (["effect_abs_over_1mm", "effect_abs_over_5mm", "effect_abs_over_10mm"], ["|C-B| >1 mm", ">5 mm", ">10 mm"], "Drainage-effect magnitude"),
        (["near_inlet_0_20m", "near_inlet_20_60m", "near_inlet_60_100m", "near_inlet_100_infm"], ["0-20", "20-60", "60-100", ">100"], "Distance to inlet (m)"),
    ]
    for ax, (conditions, ticklabels, title) in zip(axes, condition_sets):
        x = np.arange(len(conditions))
        width = 0.19
        for index, (model, label, colour) in enumerate(zip(models, labels, colours)):
            values = [np.mean(grouped[(model, condition)]) for condition in conditions]
            ax.bar(x + (index - 1.5) * width, values, width, label=label, color=colour)
        ax.set_xticks(x)
        ax.set_xticklabels(ticklabels, rotation=25, ha="right", fontsize=6.4)
        ax.set_ylabel("Conditional MAE (mm)")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.22)
    axes[0].legend(fontsize=6.2, ncol=2)
    add_panel_labels(axes, x=-0.18, y=1.04)
    save(fig, "fig09_conditioned_performance")


def fig10_mike_tradeoff() -> None:
    event_rows = read_csv(EXP / "metrics" / "event_metrics_all_seeds.csv")
    physics = read_csv(PHYS)
    models = ["Surface B", "Coupled C", "DrainLite"]
    fields = [
        ("mae_mm", "MAE (mm)"),
        ("peak_map_mae_mm", "Peak-map MAE (mm)"),
        ("final_volume_abs_error_m3", "Final-volume error ($10^3$ m$^3$)"),
        ("csi_0p15", "CSI at 0.15 m"),
    ]
    b_map = {"mae_mm": "B_vs_MIKE_mae_mm", "peak_map_mae_mm": "B_vs_MIKE_peak_map_mae_mm", "final_volume_abs_error_m3": "B_vs_MIKE_final_volume_abs_error_m3", "csi_0p15": "B_vs_MIKE_csi_0p15"}
    c_map = {key: value.replace("B_", "C_") for key, value in b_map.items()}
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 5.1), constrained_layout=True)
    for ax, (metric, ylabel) in zip(axes.flat, fields):
        b = np.mean([float(row[b_map[metric]]) for row in physics])
        c = np.mean([float(row[c_map[metric]]) for row in physics])
        d = macro(event_rows, "hybrid_all", "mike_external", metric)
        values = np.asarray([b, c, d])
        if metric == "final_volume_abs_error_m3":
            values /= 1000.0
        ax.bar(models, values, color=["#777777", "#d55e00", "#0072b2"])
        ax.set_ylabel(ylabel)
        ax.set_title({"mae_mm": "Full space-time field", "peak_map_mae_mm": "Cell-wise maxima", "final_volume_abs_error_m3": "End-of-event storage", "csi_0p15": "Flood-extent overlap"}[metric])
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.22)
    add_panel_labels(axes.flat, x=-0.14, y=1.04)
    save(fig, "fig10_mike_metric_tradeoff")


def fig11_runtime_clipping_selection() -> None:
    runtime = read_csv(EXP / "metrics" / "runtime_uncached_canonical.csv")
    clipping = read_csv(EXP / "metrics" / "clipping_sensitivity.csv")
    inventory = read_csv(EXP / "metrics" / "event_selection_inventory.csv")
    hybrid_runtime = [row for row in runtime if row["model"] == "hybrid_all" and int(row["seed"]) == CANONICAL_SEED]
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 5.2), constrained_layout=True)
    components = ["feature_seconds", "assembly_seconds", "model_seconds", "postprocess_seconds"]
    labels = ["Feature", "Assembly", "Estimator", "Post-process"]
    values = [np.mean([float(row[name]) for row in hybrid_runtime]) for name in components]
    axes[0, 0].bar(labels, values, color=["#56b4e9", "#999999", "#0072b2", "#009e73"])
    axes[0, 0].set_ylabel("Time per event (s)")
    axes[0, 0].set_title("Measured correction-time components")
    axes[0, 0].tick_params(axis="x", rotation=20)
    b = np.mean([float(row["surface_runtime_s"]) for row in hybrid_runtime])
    total = np.mean([float(row["surface_plus_correction_seconds"]) for row in hybrid_runtime])
    c = np.mean([float(row["coupled_runtime_s"]) for row in hybrid_runtime])
    axes[0, 1].bar(["Surface B", "B + DrainLite", "Coupled C"], [b, total, c], color=["#777777", "#0072b2", "#d55e00"])
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_ylabel("Time per event (s, log scale)")
    axes[0, 1].set_title("End-to-end deployment comparison")
    axes[0, 1].tick_params(axis="x", rotation=20)
    clip_groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in clipping:
        clip_groups[str(row["clip_mm"])].append(row)
    clip_order = [name for name in ["none", "500.0", "1200.0"] if name in clip_groups]
    axes[1, 0].bar(clip_order, [np.mean([float(row["mae_mm"]) for row in clip_groups[name]]) for name in clip_order], color=["#0072b2", "#e69f00", "#cc79a7"])
    axes[1, 0].set_xlabel("Residual clipping bound (mm)")
    axes[1, 0].set_ylabel("Coupled-label MAE (mm)")
    axes[1, 0].set_title("Post-processing sensitivity")
    valid = [row for row in inventory if row["public_arrays_valid"].lower() == "true"]
    for used, colour, label in [(False, "#aaaaaa", "Public only"), (True, "#d55e00", "Paired A/B/C")]:
        subset = [row for row in valid if (row["used_in_paired_v3"].lower() == "true") == used]
        axes[1, 1].scatter([float(row["mean_6h_rainfall_mm_active"]) for row in subset], [float(row["mike_global_peak_m_active"]) for row in subset], s=16 if used else 9, alpha=0.8, color=colour, label=label)
    axes[1, 1].set_xlabel("Mean six-hour rainfall (mm)")
    axes[1, 1].set_ylabel("MIKE global peak (m)")
    axes[1, 1].set_title("Event-inclusion context")
    axes[1, 1].legend(fontsize=6.5)
    for ax in axes.flat:
        ax.grid(axis="y", alpha=0.22)
    add_panel_labels(axes.flat, x=-0.14, y=1.04)
    save(fig, "fig11_runtime_clipping_event_selection")


def main() -> int:
    fig01_workflow()
    copy_v3("fig02_network_audit", "fig02_network_audit")
    copy_v3("fig03_physical_quality", "fig03_physical_quality")
    copy_v3("fig04_eight_event_hydrographs", "fig04_eight_event_hydrographs")
    copy_v3("fig05_event68_physical_maps", "fig05_event68_physical_maps")
    fig06_residual_maps()
    fig07_skill_and_seeds()
    fig08_final_hybrid_controls()
    fig09_conditional_metrics()
    fig10_mike_tradeoff()
    fig11_runtime_clipping_selection()
    copy_v3("fig10_larno_reproduction", "fig12_larno_reproduction")
    fig13_physical_sensitivity()
    print(FIG)
    return 0


def fig13_physical_sensitivity() -> None:
    rows = read_csv(V4 / "physical_sensitivity_event68/physical_sensitivity_metrics.csv")
    labels = ["Baseline", "Loss 0", "Loss 2", "Nearest", "Exclude", "n=0.015"]
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 5.3), constrained_layout=True)
    panels = [
        ("drainage_effect_mae_mm", "Mean |C-B| (mm)", 1.0),
        ("final_surface_reduction_m3", "Final B-C volume ($10^3$ m$^3$)", 0.001),
        ("C_vs_MIKE_mae_mm", "C vs MIKE MAE (mm)", 1.0),
        ("routing_continuity_error_pct", "Routing continuity error (%)", 1.0),
    ]
    for ax, (field, label, scale) in zip(axes.flat, panels):
        ax.bar(labels, [float(row[field]) * scale for row in rows], color=["#777777", "#56b4e9", "#0072b2", "#e69f00", "#cc79a7", "#009e73"])
        ax.set_ylabel(label)
        ax.tick_params(axis="x", rotation=35)
        ax.grid(axis="y", alpha=0.2)
    add_panel_labels(axes.flat, x=-0.15, y=1.04)
    save(fig, "fig13_physical_sensitivity")


if __name__ == "__main__":
    raise SystemExit(main())
