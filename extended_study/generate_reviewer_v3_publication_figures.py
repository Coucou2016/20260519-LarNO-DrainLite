#!/usr/bin/env python3
"""Generate reviewer-v3 publication figures from verified local outputs."""

from __future__ import annotations

import csv
import json
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
REV = ROOT / "extended_study" / "output" / "reviewer_major_revision_v3"
PHYS = REV / "formal_matched_full"
DRAIN = REV / "drainlite_v3"
HYBRID = REV / "drainlite_hybrid_v3"
DATASET = "region1_20m_drainage_v3_full"
FLOOD = ROOT / "LarNO-main" / "benchmark" / "urbanflood" / "flood" / DATASET
GEO = ROOT / "LarNO-main" / "benchmark" / "urbanflood" / "geodata" / DATASET
OUT = REV / "submission_package_v3"
FIG = OUT / "figures"
EVENTS = ["event1", "event20", "event65", "event66", "event67", "event68", "event69", "event70"]
DT_H = 5.0 / 60.0
CELL_AREA_M2 = 400.0


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def save(fig: plt.Figure, name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(FIG / f"{name}.{suffix}", dpi=400 if suffix == "png" else None)
    plt.close(fig)


def arrays(event: str) -> dict[str, np.ndarray]:
    directory = FLOOD / event
    return {
        "A": np.load(directory / "h_itzi_surface.npy", mmap_mode="r"),
        "B": np.load(directory / "h_itzi_surface_matched.npy", mmap_mode="r"),
        "C": np.load(directory / "h_itzi_swmm.npy", mmap_mode="r"),
        "MIKE": np.load(directory / "h_mike_ref.npy", mmap_mode="r"),
        "rain": np.load(directory / "rainfall.npy", mmap_mode="r"),
    }


def volume(h: np.ndarray, active: np.ndarray) -> np.ndarray:
    return np.asarray(h[:, active]).sum(axis=1, dtype=np.float64) * CELL_AREA_M2


def fig01_workflow() -> None:
    fig, ax = plt.subplots(figsize=(7.1, 4.0), constrained_layout=True)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    boxes = [
        (0.03, 0.68, 0.16, 0.18, "Rainfall, DEM\nand buildings", "#e8eef5"),
        (0.23, 0.68, 0.16, 0.18, "A: surface-only\noriginal roughness", "#f1f1f1"),
        (0.43, 0.68, 0.16, 0.18, "B: surface-only\nmatched roughness", "#fff0db"),
        (0.63, 0.68, 0.16, 0.18, "C: native Itzï-SWMM\nbidirectional coupling", "#e1f2e7"),
        (0.81, 0.68, 0.16, 0.18, "Target residual\nr = C - B", "#fde8e7"),
        (0.05, 0.22, 0.18, 0.18, "Training-event\nspatiotemporal prior", "#eee8f7"),
        (0.28, 0.22, 0.18, 0.18, "Dynamic state\nrainfall, depth, terrain", "#e4eff8"),
        (0.51, 0.22, 0.18, 0.18, "Static network\nmasks and hydraulics", "#e8f3ee"),
        (0.74, 0.22, 0.20, 0.18, "DrainLite hybrid\n$h = \\max(B + \\hat{r}, 0)$", "#f9edda"),
    ]
    for x, y, w, h, label, colour in boxes:
        patch = FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.008",
            facecolor=colour, edgecolor="#374151", linewidth=0.75,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=7.5)
    for x0, x1 in [(0.19, 0.23), (0.39, 0.43), (0.59, 0.63), (0.79, 0.81)]:
        ax.add_patch(FancyArrowPatch((x0, 0.77), (x1, 0.77), arrowstyle="-|>", mutation_scale=9, lw=0.9))
    for x0, x1 in ((0.23, 0.28), (0.46, 0.51), (0.69, 0.74)):
        ax.add_patch(FancyArrowPatch((x0, 0.31), (x1, 0.31), arrowstyle="-|>", mutation_scale=9, lw=0.9))
    ax.add_patch(FancyArrowPatch((0.89, 0.68), (0.84, 0.40), arrowstyle="-|>", mutation_scale=9, lw=0.9))
    ax.text(0.03, 0.93, "Paired physical controls and leakage-controlled residual learning", fontsize=11, fontweight="bold")
    ax.text(0.03, 0.57, "Physical decomposition: B-A isolates the inlet-neighbourhood roughness change; C-B isolates sewer coupling.", fontsize=7.5)
    ax.text(0.03, 0.09, "MIKE fields are used only as an external descriptive reference. The public LarNO checkpoint is reproduced separately.", fontsize=7.5)
    save(fig, "fig01_workflow")


def fig02_network() -> None:
    source = REV / "figures" / "network" / "fig_network_audit"
    FIG.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf", "svg"):
        shutil.copy2(source.with_suffix(f".{suffix}"), FIG / f"fig02_network_audit.{suffix}")


def fig03_physical_quality() -> None:
    rows = read_csv(PHYS / "physics_quality.csv")
    labels = [row["event"].replace("event", "E") for row in rows]
    x = np.arange(len(rows))
    continuity = np.array([float(row["swmm_flow_routing_continuity_error_pct"]) for row in rows])
    nonconv = np.array([float(row["swmm_steps_not_converging_pct"]) for row in rows])
    rough = np.array([float(row["roughness_effect_mae_mm"]) for row in rows])
    drain = np.array([float(row["drainage_effect_mae_mm"]) for row in rows])
    final_b = np.array([float(row["matched_control_final_volume_m3"]) for row in rows]) / 1e6
    final_c = np.array([float(row["coupled_final_volume_m3"]) for row in rows]) / 1e6
    mike_error_b = np.array([float(row["B_vs_MIKE_final_volume_abs_error_m3"]) for row in rows]) / 1e6
    mike_error_c = np.array([float(row["C_vs_MIKE_final_volume_abs_error_m3"]) for row in rows]) / 1e6
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 5.4), constrained_layout=True)
    axes[0, 0].bar(x - 0.18, continuity, 0.36, label="Continuity error")
    axes[0, 0].bar(x + 0.18, nonconv, 0.36, label="Non-converging steps")
    axes[0, 0].axhline(0, color="black", lw=0.6)
    axes[0, 0].set_ylabel("Percent")
    axes[0, 0].set_title("SWMM numerical diagnostics")
    axes[0, 0].legend(ncol=2, fontsize=6.5)
    axes[0, 1].bar(x - 0.18, rough, 0.36, label="Roughness effect |B-A|")
    axes[0, 1].bar(x + 0.18, drain, 0.36, label="Drainage effect |C-B|")
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_ylabel("MAE (mm, log scale)")
    axes[0, 1].set_title("Matched physical decomposition")
    axes[0, 1].legend(fontsize=6.5)
    axes[1, 0].plot(x, final_b, "o-", label="B: matched surface")
    axes[1, 0].plot(x, final_c, "s-", label="C: coupled")
    axes[1, 0].set_ylabel("Final surface volume ($10^6$ m$^3$)")
    axes[1, 0].set_title("Stored surface water after 6 h")
    axes[1, 0].legend(fontsize=6.5)
    axes[1, 1].bar(x - 0.18, mike_error_b, 0.36, label="B vs MIKE")
    axes[1, 1].bar(x + 0.18, mike_error_c, 0.36, label="C vs MIKE")
    axes[1, 1].set_ylabel("Absolute volume error ($10^6$ m$^3$)")
    axes[1, 1].set_title("External final-volume comparison")
    axes[1, 1].legend(fontsize=6.5)
    for ax in axes.flat:
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.grid(axis="y", alpha=0.22)
    add_panel_labels(axes.flat, x=-0.13, y=1.03)
    save(fig, "fig03_physical_quality")


def fig04_hydrographs() -> None:
    active = np.load(GEO / "active_mask.npy").astype(bool)
    t = np.arange(1, 73) * DT_H
    fig, axes = plt.subplots(4, 2, figsize=(7.1, 8.4), sharex=True, constrained_layout=True)
    colours = {"B": "#6b7280", "C": "#d55e00", "MIKE": "#111111"}
    labels = {"B": "B: matched surface", "C": "C: Itzï-SWMM", "MIKE": "MIKE reference"}
    for ax, event in zip(axes.flat, EVENTS):
        data = arrays(event)
        for key in ("MIKE", "B", "C"):
            ax.plot(t, volume(data[key], active) / 1e6, label=labels[key], color=colours[key], lw=1.05)
        rain = np.asarray(data["rain"][:, active]).mean(axis=1) * 12.0
        twin = ax.twinx()
        twin.fill_between(t, 0, rain, step="mid", color="#56b4e9", alpha=0.16, linewidth=0)
        twin.set_ylim(max(rain.max() * 3.2, 1), 0)
        twin.tick_params(axis="y", colors="#2b6f9f", labelsize=6)
        twin.set_ylabel("Rain (mm h$^{-1}$)", color="#2b6f9f", fontsize=6.5)
        ax.set_title(event)
        ax.set_ylabel("Surface volume ($10^6$ m$^3$)")
        ax.grid(alpha=0.2)
    for ax in axes[-1]:
        ax.set_xlabel("Time (h)")
    axes[0, 0].legend(ncol=3, fontsize=6.4, loc="lower right")
    add_panel_labels(axes.flat, x=-0.13, y=1.02)
    save(fig, "fig04_eight_event_hydrographs")


def fig05_event68_maps() -> None:
    source = REV / "acceptance_event68_6h_full" / "event68" / "figures" / "fig_physics_peak_maps"
    FIG.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf", "svg"):
        path = source.with_suffix(f".{suffix}")
        if path.exists() and path.stat().st_size > 5000:
            shutil.copy2(path, FIG / f"fig05_event68_physical_maps.{suffix}")


def fig06_residual_maps() -> None:
    active = np.load(GEO / "active_mask.npy").astype(bool)
    selected = ["event1", "event20", "event68", "event70"]
    fig, axes = plt.subplots(len(selected), 3, figsize=(7.1, 8.2), constrained_layout=True)
    images = []
    for row, event in enumerate(selected):
        data = arrays(event)
        pred = np.load(HYBRID / "predictions" / event / "h_clim_all_static.npy", mmap_mode="r")
        true_res = np.asarray(data["C"] - data["B"])
        pred_res = np.asarray(pred - data["B"])
        peak_t = int(np.argmax(volume(data["C"], active)))
        fields = [true_res[peak_t], pred_res[peak_t], pred_res[peak_t] - true_res[peak_t]]
        bound = float(np.quantile(np.abs(np.concatenate([field[active] for field in fields[:2]])), 0.995))
        bound = max(bound, 0.005)
        for col, field in enumerate(fields):
            shown = np.ma.masked_where(~active, field * 1000.0)
            im = axes[row, col].imshow(shown, cmap="RdBu_r", vmin=-bound * 1000, vmax=bound * 1000, origin="upper")
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            if row == 0:
                axes[row, col].set_title(["Coupled residual C-B", "DrainLite hybrid", "Prediction error"][col])
            if col == 0:
                axes[row, col].set_ylabel(f"{event}\n$t$ = {(peak_t + 1) * DT_H:.2f} h")
            images.append(im)
        cbar = fig.colorbar(images[-1], ax=axes[row, :], fraction=0.022, pad=0.01)
        cbar.set_label("Signed depth residual (mm)")
    add_panel_labels(axes.flat, x=-0.06, y=1.01)
    save(fig, "fig06_hybrid_residual_maps")


def fig07_model_skill() -> None:
    original = read_csv(DRAIN / "metrics" / "summary_coupled.csv")
    hybrid = read_csv(HYBRID / "metrics" / "summary_coupled.csv")
    lookup = {row["model"]: row for row in original + hybrid}
    order = ["surface_matched", "base", "all_static", "spatiotemporal_climatology", "clim_dynamic", "clim_all_static"]
    labels = ["Surface B", "Dynamic base", "All static", "ST prior", "Prior + dynamic", "Prior + dynamic + network"]
    colours = ["#7a7a7a", "#e69f00", "#0072b2", "#8b5cf6", "#56b4e9", "#009e73"]
    event_rows = read_csv(HYBRID / "metrics" / "event_metrics.csv")
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 3.1), constrained_layout=True)
    for ax, metric, ylabel in [
        (axes[0], "mae_mm", "MAE (mm)"),
        (axes[1], "rmse_mm", "RMSE (mm)"),
        (axes[2], "csi_0p15", "CSI at 0.15 m"),
    ]:
        values = [float(lookup[name][metric]) for name in order]
        ax.bar(np.arange(len(order)), values, color=colours, width=0.72)
        ax.set_xticks(np.arange(len(order)))
        ax.set_xticklabels(labels, rotation=50, ha="right", fontsize=6.3)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.22)
    paired = defaultdict(dict)
    for row in event_rows:
        if row["reference"] == "coupled_label":
            paired[row["event"]][row["model"]] = float(row["mae_mm"])
    for event, values in paired.items():
        axes[0].plot([3, 4, 5], [values[name] for name in ["spatiotemporal_climatology", "clim_dynamic", "clim_all_static"]], color="#333333", alpha=0.35, lw=0.6, marker="o", ms=2)
    axes[0].set_title("Held-event absolute error")
    axes[1].set_title("Sensitivity to larger errors")
    axes[2].set_title("Inundation detection")
    add_panel_labels(axes, x=-0.16, y=1.04)
    save(fig, "fig07_drainlite_skill")


def fig08_controls_importance() -> None:
    controls = read_csv(DRAIN / "metrics" / "spatial_alignment_controls.csv")
    shift_values: dict[int, list[float]] = defaultdict(list)
    shuffle = []
    for row in controls:
        name = row["control"]
        delta = float(row["delta_vs_aligned_mm"])
        if name.startswith("shift_"):
            shift_values[int(name.split("_")[1])].append(delta)
        elif name == "block_shuffle":
            shuffle.append(delta)
    importance = read_csv(DRAIN / "metrics" / "permutation_importance.csv")
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in importance:
        grouped[row["feature"]].append(float(row["importance_mae_mm"]))
    ranked = sorted(((name, np.mean(values), np.std(values)) for name, values in grouped.items()), key=lambda item: item[1], reverse=True)[:10]
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.2), constrained_layout=True)
    positions = sorted(shift_values)
    means = [np.mean(shift_values[p]) for p in positions]
    stds = [np.std(shift_values[p]) for p in positions]
    axes[0].errorbar(np.array(positions) * 20, means, yerr=stds, marker="o", capsize=2, label="Directional shifts")
    axes[0].axhline(np.mean(shuffle), color="#d55e00", ls="--", label="20-cell block shuffle mean")
    axes[0].fill_between([0, 170], np.mean(shuffle) - np.std(shuffle), np.mean(shuffle) + np.std(shuffle), color="#d55e00", alpha=0.12)
    axes[0].set_xlabel("Network displacement (m)")
    axes[0].set_ylabel("Increase in MAE (mm)")
    axes[0].set_title("Spatial alignment controls")
    axes[0].legend(fontsize=6.5)
    names = [item[0].replace("_", " ") for item in ranked][::-1]
    vals = [item[1] for item in ranked][::-1]
    errs = [item[2] for item in ranked][::-1]
    axes[1].barh(np.arange(len(names)), vals, xerr=errs, color="#0072b2", alpha=0.85)
    axes[1].set_yticks(np.arange(len(names)))
    axes[1].set_yticklabels(names, fontsize=6.2)
    axes[1].set_xlabel("Increase in held-event MAE (mm)")
    axes[1].set_title("Permutation importance")
    for ax in axes:
        ax.grid(alpha=0.22)
    add_panel_labels(axes, x=-0.15, y=1.04)
    save(fig, "fig08_controls_importance")


def fig09_mike() -> None:
    physics = read_csv(PHYS / "physics_quality.csv")
    hybrid = read_csv(HYBRID / "metrics" / "event_metrics.csv")
    hybrid_lookup = {(row["event"], row["model"]): row for row in hybrid if row["reference"] == "mike_external"}
    x = np.arange(len(EVENTS))
    series = {
        "B: matched surface": [float(row["B_vs_MIKE_mae_mm"]) for row in physics],
        "C: coupled label": [float(row["C_vs_MIKE_mae_mm"]) for row in physics],
        "DrainLite hybrid": [float(hybrid_lookup[(event, "clim_all_static")]["mae_mm"]) for event in EVENTS],
    }
    volume_series = {
        "B: matched surface": [float(row["B_vs_MIKE_final_volume_abs_error_m3"]) / 1e6 for row in physics],
        "C: coupled label": [float(row["C_vs_MIKE_final_volume_abs_error_m3"]) / 1e6 for row in physics],
        "DrainLite hybrid": [float(hybrid_lookup[(event, "clim_all_static")]["final_volume_abs_error_m3"]) / 1e6 for event in EVENTS],
    }
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.1), constrained_layout=True)
    offsets = [-0.24, 0, 0.24]
    colours = ["#7a7a7a", "#d55e00", "#009e73"]
    for (label, values), offset, colour in zip(series.items(), offsets, colours):
        axes[0].bar(x + offset, values, 0.23, label=label, color=colour)
    for (label, values), offset, colour in zip(volume_series.items(), offsets, colours):
        axes[1].bar(x + offset, values, 0.23, label=label, color=colour)
    axes[0].set_ylabel("Full-sequence MAE (mm)")
    axes[0].set_title("Pixel-time depth agreement")
    axes[1].set_ylabel("Final-volume absolute error ($10^6$ m$^3$)")
    axes[1].set_title("End-of-event water storage")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels([event.replace("event", "E") for event in EVENTS])
        ax.grid(axis="y", alpha=0.22)
    axes[0].legend(fontsize=6.3, ncol=1)
    add_panel_labels(axes, x=-0.13, y=1.04)
    save(fig, "fig09_mike_metric_tradeoff")


def fig10_larno() -> None:
    ref = np.load(ROOT / "LarNO-main" / "benchmark" / "urbanflood" / "flood" / "region1_20m" / "event68" / "h.npy", mmap_mode="r")
    pred_raw = np.load(ROOT / "LarNO-main" / "exp" / "20260220_183648_006352" / "pred_results" / "region1_20m" / "epoch_992" / "predictions_epoch_992_sample_event68.npy", mmap_mode="r")
    pred = np.transpose(pred_raw, (2, 0, 1))
    peak_t = int(np.argmax(np.asarray(ref).max(axis=(1, 2))))
    vmax = float(np.quantile(np.concatenate([np.asarray(ref[peak_t]).ravel(), np.asarray(pred[peak_t]).ravel()]), 0.995))
    err = (np.asarray(pred[peak_t]) - np.asarray(ref[peak_t])) * 1000.0
    ebound = max(float(np.quantile(np.abs(err), 0.995)), 1.0)
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.65), constrained_layout=True)
    im0 = axes[0].imshow(ref[peak_t], cmap="Blues", vmin=0, vmax=vmax, origin="upper")
    axes[1].imshow(pred[peak_t], cmap="Blues", vmin=0, vmax=vmax, origin="upper")
    im2 = axes[2].imshow(err, cmap="RdBu_r", vmin=-ebound, vmax=ebound, origin="upper")
    for ax, title in zip(axes, ["MIKE target", "LarNO checkpoint", "LarNO - MIKE"]):
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im0, ax=axes[:2], fraction=0.025, pad=0.015, label="Water depth (m)")
    fig.colorbar(im2, ax=axes[2], fraction=0.05, pad=0.02, label="Error (mm)")
    audit = json.loads((REV / "larno_event68_checkpoint_audit.json").read_text(encoding="utf-8"))
    metrics = audit["all_grid_unclamped"]
    fig.suptitle(
        f"Public LarNO event68 reproduction: MAE = {metrics['mae_m'] * 1000:.3f} mm, "
        f"CSI$_{{0.15}}$ = {metrics['csi_0p15']:.3f}; raw negative fraction = {audit['negative_fraction'] * 100:.2f}%",
        fontsize=9,
    )
    add_panel_labels(axes, x=-0.08, y=1.03)
    save(fig, "fig10_larno_reproduction")


def fig11_cost() -> None:
    rows = read_csv(DRAIN / "metrics" / "runtime_model_size.csv")
    fit = [float(row["seconds"]) for row in rows if row["model"] == "all_static_uniform" and row["stage"] == "fit"]
    inference = [float(row["seconds"]) for row in rows if row["model"] == "all_static_uniform" and row["stage"] == "full_domain_inference"]
    sizes = [float(row["model_bytes"]) / 1e6 for row in rows if row["model"] == "all_static_uniform" and row["stage"] == "fit"]
    coupled = [float(row["coupled_runtime_s"]) for row in read_csv(PHYS / "physics_quality.csv")]
    metadata = json.loads((HYBRID / "experiment_metadata.json").read_text(encoding="utf-8"))
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.8), gridspec_kw={"width_ratios": [1.55, 0.72, 0.72]}, constrained_layout=True)
    labels = ["Coupled physical\nsimulation", "DrainLite fit\nper fold", "DrainLite inference\nper event"]
    vals = [np.mean(coupled), np.mean(fit), np.mean(inference)]
    axes[0].bar(labels, vals, color=["#d55e00", "#0072b2", "#009e73"])
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Wall time (s, log scale)")
    axes[0].set_title("Observed local runtime")
    axes[1].bar(["All-static\nmodel"], [np.mean(sizes)], color="#0072b2")
    axes[1].set_ylabel("Model file size (MB)")
    axes[1].set_title("Serialized model")
    axes[1].text(0, np.mean(sizes), f"{np.mean(sizes):.2f} MB", ha="center", va="bottom", fontsize=7)
    peak_gb = metadata["peak_rss_mb"] / 1024.0
    axes[2].bar(["Complete\nhybrid run"], [peak_gb], color="#8b5cf6")
    axes[2].set_ylabel("Peak working memory (GB)")
    axes[2].set_title("Process memory")
    axes[2].text(0, peak_gb, f"{peak_gb:.2f} GB", ha="center", va="bottom", fontsize=7)
    for ax in axes:
        ax.grid(axis="y", alpha=0.22)
    add_panel_labels(axes, x=-0.15, y=1.04)
    save(fig, "fig11_computational_cost")


def main() -> int:
    FIG.mkdir(parents=True, exist_ok=True)
    fig01_workflow()
    fig02_network()
    fig03_physical_quality()
    fig04_hydrographs()
    fig05_event68_maps()
    fig06_residual_maps()
    fig07_model_skill()
    fig08_controls_importance()
    fig09_mike()
    fig10_larno()
    fig11_cost()
    print(FIG)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
