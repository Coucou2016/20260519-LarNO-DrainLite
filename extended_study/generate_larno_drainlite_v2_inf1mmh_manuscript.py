#!/usr/bin/env python3
"""Generate the journal-style LarNO-compatible DrainLite manuscript."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from publication_plot_style import configure_publication_style


configure_publication_style()
ROOT = Path(__file__).resolve().parents[1]
DATASET = "region1_20m_connected_swmm_v2_inf1mmh"
MAIN = ROOT / "extended_study" / "output" / "drainlite_connected_residual_v2_inf1mmh"
CV = ROOT / "extended_study" / "output" / "drainlite_connected_cross_validation_v2_inf1mmh"
FILTERED_CV = ROOT / "extended_study" / "output" / "drainlite_connected_cross_validation_v2_inf1mmh_quality_filtered"
QUALITY = ROOT / "extended_study" / "output" / "drainlite_quality_stratified_v2_inf1mmh"
PACKAGE = ROOT / "extended_study" / "output" / "larno_drainlite_v2_inf1mmh_package"
HYDRO = ROOT / "extended_study" / "output" / "hydrograph_timing_diagnostics_v2_inf1mmh"
SENSITIVITY = ROOT / "extended_study" / "output" / "infiltration_sensitivity"
SELECTION = ROOT / "extended_study" / "output" / "infiltration_model_selection"
PAPER_FIG = PACKAGE / "paper_figures"
OUT_MD = PACKAGE / "manuscript_larno_drainlite_v2_inf1mmh.md"
OUT_ALIAS = PACKAGE / "manuscript_larno_drainlite_v2.md"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def find(rows: list[dict[str, str]], key: str, value: str) -> dict[str, str]:
    return next((row for row in rows if row.get(key) == value), {})


def fnum(value: object, digits: int = 3) -> str:
    number = float(value)
    if abs(number) >= 1000:
        return f"{number:,.0f}"
    return f"{number:.{digits}f}"


def table(rows: list[dict[str, str]], columns: list[tuple[str, str, int | None]]) -> str:
    header = "| " + " | ".join(label for _, label, _ in columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [header, separator]
    for row in rows:
        values = []
        for key, _, digits in columns:
            value = row.get(key, "")
            if digits is None:
                values.append(str(value))
            elif key in {"n_events", "evaluation_active_cells"}:
                values.append(str(int(float(value))))
            else:
                values.append(fnum(value, digits))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def copy_figure(source: Path, name: str) -> str:
    PAPER_FIG.mkdir(parents=True, exist_ok=True)
    destination = PAPER_FIG / name
    shutil.copy2(source, destination)
    return f"paper_figures/{name}"


def draw_workflow(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis("off")

    def box(x: float, y: float, w: float, h: float, text: str, color: str) -> None:
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.025,rounding_size=0.05",
            linewidth=0.8,
            edgecolor="0.2",
            facecolor=color,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=7.6)

    def arrow(x1: float, y1: float, x2: float, y2: float, label: str = "") -> None:
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=9, lw=0.9, color="0.25"))
        if label:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.18, label, ha="center", va="bottom", fontsize=6.7)

    box(0.25, 4.85, 2.15, 1.25, "Public benchmark inputs\nRainfall, DEM", "#DCEAF7")
    box(0.25, 1.05, 2.15, 1.25, "External reference\nMIKE water depth", "#E8E8E8")
    box(3.05, 5.2, 2.25, 1.15, "ITZI surface flow\n1 mm h$^{-1}$ effective loss", "#F3E5C0")
    box(3.05, 3.05, 2.25, 1.35, "Road-aligned network\nSWMM dynamic wave\nNative ITZI exchange", "#F8D6C5")
    box(6.0, 4.0, 2.2, 1.35, "Paired physical labels\n$h_{surface}$ and $h_{coupled}$", "#E8DDF1")
    box(6.0, 1.55, 2.2, 1.15, "Signed residual\n$r=h_{coupled}-h_{surface}$", "#D8EEDB")
    box(8.85, 2.55, 2.55, 1.45, "DrainLite\nGradient-boosted residual model\nStatic drainage features", "#CEE7F2")
    box(8.85, 5.15, 2.55, 1.15, "Quality-filtered\nleave-one-event-out validation", "#F0E4D2")
    arrow(2.4, 5.48, 3.05, 5.72)
    arrow(2.4, 5.18, 3.05, 3.72)
    arrow(5.3, 5.72, 6.0, 4.9)
    arrow(5.3, 3.72, 6.0, 4.45)
    arrow(7.1, 4.0, 7.1, 2.7, "difference")
    arrow(8.2, 2.13, 8.85, 3.05)
    arrow(10.1, 4.0, 10.1, 5.15)
    arrow(2.4, 1.68, 8.85, 5.55, "external comparison")
    ax.text(0.25, 6.65, "Data provenance", fontsize=8.3, fontweight="bold")
    ax.text(8.85, 6.65, "Model evaluation", fontsize=8.3, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> int:
    PACKAGE.mkdir(parents=True, exist_ok=True)
    PAPER_FIG.mkdir(parents=True, exist_ok=True)
    metadata = json.loads(
        (ROOT / "LarNO-main" / "benchmark" / "urbanflood" / "flood" / DATASET / "metadata.json").read_text(
            encoding="utf-8"
        )
    )
    network = json.loads(
        (ROOT / "extended_study" / "output" / "connected_swmm_network" / "component_outfall_swmm_network_audit.json").read_text(
            encoding="utf-8"
        )
    )
    quality_rows = read_csv(PACKAGE / "metrics" / "v2_label_quality_grading.csv")
    quality_summary = read_csv(PACKAGE / "metrics" / "v2_label_quality_summary.csv")
    stratified = read_csv(QUALITY / "metrics" / "quality_stratified_drainlite_summary.csv")
    formal = [row for row in stratified if row["analysis_set"] == "quality_filtered_formal"]
    accepted = [row for row in stratified if row["analysis_set"] == "accepted_only_sensitivity"]
    all8 = [row for row in stratified if row["analysis_set"] == "all8_diagnostic"]
    mike = read_csv(QUALITY / "metrics" / "quality_filtered_mike_summary.csv")
    mike_events = read_csv(QUALITY / "metrics" / "quality_filtered_mike_event_metrics.csv")
    sensitivity = [
        row
        for row in read_csv(SENSITIVITY / "event68_infiltration_sensitivity_metrics.csv")
        if row["scenario"] in {"mike_reference", "connected"}
    ]
    selection = read_csv(SELECTION / "infiltration_final_selection.csv")

    surface = find(formal, "model", "surface_only")
    base = find(formal, "model", "base")
    masks = find(formal, "model", "mask_only")
    hydraulic = find(formal, "model", "hydraulic_only")
    all_static = find(formal, "model", "all_static")
    swmm_oracle = find(formal, "model", "swmm_assisted")
    mike_surface = find(mike, "model", "ITZI surface-only")
    mike_connected = find(mike, "model", "ITZI-SWMM connected")
    mike_drainlite = find(mike, "model", "DrainLite all_static")
    accepted_all_static = find(accepted, "model", "all_static")
    all8_all_static = find(all8, "model", "all_static")

    workflow = PAPER_FIG / "study_workflow.png"
    draw_workflow(workflow)
    figures = {
        "workflow": "paper_figures/study_workflow.png",
        "network": copy_figure(MAIN / "figures" / "connected_static_dem_network.png", "network_dem.png"),
        "quality": copy_figure(PACKAGE / "figures" / "v2_label_quality_grading.png", "label_quality.png"),
        "hydro": copy_figure(HYDRO / "figures" / "hydrograph_timing_summary.png", "hydrograph_timing.png"),
        "sensitivity": copy_figure(
            SENSITIVITY / "figures" / "event68_infiltration_sensitivity_hydrographs.png",
            "infiltration_sensitivity.png",
        ),
        "validation": copy_figure(
            QUALITY / "figures" / "quality_filtered_validation_summary.png", "formal_validation.png"
        ),
        "peak68": copy_figure(MAIN / "figures" / "event68_connected_peak_maps.png", "event68_peak_maps.png"),
        "peak70": copy_figure(MAIN / "figures" / "event70_connected_peak_maps.png", "event70_peak_maps.png"),
        "importance": copy_figure(
            MAIN / "figures" / "feature_importance_all_static.png", "feature_importance.png"
        ),
        "mike": copy_figure(QUALITY / "figures" / "quality_filtered_mike_summary.png", "mike_external.png"),
        "event68": copy_figure(MAIN / "figures" / "event68_cv_timeseries.png", "event68_timeseries.png"),
    }

    formal_order = ["surface_only", "base", "mask_only", "hydraulic_only", "all_static"]
    formal_main = sorted(
        [row for row in formal if row["model"] in set(formal_order)],
        key=lambda row: formal_order.index(row["model"]),
    )
    formal_table = table(
        formal_main,
        [
            ("model", "Model", None),
            ("n_events", "Events", 0),
            ("mae_mm", "MAE (mm)", 3),
            ("rmse_mm", "RMSE (mm)", 3),
            ("csi_0p03", "CSI 0.03 m", 3),
            ("csi_0p15", "CSI 0.15 m", 3),
            ("abs_peak_error_mm", "Abs. global-peak error (mm)", 3),
            ("abs_peak_time_error_h", "Abs. peak-time error (h)", 3),
        ],
    )
    quality_table = table(
        quality_rows,
        [
            ("event", "Event", None),
            ("quality_status", "Tier", None),
            ("continuity_error_pct", "Continuity error (%)", 3),
            ("nonconverging_steps_pct", "Nonconverging steps (%)", 2),
            ("peak_reduction_mm", "Peak change (mm)", 3),
            ("final_volume_reduction_m3", "Final-volume change (m3)", 0),
        ],
    )
    mike_table = table(
        mike,
        [
            ("model", "Model", None),
            ("mae_mm", "MAE (mm)", 3),
            ("rmse_mm", "RMSE (mm)", 3),
            ("csi_0p03", "CSI 0.03 m", 3),
            ("csi_0p15", "CSI 0.15 m", 3),
            ("peak_depth_error_mm", "Signed global-peak bias (mm)", 3),
            ("peak_map_mae_mm", "Peak-map MAE (mm)", 3),
            ("final_volume_error_m3", "Signed final-volume bias (m3)", 0),
        ],
    )
    sensitivity_table = table(
        sensitivity,
        [
            ("label", "Case", None),
            ("rate_mmh", "Effective loss (mm h-1)", 1),
            ("max_depth_peak_hour", "Global-peak time (h)", 2),
            ("max_depth_peak_m", "Global peak (m)", 3),
            ("volume_final_to_peak_ratio", "Final/peak volume", 3),
            ("mae_to_mike_mm", "MAE to MIKE (mm)", 3),
            ("continuity_error_pct", "Continuity error (%)", 3),
        ],
    )

    text = f"""# Drainage-aware residual correction for LarNO-compatible urban flood forecasting

**Author names and affiliations:** [待补充]

## Highlights

- Paired ITZI surface-flow and native ITZI-SWMM simulations isolate the effect of a road-aligned conceptual drainage network.
- A signed residual adapter represents both drainage removal and local surcharge without retraining a neural-operator backbone.
- Five-event leave-one-event-out validation reduced the coupled-label MAE from {fnum(surface['mae_mm'])} to {fnum(all_static['mae_mm'])} mm.
- Static drainage descriptors improved over a terrain-rainfall residual model by {fnum(all_static['improvement_vs_base_pct'])}%, indicating that dynamic pipe states remain unresolved.

## Abstract

Neural operators provide rapid forecasts of spatially distributed urban flooding, yet their predictions can omit the hydraulic influence of underground drainage when pipe-network states are absent from the model inputs and training labels. We present a lightweight drainage-aware residual method that is compatible with a LarNO-type surface predictor but does not require the neural-operator backbone to be retrained. A road-aligned conceptual drainage network was connected to the native two-way ITZI-SWMM workflow, producing paired surface-only and coupled water-depth fields under identical rainfall, topography, boundary and effective-loss settings. Their signed difference was learned with histogram-based gradient-boosted trees using surface water depth, rainfall, terrain and rasterized network descriptors. The resulting data set contains eight six-hour events on a 20 m grid at five-minute intervals. Numerical-quality screening retained five events for strict leave-one-event-out evaluation and excluded three events from both training and testing because their absolute SWMM continuity error exceeded 8%. On 43,606 cells below the 49.9 m wall-height threshold, the all-static model reduced mean absolute error relative to the coupled label from {fnum(surface['mae_mm'])} to {fnum(all_static['mae_mm'])} mm and achieved critical success indices of {fnum(all_static['csi_0p03'])} and {fnum(all_static['csi_0p15'])} at 0.03 and 0.15 m. The gain over an otherwise identical residual model without drainage descriptors was {fnum(all_static['improvement_vs_base_pct'])}%. Comparison with MIKE was used only as an external check. DrainLite moderated the signed global-peak bias, from {fnum(mike_surface['peak_depth_error_mm'])} mm for ITZI surface-only to {fnum(mike_drainlite['peak_depth_error_mm'])} mm, but did not improve the full time-space MIKE error or final-volume agreement. The results show that a lightweight adapter can recover much of the drainage-induced correction contained in the coupled simulations. They also show that static network maps do not replace dynamic pipe head, flow and surcharge information, and that the present conceptual network remains a controlled research representation rather than a surveyed municipal drainage model.

**Keywords:** urban flooding; drainage network; ITZI-SWMM coupling; residual learning; neural operator; LarNO; surrogate modelling

## 1. Introduction

Urban pluvial flooding is governed by an interaction between rainfall, surface storage, overland conveyance and underground drainage. At street scale, shallow depressions and building blocks redirect runoff over short distances, while inlets and pipes remove water from the surface or return it when the network becomes surcharged. Models that resolve only the two-dimensional surface therefore describe only part of the response. Conversely, a one-dimensional pipe model cannot determine where water spreads once it reaches the ground. Coupled surface-drainage modelling is consequently required when the timing and spatial pattern of urban inundation are sensitive to sewer capacity and surcharge.

Hydrodynamic models represent these processes explicitly, but their computational cost restricts their use in rapid forecasting. Two-dimensional solvers must satisfy stability requirements at time steps much shorter than the output interval, and a coupled pipe network adds a second hydraulic system and an exchange calculation. Reduced-complexity flood models and data-driven surrogates have been developed to alleviate this cost (Bates et al., 2010; Fraehr et al., 2023). Neural operators are particularly attractive because they learn mappings between function spaces and can, in principle, be evaluated on different discretisations (Li et al., 2021; Kovachki et al., 2023).

LarNO extends this class of models through latent autoregression and was designed for large-scale urban flood prediction at high spatial and temporal resolution (Cao et al., 2026). Its benchmark provides rainfall, topographic information and MIKE-based water-depth fields for a large urban area in Shenzhen. The benchmark is valuable for rapid flood forecasting, but the released data do not expose the time-varying state of the underground network as a set of learnable inputs. A model can therefore reproduce drainage effects only to the extent that they are already implicit in the reference water-depth labels and correlated with static surface variables. It cannot readily adapt when a different network layout, inlet distribution or hydraulic capacity is imposed.

This limitation is partly architectural, but it is first a data problem. To train a drainage-aware correction, the effect attributed to the pipe system must be separated from the surface response under the same forcing. Paired simulations provide such a separation: one member contains dynamic surface flow alone and the other contains the same surface model coupled to a pipe network. Their difference is a drainage-induced residual. The residual is signed because the network can both reduce surface water through inlet capture and increase it locally through surcharge. Treating this quantity as non-negative would suppress one of the defining behaviours of two-way coupling.

Directly retraining LarNO with additional drainage channels would require substantially more coupled events and greater computational resources than are available in the present study. A residual adapter offers a narrower test of the scientific hypothesis. The surface predictor is left unchanged, and a low-cost model estimates only the correction associated with the network. This design also separates two questions that would otherwise be conflated: whether the coupled simulations contain a learnable drainage signal, and whether a large neural operator can be trained end to end. The present experiments address the first question. ITZI surface-only water depth is used as the base field because a complete set of consistent six-hour simulations is available. A future LarNO prediction with the same variables and grid could occupy the same position in the workflow, but that substitution is not evaluated here.

The absence of surveyed pipe data introduces a second constraint. We therefore use a conceptual network aligned with the corrected road graph. The network is not intended to reproduce municipal assets pipe by pipe. Instead, it provides a controlled representation in which every retained inlet is connected to an outfall, pipe inverts follow a consistent cover rule, and both surface-only and coupled cases can be repeated under identical external conditions. This makes it possible to study whether network position and hydraulic descriptors improve residual prediction without representing the synthetic network as ground truth.

The study has three objectives. First, we construct paired ITZI surface-only and native ITZI-SWMM simulations with a connected road-aligned network and document their numerical quality. Second, we test whether a lightweight signed-residual model can reproduce the coupled water-depth field in event-wise validation. Third, we determine how much additional information is supplied by static network masks and hydraulic attributes, and compare the resulting fields with the MIKE benchmark as an external, non-training reference. Figure 1 summarises the separation between public benchmark inputs, newly computed coupled labels, residual learning and external evaluation.

![Study workflow]({figures['workflow']})

**Fig. 1. Study design and data provenance.** Public rainfall and topographic fields drive two locally computed ITZI cases. The coupled branch adds the road-aligned SWMM network through ITZI's native exchange routine. DrainLite learns the signed difference between these paired simulations. MIKE water depth remains outside the training path and is used only for external comparison.

## 2. Materials and methods

### 2.1 Study data and computational domain

The study uses a 20 m subset of the Shenzhen Region 1 benchmark distributed with LarNO. The computational array contains {metadata['shape'][0]} rows and {metadata['shape'][1]} columns, corresponding to 56,000 grid cells. Each event lasts six hours and contains {metadata['time_steps']} output frames at five-minute intervals. The eight stored events are event1, event20, event65, event66, event67, event68, event69 and event70. The rainfall array is interpreted as depth accumulated over each five-minute frame. Frame 0 is imposed from time 0 to 300 s and its stored water-depth field represents the state at 300 s; frame 71 covers 21,300--21,600 s and is stored at 21,600 s. This convention prevents the one-frame delay present in earlier diagnostic runs.

The topographic array contains elevated wall or building cells. For evaluation, the analysis mask was defined as cells with finite elevation below 49.9 m, leaving 43,606 cells. This is a model-specific non-wall threshold, not a missing-data mask. Water-depth errors, inundation classification and water volume reported in the main analysis use the same mask. The cell area is 400 m2. Public DEM, rainfall and MIKE arrays were retained as source data; only the ITZI, coupled and DrainLite fields were generated in the present workflow.

The initial surface was dry. The two-dimensional raster boundary remained closed because no `bctype` map was supplied to the ITZI configuration. Water could leave the surface through effective loss and, in the coupled case, through drainage exchange and outfalls, but not through an open two-dimensional river boundary. The simulation workflow redistributed rain falling on wall/building cells to active cells. Buildings represent approximately 22.13% of the window, so this assumption increases the active-cell rainfall by about 28.4%. It is maintained here to preserve consistency between paired cases, but it is revisited as a source of discrepancy with MIKE.

### 2.2 Road-aligned conceptual drainage network

The pipe layout was derived from the corrected, road-aligned graph. Nodes outside the 20 m window and links without valid endpoints were removed. Connected components with fewer than three nodes were discarded. Twelve retained components were each assigned one synthetic outfall at the node with the lowest candidate invert; the outfall was placed immediately outside the surface window so that it was not coupled back to the two-dimensional domain. Conduits were oriented from higher to lower invert where possible, although the dynamic-wave solver continued to permit flow reversal.

The resulting network contains {network['retained_junctions']} junctions, {network['retained_conduits']} conduits and {network['retained_outfalls']} outfalls. All {network['junctions_connected_to_any_outfall']} junctions have a graph path to an outfall; no junction is isolated. Multiple junctions can fall in the same 20 m cell, producing 866 unique coupled junction cells after rasterisation. Of {network['retained_conduits']} conduits, 2,877 intersect the raster and occupy 3,675 unique pipe cells. Junction inverts were set 2 m below the local surface. Circular pipe diameters were inherited from the conceptual source network with lower bounds of 0.8 m for ordinary links and 1.0 m for main-trunk links; synthetic outfall links used a 2.0 m diameter. Manning's roughness was 0.013 where no different source value was provided.

Eight static fields were rasterised on the model grid: inlet mask, outfall mask, pipe mask, pipe diameter, pipe slope, full-flow capacity, cover depth and distance to the nearest outfall. Local 3 x 3 and 7 x 7 averages were subsequently calculated for pipe density, inlet density and capacity. Figure 2 shows the final alignment. White areas in the elevation layer are wall cells excluded by the display mask. The black raster trace marks conduits, blue points show inlet/manhole cells and orange points indicate outfalls. Outfalls located on the domain edge are intentionally retained because they are the discharge endpoints of the conceptual components.

![Network and DEM]({figures['network']})

**Fig. 2. Road-aligned conceptual drainage network on the 20 m elevation grid.** The figure is a spatial-consistency check rather than evidence of a surveyed municipal system. Junctions and conduits follow the corrected road-derived geometry, every retained component terminates at an outfall, and the network is displayed in the same raster orientation used by ITZI and DrainLite.

### 2.3 ITZI surface flow and native ITZI-SWMM coupling

Itzï is a distributed dynamic flood model designed for raster-based overland flow and can exchange water with a one-dimensional SWMM drainage model (Courty et al., 2017). In the surface-only case, the ITZI surface solver was driven by the benchmark rainfall and DEM without drainage exchange. In the coupled case, the same surface state was connected to SWMM through `DrainageSimulation.apply_coupling_to_nodes()`. SWMM used dynamic-wave routing, which solves the one-dimensional Saint-Venant continuity and momentum equations and can represent backwater, pressurisation and flow reversal (Rossman and Simon, 2022). The routing step was 2 s, variable stepping was enabled and the maximum number of trials per step was 20.

At each coupling update, ITZI transfers the surface water level to the corresponding drainage node and applies the exchange flow returned by the network solver. Under the sign convention used by ITZI, negative exchange removes water from the surface and positive exchange represents return flow from the pipe system. This convention was verified in the copied runner; cumulative drainage volume was therefore integrated from the negative component of exchange. The coupled result is not a sink approximation. It allows capture, pipe routing, finite conveyance, nodal storage, surcharge and return flow, subject to the topology and parameters of the conceptual network.

Both physical cases included a spatially uniform effective loss of 1 mm h-1 on non-wall cells. The term is implemented with ITZI's constant-rate infiltration option, but it should be interpreted as an effective unresolved loss rather than a soil-calibrated infiltration parameter. The same loss was applied to the surface-only and coupled members so that their difference isolates network interaction under the chosen configuration.

### 2.4 Effective-loss sensitivity and label-quality control

The effective-loss rate was examined because simulations without loss retained almost all surface water until the end of the six-hour window, whereas the MIKE reference exhibited a pronounced falling limb. Event68 was run at 1, 2, 3, 4 and 5 mm h-1. Candidate rates were compared by global maximum depth, active-cell volume, time-space MAE to MIKE and SWMM continuity error. The single-event analysis was followed by an eight-event comparison of the 1 and 2 mm h-1 configurations. This diagnostic used the available event set and therefore does not constitute an independent calibration-validation split. Its purpose was to avoid adopting a rate that matched one event while depressing water depth across the others.

SWMM label quality was graded from the absolute routing continuity error. Events at or below 2% were designated accepted, values between 2 and 8% warning, and values above 8% excluded. The nonconverging-step ratio was recorded separately. Passing the continuity threshold does not imply stepwise dynamic-wave convergence; it only bounds the reported event-scale water-balance discrepancy. The primary learning experiment used accepted and warning events and removed excluded events from both the training and test sides of every fold.

### 2.5 DrainLite signed-residual model

Let $h_s(t,x)$ denote the ITZI surface-only depth and $h_c(t,x)$ the depth from the native ITZI-SWMM coupled simulation at time $t$ and cell $x$. The learning target is the signed residual in millimetres,

$$r(t,x)=1000[h_c(t,x)-h_s(t,x)].$$

A negative residual denotes net drainage-induced reduction at that location and time. A positive residual denotes local water-depth increase, which can arise from pipe surcharge, a redistribution of flow, or a shift in the timing of surface storage. The predicted depth is reconstructed as

$$\\hat h_c(t,x)=\\max[h_s(t,x)+\\hat r(t,x)/1000,0].$$

The non-negative truncation is applied only to the reconstructed water depth; it does not force the residual itself to be non-positive.

DrainLite uses `HistGradientBoostingRegressor`, a histogram-based gradient-boosted decision-tree model. This choice permits nonlinear interactions among depth, terrain and network variables while remaining practical on a CPU. The strict five-event leave-one-event-out experiment sampled at most 700 cells from each five-minute frame in each training event. Sampling favoured cells with a non-zero residual, wet cells and cells near the drainage network, while retaining background cells. Each fold used four events for training and the remaining event for testing. The model used a learning rate of 0.05, at most 220 boosting iterations, 31 leaf nodes, L2 regularisation of 0.01 and random seed 671 plus the fold index.

The base feature set comprised surface depth, rainfall, cumulative rainfall, DEM elevation, local slope, normalised row and column coordinates, normalised event time, and 3 x 3 and 7 x 7 local means of surface depth. `mask_only` added inlet, outfall and pipe masks, distance to outfall, and local pipe and inlet densities. `hydraulic_only` added diameter, pipe slope, capacity, cover depth, distance to outfall and local capacity density. `all_static` combined all eight static network fields and their local-density derivatives. A separate `swmm_assisted` diagnostic added event-complete SWMM inflow, outflow, storage, continuity and flooding summaries. Those summaries are unavailable when a forecast begins; the assisted model is consequently treated as an oracle diagnostic and is not included among the deployable main models.

### 2.6 Evaluation metrics and experimental roles

The primary target is the ITZI-SWMM coupled depth, not MIKE. Mean absolute error, root-mean-square error and critical success index were calculated separately for each event over all 72 frames and all non-wall evaluation cells, after which the event values were macro-averaged so that every event contributed equally. The critical success index was calculated at 0.03 and 0.15 m as the intersection of predicted and target inundated cells divided by their union. The global-peak error is the difference between the maximum water depth over all times and evaluation cells. Peak-time error is based on the time series of that domain-wide maximum. These definitions differ from peak-map MAE, which first takes the temporal maximum at each cell and then averages the absolute spatial difference.

Surface-water volume was obtained by summing water depth over evaluation cells and multiplying by 400 m2. Final-volume error is the predicted minus target volume at the last frame. Peak-volume error is evaluated at the frame when the target volume is maximal. Signed biases preserve whether a model is high or low; absolute errors remove that direction.

Three analyses serve different purposes. The primary result is the strict five-event leave-one-event-out experiment using event1, event67, event68, event69 and event70. The eight-event leave-one-event-out run is a diagnostic sensitivity analysis because the three low-quality labels enter training in some folds. A fixed event68-event70 test experiment provides maps and feature importance, but its training set includes excluded labels and it is therefore not used for the primary performance estimate. MIKE is evaluated on the same five events as the primary analysis and remains outside all DrainLite training features and targets.

## 3. Results

### 3.1 Hydrodynamic response and numerical quality of coupled labels

The connected network produced a visible reduction in surface storage relative to the surface-only simulation, but the magnitude differed among events. The event-scale quality screen retained three accepted events and two warning events for the primary analysis; three events were excluded. Continuity errors for event68, event69 and event70 were {fnum(find(quality_rows, 'event', 'event68')['continuity_error_pct'])}%, {fnum(find(quality_rows, 'event', 'event69')['continuity_error_pct'])}% and {fnum(find(quality_rows, 'event', 'event70')['continuity_error_pct'])}%, respectively. Event1 and event67 had continuity errors of {fnum(find(quality_rows, 'event', 'event1')['continuity_error_pct'])}% and {fnum(find(quality_rows, 'event', 'event67')['continuity_error_pct'])}% and were retained with warning status. Event20, event65 and event66 exceeded the 8% limit.

{quality_table}

**Table 1. Numerical quality and event-scale change produced by the connected ITZI-SWMM simulations.** Peak change and final-volume change are calculated relative to the paired surface-only case. Continuity error is the criterion used for the tier; the nonconverging-step ratio is reported independently.

The black line in Fig. 3 shows that nonconverging-step ratios remained high even when continuity error was small. This distinction matters. Low continuity error indicates that the total water balance is reasonably closed at the event scale, whereas the high step ratio indicates repeated difficulty in the iterative dynamic-wave solution. The coupled arrays are therefore usable as controlled research labels but carry numerical uncertainty at short time scales and around surcharged nodes.

![Label quality]({figures['quality']})

**Fig. 3. Quality grading of the eight coupled events.** Bars show signed SWMM continuity error; their colour denotes accepted, warning or excluded status. Dashed lines mark the +/-2% accepted range and dotted lines the +/-8% exclusion boundary. The black line uses the right axis and shows the proportion of nonconverging steps. The two measures should not be interpreted interchangeably.

The hydrograph diagnostic in Fig. 4 reveals a second difference from MIKE. MIKE maximum depths generally peaked between approximately two and five hours and its final-to-peak surface-volume ratio was close to one half. The surface-only ITZI runs frequently reached their largest depth at or near six hours and retained nearly all peak surface volume. Coupling reduced the final-volume ratio to approximately 0.87--0.99, depending on the event, but did not fully reproduce the MIKE recession. The network therefore produced a clear drainage signal while the closed surface boundary, effective-loss assumption, rainfall redistribution and conceptual pipe capacity continued to affect the timing.

![Hydrograph timing]({figures['hydro']})

**Fig. 4. Event-wise peak timing and recession diagnostics.** Panel (a) gives the time of the domain-wide maximum depth. Panel (b) is the final surface-water volume divided by its event maximum; a value of one indicates no recession before the six-hour endpoint. The figure uses non-wall cells for volume and includes all eight events as a physical diagnostic, not as the primary learning set.

### 3.2 Effective-loss sensitivity

The event68 sensitivity experiment confirmed that the missing falling limb could not be corrected by pipe coupling alone. Without effective loss, coupled maximum depth continued to rise until late in the event. Increasing the rate progressively lowered both maximum depth and total surface volume (Fig. 5). Rates of 4 and 5 mm h-1 produced an earlier plateau but reduced maximum depth to approximately 1.1 m, substantially below MIKE. The 2 mm h-1 run was closest to MIKE for several event68 criteria, whereas 1 mm h-1 retained a larger late-event storage.

![Infiltration sensitivity]({figures['sensitivity']})

**Fig. 5. Event68 sensitivity to the constant effective-loss rate.** Panel (a) shows the domain-wide maximum depth; panel (b) shows active-cell water volume. The black curve is MIKE, red is the no-loss coupled case, and the remaining curves are constant rates from 1 to 5 mm h-1. These curves demonstrate a strong parameter effect, but a match to one event does not establish a transferable physical infiltration rate.

{sensitivity_table}

**Table 2. Event68 effective-loss sensitivity.** `MAE to MIKE` is a diagnostic quantity and was not used to train DrainLite. The selected 1 mm h-1 setting is an effective unresolved loss applied uniformly to non-wall cells.

When the 1 and 2 mm h-1 rates were compared across all eight events, the 2 mm h-1 configuration produced a more negative mean MIKE global-peak bias and final-volume bias. The 1 mm h-1 setting was therefore retained as the less aggressive option. Because this comparison used the same finite pool of events later used for model evaluation, it served as configuration screening rather than independent hydrological calibration. The external MIKE comparison and the limitations below account for the resulting uncertainty.

### 3.3 Quality-filtered leave-one-event-out performance

The five-event surface-only baseline had an MAE of {fnum(surface['mae_mm'])} mm to the coupled target. Learning a residual from surface depth, rainfall and terrain without explicit drainage fields reduced the error to {fnum(base['mae_mm'])} mm. Adding network masks further reduced it to {fnum(masks['mae_mm'])} mm; the hydraulic-only set produced {fnum(hydraulic['mae_mm'])} mm, and the combined all-static model achieved {fnum(all_static['mae_mm'])} mm. The all-static improvement was {fnum(all_static['improvement_vs_surface_pct'])}% relative to surface-only and {fnum(all_static['improvement_vs_base_pct'])}% relative to the base residual model.

{formal_table}

**Table 3. Strict five-event leave-one-event-out performance against the ITZI-SWMM coupled label.** Excluded events were absent from both training and testing. `Global-peak error` refers to the maximum depth over all times and non-wall cells; it is not the spatially averaged peak-map error.

The corresponding all-static RMSE was {fnum(all_static['rmse_mm'])} mm. Critical success indices reached {fnum(all_static['csi_0p03'])} at 0.03 m and {fnum(all_static['csi_0p15'])} at 0.15 m. These classification scores indicate that the residual correction largely preserved the inundation footprint of the coupled label while adjusting depth. The accepted-only sensitivity gave an all-static MAE of {fnum(accepted_all_static['mae_mm'])} mm. The direction of improvement therefore did not depend on retaining the two warning events, although the three-event accepted subset is too small for a separate generalisation claim.

Figure 6 separates the main coupled-label result from the external MIKE check. Panel (a) shows the strong reduction from surface-only to all residual models. Panel (b) shows that lower error to the coupled label did not imply lower error to MIKE. Panel (c) confirms that the all-static model improved on surface-only within both accepted and warning tiers. The assisted model is displayed only to test whether complete-event SWMM summaries contain additional information; its MAE of {fnum(swmm_oracle['mae_mm'])} mm was slightly higher than all-static and provides no evidence that these oracle features improve prediction in this experiment.

![Formal validation]({figures['validation']})

**Fig. 6. Quality-filtered DrainLite validation.** Panel (a) is the mean absolute error to the native ITZI-SWMM coupled label. Panel (b) evaluates the same predictions against MIKE and is not a training score. Panel (c) separates accepted and warning events. The SWMM-assisted bar is a post-event diagnostic because its summary inputs are unavailable at forecast initialisation.

The eight-event diagnostic yielded an all-static MAE of {fnum(all8_all_static['mae_mm'])} mm and therefore preserved the direction of the residual correction. It is not directly comparable with the primary five-event result because the training folds can contain the three excluded physical labels. The formal performance estimate is consequently {fnum(all_static['mae_mm'])} mm from the strict quality-filtered analysis.

### 3.4 Spatial and temporal structure of the correction

Figure 7 examines event68, one of the accepted coupled labels. The MIKE peak map contains broader and deeper inundation in several western and southern corridors than either ITZI case. The ITZI-SWMM field is lower than surface-only across much of the active domain but retains local maxima, confirming that the connected simulation is not equivalent to applying a uniform sink. The DrainLite peak field reproduces the broad coupled pattern. Its error is concentrated along narrow inundation boundaries and at isolated deep cells.

The two difference panels use symmetric colour limits shared by event68--event70 and based on the 99.5th percentile of the corresponding absolute values. This display choice makes distributed errors visible while the arrowed colour-bar ends indicate that more extreme values exist outside the shown range. No values were clipped in the arrays or metric calculations. The predicted residual is mostly negative, consistent with drainage removal, but contains local positive regions. These positive cells are physically and statistically important because they preserve the signed formulation and allow surcharge-like behaviour.

![Event68 peak maps]({figures['peak68']})

**Fig. 7. Event68 peak-depth fields and DrainLite residual.** Panels (a)--(d) use the water-depth scale shared by event68--event70. Panel (e) is DrainLite minus the ITZI-SWMM label. Panel (f) is the predicted signed residual evaluated at the time when each cell reaches its coupled peak. The error and residual colour scales are centred at zero and use shared limits derived from their 99.5th absolute percentiles across event68--event70; this affects display only.

The event68 time series in Fig. 8 provides a complementary view. MIKE reaches its maximum depth earlier and exhibits a much larger surface-water-volume peak followed by recession. Surface-only ITZI continues to accumulate depth until near the end. The coupled and DrainLite curves track one another closely in volume and flooded area, which explains the low coupled-label MAE. Their remaining separation in maximum depth is associated with a small number of extreme cells rather than a domain-wide volume mismatch.

![Event68 time series]({figures['event68']})

**Fig. 8. Event68 hydrodynamic response.** Panel (a) is the maximum water depth at any grid cell, panel (b) is surface-water volume, and panel (c) is area above 0.03 m. The DrainLite curve is a leave-one-event-out prediction for event68. MIKE is shown only as an external response pattern.

Event70 produced the same broad behaviour but with a different distribution of residual extremes (Fig. 9). The correction remained coherent over the main inundation corridors, while larger errors occurred near steep wet-dry transitions and at the domain boundary. Agreement across event68 and event70 supports event-wise transfer of the residual pattern within this single spatial window, but it does not establish transfer to another catchment or another network.

![Event70 peak maps]({figures['peak70']})

**Fig. 9. Event70 peak-depth fields and DrainLite residual.** The panel definitions and display limits follow Fig. 7. The event is shown to test whether the spatial behaviour observed for event68 persists under a different rainfall sequence.

### 3.5 Contribution of static drainage descriptors

The improvement from base to all-static was smaller than the improvement from surface-only to base. This hierarchy is informative. Surface depth, rainfall history, DEM and local depth averages already locate most places where the coupled model differs from the surface run. Static network fields refine that estimate but do not determine the time-varying availability of pipe capacity. Figure 10 shows permutation importance for the fixed three-event diagnostic model. Surface depth and terrain-related variables dominate; distance to outfall, local pipe density and local capacity appear among the useful drainage descriptors. The ranking is not interpreted as a causal decomposition because correlated fields can exchange importance when one is permuted.

![Feature importance]({figures['importance']})

**Fig. 10. Permutation importance of the all-static residual model.** Bars show the increase in validation MAE after one feature is permuted; error bars show the standard deviation over repeated permutations. The figure uses the fixed event68--event70 diagnostic model and supports interpretation of feature use, not the primary five-event performance estimate.

Static masks tell the model where an inlet or conduit exists. Diameter, slope and nominal capacity describe a potential hydraulic state. None of these variables reports whether a node is surcharged at a given time, whether a downstream conduit is full, or whether an outfall is restricted. The modest {fnum(all_static['improvement_vs_base_pct'])}% increment is therefore consistent with the information content of the features. The absence of improvement from complete-event SWMM summaries further indicates that coarse event totals cannot substitute for spatially and temporally resolved pipe states.

### 3.6 External comparison with MIKE

MIKE comparison was conducted on the same five events and non-wall mask as the primary evaluation. Surface-only ITZI had the lowest full time-space MAE, {fnum(mike_surface['mae_mm'])} mm. The coupled ITZI-SWMM field had {fnum(mike_connected['mae_mm'])} mm and DrainLite {fnum(mike_drainlite['mae_mm'])} mm. DrainLite therefore did not improve the overall pixelwise agreement with MIKE. Its role is to approximate the selected coupled label, and the coupled label itself differs structurally from MIKE.

{mike_table}

**Table 4. External MIKE comparison on the five quality-usable events.** Signed biases preserve direction; peak-map MAE measures the spatial map of temporal maximum depth. MIKE was not used as a training label or model-selection loss.

The signed global-peak bias tells a different, narrower story. Surface-only ITZI overestimated the MIKE maximum by {fnum(mike_surface['peak_depth_error_mm'])} mm on average. The coupled simulation underestimated it by {fnum(mike_connected['peak_depth_error_mm'])} mm, while DrainLite reduced the magnitude of that signed bias to {fnum(mike_drainlite['peak_depth_error_mm'])} mm. This improvement does not extend to final volume: the mean signed final-volume bias was {fnum(mike_surface['final_volume_error_m3'], 0)} m3 for surface-only and {fnum(mike_drainlite['final_volume_error_m3'], 0)} m3 for DrainLite. The adapter moved the peak magnitude toward MIKE while inheriting the coupled model's lower late-event storage.

![MIKE external comparison]({figures['mike']})

**Fig. 11. External comparison with MIKE for the five quality-usable events.** Panel (a) shows full time-space MAE, panel (b) signed global-peak bias and panel (c) signed final-volume bias. A smaller signed bias is not equivalent to a smaller absolute error, so the three panels are interpreted jointly.

## 4. Discussion

### 4.1 What the residual model learns

The principal result is that the difference between paired surface-only and coupled simulations is sufficiently structured to be learned from a small number of events. More than half of the surface-to-coupled MAE was removed in event-wise validation. Much of that reduction was obtained by the base residual model, which indicates that the pipe effect is correlated with the evolving surface state and terrain. Water can enter the network only where surface water is available, and the locations that remain wet are controlled by the same topographic depressions and conveyance paths present in the surface simulation. A nonlinear residual model can exploit these relationships without reproducing the internal pipe equations.

The signed formulation is essential to this interpretation. A sink-only correction assumes that drainage can only lower water depth at every cell and time. Native two-way coupling does not satisfy that constraint: water routed through the network can return to the surface, and a reduction upstream can shift water toward another location. The positive residual cells in Figs. 7 and 9 are therefore not automatically model errors. Some represent coupled redistribution, although their reliability is limited by the numerical convergence of the SWMM labels.

This distinction also explains why the adapter should not be described as a replacement for SWMM. It learns the response of one fixed conceptual network under the sampled events. It does not conserve pipe mass by construction, solve nodal heads, or enforce conduit momentum. The physical solver remains responsible for generating the paired labels. DrainLite is a surrogate for the surface manifestation of that solver within the studied configuration.

### 4.2 Limits of static network information

The all-static model improved over base by {fnum(all_static['improvement_vs_base_pct'])}%, a useful but modest increment. This result is not a failure of the residual concept; it identifies the missing information. Pipe masks and diameters describe where conveyance could occur, but the instantaneous exchange depends on hydraulic head difference, downstream storage and network connectivity under load. Two cells with identical static attributes can have different exchange because one drains to a surcharged downstream reach. Such behaviour cannot be inferred reliably from local masks alone.

The next model extension should therefore prioritise dynamic network states rather than a larger collection of static proxies. Candidate variables include node head relative to ground, conduit flow, capacity utilisation, surcharge status, outfall discharge and recent cumulative exchange. These states would need to be rasterised or associated with nearby surface cells at each output time. They would also need to be available at forecast time, either from a lightweight pipe-state emulator or from a coupled forecasting system. Event-complete summary variables are not an adequate substitute, as shown by the absence of improvement in the SWMM-assisted diagnostic.

### 4.3 Relationship to LarNO

The adapter is motivated by LarNO but the present validation does not run the pretrained LarNO network. The base field is ITZI surface-only water depth. This choice isolates the learnability of the drainage residual from errors in a neural surface predictor. If a LarNO output replaced ITZI surface-only, the adapter would encounter both surface-model error and drainage correction simultaneously. Performance could be lower, particularly where LarNO and ITZI place the inundation boundary differently.

The results nevertheless define a concrete route toward drainage-aware LarNO forecasting. A LarNO prediction can provide the evolving surface-depth field, while the static network layers and rainfall summaries remain additional inputs to the residual adapter. End-to-end evaluation would require LarNO predictions for the same event and grid, followed by retraining or transfer of DrainLite on those predictions. Until that experiment is completed, `LarNO-compatible` is the appropriate description. Claims of a modified LarNO neural operator or a fully integrated LarNO-SWMM model would exceed the evidence.

### 4.4 Physical-model uncertainty and MIKE discrepancy

The MIKE comparison exposes differences in both peak timing and late-event storage. The reference generally rises and recedes earlier than ITZI. A 1 mm h-1 effective loss improves the falling limb but does not reconcile the models. Several mechanisms remain. The ITZI raster boundary is closed, the rainfall on wall cells is redistributed, and the conceptual network does not reproduce the surveyed drainage system embedded in the MIKE reference. Surface roughness and infiltration were not independently recalibrated against observations. Any of these differences can alter the balance between surface storage and outflow.

The coupled labels also have numerical limitations. Three events failed the continuity screen, and even accepted events contained many nonconverging dynamic-wave steps. Event-scale continuity can remain acceptable when short-time iterative convergence is poor. This matters most near sharp surcharge transitions, precisely where a residual model is asked to learn positive and negative corrections. Excluding the highest-error events from primary validation reduces but does not remove this uncertainty.

The external results should therefore be read by metric. DrainLite improved the signed global-peak bias but worsened full time-space MAE and final-volume agreement relative to surface-only. These findings are not contradictory: a correction can move one global extreme toward the reference while shifting many moderate-depth cells away from it. Reporting only the peak would overstate physical agreement; reporting only MAE would miss the more balanced peak magnitude. The present evidence supports learnability of the ITZI-SWMM residual, not universal superiority to MIKE.

### 4.5 Generalisability and reproducibility

Eight stored events, five quality-usable events and one spatial window are a limited basis for generalisation. Leave-one-event-out validation prevents rainfall-event leakage, but all folds share the same terrain and network. The model can therefore learn fixed spatial coordinates and recurrent low points. A stronger spatial test would train on one window or network configuration and evaluate another. Multiple alternative pipe layouts would also be needed to demonstrate sensitivity to network design rather than memorisation of one static map.

The use of a conceptual road-aligned network is both a strength and a limitation. It permits a complete connected experiment where surveyed data are unavailable, but its pipe sizes, cover and outlets are design assumptions. The results can support methodological development and controlled comparison. They cannot quantify the benefit of the actual Shenzhen drainage system. Repetition with surveyed assets, measured flood observations and independent rainfall events is needed before operational interpretation.

All reported quantitative results are traceable to local CSV and NPY outputs. The accompanying scientific-integrity audit records array shapes, units, masks, file hashes, event membership and the generating script for each principal result. Public benchmark DEM, rainfall and MIKE fields are explicitly separated from the ITZI-SWMM and DrainLite outputs generated in this study.

## 5. Conclusions

Paired ITZI surface-only and native ITZI-SWMM simulations were used to isolate the surface-water-depth response of a connected road-aligned conceptual drainage network. A lightweight signed-residual model reproduced much of that response without retraining a neural-operator backbone. Under strict five-event leave-one-event-out validation, the all-static model reduced MAE to the coupled label from {fnum(surface['mae_mm'])} to {fnum(all_static['mae_mm'])} mm and achieved CSI values of {fnum(all_static['csi_0p03'])} and {fnum(all_static['csi_0p15'])} at 0.03 and 0.15 m. Static network descriptors improved over the terrain-rainfall residual baseline by {fnum(all_static['improvement_vs_base_pct'])}%, showing that network location and nominal capacity contain useful information but explain only part of the dynamic coupling response.

The comparison with MIKE places a clear boundary on this result. DrainLite moderated signed peak bias but did not improve full time-space error or final water-volume agreement. The coupled labels also retain dynamic-wave stability warnings and represent a conceptual, not surveyed, network. Accordingly, the contribution is a drainage-aware residual adapter compatible with a future LarNO surface predictor, together with a reproducible paired-label workflow. The next substantive step is to evaluate the adapter on actual LarNO outputs and replace static-only pipe descriptors with time-varying node head, conduit flow, surcharge and outfall states.

## Data availability

The rainfall, DEM and MIKE reference fields originate from the publicly released LarNO urban-flood benchmark. The paired ITZI surface-only, ITZI-SWMM coupled and DrainLite arrays were generated in the present workspace. [待补充：公开数据归档地址和永久标识符。]

## Code availability

The complete local workflow includes network construction, coupled simulation, data-set assembly, residual training, cross-validation, metric recomputation, figure generation and report rendering. The accompanying `REPRODUCIBILITY.md` and `scientific_integrity_audit.md` identify the current entry points and evidence files. [待补充：公开代码仓库地址和发布版本标识符。]

## CRediT authorship contribution statement

[待补充：作者名单和贡献角色确认后填写。]

## Declaration of competing interests

[待补充：由全体作者确认利益冲突声明。]

## Acknowledgments

[待补充：作者确认基金、数据与软件致谢信息后填写。]

## References

Bates, P.D., Horritt, M.S., Fewtrell, T.J., 2010. A simple inertial formulation of the shallow water equations for efficient two-dimensional flood inundation modelling. Journal of Hydrology 387, 33-45.

Cao, X., Wang, B., Yao, Y., Zhang, L., Xing, Y., Mao, J., Zhang, R., Fu, G., Borthwick, A.G.L., Qin, H., 2025. U-RNN high-resolution spatiotemporal nowcasting of urban flooding. Journal of Hydrology 659, 133117.

Cao, X., Yao, Y., Wang, Z., Zhao, Z., Borthwick, A.G.L., Qin, H., 2026. Large-scale urban flood modeling and zero-shot high-resolution generalization with LarNO. Journal of Hydrology 676, 135686. https://doi.org/10.1016/j.jhydrol.2026.135686.

Courty, L.G., Pedrozo-Acuna, A., Bates, P.D., 2017. Itzi (version 17.1): an open-source, distributed GIS model for dynamic flood simulation. Geoscientific Model Development 10, 1835-1847. https://doi.org/10.5194/gmd-10-1835-2017.

Fraehr, N., Wang, Q.J., Wu, W., Nathan, R., 2023. Supercharging hydrodynamic inundation models for instant flood insight. Nature Water 1, 835-843.

Kovachki, N., Li, Z., Liu, B., Azizzadenesheli, K., Bhattacharya, K., Stuart, A., Anandkumar, A., 2023. Neural operator: Learning maps between function spaces with applications to PDEs. Journal of Machine Learning Research 24, 1-97.

Li, Z., Kovachki, N., Azizzadenesheli, K., Liu, B., Bhattacharya, K., Stuart, A., Anandkumar, A., 2021. Fourier neural operator for parametric partial differential equations. Proceedings of the 9th International Conference on Learning Representations.

McDonnell, B., Ratliff, K., Tryby, M., Wu, J., Mullapudi, A., 2020. PySWMM: The Python interface to Stormwater Management Model (SWMM). Journal of Open Source Software 5, 2292. https://doi.org/10.21105/joss.02292.

Rossman, L.A., Simon, M.A., 2022. Storm Water Management Model User's Manual Version 5.2. U.S. Environmental Protection Agency, Cincinnati, Ohio.

## Appendix A. Event membership and label-quality rules

The stored data set contains eight events. The primary model analysis uses event1, event67, event68, event69 and event70. Event20, event65 and event66 are excluded from both training and testing in the primary analysis because their absolute SWMM continuity errors exceed 8%. Accepted-only results for event68--event70 and the eight-event run are sensitivity analyses.

## Appendix B. Units and masking

Rainfall is millimetres per five-minute frame. Water depth is stored in metres. The learning target is converted to millimetres. Volume is computed in cubic metres from depth multiplied by 400 m2 per cell. Evaluation uses the 43,606 cells below the 49.9 m wall-height threshold. The same mask is applied to each model in a comparison.
"""

    OUT_MD.write_text(text, encoding="utf-8")
    OUT_ALIAS.write_text(text, encoding="utf-8")
    print(OUT_MD)
    print(OUT_ALIAS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
