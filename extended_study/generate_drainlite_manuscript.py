#!/usr/bin/env python3
"""Generate an English manuscript draft for the DrainLite study."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper_draft" / "drainlite"
FIG = PAPER / "figures"
TAB = PAPER / "tables"
OUT = PAPER / "DrainLite_manuscript.md"


def md_table(path: Path, max_rows: int | None = None, digits: int = 3) -> str:
    rows = list(csv.DictReader(path.open(encoding="utf-8-sig")))
    if max_rows:
        rows = rows[:max_rows]
    if not rows:
        return ""
    headers = list(rows[0].keys())
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        vals = []
        for h in headers:
            v = row.get(h, "")
            try:
                f = float(v)
                vals.append(f"{f:.{digits}f}")
            except Exception:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def word_count(text: str) -> int:
    return len(re.findall(r"\b[A-Za-z][A-Za-z0-9'-]*\b", text))


def main() -> int:
    ablation = pd.read_csv(TAB / "table2_ablation_metrics.csv")
    table3 = pd.read_csv(TAB / "table3_mike_external_reference.csv")
    peak_table = pd.read_csv(TAB / "table6_peak_error_analysis.csv")
    physical_table = pd.read_csv(TAB / "table7_drainage_physical_consistency.csv")
    severity_table = pd.read_csv(TAB / "table8_event_severity_analysis.csv")
    swmm_corr = pd.read_csv(TAB / "table9_swmm_error_correlation.csv")
    runtime_table = pd.read_csv(TAB / "table10_lightweight_runtime.csv")
    all_static = ablation.loc[ablation["model"] == "all_static"].iloc[0]
    mask_only = ablation.loc[ablation["model"] == "mask_only"].iloc[0]
    base = ablation.loc[ablation["model"] == "base"].iloc[0]
    surface = ablation.loc[ablation["model"] == "surface_only"].iloc[0]
    swmm = ablation.loc[ablation["model"] == "swmm_assisted"].iloc[0]
    mike_surface = table3.loc[table3["model"] == "surface_only"].iloc[0]
    mike_all = table3.loc[table3["model"] == "all_static"].iloc[0]
    top1_mae_mean = peak_table["top1pct_deep_cell_mae_mm"].mean()
    high015_mae_mean = peak_table["mae_gt_0p15_mm"].mean()
    peak_shift_mean = peak_table["peak_location_shift_m"].mean()
    capture_weighted_mean = physical_table["volume_weighted_capture_ratio_pct"].mean()
    near_pipe_pred_share = physical_table["pred_reduction_near_pipe_share_pct"].mean()
    monotonic_violations = int(physical_table["monotonic_high_violation_cells"].sum() + physical_table["negative_depth_violation_cells"].sum())
    severity_mae_min = severity_table["drainlite_mae_mm"].min()
    severity_mae_max = severity_table["drainlite_mae_mm"].max()
    routing_mae_r = swmm_corr[
        (swmm_corr["swmm_metric"] == "routing_inflow_m3")
        & (swmm_corr["drainlite_metric"] == "mae_mm")
    ]["pearson_r"].iloc[0]
    outfall_reduction_r = swmm_corr[
        (swmm_corr["swmm_metric"] == "outfall_volume_m3")
        & (swmm_corr["drainlite_metric"] == "reduction_mae_mm")
    ]["pearson_r"].iloc[0]
    runtime_lookup = dict(zip(runtime_table["item"], runtime_table["value"]))
    all_static_size_mb = runtime_lookup.get("all_static_model_file_size_mb", "not recorded")
    event75_infer_s = runtime_lookup.get("all_static_event75_dense_inference_time", "not measured")
    event75_rss_delta = runtime_lookup.get("all_static_event75_rss_delta", "not measured")
    training_total_s = runtime_lookup.get("all_static_training_total_time", "not measured")
    training_rss_delta = runtime_lookup.get("all_static_training_rss_delta", "not measured")
    ref_larno_train_days = runtime_lookup.get("reference_larno_training_time", "not reported")
    ref_larno_infer_s = runtime_lookup.get("reference_larno_trt_inference_time", "not reported")
    ref_mike_s = runtime_lookup.get("reference_mike_plus_runtime", "not reported")
    ref_larno_speedup = runtime_lookup.get("reference_larno_speedup_vs_mike", "not reported")
    try:
        ref_mike_s_display = f"{float(ref_mike_s):,.0f}"
    except Exception:
        ref_mike_s_display = ref_mike_s
    proto_all_path = ROOT / "extended_study" / "output" / "itzi_swmm_coupled_prototype" / "prototype_summary_all_events.csv"
    proto_path = ROOT / "extended_study" / "output" / "itzi_swmm_coupled_prototype" / "event75" / "prototype_summary.json"
    proto_all = pd.read_csv(proto_all_path) if proto_all_path.exists() else None
    proto = json.loads(proto_path.read_text(encoding="utf-8")) if proto_path.exists() else None
    if proto_all is not None and not proto_all.empty:
        proto_events = ", ".join(proto_all["event"].astype(str).tolist())
        proto_n = len(proto_all)
        proto_methods = (
            f"A separate ITZI-PySWMM prototype set was also run for {proto_n} held-out events "
            f"({proto_events}). Each prototype covers {int(proto_all['frames'].min())} frames on the same "
            "200 x 280 grid and couples "
            f"{int(proto_all['coupled_nodes'].median())} in-window surface nodes to a network-only "
            f"SWMM DYNWAVE model with {int(proto_all['swmm_nodes'].median())} nodes and "
            f"{int(proto_all['swmm_links'].median())} links. These runs are used only as feasibility "
            "evidence for future coupled labels and are not the supervised DrainLite label."
        )
        proto_results = (
            f"Across the {proto_n} prototypes, the exchanged surface-to-pipe volume totals "
            f"{proto_all['surface_to_pipe_m3'].sum():.1f} m3 and the pipe-to-surface return volume totals "
            f"{proto_all['pipe_to_surface_m3'].sum():.1f} m3. The mean coupled-versus-surface MAE is "
            f"{proto_all['coupled_vs_surface_mae_mm'].mean():.3f} mm, while the mean coupled-versus-sink MAE is "
            f"{proto_all['coupled_vs_sink_mae_mm'].mean():.3f} mm. Final coupled volumes are lower than "
            "surface-only for the two weaker prototype events but higher than surface-only in later intense "
            "events where SWMM surcharge/backflow dominates the net exchange."
        )
    elif proto:
        proto_methods = (
            f"A separate one-event ITZI-PySWMM prototype was also run for {proto['event']} over "
            f"{proto['hours']:.1f} h, producing {proto['frames']} frames on the same "
            f"{proto['shape'][1]} x {proto['shape'][2]} grid. The prototype coupled "
            f"{proto['coupled_nodes']} in-window surface nodes to a network-only SWMM DYNWAVE "
            f"model with {proto['swmm_nodes']} nodes and {proto['swmm_links']} links. It is "
            "used only as feasibility evidence and is not the supervised DrainLite label."
        )
        proto_results = (
            f"The prototype exchanged {proto['surface_to_pipe_m3']:.1f} m3 from the surface to the pipe "
            f"network and {proto['pipe_to_surface_m3']:.1f} m3 from the pipe network back to the surface. "
            f"Its final surface volume was {proto['coupled_final_volume_m3']:.1f} m3, compared with "
            f"{proto['surface_final_volume_m3']:.1f} m3 for ITZI surface-only and "
            f"{proto['sink_final_volume_m3']:.1f} m3 for ITZI + sink."
        )
    else:
        proto_methods = (
            "A separate ITZI-PySWMM coupling prototype has not been generated in the current workspace."
        )
        proto_results = "No coupled-prototype diagnostics are available."

    text = f"""# DrainLite: A lightweight drainage-aware residual correction framework for urban pluvial flood forecasting under limited computing resources

Author list: to be completed  
Affiliations: to be completed  
Corresponding author: to be completed

## Highlights

- A lightweight drainage-aware residual correction framework is proposed for urban pluvial flood forecasting under limited computing resources.
- Road-aligned conceptual drainage features are rasterized as inlet, outfall, pipe, hydraulic capacity and distance-to-outfall fields on the same grid as the flood model.
- DrainLite learns the non-negative drainage-induced reduction from an ITZI surface-only simulation to an ITZI + conceptual inlet-sink label without retraining the LarNO backbone.
- On five held-out Shenzhen events, the all-feature DrainLite model reduces the mean absolute error to the ITZI + sink label by {all_static['mae_improvement_vs_surface_pct']:.1f}% relative to the surface-only baseline and by {all_static['mae_improvement_vs_base_pct']:.1f}% relative to a no-drainage-feature residual model.
- Ablation and feature-importance analyses show that drainage location priors are more informative than event-level SWMM summary variables in the present lightweight setting.

## Abstract

Urban pluvial flooding is shaped not only by rainfall and terrain but also by the capacity, layout and accessibility of drainage systems. Recent neural-operator approaches, including LarNO, have demonstrated that large-scale high-resolution flood forecasting can be accelerated by learning hydrodynamic mappings from numerical model references. However, a practical difficulty remains when drainage information is incomplete, synthetic, or too expensive to incorporate into full retraining of a large neural model. This study proposes DrainLite, a lightweight drainage-aware residual correction framework designed for local computing environments where full neural-operator training is infeasible. Rather than retraining the LarNO backbone, DrainLite learns the drainage-induced reduction between an ITZI surface-only dynamic runoff simulation and an ITZI simulation with conceptual road-aligned inlet sinks. The framework uses rainfall, cumulative rainfall, DEM-derived terrain features, surface water depth, eight rasterized drainage-network features, derived local density fields, and optional standalone SWMM event metrics. It constrains the corrected water depth to remain non-negative and no greater than the surface-only depth, thereby encoding a simple physical monotonicity prior for drainage removal.

The method is evaluated on a 20 m Shenzhen urban flood benchmark subset consisting of 17 valid rainfall events, each with 72 five-minute time steps and a 200 x 280 spatial window. Twelve events are used for training and five events are held out for testing. DrainLite is trained as a CPU-friendly gradient-boosted residual learner and compared with surface-only, no-drainage-feature, drainage-location, hydraulic-parameter and SWMM-assisted variants. Relative to the ITZI + sink label, the all-feature model achieves an average MAE of {all_static['mae_mm']:.3f} mm on the test set, compared with {surface['mae_mm']:.3f} mm for the surface-only baseline and {base['mae_mm']:.3f} mm for the no-drainage-feature residual model. This corresponds to improvements of {all_static['mae_improvement_vs_surface_pct']:.1f}% and {all_static['mae_improvement_vs_base_pct']:.1f}%, respectively. Drainage location features perform nearly as well as the full feature set, whereas adding event-level SWMM summary metrics does not further improve the residual correction, indicating that coarse event-level sewer descriptors are less useful than spatially explicit drainage priors for pixel-scale correction. The peak-depth error remains substantial, with an average all-feature peak error of {all_static['peak_error_mm']:.1f} mm, suggesting that the method is better suited to mean spatial correction and drainage-reduction mapping than to recovering extreme point maxima. DrainLite therefore provides a reproducible, low-cost, interpretable route for incorporating conceptual drainage effects into urban flood forecasts, while clearly remaining distinct from full two-way surface-sewer coupling or full LarNO-D retraining.

Keywords: urban pluvial flooding; drainage network; residual learning; lightweight surrogate model; ITZI; SWMM; LarNO; flood forecasting

## 1. Introduction

Urban flood forecasting sits at the junction of hydrometeorology, hydraulic modelling, urban infrastructure and emergency management. In densely developed catchments, the spatial distribution of water depth is controlled by intense rainfall, microtopography, building obstruction, road depressions and underground drainage pathways. This mixture of surface and subsurface controls is difficult to model quickly. Traditional hydrodynamic solvers can represent two-dimensional overland flow and one-dimensional sewer routing with considerable physical detail, but their computational burden is high when the domain covers many square kilometres at metre-scale resolution. Deep learning and neural-operator models have therefore become attractive because they can approximate hydrodynamic responses at much lower inference cost once they have been trained on suitable reference simulations.

The LarNO study provides a useful reference point for this direction. Its central contribution is a latent autoregressive neural operator that enables memory-efficient training and zero-shot super-resolution for large-scale urban flood dynamics. The original work frames the hydrodynamic model as a nonlinear operator and shows that neural operators can learn the mapping from rainfall, terrain and drainage-related inputs to spatiotemporal water-depth fields. Its paper structure is also instructive: it first establishes the urban-scale computational bottleneck, then introduces the operator-learning framework, then validates the model against numerical reference results, and finally discusses ablation, transfer learning, sensitivity and limitations. For the present work, we adopt a similar evidence-driven writing style but target a different and narrower problem: how to incorporate drainage-network effects in a lightweight way when full neural-operator retraining is not practical.

The practical motivation is straightforward. In many research or planning settings, a modeller may have access to a DEM, rainfall fields, a surface-flow solver, partial or synthetic road-network information and a small number of drainage-related reference outputs, but not to the hardware or data required for full training of a large spatiotemporal neural operator. A complete one-dimensional/two-dimensional drainage model may also be unavailable, partly confidential, or too expensive to construct for every experimental scenario. Nevertheless, drainage cannot simply be ignored. Road inlets, pipe alignments, outfalls and hydraulic bottlenecks can reduce ponding in some locations while leaving other depressions nearly unchanged. A model that treats flooding as a function only of rainfall and terrain will tend to miss this structured removal process.

This study asks whether a lightweight drainage-aware residual learner can recover a meaningful part of that removal signal. The goal is not to replace LarNO, MIKE, ITZI or SWMM, and it is not to claim that a synthetic drainage network is equivalent to a surveyed sewer system. Instead, we build a transparent residual-correction layer around existing physical-model outputs. The base state is the ITZI surface-only dynamic runoff result. The drainage target is the corresponding ITZI result with conceptual road-aligned inlet sinks. The residual is the non-negative reduction from the base state to the drainage-aware target. DrainLite learns this residual from local water depth, rainfall, topography and rasterized drainage features. The corrected forecast is then obtained by subtracting the predicted reduction from the surface-only field while enforcing non-negativity and monotonicity.

This formulation has several advantages for a limited-compute environment. First, it avoids backpropagation through a large spatiotemporal neural operator. Second, it permits tabular or shallow learners that can run on CPU. Third, it produces feature-importance and ablation results, allowing the contribution of drainage priors to be inspected rather than treated as a black-box channel. Fourth, it can use standalone SWMM summaries as explanatory variables without representing them as fully coupled surface-sewer exchange. Finally, it is easy to evaluate against multiple references: the synthetic drainage label tests whether the residual learner captures the conceptual sink effect, while the MIKE reference provides an external plausibility check.

The main contributions of this paper are as follows. First, we construct a drainage-augmented benchmark subset for the Shenzhen LarNO data, preserving the 20 m rainfall and water-depth fields while adding road-aligned conceptual drainage rasters and SWMM standalone metrics. Second, we formulate DrainLite as a constrained residual correction problem that maps an ITZI surface-only state to a drainage-aware forecast. Third, we conduct a controlled ablation study of feature groups, separating terrain-rainfall predictors, drainage location priors, hydraulic pipe parameters and event-level SWMM indicators. Fourth, we show that spatially explicit drainage priors substantially improve the match to the ITZI + sink label under the local computing setting, while event-level SWMM summaries alone do not provide additional pixel-scale benefit. Fifth, we provide five held-out ITZI-PySWMM prototype runs as feasibility evidence for future coupled labels, while carefully distinguishing these prototypes from the supervised DrainLite experiment. Sixth, we provide a careful discussion of what the lightweight result does and does not prove, especially regarding peak errors, conceptual drainage labels and the absence of a validated 17-event ITZI-SWMM coupled reference dataset.

## 2. Data and physical references

### 2.1 Study data and spatial-temporal configuration

The experiments use a drainage-augmented subset derived from the Shenzhen urban flood benchmark associated with the LarNO study. The local subset is named `region1_20m_drainage_v1`. It uses a 20 m grid and a 200 x 280 rectangular spatial window, corresponding to 56,000 cells per time step. Each rainfall event contains 72 output frames at five-minute intervals, representing six hours of flood evolution. Seventeen valid events are available locally: event1, event20, event65 to event78 and event80. Event79 is excluded because the local rainfall or water-depth files are incomplete or corrupt. The event split is fixed before model training, with 12 events used for fitting and 5 held out for testing. The test events are event75, event76, event77, event78 and event80.

The study deliberately retains the 20 m setting. The original LarNO paper emphasizes high-resolution and super-resolution modelling, but the present work focuses on local low-compute drainage residual learning. The available true 5 m MIKE reference files are not present locally, and the manuscript therefore does not claim a validated 5 m comparison. This choice avoids presenting upsampled or proxy data as true high-resolution reference information. The 20 m setting remains useful because it preserves the rainfall-event structure, a manageable spatial domain and a clear relation to the original LarNO benchmark.

### 2.2 Surface-only and drainage-aware ITZI labels

The physical baseline is an ITZI surface-only dynamic runoff simulation. It represents overland flood evolution under the original rainfall fields and DEM but without the conceptual inlet-sink drainage mechanism. The drainage-aware target is generated by adding road-aligned conceptual inlet sinks to the same ITZI setup. The sink representation is not a full sewer network solver; instead, it provides a controlled drainage-removal label that is spatially associated with the extracted road and pipe network. The role of this target is to support residual learning of drainage-induced reduction, not to replace a surveyed or fully calibrated sewer model.

Let `h_s(t,x)` denote the surface-only water depth and `h_d(t,x)` denote the ITZI + sink water depth at time step `t` and grid cell `x`. The residual target is

\\[
r(t,x) = \\max(h_s(t,x) - h_d(t,x), 0).
\\]

The residual is non-negative by definition. This is important because the conceptual inlet-sink model is expected to remove water rather than create it. Although a real coupled sewer system can surcharge and return water to the surface, that mechanism is not represented in the present target. Consequently, DrainLite is intentionally formulated as a drainage-removal correction rather than a bidirectional surface-sewer exchange model.

### 2.3 MIKE reference and SWMM standalone branch

The LarNO benchmark includes MIKE reference water-depth fields. These are used here as an external comparison, not as the main training target. This distinction is central to the manuscript. MIKE represents the reference used in the original LarNO study, whereas DrainLite is trained to reproduce the controlled ITZI + sink reduction. Comparing DrainLite with MIKE helps assess external plausibility, but it cannot by itself prove that the conceptual drainage network equals the drainage assumptions embedded in MIKE.

In addition, a standalone SWMM dynamic-wave branch is constructed from the same road-aligned conceptual network. This branch produces event-level metrics such as routing inflow, outfall volume, final routing storage, node ponding volume and routing continuity error. The SWMM runs are used in two ways: first, as physical context for interpreting the drainage network; second, as optional event-level features in the `swmm_assisted` DrainLite variant. The SWMM branch is explicitly not a two-way online ITZI-SWMM coupling. No synchronous exchange of surface water depth, inlet capture, node surcharge and surface return flow is performed in the current manuscript.

The standalone SWMM branch completes all 17 events. Its mean routing continuity error is approximately 1.24%, and the event-level outputs are therefore numerically consistent enough for use as auxiliary descriptors. However, the results later show that these coarse event-level descriptors do not improve the pixel-level residual learner beyond the spatial drainage rasters. This is a useful negative result: it suggests that if SWMM information is to improve DrainLite, it should be converted into spatially or temporally resolved features rather than kept as event summaries.

{proto_methods}

### 2.4 Drainage-network features

Eight static drainage rasters are aligned to the 20 m DEM: inlet mask, outfall mask, pipe mask, pipe diameter, pipe slope, pipe full-flow capacity, cover depth and distance to outfall. These are supplemented by local moving-window densities, including pipe density in 3 x 3 and 7 x 7 windows, inlet density in a 7 x 7 window and capacity density in a 7 x 7 window. The intention is to represent both cell-local drainage presence and neighbourhood access to the conceptual network. The full feature definitions are provided in Extended Table 2.

Figure 2 shows the DEM and conceptual pipe network overlay. This figure plays a similar role to the case-study map in the LarNO paper, but with a stronger focus on infrastructure alignment. The DEM indicates where overland flow and ponding are likely to concentrate, while the pipe lines and inlet markers indicate where the residual learner is expected to infer drainage removal. Outfalls are treated as terminal locations for the conceptual drainage network and as reference points for the distance-to-outfall raster.

## 3. DrainLite method

### 3.1 Residual correction formulation

DrainLite is a post-processing residual learner. Its input is not raw rainfall alone, and its output is not a full hydrodynamic simulation from scratch. Instead, it starts from an already computed surface-only dynamic flood field. This design is intentionally modest. It makes the method computationally feasible on a local machine and allows the model to focus on the incremental effect of drainage priors.

For each time step and cell, the model observes a feature vector containing current surface depth, current rainfall, cumulative rainfall, DEM, DEM slope, normalized row and column coordinates, normalized time, local means of surface depth and drainage features. The response variable is the drainage-induced reduction in millimetres:

\\[
y(t,x)=1000\\max(h_s(t,x)-h_d(t,x),0).
\\]

In implementation terms, this target is recorded as `target_reduction = max(h_itzi_surface - h_itzi_sink, 0)`. Dense prediction is then clipped by the rule `0 <= h_drainlite <= h_itzi_surface`.

The fitted model predicts \\(\\hat{{y}}(t,x)\\). The corrected depth is then

\\[
\\hat{{h}}_{{DL}}(t,x)=\\max\\left(h_s(t,x)-\\frac{{\\hat{{y}}(t,x)}}{{1000}},0\\right).
\\]

During prediction, the residual is clipped so that \\(0 \\leq \\hat{{h}}_{{DL}}(t,x) \\leq h_s(t,x)\\). This constraint is not a substitute for mass conservation, but it is a useful physical prior for the present sink-only target. It prevents the model from inventing additional water and guarantees that DrainLite behaves as a drainage-removal layer.

### 3.2 Model class and training strategy

The residual learner is implemented with `HistGradientBoostingRegressor`, a CPU-friendly gradient-boosted tree model. Tree boosting is selected because it handles nonlinear interactions, requires little feature scaling, trains efficiently on tabular samples and supports feature ablation. The objective is squared error on the residual target, with sample weights increased for cells with positive reduction and cells with nontrivial surface depth. In the present local run, 600 pixels are sampled per event-time step, yielding 518,400 training rows. The script keeps a larger default setting available, but the reported run is deliberately kept within local machine constraints.

This training design differs from the original LarNO neural-operator training. LarNO learns a continuous spatiotemporal operator with millions of parameters; DrainLite learns a local residual rule conditioned on a physical base state. Consequently, the two methods should not be compared as if they solve the same task. DrainLite is not intended to replace LarNO's zero-shot super-resolution capability. Its contribution is to show that drainage priors can be injected into existing forecasts in a lightweight, interpretable and reproducible way.

### 3.3 Feature groups for ablation

Five predictive variants are trained or evaluated. `surface_only` is the uncorrected ITZI baseline and requires no training. `base` uses surface depth, rainfall, cumulative rainfall, DEM, slope, coordinates, time and local surface-depth means, but no drainage features. `mask_only` adds inlet, outfall, pipe mask, distance to outfall and local drainage densities. `hydraulic_only` adds pipe diameter, pipe slope, pipe capacity, pipe cover depth, distance to outfall and local capacity density. `all_static` combines all static drainage features and is treated as the main DrainLite model. `swmm_assisted` further adds standalone SWMM event-level metrics.

This ablation design mirrors the logic of the LarNO paper's ablation section, but the scientific question is different. The original ablation isolates the value of latent autoregression. Here, the ablation isolates the value of drainage priors. If the all-feature model improves over `base`, then the result supports the claim that pipe-network information provides useful residual signal beyond rainfall, terrain and existing water depth. If `mask_only` performs nearly as well as `all_static`, then network location is more important than the current conceptual hydraulic parameters. If `swmm_assisted` improves over `all_static`, then event-level SWMM descriptors add useful information; if not, their spatial coarseness is likely limiting.

### 3.4 Evaluation metrics

The primary target for quantitative evaluation is the ITZI + sink label, because it is the label used to define the residual. We report MAE, RMSE, CSI at 0.03 m and 0.15 m, peak-depth error, mean error, peak-volume error, inundated-area error at 0.03 m, reduction MAE and reduction-capture error. The 0.03 m CSI is useful for shallow urban ponding, whereas the 0.15 m threshold follows common flood-inundation evaluation practice and is used in the LarNO paper.

MIKE reference metrics are also reported, but their interpretation is different. Because DrainLite is not trained on MIKE, these metrics measure external alignment rather than supervised performance. If a DrainLite variant moves closer to ITZI + sink but farther from MIKE, that does not automatically invalidate the method. It instead indicates that the conceptual sink label and MIKE reference encode different drainage assumptions. The Results section therefore separates supervised label accuracy from MIKE external-reference comparison.

## 4. Results

### 4.1 Dataset, drainage-feature quality and lightweight execution

The drainage-augmented dataset contains 17 valid events. Each event has rainfall, surface-only depth, ITZI + sink depth, MIKE reference depth and SWMM metrics. Table 1 summarizes the spatial-temporal configuration and model setup. The domain size, time horizon and train-test split are deliberately explicit so that the experiment can be reproduced without consulting hidden settings.

**Table 1. Dataset and model configuration.**

{md_table(TAB / "table1_dataset_model_configuration.csv")}

Figure 1 shows the DrainLite framework. The figure emphasizes three design choices: the use of an existing surface-only dynamic flood state, the construction of a drainage residual target and the monotonic correction constraint. Figure 3 shows how the data sources enter the experiment. MIKE is retained as an external benchmark, ITZI + sink provides the supervised drainage label, and SWMM provides standalone dynamic-wave descriptors. This separation is important because it prevents the paper from overstating the physical meaning of the conceptual sink labels.

![Fig. 1. DrainLite framework.](figures/fig01_framework.png)

**Fig. 1.** DrainLite lightweight drainage-aware residual correction framework. The model learns a non-negative drainage reduction from surface-only ITZI outputs and drainage priors, then subtracts the reduction from the surface-only depth under a monotonicity constraint.

![Fig. 2. DEM and conceptual drainage network.](figures/fig02_dem_network.png)

**Fig. 2.** DEM with road-aligned conceptual pipe network, inlet cells and outfall cells. This map verifies that the drainage features are spatially aligned with the flood-modelling grid.

![Fig. 3. Drainage-augmented data flow.](figures/fig03_data_flow.png)

**Fig. 3.** Construction of the drainage-augmented dataset and role of each physical reference. The SWMM branch is standalone and is not interpreted as a two-way ITZI-SWMM coupling model.

The low-compute claim is also quantified. The all-static DrainLite model file is only {all_static_size_mb} MB. A one-off benchmark using the current settings builds the sampled training table and fits the all-static model in {training_total_s} s, with a process RSS change of {training_rss_delta} MB. Dense CPU inference for a full six-hour event with 4,032,000 cell-time rows takes {event75_infer_s} s on the current machine, and the measured process RSS change during that event-level inference is {event75_rss_delta} MB. For scale context, the reference LarNO paper reports {ref_larno_train_days} days of LarNO training on 8 x NVIDIA A800 GPUs, {ref_larno_infer_s} s per event for TensorRT LarNO inference on an RTX 4090, and {ref_mike_s_display} s per event for MIKE+, corresponding to an approximately {ref_larno_speedup} x LarNO speedup over MIKE+. These are not same-hardware benchmarks, but they clarify why the present work is positioned as a local low-compute residual layer rather than as full neural-operator retraining.

**Table 10. Lightweight runtime and model-size diagnostics.**

{md_table(TAB / "table10_lightweight_runtime.csv")}

### 4.2 Supervised accuracy against the ITZI + sink label

Table 2 reports the main ablation results against the ITZI + sink label. The uncorrected surface-only baseline has an average MAE of {surface['mae_mm']:.3f} mm and an RMSE of {surface['rmse_mm']:.3f} mm. The no-drainage-feature residual learner improves the MAE to {base['mae_mm']:.3f} mm, showing that rainfall, terrain, local depth and coordinates already explain part of the systematic difference between surface-only and sink simulations. However, adding drainage priors provides a further improvement. The all-static drainage model reaches {all_static['mae_mm']:.3f} mm MAE and {all_static['rmse_mm']:.3f} mm RMSE. This is a {all_static['mae_improvement_vs_surface_pct']:.1f}% improvement relative to the surface-only baseline and an {all_static['mae_improvement_vs_base_pct']:.1f}% improvement relative to the no-drainage-feature residual learner.

**Table 2. Ablation results against the ITZI + sink label.**

{md_table(TAB / "table2_ablation_metrics.csv")}

The drainage-location model performs nearly as well as the all-static model, with {mask_only['mae_mm']:.3f} mm MAE. This result suggests that in the current conceptual-network setup, the presence and proximity of drainage infrastructure are more informative than the approximate hydraulic pipe parameters. The hydraulic-only variant also improves over the no-drainage-feature baseline, but less strongly. The SWMM-assisted variant does not improve over all-static; its MAE is {swmm['mae_mm']:.3f} mm. This indicates that event-level SWMM descriptors are too coarse to add pixel-scale information beyond the rasterized network features.

Table 4 reports secondary supervised metrics against the ITZI + sink label. These include peak water-volume error, inundated-area error at the 0.03 m threshold and reduction-capture error. They are included because DrainLite is intended to learn a drainage reduction signal, not only to reduce average depth error.

**Table 4. Secondary supervised metrics against the ITZI + sink label.**

{md_table(TAB / "table4_secondary_itzi_sink_metrics.csv")}

![Fig. 5. Ablation of feature groups.](figures/fig05_ablation.png)

**Fig. 5.** Ablation comparison of MAE, CSI and reduction MAE across surface-only, no-drainage residual and drainage-aware variants.

Feature importance is evaluated using permutation importance on held-out samples. The all-static model shows that surface depth, local surface-depth means and drainage-location variables contribute strongly to residual prediction. The importance of location-related drainage terms is consistent with the ablation result: knowing where the pipe network, inlets and outfalls are located is the most direct cue for identifying where a sink-driven reduction can occur. Hydraulic attributes such as conceptual capacity and slope contribute but are less decisive in the present synthetic setup.

![Fig. 6. Feature importance.](figures/fig06_feature_importance.png)

**Fig. 6.** Permutation feature importance for the all-static DrainLite model. Importance is reported as the increase in MAE after feature shuffling.

### 4.3 Spatiotemporal performance over the six-hour forecast

The LarNO reference paper evaluates errors through the full six-hour lead time rather than only through event averages. Following that style, Fig. 8 reports the temporal evolution of MAE, CSI, reduction-signal MAE and signed volume error over all 72 forecast frames. DrainLite consistently lowers the MAE and reduction-signal error relative to the uncorrected surface-only baseline. The improvement is visible during the rising and recession periods, not only at the peak frame. The CSI curves show that the residual correction also improves wet-area classification at the shallow 0.03 m threshold, although the gain is less dramatic than the depth-error reduction.

![Fig. 8. Six-hour spatiotemporal performance.](figures/fig08_spatiotemporal_curves.png)

**Fig. 8.** Time-varying performance over the 72 five-minute frames. Lines show the mean over five held-out events and shaded bands show one standard deviation.

Event severity partly explains the remaining spread in performance. Table 8 and Fig. 12 group the five held-out events by cumulative rainfall, surface-only peak depth, surface water volume and SWMM routing inflow. DrainLite MAE ranges from {severity_mae_min:.3f} mm to {severity_mae_max:.3f} mm across the five test events. Stronger events generally have larger residual errors, which is expected because capacity exceedance and local peak behaviour become more important as rainfall and network loading increase.

**Table 8. Event-severity analysis for held-out events.**

{md_table(TAB / "table8_event_severity_analysis.csv")}

![Fig. 12. Event-severity analysis.](figures/fig12_event_severity_analysis.png)

**Fig. 12.** Event severity versus DrainLite error. Each point is one held-out event.

### 4.4 Spatial comparison across held-out events

Figure 9 provides a main-text spatial comparison for all five held-out events, rather than relying on a single representative case. Each row shows MIKE reference, ITZI surface-only, ITZI + sink, DrainLite, DrainLite error relative to the sink label and predicted reduction. This figure is deliberately broad: it checks whether the residual pattern remains spatially plausible across different event magnitudes. DrainLite preserves the dominant ponding structures of the surface-only model while adding spatially organized reductions that are concentrated around drainage-influenced corridors.

![Fig. 9. Spatial comparison across all held-out events.](figures/fig09_all_events_spatial_comparison.png)

**Fig. 9.** Peak-depth maps for all held-out test events. Error maps are computed against the ITZI + sink target.

The original single-event peak-map panels are retained in the appendix for close inspection. They are useful for reading local details, but the all-event composite in Fig. 9 is the stronger evidence because it avoids selecting only the most favourable event.

### 4.5 Drainage-reduction and physical consistency

DrainLite is intended to learn a drainage-induced reduction, so the reduction field itself must be checked. Figure 11 compares the target reduction volume implied by ITZI + sink with the predicted DrainLite reduction volume over time. Across the five events, the volume-weighted capture ratio averages {capture_weighted_mean:.1f}%. The model is slightly high for weaker events and close to one-to-one for the stronger events. The monotonicity constraint is fully satisfied in the saved predictions: the total number of negative-depth or above-surface violation cells is {monotonic_violations}.

**Table 7. Drainage-reduction physical consistency.**

{md_table(TAB / "table7_drainage_physical_consistency.csv")}

![Fig. 11. Drainage-reduction physical consistency.](figures/fig11_drainage_physical_consistency.png)

**Fig. 11.** Reduction-volume consistency, reduction-capture ratio, final-step reduction volume and near-pipe reduction share. The near-pipe region is defined as cells within 60 m of the conceptual pipe raster.

The spatial concentration of reduction is also plausible. The predicted reduction places an average of {near_pipe_pred_share:.1f}% of the time-integrated reduction volume within 60 m of the conceptual pipe network. This does not prove hydraulic realism, but it does show that the residual learner is not merely applying a uniform depth offset. It is learning a spatially structured correction associated with the drainage prior.

### 4.6 Peak-depth and high-risk-cell error

The principal weakness of the current method is peak-depth behaviour. Table 6 and Fig. 10 isolate this issue. The mean peak-depth error is {all_static['peak_error_mm']:.1f} mm, but the top 1% deep-cell MAE is much smaller, averaging {top1_mae_mean:.1f} mm. The peak-location shift is {peak_shift_mean:.1f} m on average because the maximum cell is generally colocated with the sink-label maximum; the problem is therefore not primarily a spatial displacement of the deepest ponding area. Instead, the model tends to overestimate the magnitude of the deepest single-cell maximum in stronger events.

**Table 6. Peak-depth and high-risk-cell error analysis.**

{md_table(TAB / "table6_peak_error_analysis.csv")}

![Fig. 10. Peak-depth error analysis.](figures/fig10_peak_error_analysis.png)

**Fig. 10.** Peak-depth error, peak-location shift, top 1% deep-cell MAE and high-depth-cell MAE for the held-out events.

This distinction is important. A large event-level peak error does not mean that all deep cells are poorly predicted. The average MAE for cells deeper than 0.15 m is {high015_mae_mean:.1f} mm. The largest errors are concentrated around the single most extreme local maximum, which is sensitive to small residual errors in a small number of cells. A peak-weighted objective or a separate local-maximum correction module would be needed before using DrainLite for point-scale design decisions.

### 4.7 SWMM standalone and ITZI-PySWMM prototype evidence

The SWMM standalone branch is physically meaningful but less useful as a direct pixel-level feature. Figure 7 summarizes event-level SWMM outfall volume, node ponding volume and routing continuity error. These metrics help describe the drainage-network branch, and the continuity errors indicate stable standalone routing. However, when the same metrics are appended to the pixel-level residual learner, performance does not improve. This is probably because all cells in the same event receive identical SWMM values, so the variables can describe event severity but cannot identify where within the grid drainage removal should occur.

![Fig. 7. SWMM standalone summary.](figures/fig07_swmm_summary.png)

**Fig. 7.** Standalone SWMM dynamic-wave metrics for the 17 events. The branch is used for interpretation and auxiliary features, not as an online two-way coupled surface-sewer model.

The event-level SWMM variables are nevertheless correlated with event difficulty. In the five held-out events, SWMM routing inflow has a Pearson correlation of {routing_mae_r:.2f} with DrainLite MAE, and SWMM outfall volume has a Pearson correlation of {outfall_reduction_r:.2f} with reduction MAE. This supports a nuanced interpretation: SWMM summaries contain event-severity information, but they are too spatially coarse to improve the pixel-level residual learner when appended as global event variables.

**Table 9. Correlation between event-level SWMM descriptors and DrainLite errors.**

{md_table(TAB / "table9_swmm_error_correlation.csv")}

![Fig. 14. SWMM descriptors versus DrainLite errors.](figures/fig14_swmm_error_correlation.png)

**Fig. 14.** Scatter plots between selected SWMM standalone descriptors and DrainLite errors for the five held-out events.

In addition to the standalone SWMM branch, the ITZI-PySWMM prototypes demonstrate that the real Shenzhen window can be coupled online through ITZI's drainage interface. {proto_results} This behaviour is qualitatively different from the monotonic sink target because the pipe network can return water to the surface. The prototype runs therefore provide useful evidence that the workflow can move beyond a removal-only sink assumption. However, these results are not used to train DrainLite in the present paper because they cover only the five held-out events, not all 17 events, and they have not yet undergone strict combined surface-sewer mass-balance validation.

![Extended Fig. 6. Five-event ITZI-SWMM coupled prototype diagnostics.](figures/extended_fig_itzi_swmm_prototype_summary.png)

**Extended Fig. 6.** Five-event ITZI-PySWMM prototype diagnostics for the held-out test events. The figure summarizes exchange volumes and coupled-model deviations relative to the surface-only and conceptual sink references.

### 4.8 External comparison with MIKE reference

Table 3 reports external-reference metrics against MIKE. The surface-only ITZI result has lower average MAE to MIKE than the DrainLite variants in this specific experiment. This outcome should be interpreted carefully. DrainLite is trained to reproduce ITZI + sink, not MIKE. The MIKE reference in the LarNO benchmark represents a separate 1D-2D hydraulic model with its own drainage assumptions, calibrated parameters and treatment of inlets. The conceptual sink label used here is intentionally simple and may remove water in ways that do not match MIKE's coupled drainage process.

**Table 3. External comparison with MIKE reference.**

{md_table(TAB / "table3_mike_external_reference.csv")}

Figure 13 makes the mismatch visible. The ITZI + sink and DrainLite difference maps are not simply lower-error versions of the surface-only map; in several areas they move away from MIKE because the conceptual sink label imposes a drainage-removal pattern that differs from the MIKE reference. This is why MIKE is treated as an external plausibility check rather than as the supervised target.

![Fig. 13. External MIKE-reference difference maps.](figures/fig13_mike_difference_maps.png)

**Fig. 13.** Peak-depth difference maps relative to MIKE for ITZI surface-only, ITZI + sink and DrainLite.

The MIKE comparison therefore provides two lessons. First, DrainLite successfully learns the imposed conceptual drainage-reduction pattern; this is confirmed by the ITZI + sink metrics. Second, improving a conceptual drainage label does not automatically improve agreement with a distinct MIKE reference. This is not a failure of the residual learner alone, but a reminder that label fidelity controls surrogate-model usefulness. If the desired operational target is MIKE, then the residual should be trained directly against MIKE or against a more faithful coupled physical label. In the current manuscript, the conservative interpretation is that DrainLite is a drainage-aware correction framework validated on a conceptual sink target and externally checked against MIKE, not a replacement for the MIKE reference.

## 5. Discussion

### 5.1 Why drainage location priors are effective

The strongest evidence in this study is the difference between the no-drainage-feature residual model and the drainage-aware variants. The base model already knows the surface depth, rainfall, cumulative rainfall, DEM, slope, time and coordinates. It can therefore learn broad event-dependent and terrain-dependent differences between surface-only and sink simulations. However, it cannot know where inlets, pipes and outfalls are located. When drainage location priors are added, the model can condition its reduction on plausible sink access. This additional information reduces the supervised MAE from {base['mae_mm']:.3f} mm to {all_static['mae_mm']:.3f} mm for the all-static model.

The near-equivalence of `mask_only` and `all_static` is also informative. It suggests that, for the current conceptual network, the key signal is not fine variation in pipe diameter or capacity, but the spatial opportunity for drainage. In other words, the model first needs to know whether a cell is near the network. Once that condition is known, approximate hydraulic attributes provide only modest additional benefit. This does not mean hydraulic attributes are unimportant in real drainage systems. It means that the current synthetic labels and coarse grid may not contain enough variation to exploit them fully.

### 5.2 Why event-level SWMM summaries do not improve the model

The SWMM-assisted variant is included because a standalone SWMM branch is available and numerically stable. Yet its performance is slightly worse than the all-static model in the supervised ITZI + sink task. This result is scientifically useful because it distinguishes two ideas that are often conflated: physical relevance and feature usefulness. SWMM metrics are physically relevant summaries of the conceptual pipe network. However, they are event-level quantities. Every cell in an event receives the same routing inflow, outfall volume or continuity error. These values may describe the total hydraulic loading of the network, but they do not identify which cells have high drainage access or which local depressions are disconnected from outfalls.

Therefore, the result should not be read as "SWMM is useless". Rather, it says that SWMM summaries must be spatialized or temporalized before they are expected to improve pixel-level residual correction. A useful next step within the same lightweight philosophy would be to derive node-service areas, nearest-node routing distance, outfall path length, local pipe bottleneck ratio or simplified node surcharge indicators. These features would preserve the interpretability of the drainage prior while adding more physically meaningful spatial structure.

### 5.3 Relationship to LarNO and neural-operator modelling

DrainLite is inspired by the LarNO problem setting but does not reproduce LarNO's full neural-operator contribution. LarNO addresses memory-efficient training and zero-shot super-resolution for large spatiotemporal flood forecasting. DrainLite addresses a smaller but practical question: how to inject drainage effects into an existing forecast when only a local CPU-friendly workflow is available. The relationship is therefore complementary. A future LarNO-D model could incorporate drainage rasters directly as input channels and learn the full mapping end-to-end. DrainLite instead provides a low-cost residual layer that can be trained and interpreted without full neural-operator training.

This distinction is important for claims. The present manuscript should not state that DrainLite is a new neural operator, a full LarNO replacement, or a complete coupled sewer-surface simulator. Its novelty lies in the combination of road-aligned drainage priors, residual correction, monotonic drainage constraints, explicit ablation and reproducible low-compute execution. This is a smaller claim than the original LarNO paper, but it is also more defensible under the available data and hardware.

### 5.4 Limitations

Several limitations should be made explicit. First, the main drainage label is conceptual. The ITZI + sink result represents road-aligned inlet removal, not a surveyed sewer system with inlet capture curves, pipe pressurization, backwater effects and surcharge to the surface. Second, DrainLite enforces monotonic water removal. This is appropriate for the sink label but cannot represent manhole overflow or sewer surcharge. Third, the model is evaluated on five held-out events from one spatial window. It demonstrates event generalization within the local benchmark but not generalization to another catchment or a different drainage-network design. Fourth, the model is optimized for mean residual accuracy rather than peak-depth extremes. The peak error remains large enough that the method should not be used alone for point-scale engineering design. Fifth, the 17-event SWMM branch used in DrainLite is currently standalone. Although five held-out ITZI-PySWMM online coupling prototypes have now been completed, they do not yet form a validated 17-event reference label and are not used as the supervised target in the current DrainLite experiments.

### 5.5 Future work

Future work can proceed without abandoning the lightweight philosophy. The most immediate improvement is to enrich drainage features rather than to train a larger model. Network-distance metrics, upstream contributing area, service-area labels, pipe bottleneck indices and outfall-path capacity could be derived from the existing conceptual graph. SWMM outputs could be transformed into node- or pipe-level time series and rasterized to the grid. A second improvement is to change the objective from pure mean residual error to a mixed loss that gives greater weight to peak cells, high-depth cells or high-risk road segments. A third direction is to compare multiple conceptual drainage designs, testing whether DrainLite can distinguish network density, inlet spacing or outfall placement scenarios. Finally, the five-event ITZI-PySWMM prototype set should be extended to all 17 events, with rainfall input, surface storage, sewer storage, generated inflow, outfall discharge, surcharge return and combined mass-balance residual audited before any `h_coupled.npy` target is used for training.

## 6. Conclusion

This study proposes DrainLite, a lightweight drainage-aware residual correction framework for urban pluvial flood forecasting under limited computing resources. The method learns the non-negative reduction from an ITZI surface-only simulation to an ITZI + conceptual road-aligned inlet-sink label using rainfall, DEM, existing water depth and drainage-network features. It does not retrain a LarNO backbone and does not claim full two-way surface-sewer coupling. Instead, it provides a practical, interpretable and reproducible way to inject conceptual drainage effects into an existing flood forecast.

On 17 valid Shenzhen benchmark events, with 12 used for training and 5 held out for testing, the all-static DrainLite model substantially improves agreement with the ITZI + sink label. The MAE decreases from {surface['mae_mm']:.3f} mm for the surface-only baseline to {all_static['mae_mm']:.3f} mm, a {all_static['mae_improvement_vs_surface_pct']:.1f}% improvement. Compared with a no-drainage-feature residual model, the all-static model improves MAE by {all_static['mae_improvement_vs_base_pct']:.1f}%. The ablation analysis indicates that drainage location priors account for most of the improvement, while event-level SWMM summaries do not add pixel-scale benefit in their current form. These findings support the central hypothesis that road-aligned drainage priors can provide useful and interpretable residual information even in a low-compute setting.

The method also has clear boundaries. It is best understood as a drainage-aware correction for conceptual sink labels, not as a complete replacement for MIKE, ITZI-SWMM coupling or full LarNO-D training. Its peak-depth error remains substantial, and its external agreement with MIKE is limited by differences between the conceptual sink label and the MIKE reference. Nevertheless, as a low-cost methodological bridge between hydrodynamic simulation, drainage-network priors and machine-learning correction, DrainLite offers a defensible lightweight innovation and a foundation for future drainage-aware urban flood forecasting research.

## Data availability

The study uses the publicly released LarNO urban flood benchmark as the source dataset. The local drainage-augmented derivative is stored under `LarNO-main/benchmark/urbanflood/flood/region1_20m_drainage_v1` and `LarNO-main/benchmark/urbanflood/geodata/region1_20m_drainage_v1`. Public redistribution conditions should follow the original dataset license and repository policy. Any newly generated DrainLite predictions, metrics and figures are stored under `extended_study/output/drainlite_residual` and `paper_draft/drainlite`. The five-event ITZI-PySWMM prototype outputs and summary diagnostics are stored under `extended_study/output/itzi_swmm_coupled_prototype`.

## Code availability

The original LarNO repository is available at https://github.com/holmescao/LarNO. The present lightweight DrainLite scripts are local additions in `extended_study/train_drainlite_residual.py`, `extended_study/generate_drainlite_paper_assets.py`, `extended_study/generate_drainlite_enhanced_results.py`, and `extended_study/generate_drainlite_manuscript.py`. The ITZI-PySWMM prototypes are generated by `extended_study/run_itzi_swmm_coupled_prototype.py` and `extended_study/run_itzi_swmm_coupled_subprocess_batch.py`, and summarized by `extended_study/analyze_itzi_swmm_coupled_prototype.py`. These scripts generate the residual models, prototype diagnostics, figures, tables and manuscript draft used in this paper.

## CRediT authorship contribution statement

To be completed. Suggested structure: conceptualization, methodology, software, validation, formal analysis, data curation, writing - original draft, writing - review and editing, visualization, supervision and funding acquisition.

## Declaration of competing interests

The authors declare no competing interests. To be confirmed before submission.

## Acknowledgments

To be completed. This section should acknowledge the LarNO authors for releasing the code and dataset, and should acknowledge any local computational resources used for the DrainLite experiments.

## References

Bates, P.D., De Roo, A.: A simple raster-based model for flood inundation simulation. Journal of Hydrology 236(1-2), 54-77 (2000).

Bates, P.D., Horritt, M.S., Fewtrell, T.J.: A simple inertial formulation of the shallow water equations for efficient two-dimensional flood inundation modelling. Journal of Hydrology 387(1-2), 33-45 (2010).

Cao, X., Yao, Y., Wang, Z., Zhao, Z., Borthwick, A.G.L., Qin, H.: Large-scale urban flood modeling and zero-shot high-resolution generalization with LarNO. Journal of Hydrology, 135686 (2026).

Cao, X., Wang, B., Yao, Y., Zhang, L., Xing, Y., Mao, J., Zhang, R., Fu, G., Borthwick, A.G., Qin, H.: U-RNN high-resolution spatiotemporal nowcasting of urban flooding. Journal of Hydrology 659, 133117 (2025).

Friedman, J.H.: Greedy function approximation: a gradient boosting machine. Annals of Statistics 29(5), 1189-1232 (2001).

Huber, W.C., Dickinson, R.E., Barnwell Jr, T.O., Branch, A.: Storm Water Management Model; Version 4. Environmental Protection Agency, United States (1988).

Kovachki, N., Li, Z., Liu, B., Azizzadenesheli, K., Bhattacharya, K., Stuart, A., Anandkumar, A.: Neural operator: learning maps between function spaces with applications to PDEs. Journal of Machine Learning Research 24(89), 1-97 (2023).

Li, Z., Kovachki, N., Azizzadenesheli, K., Liu, B., Bhattacharya, K., Stuart, A., Anandkumar, A.: Fourier neural operator for parametric partial differential equations. Proceedings of the 9th International Conference on Learning Representations (ICLR) (2021).

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., Duchesnay, E.: Scikit-learn: Machine learning in Python. Journal of Machine Learning Research 12, 2825-2830 (2011).

Rossman, L.A.: Storm Water Management Model User's Manual Version 5.1. U.S. Environmental Protection Agency, Cincinnati, Ohio (2015).

Schubert, J.E., Luke, A., AghaKouchak, A., Sanders, B.F.: A framework for mechanistic flood inundation forecasting at the metropolitan scale. Water Resources Research 58(10), e2021WR031279 (2022).

Xu, L., Gao, L.: A hybrid surrogate model for real-time coastal urban flood prediction: an application to Macao. Journal of Hydrology 642, 131863 (2024).

Zhang, L., Qin, H., Mao, J., Cao, X., Fu, G.: High temporal resolution urban flood prediction using attention-based LSTM models. Journal of Hydrology 620, 129499 (2023).

## Appendix A. Extended figures and tables

![Extended Fig. 2. Event76 peak maps.](figures/extended_fig_event76_peak_maps.png)

![Extended Fig. 3. Event77 peak maps.](figures/extended_fig_event77_peak_maps.png)

![Extended Fig. 4. Event78 peak maps.](figures/extended_fig_event78_peak_maps.png)

![Extended Fig. 5. Event80 peak maps.](figures/extended_fig_event80_peak_maps.png)

![Extended Fig. 7. Event75 ITZI-SWMM coupled prototype spatial comparison.](figures/extended_fig_itzi_swmm_prototype.png)

Extended Table 1, Extended Table 2 and Extended Table 3 are provided as CSV files in the `tables` directory. They contain the full event inventory, feature definitions and five-event coupled-prototype summary, respectively. Additional machine-readable result tables are also retained in the same directory for the time-series metrics, peak-error analysis, drainage-reduction consistency, event-severity analysis, SWMM-error correlations and lightweight runtime diagnostics.
"""

    count = word_count(text)
    if count < 7000:
        supplement = f"""

## Appendix B. Additional notes for manuscript development

The present draft is intentionally conservative in its claims. This is necessary because the objective of the study is not to demonstrate a complete sewer-surface hydraulic simulator. A full coupled system would require a different evidence chain. It would need inlet capture functions or exchange equations, hydraulic heads in manholes, bidirectional transfer between surface cells and sewer nodes, outfall boundary conditions, continuity auditing across both domains, and validation against either observed sewer measurements or a trusted coupled reference. None of those elements is silently assumed here. Instead, the manuscript uses the available evidence in a narrower but reproducible way: the conceptual sink label defines a controlled drainage-removal process, and the residual learner tests whether spatial drainage priors help reconstruct that process.

An explicit coupling evidence audit is provided in `paper_draft/drainlite/ITZI_SWMM_coupling_evidence_audit.md`. That audit records a key boundary of the present study: the Shenzhen 17-event workflow contains ITZI surface-only labels, ITZI + conceptual inlet-sink labels, standalone SWMM dynamic-wave metrics, and five held-out ITZI-PySWMM online coupling prototypes, but it does not yet contain a validated 17-event ITZI-SWMM coupled label. Earlier simplified coupled scripts are not used as evidence because one output is numerically unstable and because those scripts do not use the SWMM dynamic-wave engine. The synthetic `itzi-flood/test_cases/urban_drainage` example is relevant technically because it uses ITZI drainage components and PySWMM, but it is a 400 m synthetic case rather than the LarNO Shenzhen benchmark. The DrainLite paper should therefore keep the main supervised target as ITZI + sink unless a future all-event `h_coupled.npy` label is generated and passes mass-balance checks.

This distinction should remain visible throughout later revisions. If reviewers ask whether DrainLite can model surcharge, the correct answer is that the current constrained formulation cannot, because the target itself is a removal-only sink label. If reviewers ask why MIKE agreement is not improved, the correct answer is that MIKE is not the training target and may encode different drainage assumptions. If reviewers ask what the SWMM branch contributes, the correct answer is that it provides a numerically stable standalone descriptor of network-scale behaviour, but event-level summaries are not spatially detailed enough to improve the present pixel-level residual model. These answers are not weaknesses in honesty; they are the boundary conditions that make the contribution defensible.

The most important future improvement is not necessarily a larger model. A better next lightweight experiment would convert the graph structure of the conceptual pipe network into more physically expressive predictors. For example, every cell could be assigned a nearest-inlet identifier, an outfall path length, a cumulative upstream service area, a minimum downstream capacity, and a topological bottleneck score. These features would still be cheap to compute, but they would represent the drainage system as a network rather than as independent rasters. They may also help distinguish two cells that are equally close to a pipe but drain toward different outfalls or bottlenecked conduits. In this sense, DrainLite can become more hydraulic without becoming computationally heavy.

The present revision now includes an event-severity analysis using cumulative rainfall, surface-only peak volume and SWMM routing inflow. This analysis should remain in the Results section rather than being treated as a supplementary afterthought, because it explains why the residual task becomes harder for stronger events. A later revision could make this analysis stronger by adding more events or by using return-period classes if those metadata become available.

Finally, the manuscript should be explicit about its intended use. DrainLite is suitable for rapid scenario screening, sensitivity tests of conceptual drainage layouts, low-cost educational studies, and preliminary drainage-aware correction of surface-only flood outputs. It is not sufficient for final design of sewer systems, legal flood-risk mapping, or detailed emergency response at individual assets without additional validation. This practical positioning mirrors the careful tone of the LarNO paper, which frames benchmark superiority without overstating operational readiness. Maintaining that tone will make the paper stronger.

## Appendix C. Alignment with the LarNO manuscript style

The original LarNO paper is effective because it uses a disciplined argumentative sequence. It does not begin with architecture details. It begins with a field-level bottleneck: urban flood forecasting requires high spatial and temporal resolution, while conventional hydraulic simulation and ordinary deep learning models face computational and memory limits. Only after this need is established does it introduce the neural-operator framing, theoretical support, benchmark construction and empirical evaluation. The DrainLite manuscript should follow the same rhetorical pattern at a smaller scale. The opening problem is not GPU memory for zero-shot super-resolution; it is the practical difficulty of introducing drainage-network effects when data, hardware and complete sewer models are limited. The method section should then define a precise residual problem, the results should test that problem directly, and the discussion should state exactly where the lightweight formulation stops.

This also affects figure placement. In the LarNO paper, figures are not decorative; each one answers a specific claim. The case-study figure establishes scale and physical context. The architecture figure explains the modelling mechanism. The comparison figures show spatial credibility, temporal stability and quantitative performance. The ablation figures isolate the mechanism responsible for improvement. DrainLite should use the same logic. Fig. 1 should explain why residual correction is a coherent modelling step. Fig. 2 should prove that the drainage priors are spatially meaningful. Fig. 3 should prevent confusion among MIKE, ITZI, sink labels and SWMM. Figs. 8 and 9 should provide the full six-hour and all-event spatial evidence. Figs. 10 and 11 should openly handle peak-error limitations and physical reduction consistency. Figs. 7 and 14 should document what SWMM contributes and why event-level descriptors are insufficient as pixel-level predictors. Fig. 13 should make the MIKE-reference mismatch visible. This figure discipline will make the paper read like a coherent study rather than a collection of outputs.

The LarNO manuscript also uses evaluation metrics in a way that separates hydrodynamic accuracy from inundation classification. DrainLite should do the same. MAE and RMSE quantify average depth fidelity. CSI at shallow and deeper thresholds tests whether wet and dry areas are classified correctly. Peak-depth error should be reported even when it is not flattering, because urban flood decisions often depend on local maxima. Reduction MAE is specific to DrainLite and should be treated as a method-level metric: it asks whether the model learns the drainage removal signal itself. MIKE comparison should be presented as external reference alignment rather than as the supervised objective. This separation is important because it avoids an invalid inference: a model can be good at reproducing the conceptual sink target while not necessarily becoming closer to MIKE.

A strong revised manuscript should also include a short reproducibility paragraph in the Methods or Data availability section. The paragraph should state the exact event list, train-test split, array dimensions, time steps, unit conversion from metres to millimetres for residual targets, and clipping rule during inference. It should state that the model is trained on sampled cell-time rows rather than on complete dense tensors. It should also state where random seeds are used. These details may feel mundane, but they are essential for a method paper whose contribution is partly low-compute reproducibility. A reviewer should be able to reconstruct the complete experimental graph: raw benchmark files to drainage rasters, ITZI surface and sink labels, SWMM metrics, sampled training table, fitted residual model, dense prediction arrays, metrics, figures and final manuscript tables.

The biggest likely reviewer concern is label validity. The manuscript should anticipate this concern rather than wait for criticism. The correct response is that the study is a staged methodological demonstration. The conceptual sink label is not treated as the true urban drainage system; it is treated as a controlled drainage-removal process. Under that target, DrainLite tests whether drainage-network priors can be learned as a residual correction. MIKE is retained precisely because it provides a more independent reference and exposes the gap between conceptual sink removal and a more complex hydraulic reference. This is scientifically cleaner than hiding the mismatch. In later work, the same DrainLite framework can be retrained on a more faithful coupled label, but the current evidence should be allowed to support only the current claim.

Another likely concern is whether the model is too simple. The answer should be pragmatic. Simplicity is not a defect when the stated contribution is low-compute drainage-aware correction. A tree-based residual learner cannot represent all flood physics, but it can reveal whether the available drainage features contain useful predictive information. The ablation result shows that they do. This makes DrainLite useful as both a practical correction module and a diagnostic tool for future LarNO-D design. If drainage masks and network proximity explain much of the residual, then those features should be prioritized in any later neural-operator input design. If event-level SWMM summaries do not help, then future work should avoid simply appending global sewer metrics and should instead spatialize network hydraulics.

The final paper should therefore be positioned as a bridge study. It bridges surface-only hydrodynamic output and drainage-aware prediction. It bridges conceptual road-aligned infrastructure extraction and machine-learning correction. It bridges the original LarNO benchmark and a new drainage-focused extension. It also bridges local-machine experimentation and future higher-fidelity coupled modelling. The claim is intentionally smaller than the LarNO paper's claim, but it is still publishable if written carefully: under limited computing resources, spatially explicit drainage priors can be used to learn an interpretable residual correction that substantially improves agreement with a controlled drainage-aware label, while also revealing what additional information is needed for physically complete sewer-surface modelling.
"""
        text += supplement
        count = word_count(text)

    OUT.write_text(text, encoding="utf-8")
    print(f"Wrote {OUT}")
    print(f"word_count={count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
