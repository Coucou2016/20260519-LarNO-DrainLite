#!/usr/bin/env python3
"""Generate a manuscript-style Markdown draft for the LarNO-DrainLite v2 package."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
V2_OUT = ROOT / "extended_study" / "output" / "larno_drainlite_v2_package"
MAIN_OUT = ROOT / "extended_study" / "output" / "drainlite_connected_residual"
CV_OUT = ROOT / "extended_study" / "output" / "drainlite_connected_cross_validation"
HYDRO_OUT = ROOT / "extended_study" / "output" / "hydrograph_timing_diagnostics"
INF_PILOT_OUT = ROOT / "extended_study" / "output" / "infiltration_pilot"
DATA_DIR = ROOT / "LarNO-main" / "benchmark" / "urbanflood" / "flood" / "region1_20m_connected_swmm_v1"
GEO_DIR = ROOT / "LarNO-main" / "benchmark" / "urbanflood" / "geodata" / "region1_20m_connected_swmm_v1"
MANUSCRIPT = V2_OUT / "manuscript_larno_drainlite_v2.md"


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def find_row(rows: list[dict[str, str]], key: str, value: str) -> dict[str, str]:
    for row in rows:
        if row.get(key) == value:
            return row
    return {}


def num(value: object, digits: int = 3) -> str:
    try:
        x = float(value)
    except Exception:
        return "待补充"
    if abs(x) >= 1000:
        return f"{x:,.0f}"
    return f"{x:.{digits}f}"


def md_table(rows: list[dict[str, str]], columns: list[tuple[str, str]], digits: int = 3) -> str:
    if not rows:
        return "待补充：表格数据尚未生成。\n"
    header = "| " + " | ".join(label for _, label in columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [header, sep]
    for row in rows:
        vals = []
        for key, _ in columns:
            value = row.get(key, "")
            if key in {
                "event",
                "model",
                "quality_status",
                "rule",
                "feature",
                "use_as_training_label",
                "must_disclose_stability_warning",
            }:
                vals.append(str(value))
            else:
                vals.append(num(value, digits))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"


def main() -> int:
    V2_OUT.mkdir(parents=True, exist_ok=True)
    quality = read_csv(V2_OUT / "metrics" / "v2_label_quality_grading.csv")
    quality_summary = read_csv(V2_OUT / "metrics" / "v2_label_quality_summary.csv")
    mike_summary = read_csv(V2_OUT / "metrics" / "v2_mike_reference_summary.csv")
    hydro_rows = read_csv(HYDRO_OUT / "hydrograph_timing_summary.csv")
    rainfall_accounting = read_csv(HYDRO_OUT / "rainfall_accounting_summary.csv")
    infiltration_pilot = read_csv(INF_PILOT_OUT / "event68_infiltration_pilot_summary.csv")
    cv_summary = read_csv(CV_OUT / "metrics" / "connected_residual_cv_summary.csv")
    ablation = read_csv(MAIN_OUT / "metrics" / "connected_residual_ablation_summary.csv")
    importance = read_csv(MAIN_OUT / "metrics" / "feature_importance_all_static.csv")[:10]

    all_static_cv = find_row(cv_summary, "model", "all_static")
    base_cv = find_row(cv_summary, "model", "base")
    surface_cv = find_row(cv_summary, "model", "surface_only")
    surface_mike = find_row(mike_summary, "model", "ITZI surface-only")
    connected_mike = find_row(mike_summary, "model", "ITZI-SWMM connected")
    drainlite_mike = find_row(mike_summary, "model", "DrainLite all_static CV")

    def model_mean(model: str, field: str) -> float:
        values = []
        for row in hydro_rows:
            if row.get("model") != model:
                continue
            try:
                values.append(float(row[field]))
            except Exception:
                pass
        return sum(values) / len(values) if values else float("nan")

    mike_hpeak_h = model_mean("MIKE reference", "max_depth_peak_hour")
    surf_hpeak_h = model_mean("ITZI surface-only", "max_depth_peak_hour")
    connected_hpeak_h = model_mean("ITZI-SWMM connected", "max_depth_peak_hour")
    mike_vol_ratio = model_mean("MIKE reference", "volume_final_to_peak_ratio")
    surf_vol_ratio = model_mean("ITZI surface-only", "volume_final_to_peak_ratio")
    connected_vol_ratio = model_mean("ITZI-SWMM connected", "volume_final_to_peak_ratio")
    rainfall_accounting_first = rainfall_accounting[0] if rainfall_accounting else {}

    peak_reductions = [float(r["peak_reduction_mm"]) for r in quality if r.get("peak_reduction_mm")]
    volume_reductions = [float(r["final_volume_reduction_m3"]) for r in quality if r.get("final_volume_reduction_m3")]
    peak_range = f"{num(min(peak_reductions))}-{num(max(peak_reductions))}" if peak_reductions else "待补充"
    volume_range = f"{num(min(volume_reductions))}-{num(max(volume_reductions))}" if volume_reductions else "待补充"

    text = f"""# LarNO-DrainLite: A Lightweight Drainage-Residual Correction Framework for Road-Aligned Conceptual Urban Drainage Effects

## Highlights

- A road-aligned conceptual drainage network was coupled with ITZI through native SWMM dynamic-wave routing to generate drainage-aware labels.
- A lightweight LarNO-compatible residual model was developed to learn signed drainage effects without retraining the LarNO backbone.
- Leave-one-event-out validation over eight events reduced the mean absolute error to the ITZI-SWMM drainage label from {num(surface_cv.get("mae_mm"))} mm to {num(all_static_cv.get("mae_mm"))} mm.
- Static drainage features improved the base residual model by {num(all_static_cv.get("mae_improvement_vs_base_pct"))}% in mean absolute error, showing measurable but limited marginal contribution.
- MIKE reference comparison was retained as an external plausibility check rather than as a training target.

## Abstract

Large-scale neural urban flood forecasting models such as LarNO learn rainfall-topography-to-water-depth mappings from high-resolution hydrodynamic reference data. However, the released benchmark setting does not explicitly expose a learnable representation of urban drainage networks. This study develops LarNO-DrainLite, a lightweight residual correction framework that injects drainage-network effects into surface flood prediction under low-computing-resource conditions. A road-aligned conceptual drainage network is first constructed from the adjusted road prior and converted into a connected SWMM network. The network is coupled with the ITZI two-dimensional surface-flow solver through ITZI's native SWMM coupling interface, producing paired surface-only and drainage-connected labels for eight rainfall events. DrainLite then learns the signed residual between the coupled and uncoupled water-depth fields using surface depth, rainfall, terrain, and rasterized drainage features.

The connected ITZI-SWMM simulations produced event-level peak-depth reductions ranging from {peak_range} mm and final surface-water volume reductions ranging from {volume_range} m3. In leave-one-event-out validation, the all-static DrainLite model reduced the mean absolute error against the ITZI-SWMM drainage label from {num(surface_cv.get("mae_mm"))} mm for the uncorrected surface-only result to {num(all_static_cv.get("mae_mm"))} mm, corresponding to a {num(all_static_cv.get("mae_improvement_vs_surface_pct"))}% improvement. Relative to a base residual model without drainage features, the all-static model improved mean absolute error by {num(all_static_cv.get("mae_improvement_vs_base_pct"))}%. Compared with MIKE reference data, DrainLite did not improve full time-series pixelwise mean absolute error, but it substantially reduced final-volume bias and mitigated surface-only peak-depth overestimation. These results support the feasibility of a low-cost drainage-residual correction route while also showing that stronger physical labels and dynamic drainage-state features are needed before claiming a fully drainage-aware LarNO backbone.

## Keywords

Urban flood forecasting; LarNO; ITZI; SWMM; drainage network; residual learning; lightweight model; road-aligned conceptual drainage

## 1. Introduction

Urban pluvial flooding is controlled by both above-ground flow paths and underground drainage capacity. The original LarNO study demonstrated that neural operators can provide fast, high-resolution urban flood forecasts by learning from MIKE reference simulations. Its released benchmark inputs emphasize rainfall history, cumulative rainfall, and terrain information. The current extension addresses a specific gap: how to introduce drainage-network effects when the available computing hardware is insufficient for full LarNO retraining and when surveyed municipal drainage data are unavailable.

This study therefore does not attempt to replace LarNO with a new large model. Instead, it proposes a lightweight correction route. The key question is whether a model can learn the difference between an uncoupled surface-flow result and a drainage-coupled result using explainable drainage features. If successful, such a model can be attached after LarNO or other surface flood predictors and used as a practical bridge toward drainage-aware neural flood forecasting.

The contributions are as follows. First, a road-aligned conceptual drainage network is generated and corrected so that each retained inlet or junction has a hydraulic path to an outfall. Second, ITZI's native SWMM coupling is used to produce paired surface-only and drainage-connected labels. Third, a signed residual model is trained and validated by leave-one-event-out testing. Fourth, MIKE reference data are used to check external plausibility, while the paper explicitly separates this check from the DrainLite training objective.

## 2. Data and Methods

### 2.1 Benchmark Context and Study Data

The base data follow the LarNO Shenzhen benchmark setting at 20 m spatial resolution. The drainage-enhanced dataset is saved at:

`{DATA_DIR}`

The corresponding static geodata are saved at:

`{GEO_DIR}`

Each event contains 72 temporal frames. The main arrays are `h_itzi_surface.npy` for the uncoupled ITZI surface-flow result, `h_itzi_swmm_connected.npy` for the native ITZI-SWMM coupled result, `h_mike_ref.npy` for the MIKE external reference, and `h.npy` for the current drainage-aware training label. The current formal label set includes eight events: event1, event20, event65, event66, event67, event68, event69, and event70.

### 2.2 Road-Aligned Conceptual Drainage Network

Because surveyed pipe-network data are unavailable, the drainage network is conceptualized from the adjusted road-aligned network. The objective is not to reproduce a real municipal sewer map, but to provide a spatially meaningful drainage prior. The network correction keeps connected main components, removes isolated nodes and broken pipes, and assigns each component at least one outfall. Static drainage features are rasterized to the same 20 m grid as the terrain, including inlet mask, outfall mask, pipe mask, pipe diameter, pipe slope, pipe capacity, pipe cover depth, and distance to outfall.

### 2.3 ITZI-SWMM Coupled Label Generation

The fixed physical simulation route uses ITZI's two-dimensional surface-flow solver and SWMM dynamic-wave routing through ITZI's native drainage coupling. Surface-only simulations provide the baseline hydrodynamic result. Connected ITZI-SWMM simulations provide the drainage-aware label. The signed residual is defined as:

`residual = h_itzi_swmm_connected - h_itzi_surface`

A negative residual means that the drainage system reduces surface water depth. A positive residual can occur where pipe surcharge, local backwater, or redistribution raises water depth relative to the surface-only case.

### 2.4 LarNO-DrainLite Residual Model

DrainLite is positioned as a LarNO-compatible post-processing module. It can accept an existing surface flood prediction from LarNO, ITZI, or a similar predictor and then estimate the drainage-induced residual. In the current demonstration, ITZI surface-only results are used as the base prediction because they are available for all eight formal events.

The model input includes surface water depth, rainfall intensity, cumulative rainfall, digital elevation, terrain slope, normalized row and column coordinates, local mean water depths, and rasterized drainage features. The output is a signed residual in millimetres. The corrected water depth is:

`h_drainlite = max(h_surface + residual_mm / 1000, 0)`

### 2.5 Evaluation

The primary target is the ITZI-SWMM connected label. Metrics include mean absolute error, root mean square error, critical success index at 0.03 m and 0.15 m thresholds, peak-depth error, signed-residual error, and surface-volume bias. MIKE reference metrics are computed separately as an external plausibility check.

## 3. Results

### 3.1 Physical-Label Quality

The SWMM stability grading is summarized below. The accepted category means that the absolute continuity error is no greater than 2%. The warning category means that the event remains usable for exploratory modelling but needs explicit numerical-stability disclosure. The excluded category is reserved for failed or unstable events and is not present in the current eight-event set.

{md_table(quality_summary, [("quality_status", "Quality"), ("n_events", "Events"), ("rule", "Rule")])}

The event-level quality table is:

{md_table(quality, [
    ("event", "Event"),
    ("quality_status", "Quality"),
    ("continuity_error_pct", "Continuity error (%)"),
    ("nonconverging_steps_pct", "Nonconverging steps (%)"),
    ("external_inflow_m3", "External inflow (m3)"),
    ("external_outflow_m3", "External outflow (m3)"),
    ("final_stored_m3", "Final stored water (m3)"),
    ("peak_reduction_mm", "Peak reduction (mm)"),
    ("final_volume_reduction_m3", "Final volume reduction (m3)"),
], digits=3)}

The drainage effect is physically visible: all eight events show positive surface-volume reduction, and peak-depth reduction is also apparent in most events. However, seven events are marked as warning because the SWMM continuity error exceeds 2% or the nonconverging-step percentage is high. This is a key limitation for paper writing. The current labels are suitable for demonstrating a drainage-residual learning pathway, but should not be described as a fully calibrated municipal drainage reference.

### 3.2 Main DrainLite Ablation

The main train-test split compares a surface-only baseline, a base residual model without drainage features, several drainage-feature groups, and a SWMM-assisted version.

{md_table(ablation, [
    ("model", "Model"),
    ("n_events", "Events"),
    ("mae_mm", "MAE (mm)"),
    ("rmse_mm", "RMSE (mm)"),
    ("csi_0p03", "CSI 0.03 m"),
    ("csi_0p15", "CSI 0.15 m"),
    ("peak_error_mm", "Peak error (mm)"),
    ("mae_improvement_vs_surface_pct", "Improvement vs surface (%)"),
    ("mae_improvement_vs_base_pct", "Improvement vs base (%)"),
], digits=3)}

The all-static model is the main DrainLite version because it uses the complete static drainage feature set. The result shows that the residual correction is not merely reproducing the surface-only field. It learns a systematic difference between surface-only and drainage-connected simulations. The additional drainage features provide a smaller but measurable improvement over the base residual model.

### 3.3 Leave-One-Event-Out Validation

Leave-one-event-out validation is stricter than a single hold-out split because every event is tested once after being excluded from training.

{md_table(cv_summary, [
    ("model", "Model"),
    ("n_events", "Events"),
    ("mae_mm", "MAE (mm)"),
    ("rmse_mm", "RMSE (mm)"),
    ("csi_0p03", "CSI 0.03 m"),
    ("csi_0p15", "CSI 0.15 m"),
    ("peak_error_mm", "Peak error (mm)"),
    ("mae_to_mike_mm", "MAE to MIKE (mm)"),
    ("mae_improvement_vs_surface_pct", "Improvement vs surface (%)"),
    ("mae_improvement_vs_base_pct", "Improvement vs base (%)"),
], digits=3)}

The all-static model satisfies the current acceptance criteria. Its mean absolute error is {num(all_static_cv.get("mae_mm"))} mm, improving by {num(all_static_cv.get("mae_improvement_vs_surface_pct"))}% over surface-only and by {num(all_static_cv.get("mae_improvement_vs_base_pct"))}% over the base residual model. This establishes the central result of the current lightweight innovation.

### 3.4 Feature Importance

The leading all-static features are:

{md_table(importance, [
    ("feature", "Feature"),
    ("importance_mean", "Importance mean (mm)"),
    ("importance_std", "Importance standard deviation (mm)"),
], digits=4)}

Surface depth remains the strongest feature because drainage effects can only occur where surface water is present or likely to accumulate. Terrain and coordinates describe where runoff concentrates. Drainage features such as distance to outfall, pipe density, capacity density, and inlet density provide additional spatial priors. The contribution is meaningful but limited, implying that static pipe-network descriptors alone cannot fully represent dynamic drainage states.

### 3.5 MIKE External Reference

The MIKE comparison is reported separately:

{md_table(mike_summary, [
    ("model", "Model"),
    ("n_events", "Events"),
    ("mae_mm", "MAE (mm)"),
    ("rmse_mm", "RMSE (mm)"),
    ("csi_0p03", "CSI 0.03 m"),
    ("csi_0p15", "CSI 0.15 m"),
    ("peak_depth_error_mm", "Peak depth error (mm)"),
    ("peak_map_mae_mm", "Peak-map MAE (mm)"),
    ("final_volume_error_m3", "Final volume error (m3)"),
], digits=3)}

The MIKE reference analysis gives a nuanced result. ITZI surface-only has the smallest average pixelwise time-series MAE to MIKE at {num(surface_mike.get("mae_mm"))} mm. DrainLite all-static has {num(drainlite_mike.get("mae_mm"))} mm, and ITZI-SWMM connected has {num(connected_mike.get("mae_mm"))} mm. Therefore, the current study should not claim that DrainLite universally improves MIKE agreement. At the same time, the drainage-aware results reduce the final-volume bias from {num(surface_mike.get("final_volume_error_m3"))} m3 to about {num(drainlite_mike.get("final_volume_error_m3"))} m3, and reduce the mean peak-depth overestimation from {num(surface_mike.get("peak_depth_error_mm"))} mm to {num(drainlite_mike.get("peak_depth_error_mm"))} mm for DrainLite and {num(connected_mike.get("peak_depth_error_mm"))} mm for ITZI-SWMM. This supports the physical plausibility of the drainage correction in terms of water-volume and peak-bias control.

### 3.6 Hydrograph Timing Diagnostic

A further diagnostic was performed because the MIKE reference exhibits a clear rising and falling limb, whereas the ITZI simulations tend to store water until the end of the six-hour window. The average peak time of the maximum-depth hydrograph is {num(mike_hpeak_h)} h for MIKE, {num(surf_hpeak_h)} h for ITZI surface-only, and {num(connected_hpeak_h)} h for ITZI-SWMM connected. The final-to-peak surface-volume ratio is {num(mike_vol_ratio)} for MIKE, {num(surf_vol_ratio)} for ITZI surface-only, and {num(connected_vol_ratio)} for ITZI-SWMM connected.

This confirms that the current ITZI labels remain too storage-dominated relative to MIKE. The most direct causes are the absence of calibrated infiltration or loss terms, the absence of explicit two-dimensional open-boundary outflow, limited capacity and conceptual nature of the generated drainage network, and a rainfall-loading timing issue found in the copied runner. The runner has now been corrected so that the first rainfall frame is applied at `t = 0` and the final frame is fully integrated. Existing labels generated before this correction should therefore be treated as pre-fix diagnostic labels until the events are rerun.

Rainfall accounting also revealed that the current runner redistributes rainfall falling on building cells to non-building active cells. Buildings account for {num(rainfall_accounting_first.get("building_fraction_pct"))}% of the current grid, so this assumption increases active-cell rainfall by a factor of {num(rainfall_accounting_first.get("redistribution_factor_on_active_cells"))}. If MIKE does not use the same no-loss and instantaneous roof-runoff assumption, ITZI will overstate surface storage and weaken the falling limb.

### 3.7 First Infiltration Pilot

To test whether infiltration/loss is a primary missing process, ITZI's `InfConstantRate` model was connected to the copied runner and applied to event68 with a constant active-cell infiltration rate of 5 mm/h. This pilot is not a final calibration, but it directly tests the direction of the correction.

{md_table(infiltration_pilot, [
    ("model", "Model"),
    ("max_depth_peak_hour", "Maximum-depth peak time (h)"),
    ("max_depth_peak_m", "Maximum-depth peak (m)"),
    ("max_depth_final_to_peak_ratio", "Final/peak maximum depth"),
    ("volume_peak_hour", "Volume peak time (h)"),
    ("volume_final_to_peak_ratio", "Final/peak volume"),
    ("mae_to_mike_mm", "MAE to MIKE (mm)"),
    ("peak_map_mae_to_mike_mm", "Peak-map MAE to MIKE (mm)"),
], digits=3)}

The pilot confirms the user's interpretation: adding infiltration creates a falling limb and shifts the hydrograph peak much closer to MIKE. For event68, maximum-depth peak timing moves from about 5.92 h without infiltration to about 3.42 h with 5 mm/h infiltration, while MIKE peaks at 3.17 h. However, 5 mm/h is too strong because peak depth and surface-water volume are underpredicted. The next step is therefore not to fix this value, but to calibrate a smaller effective rate or spatially variable loss field.

## 4. Discussion

The current results support a pragmatic interpretation. The lightweight residual model is successful for its intended target: approximating the signed drainage effect generated by the fixed ITZI-SWMM conceptual-network simulations. It is also useful as a low-computing-resource route because it avoids retraining the full LarNO backbone and can be trained with tabular machine-learning methods.

The limited marginal gain of static drainage features is not a failure; it is an important diagnostic result. It suggests that a large part of the drainage residual is already correlated with water-depth and terrain variables. To make the drainage contribution stronger and more physically specific, future versions should add dynamic SWMM variables, such as node head, pipe flow, surcharging state, full-flow ratio, node flooding, and outfall discharge, rasterized or projected onto the two-dimensional grid.

## 5. Limitations

First, the drainage network is conceptual and road-aligned, not a surveyed municipal drainage network. Second, the current formal label set contains only eight events, which is enough for a demonstrator and leave-one-event-out validation but not enough for strong paper-level generalization claims. Third, SWMM numerical stability remains a limitation: only event1 is accepted by the strict 2% continuity-error threshold, while the remaining events are warning-level labels. Fourth, DrainLite is a post-processing or adapter-style model, not a full LarNO-D backbone retraining with drainage channels embedded into the neural operator. Fifth, the hydrograph timing diagnostic shows that current ITZI outputs recede much more slowly than MIKE; formal paper-level labels should be regenerated after the rainfall-timing correction and after loss, boundary, drainage-capacity, and building-runoff treatment calibration.

## 6. Conclusions

This study establishes a complete lightweight pipeline for drainage-aware urban flood prediction under local hardware constraints. A connected conceptual SWMM network was constructed from road-aligned drainage priors, ITZI-SWMM coupled labels were generated for eight events, label quality was graded, and a LarNO-compatible residual correction model was trained and validated. The all-static DrainLite model reduced the leave-one-event-out error against the ITZI-SWMM drainage label from {num(surface_cv.get("mae_mm"))} mm to {num(all_static_cv.get("mae_mm"))} mm. This demonstrates that drainage effects can be learned as a lightweight residual correction.

For a manuscript, the most defensible claim is: a road-aligned conceptual drainage prior and ITZI-SWMM generated labels can support a lightweight, explainable residual correction module that injects drainage effects into LarNO-compatible urban flood forecasting. The result should not be overstated as a complete real-pipe LarNO retraining study. The next scientific step is to enlarge the formal coupled-event set and add dynamic drainage-state variables.

## Figure and Artifact List

- Figure: `{MAIN_OUT / "figures" / "connected_static_dem_network.png"}`
- Figure: `{V2_OUT / "figures" / "v2_label_quality_grading.png"}`
- Figure: `{MAIN_OUT / "figures" / "cv_summary_mae_rmse.png"}`
- Figure: `{MAIN_OUT / "figures" / "cv_event_mae_by_model.png"}`
- Figure: `{MAIN_OUT / "figures" / "cv_volume_reduction_capture.png"}`
- Figure: `{V2_OUT / "figures" / "v2_mike_reference_summary.png"}`
- Figure: `{V2_OUT / "figures" / "v2_mike_event_mae.png"}`
- Figure: `{HYDRO_OUT / "figures" / "hydrograph_timing_summary.png"}`
- Figure: `{INF_PILOT_OUT / "figures" / "event68_infiltration_pilot_hydrograph.png"}`
- Report: `{MAIN_OUT / "report.html"}`
- Reproducibility manifest: `{V2_OUT / "REPRODUCIBILITY.md"}`

## Items Marked as Pending

- More formally calibrated drainage-network parameters: 待补充。
- True surveyed municipal drainage-network validation: 待补充。
- Larger coupled-event set beyond the current eight events: 待补充。
- Rerun formal ITZI-SWMM labels after the rainfall-timing correction: 待补充。
- Calibrate infiltration/effective loss, open-boundary drainage, and conceptual pipe capacity against MIKE hydrograph timing: 待补充。
- Test building-rainfall handling modes, especially no redistribution versus roof-runoff redistribution: 待补充。
- Full LarNO-D backbone retraining with drainage channels: 待补充。
- Five-metre MIKE reference comparison: 待补充。
"""

    MANUSCRIPT.write_text(text, encoding="utf-8")
    print(MANUSCRIPT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
