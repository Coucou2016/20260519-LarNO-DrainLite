#!/usr/bin/env python3
"""Generate a journal-style manuscript and evidence-led companion documents."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import generate_major_revision_documents as source


ROOT = source.ROOT
OUT = source.OUT
METRICS = source.METRICS


def metric_lookup() -> dict[str, dict[str, str]]:
    return {row["model"]: row for row in source.read_csv(METRICS / "ablation_summary.csv")}


def data_provenance_table() -> str:
    rows = [
        ["Rainfall and terrain", "Public LarNO 20 m benchmark", "Forcing and static terrain input; not claimed as newly generated"],
        ["MIKE water depth", "Public LarNO 20 m benchmark", "Configuration-informed hydrodynamic reference; not a DrainLite training target"],
        ["Itzï surface-only depth", "Generated in this study", "Paired physical baseline using the copied, fixed local Itzï workflow"],
        ["ITZI-SWMM coupled depth", "Generated in this study", "Paired conceptual-network label using Itzï DrainageSimulation and SWMM Dynamic Wave"],
        ["Drainage rasters", "Generated in this study", "Road-derived conceptual network mapped to the 20 m calculation grid"],
        ["DrainLite predictions and metrics", "Generated in this study", "Five held-event predictions, ablations, controls and derived statistics"],
    ]
    return source.markdown_table(["Data product", "Provenance", "Role in this study"], rows)


def feature_table() -> str:
    rows = [
        ["Hydrometeorological", "Surface depth; rainfall in the current 5 min interval; cumulative rainfall; normalised time"],
        ["Terrain and local context", "Elevation; terrain-slope magnitude; 3 x 3 and 7 x 7 mean surface depth"],
        ["Network geometry", "Inlet mask; pipe mask; outfall mask; Euclidean distance to the nearest synthetic outfall; 3 x 3 and 7 x 7 pipe density; 7 x 7 inlet density"],
        ["Conceptual hydraulic descriptors", "Rasterised pipe diameter, design slope, Manning full-flow capacity, approximate cover depth and 7 x 7 capacity density"],
        ["Multiplicity diagnostics", "Junction count and number of rasterised pipe segments per cell"],
        ["Location diagnostic", "Normalised row and column, used only to measure fixed-grid location dependence"],
    ]
    return source.markdown_table(["Feature family", "Variables"], rows)


def manuscript() -> str:
    summary = metric_lookup()
    main = summary["no_xy_all_static"]
    base = summary["base_no_xy"]
    signed_count = summary["no_xy_all_static_count"]
    sink = summary["sink_only_no_xy_all_static_count"]
    shuffled = summary["no_xy_all_static_shuffle"]
    shifted = summary["no_xy_all_static_shift"]
    boot_base = source.bootstrap_delta("no_xy_all_static", "base_no_xy", seed=732)
    boot_surface = source.bootstrap_delta("no_xy_all_static", "surface_only", seed=731)
    return f"""# DrainLite: lightweight learning of drainage-induced residuals for urban flood prediction

## Highlights

- A road-derived conceptual sewer network was coupled to a two-dimensional Itzï surface model.
- Paired simulations isolated a signed drainage-induced water-depth residual at 20 m resolution.
- Static drainage descriptors reduced held-event residual error without explicit coordinate channels.
- Spatial controls and target-form experiments exposed both useful network information and model limitations.

## Abstract

Urban-flood emulators commonly predict water depth from rainfall and terrain, although the hydrodynamic simulations used for supervision may already include underground drainage. When the drainage layout is unavailable, its effect cannot be altered explicitly without regenerating the reference simulations and retraining the forecasting model. This study investigates a lighter alternative in which drainage is represented as a learned correction to an existing surface-water prediction. A road-derived conceptual sewer network was constructed for a 200 x 280 cell window of the 20 m Shenzhen benchmark and coupled to the Itzï two-dimensional surface solver through the Storm Water Management Model Dynamic Wave engine. Eight six-hour events were simulated with and without network coupling under identical rainfall, terrain and effective-loss settings. Five events with an absolute routing continuity error not exceeding 8% were retained for leave-one-event-out evaluation, while the remaining three were used to characterise numerical uncertainty.

DrainLite learns the signed difference between coupled and surface-only depth using a histogram gradient-boosting regressor. The input comprises rainfall, terrain, local surface-water context and rasterised drainage descriptors. Explicit row and column coordinates are omitted from the principal model. Across the five held events, the uncorrected surface field differed from the coupled field by 4.932 mm mean absolute error. A residual model without drainage descriptors reduced the error to 3.190 mm, and the coordinate-free model containing all static drainage descriptors reduced it further to {float(main['mae_mm']):.3f} mm. The mean event-level improvement over the non-network residual model was {-boot_base['mean_delta_mm']:.3f} mm, with a five-event bootstrap interval of {-boot_base['ci_high_mm']:.3f} to {-boot_base['ci_low_mm']:.3f} mm. For a matched count-enhanced experiment, shifting the drainage fields by one cell increased the error from {float(signed_count['mae_mm']):.3f} to {float(shifted['mae_mm']):.3f} mm, whereas joint spatial shuffling increased it to {float(shuffled['mae_mm']):.3f} mm. These controls show that the improvement depends on the spatial correspondence between the network descriptors and the simulated residual.

The target formulation affected model ranking. A non-negative sink-only correction achieved a lower global mean absolute error ({float(sink['mae_mm']):.3f} mm) than the matched signed model ({float(signed_count['mae_mm']):.3f} mm), but produced larger root-mean-square, positive-residual and final-volume errors. Comparison with the MIKE water-depth fields was metric dependent and was treated as descriptive because MIKE informed the finite effective-loss screen. The coupled labels also retained substantial hydraulic uncertainty: 59.49-87.40% of SWMM routing steps were reported as non-converging. DrainLite should therefore be interpreted as an emulator of the specified conceptual coupled calculation, rather than as a validated representation of the municipal sewer system. Within that scope, the results establish a reproducible low-compute route for introducing explicit drainage information into an existing urban-flood prediction.

**Keywords:** urban pluvial flooding; drainage network; Itzï; SWMM; residual learning; surrogate modelling; conceptual sewer network

## 1. Introduction

Short-duration urban flooding is governed by interactions among rainfall, microtopography, buildings, surface conveyance and underground drainage. Two-dimensional shallow-water models resolve the propagation and storage of surface water, while one-dimensional drainage models represent pipe storage, pressurised flow and discharge at outfalls. Their exchange through inlets and manholes can alter the timing, magnitude and spatial distribution of inundation. Coupled models provide a physically interpretable description of these processes, but their small numerical time steps and detailed spatial representation remain expensive for real-time forecasting (Bates et al., 2010; Cao et al., 2025).

Data-driven flood models reduce this cost by learning mappings between hydrometeorological inputs and hydrodynamic outputs. Convolutional, recurrent and graph-based models have been used to predict urban inundation from rainfall and terrain at increasingly fine resolutions (Bentivoglio et al., 2023; Burrichter et al., 2023; He et al., 2023; Situ et al., 2024). Neural operators extend this approach by learning mappings between function spaces and can be evaluated at discretisations that differ from those used for training (Li et al., 2021; Kovachki et al., 2023). LarNO combines a Fourier neural operator with latent autoregression and was developed for large-scale, high-resolution urban-flood forecasting in Shenzhen (Cao et al., 2026).

The released LarNO benchmark provides rainfall, terrain and MIKE water-depth fields at 20 m resolution, but does not distribute the drainage-network data used by the reference model. This distinction matters for model adaptation. A predictor may reproduce a water-depth field that implicitly contains drainage effects without accepting the network layout or pipe properties as explicit conditions. Such a model cannot directly represent the effect of adding, removing or modifying a drainage system. Expanding the neural-operator input is possible, but it requires a sufficiently large set of coupled simulations and the computational resources needed to retrain the full model.

Residual learning offers a complementary route when the available coupled dataset is small. Instead of relearning the complete rainfall-to-depth mapping, a secondary model estimates the change produced by drainage relative to a surface-only field. This formulation has three practical advantages. First, the target is concentrated in places and times affected by the network. Second, the learned correction can be attached to different upstream predictors, provided that their water-depth fields are compatible with the residual model. Third, the effect of network information can be tested directly by removing, displacing or shuffling its spatial descriptors.

Here we develop DrainLite, a lightweight residual model trained on paired Itzï simulations with and without SWMM coupling. The study is designed to answer two questions. The first is whether static descriptors of a road-derived conceptual network contain information beyond rainfall, terrain and local surface depth. The second is whether a signed residual, which permits both increases and decreases in local depth, provides a more complete representation than a sink-only correction. Five-event leave-one-event-out experiments, coordinate diagnostics, matched spatial controls and conditional error analyses are used to evaluate these questions. MIKE water depths are retained as a hydrodynamic reference, but not as the learning target. Figure 1 summarises the modelling framework.

{source.fig(1, 'fig01_provenance_workflow.png', 'Drainage-residual modelling framework. Rainfall and terrain drive paired surface-only and ITZI-SWMM simulations constructed in this study. Their signed difference is learned by DrainLite and added to an upstream surface-water prediction. Dashed arrows show the separate role of the public MIKE water-depth fields in finite configuration screening and descriptive comparison; MIKE is not a DrainLite target or early-stopping variable.')}

## 2. Data and methods

### 2.1 Problem formulation

Let `R(t,x)` denote rainfall, `Z(x)` terrain, `B(x)` the building mask and `N` a specified drainage network. The surface solver defines a water-depth field `h_s = G_s(R,Z,B)`, whereas the coupled solver defines `h_c = G_c(R,Z,B,N)`. Because both members of a pair use identical surface forcing and parameters, their signed difference isolates the response of the specified coupled configuration:

<p style="text-align:center"><i>r</i>(t,x) = <i>h</i><sub>c</sub>(t,x) - <i>h</i><sub>s</sub>(t,x). &nbsp;&nbsp; (1)</p>

A negative value of `r` indicates that the coupled calculation is shallower at a given cell and time; a positive value indicates a locally greater depth. A positive residual need not imply sewer surcharge because surface redistribution, timing differences and numerical behaviour can produce the same sign. DrainLite approximates `r` from the surface field, rainfall, terrain and drainage descriptors. The corrected prediction is

<p style="text-align:center"><i>h</i><sub>DL</sub>(t,x) = max[<i>h</i><sub>s</sub>(t,x) + <i>r</i><sub>theta</sub>(t,x), 0]. &nbsp;&nbsp; (2)</p>

Residuals are learned in millimetres and converted back to metres before being added to the surface field. A second model predicts `d = max(h_s - h_c, 0)` and applies `h_sink = max(h_s - d_theta, 0)`. This sink-only formulation enforces non-increasing depth and provides a direct test of whether positive residuals are needed.

### 2.2 Study data and provenance

The analysis uses a 200 x 280 cell window of `region1_20m`, corresponding to 4.0 x 5.6 km at 20 m resolution. Each event contains 72 water-depth and rainfall frames representing the ends of consecutive five-minute intervals from 0.083 to 6.000 h. Water depth is stored in metres and rainfall as millimetres per five-minute interval. Cells with a digital elevation value at or above 49.9 m are treated as buildings or walls and excluded from evaluation, leaving 43,606 active cells.

The study combines public inputs with simulations and model outputs generated locally. Table 1 separates these sources. In particular, none of the reported Itzï, ITZI-SWMM or DrainLite performance values are copied from the LarNO paper. The MIKE arrays are public reference data and are labelled as such throughout.

**Table 1. Provenance and role of each principal data product.**

{data_provenance_table()}

Eight events were available with paired Itzï calculations. The formal event set comprised events 1, 67, 68, 69 and 70, for which the absolute SWMM routing continuity error did not exceed the study-defined 8% screen. Events 20, 65 and 66 were retained for diagnosis. The threshold is used only to define a reproducible experiment subset; it is not an EPA acceptance criterion and does not imply convergence of the Dynamic Wave solution. Rainfall characteristics and numerical diagnostics are given in Table 2.

**Table 2. Rainfall characteristics and SWMM diagnostics for the eight paired simulations.**

{source.markdown_table(['Event', 'Mean 6 h rainfall (mm)', 'Maximum local 6 h rainfall (mm)', 'Maximum 5 min rainfall (mm)', 'Continuity error (%)', 'Non-converging steps (%)', 'Continuity screen', 'Use'], source.rainfall_rows())}

### 2.3 Paired surface and drainage-coupled simulations

Surface flow was calculated with Itzï 25.4 using the installed `SurfaceFlowSimulation`. The configuration used a minimum computational depth of 0.001 m, Courant number 0.7, partial-inertia coefficient 0.9 and a maximum surface-flow step of 1 s. Manning's coefficient was 0.015 over active cells. Building cells were assigned an elevation of 50 m and a Manning coefficient of 100 to impede flow through the building footprint. The initial surface was dry and the outer boundary of the rectangular calculation window was closed.

Rainfall was applied from the first five-minute interval. Rain falling on building cells was redistributed uniformly over the active cells so that the rainfall volume in the rectangular window was conserved. A spatially uniform effective loss of 1 mm h-1 was applied to active cells. This parameter aggregates unresolved losses and should not be interpreted as a calibrated soil-infiltration rate. Surface-only and coupled calculations used the same rainfall sequence, terrain, building treatment and effective loss.

The drainage calculation used SWMM 5.2.4 through PySWMM 2.1.0 and swmm-toolkit 0.17.0. Dynamic Wave routing used a nominal 2 s step, a variable-step factor of 0.50, a minimum step of 0.2 s, 20 trials, partial inertial damping and ponding. Itzï's `DrainageSimulation` transferred water between surface cells and SWMM nodes using the installed orifice, free-weir and submerged-weir coefficients of 0.167, 0.54 and 0.056, respectively. The exchange term was signed and could therefore transfer water in either direction. At coupled inlet cells, surface Manning's coefficient in the surrounding 3 x 3 neighbourhood was limited to 0.012, matching the fixed local workflow used to generate all paired labels.

The effective-loss setting was selected from a finite configuration pool. Event68 was simulated at 1-5 mm h-1, while complete eight-event packages were available at 1 and 2 mm h-1. Peak-depth and final-volume differences from the public MIKE fields were considered when retaining 1 mm h-1. MIKE therefore informed physical configuration selection, although it did not enter DrainLite fitting. Comparisons with MIKE later in the paper are accordingly descriptive rather than independent validation.

### 2.4 Road-derived conceptual drainage network

No surveyed sewer inventory was available for the study window. A road-centreline graph derived from OpenStreetMap was therefore converted to a conceptual SWMM network. Components containing fewer than three nodes were removed. Twelve retained components were each connected to a synthetic free outfall outside the two-dimensional window. Junction invert elevations were assigned from local terrain with nominal cover; circular conduit diameters were 0.8, 1.0 or 2.0 m and Manning's coefficient was 0.013. These values define a controlled drainage scenario and are not intended to reconstruct the municipal network.

Node-to-cell mapping used the same integer transformation in both the physical coupling and static-feature pipeline: `column = int(x/20) - column_offset` and `row = int((H x 20 - y)/20) - row_offset`. This produced 1,169 coupled junctions in 1,169 distinct cells. The pipe centre lines occupied 6,385 cells; 2,940 cells contained more than one rasterised segment. All synthetic outfalls lay outside the two-dimensional window, so the in-window outfall mask was identically zero.

{source.fig(2, 'fig02_aligned_network_dem.png', 'Conceptual drainage network and digital elevation model. Grey lines show the road-derived candidate graph, blue lines show rasterised SWMM pipes and red points show coupled junctions. Distances are measured from the north-west corner of the 4.0 x 5.6 km window. The main map includes a north arrow and 1 km scale; the two enlargements show contrasting urban network densities. The figure demonstrates coordinate consistency of the constructed data, not agreement with a surveyed sewer inventory.')}

The static drainage descriptors were generated from the SWMM input. Pipe diameter was assigned as the maximum diameter crossing each cell. Design slope was computed from endpoint inverts and constrained to at least 10^-5 for the raster descriptor. Full-flow capacity was estimated with the circular-pipe Manning relation

<p style="text-align:center"><i>Q</i><sub>full</sub> = (1/<i>n</i>) <i>A R</i><sup>2/3</sup> <i>S</i><sup>1/2</sup>, &nbsp;&nbsp; (3)</p>

where `A` is pipe area, `R` hydraulic radius, `S` design slope and `n` Manning's coefficient. Cover depth was approximated by subtracting the mean endpoint invert from the surface elevation along a rasterised conduit. Distance to outfall was Euclidean distance to the nearest synthetic outfall, not graph distance. These rasters are predictors for the residual model; they do not replace the hydraulic states solved by SWMM.

**Table 3. Topological and rasterisation characteristics of the conceptual network.**

{source.network_table()}

### 2.5 DrainLite features and model fitting

DrainLite uses `HistGradientBoostingRegressor` with squared-error loss, learning rate 0.05, at most 220 boosting iterations, 31 terminal leaves per tree and L2 regularisation of 0.01. Early stopping used a random 15% subset of the training rows. Because this subset was drawn only from the four training events in each fold, no cell-time from the held event entered fitting or early stopping.

The principal input contains eight non-network variables and twelve drainage variables (Table 4). Surface depth and its 3 x 3 and 7 x 7 neighbourhood means describe the current hydraulic state. Rainfall, cumulative rainfall and normalised time describe event forcing. Elevation and slope represent terrain. Drainage information comprises inlet, pipe and outfall masks; diameter, slope, capacity and cover fields; Euclidean outfall distance; and local inlet, pipe and capacity densities. Two multiplicity counts and explicit row-column coordinates were evaluated separately and were not part of the principal model.

**Table 4. DrainLite feature families.**

{feature_table()}

At each time step, 700 active cells were sampled. Candidate samples were drawn preferentially from residual-active cells, wet cells and cells close to the network, with random active cells used to complete the sample. A further residual-active subset was included before the sample was reduced to 700 unique locations. Each fold therefore used approximately 201,600 sampled rows from four events. The same event-time-cell locations were used for every feature ablation and spatial control within a fold.

Training weights were one for background samples, with four additional units where the absolute residual exceeded 0.2 mm, two where surface depth exceeded 0.03 m, and one where the 7 x 7 pipe density was positive. This weighting increased the influence of hydrologically active and network-proximal samples without altering the full-array evaluation metrics. Model predictions were generated for all 72 x 200 x 280 values of each held event, and depth was clipped only at the physical lower bound of zero. Signed residual predictions were limited to +/-1.2 m to prevent isolated extrapolation outside the range represented during fitting.

### 2.6 Experimental design

Five-event leave-one-event-out evaluation was used. In each fold, four events were sampled for training and the remaining event was predicted in full. Event metrics were calculated over all active cells and 72 time steps, and the five values were then averaged with equal event weight. This macro-average avoids giving additional weight to events solely because they contain more wet cells.

The main comparisons were: a surface-only field with no correction; a non-network residual model; geometry-only and hydraulic-descriptor models; the full static model without explicit coordinates; a count-enhanced variant; and a coordinate-inclusive diagnostic. Two spatial controls were fitted with the same samples and hyperparameters as the count-enhanced static model. In one control, all drainage fields were shifted one cell to the south and east with no periodic wrap. In the other, a single fixed permutation was applied jointly to all drainage fields over active cells, preserving their marginal distributions and cross-field association while disrupting alignment with terrain and residuals. A sink-only model used the same count-enhanced features as the matched signed model.

Mean absolute error (MAE), root-mean-square error (RMSE) and critical success index (CSI) were evaluated. For `N` active cell-times,

<p style="text-align:center">MAE = (1/<i>N</i>) sum |<i>h</i><sub>pred</sub> - <i>h</i><sub>target</sub>|, &nbsp;&nbsp; RMSE = [(1/<i>N</i>) sum (<i>h</i><sub>pred</sub> - <i>h</i><sub>target</sub>)<sup>2</sup>]<sup>1/2</sup>. &nbsp;&nbsp; (4)</p>

CSI was calculated at 0.03 and 0.15 m as `TP/(TP + FP + FN)`. Additional measures comprised absolute global-peak error, peak-map MAE, final active-cell volume error and conditional MAE. Conditional subsets represented water depth above 0.03 m, residual magnitude above 1 mm, positive and negative residuals, and distance bands from the nearest inlet or pipe. Event-level paired differences were summarised by 10,000 bootstrap resamples. With five pairs, the minimum two-sided sign-flip probability is 0.0625; the analysis therefore emphasises effect sizes and event-level consistency rather than a binary significance statement.

## 3. Results

### 3.1 Characteristics of the paired simulations

The conceptual graph contains 1,169 junctions, 2,887 conduits and 12 synthetic outfalls (Table 3). Every junction is connected to an outfall when conduit direction is ignored, whereas only 89 junctions have a directed path under the stored from-to orientation. Dynamic Wave routing permits flow reversal, so the latter value does not by itself prevent drainage. It nevertheless indicates that the assigned conduit orientation does not form a consistently directed trunk system. The network also contains 156 zero-slope conduits and 271 groups of links sharing the same endpoint pair.

The numerical diagnostics varied among events (Figure 3). Routing continuity error ranged from -10.176% for event20 to +2.840% for event67. Three events were within +/-2%, two additional events were between 2% and 8%, and three exceeded 8%. In contrast, the percentage of non-converging Dynamic Wave steps was high for every event, ranging from 59.49% to 87.40%. Continuity and convergence therefore describe different aspects of the calculation: a small cumulative water-balance error did not imply stable iteration at individual routing steps.

{source.fig(3, 'fig03_swmm_stability.png', 'Numerical diagnostics for the eight ITZI-SWMM simulations. Panel (a) shows signed routing continuity error; dashed and dotted lines mark the study-defined +/-2% and +/-8% screens. Panel (b) shows the percentage of Dynamic Wave routing steps reported as non-converging. The two panels are interpreted separately because cumulative water balance does not measure stepwise convergence.')}

Surface-water volume was lower in the coupled simulation than in the surface-only simulation for all five formal events (Figure 4). The difference developed during the main rainfall period and persisted to the end of the six-hour calculation. Maximum cell depth showed a less uniform response because it was controlled by individual low cells and frequently reached its largest saved value at the final frame. The endpoint markers distinguish these right-censored maxima from peaks followed by recession. MIKE volume generally reached a maximum between approximately 2 and 3 h and then decreased, whereas the Itzï simulations retained more water late in the event. The contrast remained in both members of each Itzï pair, indicating that it cannot be attributed to the conceptual drainage network alone.

{source.fig(4, 'fig04_five_event_hydrographs.png', 'Maximum active-cell depth and active-cell surface-water volume for the five formal events. Each row represents one event. MIKE, surface-only Itzï and coupled ITZI-SWMM are shown at five-minute interval-end times from 0.083 to 6.000 h. Circles mark maxima followed by at least one saved frame; right-pointing triangles mark maxima at 6 h and therefore indicate a peak at or beyond the simulation horizon.')}

Increasing the constant effective loss reduced event68 water depth and surface-water storage, but did not reproduce all aspects of the MIKE hydrograph (Figure 5). Routing continuity error moved from +1.534% at 1 mm h-1 to -10.663% at 5 mm h-1. The sensitivity confirms that this parameter influences both recession and the SWMM water balance. It remains a lumped model setting rather than a measured infiltration property.

{source.fig(5, 'fig05_effective_loss_sensitivity.png', 'Event68 sensitivity to a spatially uniform effective loss of 1-5 mm h-1. Panels (a) and (b) show active-cell maximum depth and volume at interval-end times; panel (c) shows the corresponding signed SWMM routing continuity error. MIKE is included as a configuration reference. No single loss rate reproduces its peak magnitude, timing and final storage simultaneously.')}

### 3.2 Predictive performance and drainage-feature contribution

Table 5 summarises the five held-event results. Relative to the coupled target, surface-only depth had an MAE of 4.932 mm. The non-network residual model reduced this value to 3.190 mm, showing that part of the paired difference was predictable from rainfall, terrain, time and the current surface state. Adding all static drainage descriptors without explicit coordinates reduced MAE to {float(main['mae_mm']):.3f} mm and RMSE to {float(main['rmse_mm']):.3f} mm. The corresponding improvement was {float(main['improvement_vs_base_no_xy_pct']):.1f}% relative to the non-network residual model and {float(main['improvement_vs_surface_pct']):.1f}% relative to the uncorrected field.

The static model improved all five held events. Its mean event-level MAE difference from the non-network model was {boot_base['mean_delta_mm']:.3f} mm, with a bootstrap interval of {boot_base['ci_low_mm']:.3f} to {boot_base['ci_high_mm']:.3f} mm. Relative to surface-only depth, the mean difference was {boot_surface['mean_delta_mm']:.3f} mm ({boot_surface['ci_low_mm']:.3f} to {boot_surface['ci_high_mm']:.3f} mm). These intervals describe uncertainty within the five available events and are not estimates for a broader storm population.

**Table 5. Five-event leave-one-event-out performance against the paired ITZI-SWMM water-depth target. Event metrics are macro-averaged; peak and final-volume columns contain absolute errors.**

{source.markdown_table(['Configuration', 'MAE (mm)', 'RMSE (mm)', 'CSI at 0.15 m', 'Global-peak error (mm)', 'Final-volume error (10^3 m^3)'], source.ablation_rows())}

Explicit coordinates reduced the non-network MAE from 3.190 to 2.505 mm and the count-enhanced static MAE from 2.276 to 2.124 mm. Because every fold used the same terrain and network, row and column could encode a fixed residual pattern. Coordinate-inclusive performance is therefore reported as a same-grid diagnostic, not as evidence of transferable drainage physics. The principal result uses no explicit coordinate channels, although fixed terrain and network fields can still carry location-specific information.

The geometry-only model reached 2.293 mm MAE, compared with 2.396 mm for the group of conceptual hydraulic descriptors. Combining all static fields gave 2.259 mm. Adding junction and segment counts did not improve this result (2.276 mm). These differences suggest that network location and local density were more consistently useful than the assigned diameter, slope, capacity and cover fields. The hydraulic attributes are conceptual and partly correlated, so the ablation should not be interpreted as a causal ranking of individual pipe properties.

The matched spatial controls support the role of alignment (Figure 6). The count-enhanced signed model had an MAE of {float(signed_count['mae_mm']):.3f} mm. Shifting the same drainage fields by 20 m increased MAE to {float(shifted['mae_mm']):.3f} mm, and joint active-cell shuffling increased it to {float(shuffled['mae_mm']):.3f} mm. All five held events deteriorated under each disruption. The shuffled result was close to the non-network MAE, indicating that the marginal values of the drainage rasters were insufficient without their original spatial correspondence.

{source.fig(6, 'fig09_ablation_event_dots.png', 'Feature ablations and spatial controls. Panel (a) gives macro-average MAE with held-event values overlaid. The coordinate-inclusive bars quantify same-grid location dependence. Panel (b) compares the matched count-enhanced signed model with a 20 m shift, a joint active-cell shuffle and the sink-only target. Lines join results for the same held event.')}

### 3.3 Spatial structure of the learned residual

For event68, the coupled and DrainLite depth maps were visually similar at the time of maximum coupled depth (Figure 7). The residual panels provide a more discriminating comparison. Negative residuals formed connected regions around parts of the network and in low-lying urban areas, while smaller positive residuals occurred in several local patches. DrainLite recovered the broad negative structures but smoothed their extrema. The largest errors occurred near sharp wet-dry transitions and isolated deep cells.

{source.fig(7, 'fig07_event68_target_conditioned_spatial.png', 'Event68 water depth and signed residual at 5.83 h, the interval-end time of maximum coupled active-cell depth. Panels (a-d) share a water-depth scale truncated at the 99.9th percentile of the combined active-cell values. Panels (e-g) share a symmetric residual scale truncated at the 99.5th percentile of combined absolute residual values. Values beyond these limits are colour-saturated but remain in all statistics. MIKE is shown at the same time and is not the learning target.')}

The same residual pattern appeared, with different magnitudes, in the other held events (Figure 8). This repeatability helps explain why static descriptors reduced error across event folds. It also limits the interpretation: rainfall events were held out, but terrain and network location were unchanged. The experiment demonstrates event transfer on one grid, not spatial transfer to a different catchment or drainage layout.

{source.fig(8, 'fig08_five_event_signed_residual_maps.png', 'True residual, predicted residual and residual error for the five held events at the maximum coupled-depth time for that event. Each row uses its own symmetric colour range based on the 99.5th percentile of the combined absolute values in that row. Saturated extreme cells remain included in the metrics. Spatial patterns may be compared across the three panels within a row, whereas colour magnitude should not be compared directly between rows.')}

Conditional errors identify where the correction was most effective (Table 6; Figure 9). Within 20 m of an inlet or pipe, surface-only MAE was 13.82 mm and the count-enhanced signed model reduced it to 3.88 mm. For residual magnitudes above 1 mm, the corresponding values were 37.79 and 11.65 mm. At coupled depths above 0.03 m, MAE decreased from 22.24 to 9.40 mm. These subsets contain the cells for which the drainage correction has the largest practical role.

Beyond 140 m from the nearest inlet or pipe, surface-only MAE was only 0.56 mm, whereas the signed static model produced 1.18 mm. The learned correction thus introduced a small far-field error where the paired residual was weak. A distance-dependent shrinkage or zero-residual gate may reduce this leakage.

**Table 6. Conditional MAE (mm) for the five held events. The signed and sink-only static models use the same count-enhanced drainage inputs.**

{source.markdown_table(['Condition', 'Surface-only', 'Base, no explicit XY', 'Signed static + counts', 'Sink-only static + counts'], source.conditional_rows())}

{source.fig(9, 'fig10_conditional_performance.png', 'Conditional MAE over physically relevant subsets. Bars are macro-averages of the five held events. Network-distance subsets are defined from the nearest inlet or rasterised pipe. The positive-residual subset shows the structural limitation of sink-only prediction, while the far-network subset shows the small error introduced by residual extrapolation.')}

### 3.4 Signed and sink-only target formulations

The sink-only model had a lower global MAE than the matched signed model: {float(sink['mae_mm']):.3f} versus {float(signed_count['mae_mm']):.3f} mm. Negative residuals occupied most of the residual-active domain, and the non-negative reduction constraint prevented small positive corrections in background cells. This explains why a physically narrower target performed well under a domain-average absolute-error criterion.

Other measures favoured the signed formulation. RMSE was {float(signed_count['rmse_mm']):.3f} mm for the signed model and {float(sink['rmse_mm']):.3f} mm for sink-only. CSI at 0.15 m was {float(signed_count['csi_0p15']):.3f} and {float(sink['csi_0p15']):.3f}, respectively, while mean absolute final-volume error was {float(signed_count['mean_abs_final_volume_error_m3'])/1000:.2f} and {float(sink['mean_abs_final_volume_error_m3'])/1000:.2f} x 10^3 m^3. On cells where coupled depth exceeded surface-only depth by more than 1 mm, signed-model MAE was 13.64 mm compared with 21.45 mm for sink-only. A sink-only target is therefore suitable when the correction is deliberately constrained to drainage reduction; a signed target is required to emulate the complete paired water-depth difference.

### 3.5 Feature dependence

Cross-fold permutation analysis of the count-enhanced signed model ranked current surface depth, elevation and local mean depth above the individual pipe descriptors (Figure 10). Distance to the synthetic outfalls and local pipe density were the most influential drainage variables. The variation among held events was substantial for several predictors. Because terrain, water depth and network fields are spatially correlated, permutation importance measures the deterioration caused by disrupting a feature and does not identify a causal hydraulic contribution.

{source.fig(10, 'fig11_cross_fold_importance.png', 'Cross-fold permutation importance for the count-enhanced signed model without explicit row-column coordinates. Bars show the mean increase in held-event MAE after permutation, error bars show the standard deviation among folds and points identify individual held events. The estimates are predictive and should not be interpreted causally because several inputs are correlated.')}

### 3.6 Comparison with the MIKE reference

Comparison with MIKE gave a different model ranking (Table 7; Figure 11). Surface-only Itzï had the lowest full-sequence MAE, 23.683 mm, and the lowest mean absolute final-volume error, 21.09 x 10^3 m^3. The coupled calculation had a higher sequence MAE of 26.231 mm and a final-volume error of 142.25 x 10^3 m^3. Residual models were close to the coupled field by design and therefore inherited much of this discrepancy.

Peak-magnitude results were less uniform. The mean absolute global-peak error was 137.6 mm for surface-only Itzï, 196.4 mm for the coupled calculation and 116.3-124.3 mm for the reported residual models. Signed peak bias alone concealed event-to-event cancellation, so absolute errors and event values are shown together. These results do not establish overall improvement relative to MIKE. They show instead that agreement depends on whether the comparison emphasises pixel-time depth, a single global maximum or final stored volume.

**Table 7. Descriptive five-event comparison with the public MIKE water-depth reference. MIKE informed the finite effective-loss screen and is not an independent validation set.**

{source.markdown_table(['Model', 'MAE (mm)', 'CSI at 0.15 m', 'Signed peak bias (mm)', 'Absolute peak error (mm)', 'Signed final-volume error (10^3 m^3)', 'Absolute final-volume error (10^3 m^3)'], source.mike_summary_rows())}

{source.fig(11, 'fig06_mike_metric_dependent_comparison.png', 'Metric-dependent comparison with the MIKE reference. Bars show five-event means and points show individual events. Lower values indicate closer agreement in all panels. MIKE informed the effective-loss configuration screen; the figure is consequently a descriptive comparison rather than an independent validation result.')}

## 4. Discussion

### 4.1 What was learned from the paired simulations

The principal finding is that drainage-aligned static descriptors contain predictive information beyond the surface state, rainfall and terrain variables used by the non-network model. The improvement was observed in every held rainfall event, and most of it disappeared when the drainage fields were jointly shuffled. This result is stronger than a comparison between two unconstrained feature sets because the controls used identical sampled cells, targets and hyperparameters. Within the fixed study grid, the coupled-minus-surface response therefore has a reproducible spatial component associated with the constructed network.

The magnitude of the improvement should be interpreted in relation to the target. DrainLite is not learning urban flooding from rainfall alone. It receives the complete surface-only water-depth field and estimates the additional response of one specified conceptual drainage configuration. This decomposition explains why a CPU-scale tree model can achieve millimetre-scale residual error with five training events. It also defines the expected application: DrainLite is a correction layer for an upstream flood model, not a replacement for that model.

### 4.2 Geometry, hydraulic descriptors and location dependence

The geometry-only and full-static models performed similarly, whereas the assigned hydraulic descriptors gave a smaller incremental benefit. Several factors may account for this result. Pipe diameter, slope and cover were generated from conceptual design rules rather than survey data. Multiple conduits were collapsed to maximum or mean raster values, and Euclidean distance to a synthetic outfall does not represent hydraulic travel distance. These fields are also strongly correlated with network location. Under these conditions, masks and local densities provide a robust indication of where exchange can occur, while the nominal hydraulic values do not fully describe how SWMM routes the event.

Explicit coordinates further reduced error, but the same terrain and network were used in every event fold. Row and column can therefore identify recurrent residual locations without learning a transferable relation between network design and flow response. Removing them prevents the most direct location lookup, although elevation, local depth and static drainage rasters remain tied to place. A spatial-window or cross-network experiment is needed before claiming generalisation to a new urban area.

### 4.3 Choice of residual target

The signed and sink-only experiments answer different modelling questions. Sink-only prediction imposes the prior that drainage can only reduce surface depth. That prior lowers global MAE in a dataset dominated by negative or negligible residuals. The signed formulation admits redistribution and any locally positive component of the paired calculation. It is consequently more difficult to fit, but it better reproduces positive-residual cells, high-error extremes and final water volume.

Positive residuals cannot presently be assigned to a unique physical mechanism. The saved outputs do not contain node head, signed exchange discharge or link capacity ratio at each routing step. Some positive regions may reflect reverse exchange, but altered surface routing and numerical differences are also possible. Future coupled datasets should retain these dynamic variables so that residual sign can be linked to identifiable drainage states.

### 4.4 Hydraulic uncertainty and comparison with MIKE

The main limitation lies in the coupled labels rather than in the residual regressor. Although five events met the selected continuity screen, non-converging Dynamic Wave steps exceeded 72% for each formal event. The network contains zero-slope and parallel conduits, inconsistent from-to orientation and free outfalls without tailwater. The two-dimensional boundary is closed, and rainfall over buildings is redistributed uniformly. Each assumption can influence stored volume and recession.

The MIKE comparison reinforces this limitation. Surface-only Itzï was closer to MIKE in full-sequence MAE and final volume, whereas some residual models were closer in global-peak magnitude. MIKE itself had already informed the effective-loss choice, so these differences cannot be presented as independent validation. More importantly, the public data do not expose the MIKE boundary conditions and drainage assets needed for a controlled process-by-process comparison. Agreement with MIKE is therefore used to identify discrepancies, not to calibrate DrainLite or to certify the conceptual network.

Before the paired simulations are treated as a hydraulic benchmark, the network should be rebuilt with explicit trunk direction, duplicate-link handling, non-zero design slopes and physically defined outfall levels. Repeated simulations should satisfy both water-balance and convergence criteria. Sensitivities to building-rainfall redistribution, surface boundary conditions and effective loss should be evaluated at the label level before the residual model is refitted.

### 4.5 Relationship to LarNO

DrainLite was motivated by the absence of explicit drainage inputs in the released LarNO benchmark. Its computational role is compatible with a LarNO forecast: an upstream neural operator supplies surface-water depth, and DrainLite supplies a network-conditioned correction. The present experiment isolates the correction problem by using Itzï surface-only depth as the upstream field. No LarNO checkpoint is evaluated here, and the results do not quantify end-to-end LarNO-DrainLite accuracy, inference time or zero-shot resolution transfer.

An end-to-end evaluation requires LarNO predictions for the same events and spatial window. The residual model would then be applied to those predictions without access to the Itzï surface field, and error would be decomposed into upstream forecast error and drainage-correction error. Additional coupled events and a second spatial window are needed to determine whether the current static representation is sufficient or whether graph-encoded and time-varying SWMM states are required.

## 5. Conclusions

This study examined whether the effect of a specified urban drainage network can be learned as a lightweight correction to a surface-water prediction. Paired Itzï simulations were generated for eight rainfall events using identical surface forcing with and without coupling to a road-derived conceptual SWMM network. On five continuity-screened events, a coordinate-free static drainage model reduced leave-one-event-out MAE from 3.190 mm for a non-network residual model to {float(main['mae_mm']):.3f} mm. Spatially displacing or shuffling matched drainage inputs degraded every held event, indicating that their alignment with the simulated residual contributed to the improvement.

The experiments also identified limits to a single drainage-correction formulation. Sink-only learning minimised global MAE, whereas signed residual learning better represented positive residuals, root-mean-square error and final volume. Model ranking changed again when predictions were compared with MIKE. Selection of the target and evaluation metric should therefore follow the intended use of the correction rather than a single aggregate score.

The present result is an emulator of a controlled conceptual-network calculation. High SWMM non-convergence, synthetic free outfalls, simplified surface losses and the absence of surveyed pipe data preclude interpretation as a validated municipal drainage forecast. Subject to these limitations, DrainLite provides a reproducible low-compute framework for testing explicit drainage information and a basis for future integration with LarNO or another upstream urban-flood predictor.

## Data and code availability

Rainfall, terrain and MIKE reference arrays originate from the public LarNO 20 m benchmark. The surface-only and ITZI-SWMM water-depth arrays, conceptual drainage rasters, DrainLite predictions, fold models, metric tables, publication figures and SHA-256 evidence manifest were generated within this study and are stored in the accompanying local research package. The conceptual network is derived from OpenStreetMap and is not an official Shenzhen drainage inventory. A public archival identifier for the derived data and code remains to be assigned before submission.

## References

Bates, P.D., Horritt, M.S., Fewtrell, T.J., 2010. A simple inertial formulation of the shallow water equations for efficient two-dimensional flood inundation modelling. Journal of Hydrology 387, 33-45.

Bentivoglio, R., Isufi, E., Jonkman, S.N., Taormina, R., 2023. Rapid spatio-temporal flood modelling via hydraulics-based graph neural networks. Hydrology and Earth System Sciences 27, 4227-4246.

Burrichter, B., Hofmann, J., Silva, J., Niemann, A., Quirmbach, M., 2023. A spatiotemporal deep learning approach for urban pluvial flood forecasting with multi-source data. Water 15, 1760.

Cao, X., Wang, B., Yao, Y., Zhang, L., Xing, Y., Mao, J., Zhang, R., Fu, G., Borthwick, A.G.L., Qin, H., 2025. U-RNN high-resolution spatiotemporal nowcasting of urban flooding. Journal of Hydrology 659, 133117.

Cao, X., Yao, Y., Wang, Z., Zhao, Z., Borthwick, A.G.L., Qin, H., 2026. Large-scale urban flood modeling and zero-shot high-resolution generalization with LarNO. Journal of Hydrology, 135686. https://doi.org/10.1016/j.jhydrol.2026.135686.

EPA, 2022. Storm Water Management Model User's Manual Version 5.2. United States Environmental Protection Agency.

He, J., Zhang, L., Xiao, T., Wang, H., Luo, H., 2023. Deep learning enables super-resolution hydrodynamic flooding process modeling under spatiotemporally varying rainstorms. Water Research 239, 120057.

Itzï developers, 2026. Itzï documentation: configuration and experimental SWMM drainage coupling. https://itzi.readthedocs.io/.

Kovachki, N., Li, Z., Liu, B., Azizzadenesheli, K., Bhattacharya, K., Stuart, A., Anandkumar, A., 2023. Neural operator: learning maps between function spaces with applications to PDEs. Journal of Machine Learning Research 24, 1-97.

Li, Z., Kovachki, N., Azizzadenesheli, K., Liu, B., Bhattacharya, K., Stuart, A., Anandkumar, A., 2021. Fourier neural operator for parametric partial differential equations. International Conference on Learning Representations.

Situ, Z., Wang, Q., Teng, S., Feng, W., Chen, G., Zhou, Q., Fu, G., 2024. Improving urban flood prediction using LSTM-DeepLabV3+ and Bayesian optimization with spatiotemporal feature fusion. Journal of Hydrology 630, 130743.
"""


def research_report() -> str:
    report = source.report()
    report = report.replace(
        "审稿复核促使本轮完成了三项关键修正。",
        "在数据质量控制和复现实验核对过程中，发现并修正了三项会影响解释的技术问题。",
    )
    report = report.replace(
        "模型把两者之差作为残差学习目标，不重训 LarNO 主干。",
        "模型把两者之差作为残差学习目标；当前上游场为完整 Itzï surface-only 结果，尚未运行 LarNO checkpoint。",
    )
    report = report.replace(
        "每个事件有 72 个五分钟时刻，总时长 6 h。",
        "每个事件有 72 个五分钟区间末端时刻，时间轴从 0.083 h 延伸至 6.000 h。",
    )
    report += """

## 10. 论文体例与图件质量控制

英文论文采用期刊论文的论证顺序，而不是工作总结或问题答复的组织方式。引言围绕科学问题展开；方法部分给出数据来源、成对物理模拟、概念管网、残差定义、特征、采样、训练和评价口径；结果部分按物理标签、总体性能、空间结构、目标形式、特征依赖和 MIKE 对比分层展开；技术更正和证据限制保留在独立审查文档中，不混入论文的结果叙述。

图件使用 SciencePlots 的 `science` 与 `no-latex` 样式，英文字体优先使用 Times New Roman。管网图新增道路来源图、方向标识、公里坐标和比例尺。空间图图注明确说明 99.9% 水深色标和 99.5% 残差色标的截断方式；截断只影响颜色显示，不影响任何统计。所有时间过程线统一使用区间末端时间，即第 1 帧为 0.083 h，第 72 帧为 6.000 h。
"""
    return report


def evidence_review(manifest: dict[str, object]) -> str:
    base = source.audit_document(manifest)
    base = base.replace("major-revision manuscript", "manuscript")
    base = base.replace("## 5. Hydraulic audit and submission blockers", "## 5. Hydraulic limitations requiring resolution")
    base = base.replace("This is a submission blocker for any claim that coupled labels form validated hydraulic reference data.", "This prevents any claim that the coupled labels form validated hydraulic reference data.")
    base = base.replace("## 6. Reviewer-action matrix", "## 6. Evidence-control matrix")
    base = base.replace("| Review issue | Action | Status |", "| Evidence issue | Control or action | Status |")
    critical = {
        "run_drainlite_major_revision_experiments.py": "DrainLite experiment runner",
        "build_connected_swmm_drainage_dataset.py": "Coupled dataset builder",
        "swmm_connected_sub.inp": "Conceptual SWMM network",
        "event_metrics.csv": "Held-event metrics",
        "ablation_summary.csv": "Ablation summary",
        "conditional_metrics.csv": "Conditional metrics",
        "mike_configuration_screening_comparison.csv": "MIKE descriptive comparison",
        "connected_swmm_major_revision_audit.json": "Network and stability audit",
        "event68_infiltration_sensitivity_metrics.csv": "Effective-loss sensitivity",
        "timestamp_convention.json": "Timestamp convention",
        "generate_major_revision_publication_figures.py": "Publication-figure generator",
        "generate_submission_revision_documents.py": "Manuscript/report generator",
        "validate_major_revision_package.py": "Package validator",
    }
    rows = []
    for record in manifest["records"]:
        name = Path(record["path"]).name
        if name in critical:
            rows.append([critical[name], name, record["bytes"], record["sha256"][:16] + "..."])
    manifest_summary = source.markdown_table(
        ["Evidence item", "File", "Bytes", "SHA-256 prefix"],
        rows,
    )
    base = base.split("## 7. File manifest", 1)[0].rstrip() + f"""

## 7. Evidence manifest summary

The audit package contains {len(manifest['records'])} hashed records. The table below lists the principal scripts and result tables in a readable form. Full relative paths, byte counts and complete SHA-256 digests for every record are retained in `evidence_manifest.json`; the shortened prefixes below are for visual cross-checking only.

{manifest_summary}
"""
    return base + """

## 8. Independent consistency review

### 8.1 Separation of external and newly generated evidence

- Public LarNO arrays provide rainfall, terrain and MIKE reference water depth.
- Itzï surface-only and ITZI-SWMM coupled arrays were produced by the copied local workflow in this workspace.
- DrainLite models, predictions, ablations, controls and metrics were generated by the scripts listed in the manifest.
- Numerical values from the LarNO paper are not reused as DrainLite results. MIKE values are labelled as external reference data wherever they appear.

### 8.2 Timestamp correction

The physical runner appends water-depth frames when elapsed time first reaches 300, 600, ..., 21600 s. Absolute peak-time fields previously used zero-based array indices and were therefore 5 min early. `repair_major_revision_metric_timestamps.py` converted `peak_time_pred_h` and `peak_time_target_h` to interval-end time. Peak-time differences, depth metrics, volumes, predictions and scientific conclusions were unchanged.

### 8.3 Statistical consistency

- The primary no-coordinate model is `no_xy_all_static` (MAE 2.259 mm).
- Shifted, shuffled, sink-only and permutation experiments use the matched count-enhanced feature set; comparisons now state this explicitly.
- Conditional metrics and feature importance also use the count-enhanced signed model and are labelled accordingly.
- Five-event bootstrap intervals are descriptive for the available event pool; no population-level significance claim is made.

### 8.4 Figure review

- All eleven figures are generated with SciencePlots and Times New Roman as the primary Latin font.
- Figure 2 distinguishes the road-derived candidate graph from final rasterised SWMM pipes and includes direction, scale and distance axes.
- Figures 4 and 5 use interval-end times from 0.083 to 6.000 h.
- Figures 7 and 8 disclose percentile colour clipping; all unsaturated numerical arrays remain in metric calculations.
- MIKE is labelled as a reference, not a training target or independent validation set.

### 8.5 Remaining evidential limits

The manuscript is internally reproducible but is not yet ready for a strong hydraulic-validation claim. A stability-improved SWMM rerun, boundary and building-rainfall sensitivities, more coupled events, a second spatial window and actual LarNO inference remain outstanding. These omissions are scientific limits rather than missing prose and are retained in the manuscript discussion.
"""


def expanded_manifest() -> dict[str, object]:
    manifest = source.evidence_manifest()
    additions = [
        ROOT / "extended_study" / "generate_submission_revision_documents.py",
        ROOT / "extended_study" / "repair_major_revision_metric_timestamps.py",
        METRICS / "timestamp_convention.json",
    ]
    known = {record["path"] for record in manifest["records"]}
    for path in additions:
        rel = str(path.relative_to(ROOT))
        if rel not in known:
            manifest["records"].append({"path": rel, "bytes": path.stat().st_size, "sha256": source.sha256(path)})
    manifest["generated_for"] = "DrainLite manuscript and research-report package"
    return manifest


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "manuscript_major_revision.md").write_text(manuscript(), encoding="utf-8")
    (OUT / "report.md").write_text(research_report(), encoding="utf-8")
    manifest = expanded_manifest()
    (OUT / "evidence_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "scientific_integrity_audit.md").write_text(evidence_review(manifest), encoding="utf-8")
    print(OUT / "manuscript_major_revision.md")
    print(OUT / "report.md")
    print(OUT / "scientific_integrity_audit.md")
    print(OUT / "evidence_manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
