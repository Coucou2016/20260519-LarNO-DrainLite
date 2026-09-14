#!/usr/bin/env python3
"""Build the major-revision manuscript, research report, and evidence audit."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "extended_study" / "output" / "larno_drainlite_major_revision"
FIG = OUT / "publication_figures"
METRICS = OUT / "metrics"
FLOOD = ROOT / "LarNO-main" / "benchmark" / "urbanflood" / "flood" / "region1_20m_connected_swmm_v2_inf1mmh"
GEO = ROOT / "LarNO-main" / "benchmark" / "urbanflood" / "geodata" / "region1_20m_connected_swmm_v2_inf1mmh"
EVENTS = ["event1", "event67", "event68", "event69", "event70"]
ALL_EVENTS = ["event1", "event20", "event65", "event66", "event67", "event68", "event69", "event70"]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def mean(rows: list[dict[str, str]], model: str, key: str) -> float:
    values = [float(row[key]) for row in rows if row["model"] == model]
    return float(np.mean(values))


def bootstrap_delta(candidate: str, reference: str, repeats: int = 10000, seed: int = 731) -> dict[str, object]:
    rows = read_csv(METRICS / "event_metrics.csv")
    lookup = {(row["event"], row["model"]): float(row["mae_m"]) * 1000.0 for row in rows}
    deltas = np.array([lookup[(event, candidate)] - lookup[(event, reference)] for event in EVENTS])
    rng = np.random.default_rng(seed)
    samples = deltas[rng.integers(0, len(deltas), size=(repeats, len(deltas)))].mean(axis=1)
    return {
        "candidate": candidate,
        "reference": reference,
        "mean_delta_mm": float(deltas.mean()),
        "ci_low_mm": float(np.quantile(samples, 0.025)),
        "ci_high_mm": float(np.quantile(samples, 0.975)),
        "wins": int(np.sum(deltas < 0)),
        "deltas": {event: float(value) for event, value in zip(EVENTS, deltas)},
    }


def rainfall_rows() -> list[list[object]]:
    quality = {row["event"]: row for row in read_csv(OUT / "network_audit" / "event_numerical_stability.csv")}
    rows = []
    for event in ALL_EVENTS:
        rain = np.load(FLOOD / event / "rainfall.npy")
        total = rain.sum(axis=0)
        q = quality[event]
        status = "continuity-pass" if q["continuity_screen"] == "pass_2pct" else "continuity-warning" if q["continuity_screen"] == "warning_8pct" else "continuity-fail"
        rows.append(
            [
                event,
                f"{float(total.mean()):.2f}",
                f"{float(total.max()):.2f}",
                f"{float(rain.max()):.2f}",
                f"{float(q['continuity_error_pct']):+.3f}",
                f"{float(q['nonconverging_steps_pct']):.2f}",
                status,
                "formal LOEO" if event in EVENTS else "diagnostic only",
            ]
        )
    return rows


def ablation_rows() -> list[list[object]]:
    summary = {row["model"]: row for row in read_csv(METRICS / "ablation_summary.csv")}
    labels = [
        ("surface_only", "Surface-only"),
        ("base_no_xy", "Base, coordinates removed"),
        ("base_xy", "Base, coordinates included"),
        ("no_xy_masks", "Masks, coordinates removed"),
        ("no_xy_hydraulic", "Hydraulic fields, coordinates removed"),
        ("no_xy_all_static", "All static fields, coordinates removed"),
        ("no_xy_all_static_count", "All static + count fields, coordinates removed"),
        ("all_static_xy", "All static + count fields + coordinates"),
        ("no_xy_all_static_shift", "All static shifted by 20 m"),
        ("no_xy_all_static_shuffle", "All static jointly shuffled"),
        ("sink_only_no_xy_all_static_count", "Sink-only target, static + counts"),
    ]
    return [
        [
            label,
            f"{float(summary[key]['mae_mm']):.3f}",
            f"{float(summary[key]['rmse_mm']):.3f}",
            f"{float(summary[key]['csi_0p15']):.3f}",
            f"{float(summary[key]['mean_abs_peak_error_mm']):.1f}",
            f"{float(summary[key]['mean_abs_final_volume_error_m3']) / 1000:.2f}",
        ]
        for key, label in labels
    ]


def mike_summary_rows() -> list[list[object]]:
    rows = read_csv(METRICS / "mike_configuration_screening_comparison.csv")
    models = [
        ("ITZI surface-only", "ITZI surface-only"),
        ("ITZI-SWMM coupled", "ITZI-SWMM coupled"),
        ("base_no_xy", "Base, no coordinates"),
        ("no_xy_all_static_count", "Signed static, no coordinates"),
        ("all_static_xy", "Static + coordinates"),
        ("sink_only_no_xy_all_static_count", "Sink-only, no coordinates"),
    ]
    return [
        [
            label,
            f"{mean(rows, key, 'mae_mm'):.3f}",
            f"{mean(rows, key, 'csi_0p15'):.3f}",
            f"{mean(rows, key, 'global_peak_bias_mm'):+.1f}",
            f"{mean(rows, key, 'global_peak_abs_error_mm'):.1f}",
            f"{mean(rows, key, 'final_volume_bias_m3') / 1000:+.2f}",
            f"{mean(rows, key, 'final_volume_abs_error_m3') / 1000:.2f}",
        ]
        for key, label in models
    ]


def conditional_rows() -> list[list[object]]:
    source = read_csv(METRICS / "conditional_metrics.csv")
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in source:
        grouped[(row["model"], row["condition"])].append(float(row["mae_mm"]))
    conditions = [
        ("all_active", "All active cell-times"),
        ("target_severe_30mm", "Coupled target > 30 mm"),
        ("residual_active_1mm", "|Coupled - surface| > 1 mm"),
        ("negative_residual_1mm", "Coupled - surface < -1 mm"),
        ("positive_residual_1mm", "Coupled - surface > 1 mm"),
        ("network_0_20m", "Within 20 m of inlet or pipe"),
        ("network_over_140m", "More than 140 m from inlet or pipe"),
    ]
    models = ["surface_only", "base_no_xy", "no_xy_all_static_count", "sink_only_no_xy_all_static_count"]
    return [
        [label] + [f"{np.mean(grouped[(model, condition)]):.2f}" for model in models]
        for condition, label in conditions
    ]


def network_table() -> str:
    audit = json.loads((OUT / "network_audit" / "connected_swmm_major_revision_audit.json").read_text(encoding="utf-8"))
    topology = audit["topology"]
    raster = audit["rasterization"]
    rows = [
        ["Junctions / outfalls / conduits", f"{topology['junctions']} / {topology['outfalls']} / {topology['conduits']}"],
        ["Undirected components", topology["undirected_components"]],
        ["Junctions with undirected path to an outfall", f"{topology['junctions_with_undirected_path_to_outfall']} / {topology['junctions']}"],
        ["Junctions with directed path to an outfall", f"{topology['junctions_with_directed_path_to_outfall']} / {topology['junctions']}"],
        ["Parallel endpoint groups / excess links", f"{topology['parallel_endpoint_groups']} / {topology['parallel_endpoint_excess_links']}"],
        ["Zero-slope / adverse-slope conduits", f"{topology['zero_slope_conduits']} / {topology['adverse_slope_conduits']}"],
        ["Free synthetic outfalls", topology["outfalls"]],
        ["Coupled junctions / unique cells / multi-junction cells", f"{raster['coupled_junction_count']} / {raster['unique_coupled_junction_cells']} / {raster['multi_junction_cells']}"],
        ["Pipe cells / overlapping pipe cells", f"{raster['pipe_cells']} / {raster['overlapping_pipe_cells']}"],
        ["Maximum rasterised segments in one cell", raster["maximum_segments_per_cell"]],
    ]
    return markdown_table(["Property", "Value"], rows)


def experiment_table() -> str:
    rows = [
        ["Surface-only", "No learned correction", "Hydrodynamic baseline"],
        ["Base, no XY", "Surface depth, current/cumulative rainfall, elevation, terrain slope, time, 3x3 and 7x7 mean depth", "Non-network residual baseline"],
        ["Base + XY", "Base plus normalised row and column", "Tests fixed-location memorisation"],
        ["Masks, no XY", "Base plus inlet, outfall, pipe masks, distance and local densities", "Tests network geometry"],
        ["Hydraulic, no XY", "Base plus diameter, slope, capacity, cover and capacity density", "Tests conceptual hydraulic fields"],
        ["All static, no XY", "Base plus all eight original static drainage fields and local densities", "Primary deployable signed model"],
        ["All static + counts", "All static plus junction count and overlapping segment count", "Tests multiplicity preservation"],
        ["Shifted / shuffled", "Same samples; only drainage fields displaced or permuted", "Negative controls for spatial alignment"],
        ["Sink-only", "Predicts max(surface - coupled, 0)", "Tests whether positive residuals are needed"],
    ]
    return markdown_table(["Experiment", "Inputs or target", "Purpose"], rows)


def fig(number: int, filename: str, caption: str) -> str:
    return f"![Figure {number}. {caption}](publication_figures/{filename})\n\n**Figure {number}. {caption}**"


def manuscript() -> str:
    summary = {row["model"]: row for row in read_csv(METRICS / "ablation_summary.csv")}
    main = summary["no_xy_all_static"]
    base = summary["base_no_xy"]
    sink = summary["sink_only_no_xy_all_static_count"]
    signed_count = summary["no_xy_all_static_count"]
    bootstrap_surface = bootstrap_delta("no_xy_all_static", "surface_only")
    bootstrap_base = bootstrap_delta("no_xy_all_static", "base_no_xy", seed=732)
    metadata = json.loads((OUT / "major_revision_experiment_metadata.json").read_text(encoding="utf-8"))
    return f"""# Learning drainage residuals from paired ITZI-SWMM simulations for lightweight urban flood forecasting

## Abstract

Drainage information is often absent from public urban-flood benchmarks even when the numerical model used to produce the reference water depths contains a sewer representation. This omission makes it difficult to adapt a flood predictor to a new or modified drainage layout without regenerating a complete training corpus. We examine whether the drainage effect can instead be learned as a lightweight correction to a pre-existing surface-water prediction. A road-aligned conceptual network was constructed over a 20 m grid in Shenzhen and coupled to the Itzï two-dimensional surface solver through the Storm Water Management Model dynamic-wave engine. Paired surface-only and coupled simulations were generated for eight six-hour rainfall events. Five events that passed a study-defined water-balance screen were used in leave-one-event-out experiments; three further events were retained for hydraulic diagnosis. The target was the signed depth residual between coupled and surface-only simulations. A histogram gradient-boosting regressor used rainfall, terrain, local surface depth and rasterised drainage descriptors, but did not use row or column coordinates in the primary configuration.

The surface-only discrepancy to the coupled label was 4.932 mm mean absolute error. A non-network residual model reduced this value to 3.190 mm, while the coordinate-free static drainage model reached {float(main['mae_mm']):.3f} mm. The latter improved all five held-out events relative to the non-network model; the event-level mean reduction was {-bootstrap_base['mean_delta_mm']:.3f} mm (bootstrap 95% interval {-bootstrap_base['ci_high_mm']:.3f} to {-bootstrap_base['ci_low_mm']:.3f} mm). Jointly shuffling the drainage rasters returned the error to 3.195 mm, showing that the gain depended on spatial correspondence rather than on feature marginal distributions. A sink-only target achieved a slightly lower global mean absolute error ({float(sink['mae_mm']):.3f} mm) but had a larger root-mean-square error, poorer performance where the paired simulations produced positive residuals, and a larger final-volume error. The ranking was therefore metric dependent. Comparisons with the available MIKE reference did not constitute independent validation because MIKE informed a finite 1 versus 2 mm h−1 effective-loss screen; moreover, DrainLite did not improve MIKE pixel-time error over surface-only Itzï.

The results support drainage residual learning as a low-cost emulator of a specified conceptual coupled model. They do not validate the conceptual network as a municipal sewer representation. Dynamic-wave reports showed 59.49-87.40% routing steps not converging despite acceptable continuity errors for several events, and the network contained free synthetic outfalls, parallel links and zero-slope conduits. These limitations set the present work at proof-of-concept level and define the hydraulic revisions required before claims of operational drainage-aware forecasting can be made.

**Keywords:** urban flooding; drainage network; Itzï; SWMM; residual learning; lightweight surrogate; conceptual sewer network

## 1. Introduction

Urban pluvial flooding develops through an interaction between rainfall, fine-scale topography, buildings, surface conveyance and underground drainage. Two-dimensional shallow-water solvers resolve the surface component, while one-dimensional sewer models represent pipe storage, conveyance, outfall discharge and, where coupling is bidirectional, exchange through inlets and surcharge points. At neighbourhood and city scales this coupled calculation is computationally demanding. Data-driven emulators offer much faster forecasts, but their response is tied to the physical processes represented in their training labels.

LarNO was introduced as a latent autoregressive neural operator for large-scale, high-resolution urban-flood forecasting. Its published Shenzhen experiments used rainfall, cumulative rainfall and terrain as model inputs, with water-depth fields generated by MIKE Plus as supervision. The released 20 m benchmark contains 72 five-minute frames per event, but the accompanying data card states that confidential drainage-network data are not distributed. Thus the benchmark supports reproduction of a rainfall-terrain-to-depth mapping, but it does not provide the explicit pipe layout and hydraulic state needed to alter drainage assumptions. This distinction is important: a model may reproduce labels that already contain a drainage response without being able to condition a new prediction on a different network.

One response would be to expand the neural operator input and retrain it on many coupled simulations. That route is scientifically direct, but it is not practical on the available 4 GB graphics processor and cannot be justified with only a small number of newly generated coupled events. The present study asks a narrower question: can a lightweight model learn the difference between paired surface-only and drainage-coupled simulations while leaving the upstream flood predictor unchanged? The approach is compatible with LarNO output, but LarNO inference is not executed in this experiment. The base fields used here are complete Itzï surface simulations so that errors in the drainage correction can be isolated from errors in a neural-operator backbone.

Residual modelling is attractive because the target is localised and physically interpretable. If `h_s` denotes the surface-only depth and `h_c` the depth from the paired coupled simulation, the signed target is `r = h_c - h_s`. Negative values denote a lower coupled depth at that cell and time; positive values denote a higher coupled depth. Positive residuals are not, by themselves, evidence of sewer surcharge. They can also arise from redistribution of surface water, altered roughness around inlet cells, timing differences, or numerical behaviour. The role of a signed target must therefore be tested against a simpler sink-only alternative rather than assumed.

Three confounders are addressed explicitly. First, normalised row and column coordinates can allow a flexible regressor to memorise a fixed residual climatology. Coordinate-free and coordinate-inclusive models are therefore separated. Second, static drainage rasters must use the same coordinate-to-cell transformation as the coupling code. The earlier round-to-nearest conversion was replaced by the runner's integer conversion. Third, a gain from drainage features is only persuasive if it disappears when their spatial relationship to the target is disrupted. Fixed-sample shifted and jointly shuffled controls were therefore included.

The objectives were to: (1) construct paired surface-only and coupled labels from the same rainfall forcing and terrain; (2) audit the topology, rasterisation and numerical stability of the conceptual network; (3) quantify the incremental information supplied by static drainage features without coordinate leakage; (4) compare signed and sink-only targets; and (5) place MIKE comparisons within their correct role as configuration-informed descriptive checks rather than independent validation. Figure 1 summarises the resulting provenance.

{fig(1, 'fig01_provenance_workflow.png', 'Provenance of the paired simulations, residual model and MIKE comparisons. Solid arrows denote the learning path. Dashed arrows show that MIKE informed the finite effective-loss screen and was subsequently reused for descriptive comparison; it was not a supervised target or early-stopping loss.')}

## 2. Data and methods

### 2.1 Study data and event selection

The analysis used a 200 x 280 subwindow of the released Shenzhen `region1_20m` arrays. The cell size was 20 m, giving a 4.0 x 5.6 km rectangular window. Each event comprised 72 frames at five-minute intervals. Water depth was stored in metres and rainfall in millimetres per five-minute interval. Building or wall cells were represented by elevations at or above 49.9 m and were excluded from evaluation. The active mask contained 43,606 of 56,000 cells.

Eight events were available with paired Itzï outputs. Event selection was not random. Events 1, 67, 68, 69 and 70 were retained for formal leave-one-event-out analysis because their absolute SWMM routing continuity error was at most 8%; events 20, 65 and 66 exceeded this study-defined screen and were used only for diagnosis. The labels `continuity-pass`, `continuity-warning` and `continuity-fail` refer solely to absolute continuity errors of at most 2%, 2-8% and greater than 8%, respectively. They are not software acceptance criteria and do not imply that the dynamic-wave iterations converged.

**Table 1. Rainfall summaries and SWMM numerical diagnostics. Rainfall statistics are computed from the local 20 m arrays.**

{markdown_table(['Event', 'Mean 6 h rain (mm)', 'Maximum local 6 h rain (mm)', 'Maximum 5 min rain (mm)', 'Continuity error (%)', 'Routing steps not converging (%)', 'Water-balance screen', 'Use'], rainfall_rows())}

### 2.2 Surface-flow and drainage calculations

The two-dimensional model used Itzï 25.4. Surface flow was advanced with the installed `SurfaceFlowSimulation`, using `hmin = 0.001 m`, Courant number 0.7, partial-inertia weighting `theta = 0.9`, maximum surface-flow time step 1 s and Manning coefficient 0.015 on active cells. Building cells had an elevation of 50 m and Manning coefficient 100. The rectangular two-dimensional boundary was closed. Initial water depth was zero.

The drainage branch used EPA SWMM 5.2.4 through PySWMM 2.1.0 and swmm-toolkit 0.17.0. Routing was `DYNWAVE`, with a nominal two-second routing step, variable-step factor 0.50, minimum step 0.2 s, 20 trials, partial inertial damping and ponding enabled. Itzï's native `DrainageSimulation` exchanged flow between surface cells and SWMM nodes. The installed coefficients were 0.167 for orifice exchange, 0.54 for free-weir exchange and 0.056 for submerged-weir exchange. The exchange can act in either direction. These values and the coupling interface follow the Itzï implementation; the current Itzï documentation describes the SWMM coupling functionality as experimental.

Rainfall was imposed from the first five-minute interval. Rain falling on building cells was summed and redistributed uniformly over active cells to conserve the rainfall volume within the rectangular window. A constant 1 mm h−1 effective loss was applied to active cells. This term is described as an effective loss, not calibrated soil infiltration: the available data do not identify its physical partition among infiltration, interception, unresolved drainage and open-boundary export. Surface-only and coupled members of each pair used identical rainfall timing, effective loss and terrain.

### 2.3 Conceptual drainage network

No surveyed municipal pipe inventory was available. A road-aligned graph derived from OpenStreetMap was converted to a conceptual SWMM network. Components with fewer than three nodes were discarded. Each of 12 retained components was connected to a free synthetic outfall outside the two-dimensional window. Junction inverts were based on local terrain with a nominal cover, circular pipe diameters were 0.8, 1.0 or 2.0 m, and Manning roughness was 0.013. These values define an experiment rather than a reconstruction of the Shenzhen sewer system.

The coordinate audit changed the static-data pipeline. The coupling runner maps a node by `col = int(x/20) - col_offset` and `row = int((H*20 - y)/20) - row_offset`. Static rasters now use this exact rule. Window-exterior outfalls remain outside instead of being clipped onto boundary cells. The rebuilt inlet raster contains all 1,169 coupled junctions in 1,169 unique cells; the earlier nearest-cell implementation retained only 866 unique inlet cells. Pipe multiplicity remains substantial: 2,940 pipe cells contain more than one rasterised segment.

{fig(2, 'fig02_aligned_network_dem.png', 'Aligned conceptual network and digital elevation model. Grey cells are buildings or walls. Blue lines show rasterised conceptual pipes and red points show coupled junctions. The two zooms expose both sparse hillside links and dense urban links; the figure demonstrates coordinate consistency, not agreement with a surveyed sewer inventory.')}

**Table 2. Topological and rasterisation audit of the conceptual network.**

{network_table()}

### 2.4 Effective-loss screen and MIKE reference

The effective-loss configuration was chosen from a finite set rather than calibrated against observations. Event68 was simulated at 1-5 mm h−1, and complete eight-event packages were available at 1 and 2 mm h−1. MIKE peak and final-volume discrepancies were considered when retaining 1 mm h−1 as the working configuration. Consequently, MIKE was not used as a machine-learning target or early-stopping loss, but it did enter physical configuration screening. Later comparisons on the same event pool are descriptive and cannot be interpreted as independent calibration and validation.

The event68 sensitivity metadata were regenerated after an audit found that the earlier script took the first row of a multi-event CSV. That error paired event68 depth arrays with event1 SWMM metadata. The corrected continuity errors at 1-5 mm h−1 are +1.534, -1.961, -5.850, -8.913 and -10.663%, respectively. The correction changes the event68-only ranking under the earlier score; it does not retroactively establish an independently selected optimum. The 1 mm h−1 package is retained here because the eight-event screen had already favoured its less aggressive peak and volume response, but the choice is now reported as MIKE-informed and configuration-pool specific.

### 2.5 DrainLite residual model

DrainLite uses a histogram gradient-boosting regressor with squared-error loss, learning rate 0.05, at most 220 boosting iterations, 31 leaf nodes and L2 regularisation 0.01. At each time step, 700 active pixels were sampled with emphasis on changed cells, wet cells and network proximity. Crucially, each feature ablation and negative control used the same event-time-pixel indices. Sample weights increased for residual magnitudes above 0.2 mm and surface depths above 30 mm.

The primary target was `1000(h_c - h_s)` in millimetres. The final prediction was `max(h_s + r_hat/1000, 0)`. A sink-only comparator learned `1000 max(h_s - h_c, 0)` and predicted `max(h_s - d_hat/1000, 0)`. It therefore cannot represent positive coupled-minus-surface residuals. The primary deployable model excluded row and column coordinates. Count-enhanced and coordinate-inclusive models were diagnostics.

**Table 3. DrainLite experiment matrix. All learned configurations use identical sampled indices within each fold.**

{experiment_table()}

### 2.6 Validation and metrics

Five-event leave-one-event-out validation was used. Each fold trained on four events and predicted all 72 frames of the fifth event. No cell-time from the held event was used in fitting or early stopping. Reported errors are first computed for each held event and then macro-averaged across the five events. This prevents events with more wet pixels from dominating solely through sample count.

Metrics included mean absolute error, root-mean-square error, critical success index at 0.03 and 0.15 m, absolute global-peak error, peak-time error and final active-cell volume error. Conditional metrics were computed over full arrays for severe target depths, residual-active cells, positive and negative residual subsets, and distance bands from the nearest inlet or pipe. Event-level uncertainty was assessed with 10,000 paired bootstrap resamples of the five held events. With only five pairs, an exact two-sided sign-flip test has a minimum attainable p-value of 0.0625 when all events improve; intervals and event dots are therefore more informative than a dichotomous significance claim.

## 3. Results

### 3.1 Paired hydrodynamic response

Figure 3 separates water-balance screening from numerical convergence. Three events were within 2% continuity error and two more were within 8%, but every event had a high proportion of nonconverging routing steps. Even among the five retained events, the fraction was 72.76-87.04%. The paired labels are therefore useful for testing whether a learner can emulate this particular coupled calculation, but they are not a numerically validated hydraulic truth set.

{fig(3, 'fig03_swmm_stability.png', 'SWMM water-balance and numerical-stability diagnostics for all eight events. Panel (a) shows signed routing continuity error with study-defined ±2 and ±8% screens. Panel (b) shows the percentage of Dynamic Wave routing steps reported as not converging. Passing panel (a) does not compensate for the high values in panel (b).')}

The process curves reveal a systematic difference between the MIKE reference and the Itzï calculations (Figure 4). MIKE water volume generally peaks near 2-3 h and then recedes. Surface-only and coupled Itzï volumes rise more gradually and recede less strongly. Coupling lowers the stored surface volume in every formal event, but the domain maximum depth is dominated by individual depression or boundary cells and often peaks late. A right-pointing triangle marks a maximum at the final saved frame; such a point is censored and means that the true peak may occur at or after 6 h.

{fig(4, 'fig04_five_event_hydrographs.png', 'Five-event process curves. The left column is domain maximum depth and the right column is active-cell surface-water volume. Circles mark interior maxima. Right-pointing triangles mark endpoint maxima at 6 h and should be read as “peak at or beyond the simulation horizon”, not as a resolved peak time.')}

The effective-loss sensitivity changes recession and retained volume, but no constant value reconciles every MIKE characteristic (Figure 5). Increasing the loss generally reduces surface storage and makes continuity error more negative. This behaviour is why the term is treated as a lumped effective loss. It cannot be interpreted as a measured soil property.

{fig(5, 'fig05_effective_loss_sensitivity.png', 'Event68 effective-loss sensitivity. Panels (a) and (b) compare maximum depth and active-cell volume for 1-5 mm h−1 with MIKE. Panel (c) gives the corrected event68 SWMM continuity error. The curves show a configuration trade-off rather than an independently calibrated parameter.')}

### 3.2 Coordinate-free drainage residual learning

The ablation results are given in Table 4 and Figure 6. Surface-only water depth differed from the coupled target by 4.932 mm MAE. The coordinate-free base regressor reduced this to 3.190 mm by learning systematic dependence on surface depth, local depth context, rainfall, elevation and time. Adding all static drainage fields reduced MAE further to {float(main['mae_mm']):.3f} mm, a {float(main['improvement_vs_base_no_xy_pct']):.1f}% improvement over the coordinate-free base and {float(main['improvement_vs_surface_pct']):.1f}% over surface-only.

The primary model improved every held event. Its event-level difference from surface-only averaged {bootstrap_surface['mean_delta_mm']:.3f} mm, with a 95% bootstrap interval of {bootstrap_surface['ci_low_mm']:.3f} to {bootstrap_surface['ci_high_mm']:.3f} mm. Relative to the coordinate-free base, the mean difference was {bootstrap_base['mean_delta_mm']:.3f} mm ({bootstrap_base['ci_low_mm']:.3f} to {bootstrap_base['ci_high_mm']:.3f} mm). These intervals describe the observed five-event pool; they should not be extrapolated to a population of storms.

**Table 4. Five-event leave-one-event-out performance against the ITZI-SWMM coupled label. Peak and final-volume columns are macro-averaged absolute errors.**

{markdown_table(['Configuration', 'MAE (mm)', 'RMSE (mm)', 'CSI at 0.15 m', 'Abs. global-peak error (mm)', 'Abs. final-volume error (10³ m³)'], ablation_rows())}

The coordinate analysis changes the interpretation of the earlier result. Coordinates lowered the base error from 3.190 to 2.505 mm, showing that fixed location carries predictive information. With all static and count fields, adding coordinates lowered error from 2.276 to 2.124 mm. This is useful within the same grid but is not evidence that the model understands drainage hydraulics; it can encode a location-specific residual climatology. The coordinate-free model is therefore the primary result.

Static multiplicity counts did not help: adding junction and pipe-segment counts changed MAE from 2.259 to 2.276 mm. By contrast, masks alone reached 2.293 mm and the hydraulic-field group reached 2.396 mm. These outcomes indicate that network location and density are informative, while the assigned conceptual diameters, slopes and capacities contribute less reliably.

### 3.3 Spatial negative controls

The negative controls provide stronger evidence than raw feature importance. Shifting all drainage fields by one 20 m cell raised MAE to 2.425 mm. Jointly shuffling them within the active mask raised MAE to 3.195 mm, essentially the coordinate-free base value. Sampling locations, target values and training hyperparameters were held fixed. Thus the improvement is not explained by the marginal distributions of pipe density or distance; it depends on their spatial alignment with the residual.

{fig(6, 'fig09_ablation_event_dots.png', 'Feature ablation, coordinate leakage and spatial controls. Panel (a) shows macro-average bars and five held-event dots. Panel (b) compares the coordinate-free aligned model with a 20 m shift, joint active-cell shuffle and sink-only target. Every event degrades after shifting or shuffling the drainage fields.')}

### 3.4 Signed versus sink-only targets

The sink-only model produced the lowest coordinate-free global MAE, {float(sink['mae_mm']):.3f} mm, compared with {float(signed_count['mae_mm']):.3f} mm for the signed model with the same count-enhanced inputs. This result contradicts the initial assumption that a signed target would necessarily be superior. The explanation is class imbalance: negative residuals dominate the full active-cell-time domain, and a sink-only constraint avoids small positive corrections in otherwise unaffected cells.

The advantage reverses for other quantities. Signed prediction had RMSE {float(signed_count['rmse_mm']):.3f} mm versus {float(sink['rmse_mm']):.3f} mm for sink-only, CSI at 0.15 m of {float(signed_count['csi_0p15']):.3f} versus {float(sink['csi_0p15']):.3f}, and mean absolute final-volume error of {float(signed_count['mean_abs_final_volume_error_m3'])/1000:.2f} versus {float(sink['mean_abs_final_volume_error_m3'])/1000:.2f} x 10³ m³. On cells where the paired target exceeded the surface-only depth by more than 1 mm, signed-model MAE was 13.64 mm and sink-only MAE was 21.45 mm. The signed model is consequently retained when spatial redistribution is part of the modelling objective, whereas sink-only is a competitive choice when minimising global MAE is the sole criterion.

### 3.5 Spatial residual fields and conditional performance

At the event68 target-conditioned time, the DrainLite water-depth field remains close to the coupled target at domain scale (Figure 7). Residual maps reveal information that is not visible in the depth panels: strong negative residuals occur in several connected low-lying regions, whereas smaller positive patches appear elsewhere. The predicted residual recovers the main negative structures but smooths local extremes. Residual error is concentrated near sharp wet-dry boundaries and isolated deep cells.

{fig(7, 'fig07_event68_target_conditioned_spatial.png', 'Event68 maps at 5.83 h, selected from the coupled-target maximum-depth curve. Panels (a-d) share a water-depth scale. Panels (e-g) share a signed residual scale. MIKE is shown at the same clock time but is not the learning target. Positive residual means coupled depth exceeds surface-only depth; it is not proof of sewer surcharge.')}

The five-event grid confirms that these patterns repeat, but it also shows that the model follows a stable residual template across storms (Figure 8). This consistency explains both the value of static drainage fields and the additional gain from coordinates. It also cautions against claiming spatial generalisation: all folds use the same terrain and conceptual network, while only the rainfall event is held out.

{fig(8, 'fig08_five_event_signed_residual_maps.png', 'True residual, predicted residual and residual error for all five held events at each coupled-target maximum-depth time. Colour limits are set separately by event to reveal structure; panels should be compared spatially within rows, not by absolute colour between rows.')}

Conditional analysis localises the gain (Table 5 and Figure 9). Near the network, surface-only MAE was 13.82 mm and the signed static model reduced it to 3.88 mm. On residual-active cells, the corresponding reduction was 37.79 to 11.65 mm. On target depths above 30 mm it reduced error from 22.24 to 9.40 mm. The signed target was especially valuable on positive-residual cells, where sink-only is structurally unable to add depth.

The picture is different more than 140 m from an inlet or pipe. Surface-only MAE was only 0.56 mm there, while the signed model gave 1.18 mm. The residual learner therefore introduces a small far-field correction error. This spatial leakage is hidden by the domain-average improvement and should be addressed with distance-aware shrinkage or a learned zero-residual gate.

**Table 5. Conditional five-event MAE in millimetres. Values are macro-averaged over held events.**

{markdown_table(['Condition', 'Surface-only', 'Base, no XY', 'Signed static, no XY', 'Sink-only, no XY'], conditional_rows())}

{fig(9, 'fig10_conditional_performance.png', 'Conditional error. The largest gains occur in severe, residual-active and network-proximal subsets. The positive-residual subset exposes the limitation of sink-only prediction. The far-network subset shows that learned corrections can add error where little drainage effect is present.')}

Permutation importance ranks current surface depth, elevation and local mean depth above individual pipe attributes (Figure 10). Distance to outfall and pipe density are the highest-ranked drainage descriptors. Importance is computed separately on each held event, and dots show fold variability. Because the features are correlated, the bars measure predictive disruption under permutation, not causal hydraulic contributions.

{fig(10, 'fig11_cross_fold_importance.png', 'Cross-fold permutation importance for the coordinate-free count-enhanced signed model. Bars are five-fold means, black error bars show between-fold standard deviation and dots are held-event estimates. Correlation among terrain, depth and network fields prevents causal interpretation.')}

### 3.6 MIKE-informed descriptive comparison

No single model is best under all MIKE metrics (Figure 11 and Table 6). Surface-only Itzï has the lowest full-sequence MAE, 23.683 mm, and the smallest mean absolute final-volume error, 21.09 x 10³ m³. The coupled simulation and residual models have larger pixel-time errors and much larger final-volume discrepancies. Some residual models reduce absolute global-peak error, but the event-to-event spread is large. A signed mean peak bias alone would be misleading because positive and negative events cancel; absolute errors and event dots are therefore reported together.

**Table 6. Five-event descriptive comparison with MIKE. MIKE informed the finite effective-loss screen, so these are not independent validation results.**

{markdown_table(['Model', 'MAE (mm)', 'CSI at 0.15 m', 'Signed peak bias (mm)', 'Abs. peak error (mm)', 'Signed final-volume error (10³ m³)', 'Abs. final-volume error (10³ m³)'], mike_summary_rows())}

{fig(11, 'fig06_mike_metric_dependent_comparison.png', 'Metric-dependent comparison with MIKE across the five formal events. Bars are event means and dots are events. Lower values are better in all three panels. The figure demonstrates that drainage-label fidelity and MIKE fidelity are distinct objectives.')}

## 4. Discussion

### 4.1 What the residual experiment establishes

The central result is narrower, and more defensible, than an end-to-end drainage-aware forecasting claim. Given paired outputs from a fixed conceptual ITZI-SWMM configuration, a CPU-scale regressor can emulate much of the coupled-minus-surface residual. The coordinate-free static model improves all five held rainfall events, and the gain disappears when drainage rasters are shuffled. This supports the presence of a learnable spatial drainage signal in the paired labels.

The experiment does not establish that the inferred signal corresponds to the real Shenzhen sewer system. The network is derived from roads, not from survey or asset records. Pipe diameters, cover and outfalls are assigned. The free outfalls have no tailwater. The two-dimensional boundary is closed, whereas MIKE may exchange water with rivers or open boundaries. These choices can alter both peak timing and recession. The resulting model should be described as an emulator of the specified conceptual coupled calculation.

### 4.2 Hydraulic uncertainty is the limiting issue

The largest obstacle is not machine-learning error. Dynamic Wave nonconvergence is high in every event. The graph audit also found that all junctions connect to an outfall in the undirected sense, but only 89 of 1,169 have a directed path under the current conduit orientation. SWMM can reverse flow under dynamic-wave routing, so directed reachability is not a strict feasibility condition, yet the combination of direction inconsistency, 156 zero-slope conduits, parallel links and free outfalls is a credible source of instability.

Accordingly, the water-balance screen is retained only to define a reproducible subset; it is not presented as validation. Before the coupled arrays are used as a hydraulic benchmark, the network should be rebuilt with unambiguous trunk direction, duplicate-lane handling, monotonic invert profiles, nonzero design slopes and physically specified outfall levels. At least three events should then be rerun until both continuity and nonconvergence diagnostics are acceptable, with node heads, link flows, exchange flows and outfall discharge saved at routing time steps.

### 4.3 Why signed and sink-only models answer different questions

Global MAE favours sink-only because most corrected depth is negative and large parts of the domain have little residual. A signed model is harder to fit and can add small far-field errors. However, paired coupled simulations contain positive residuals. These may reflect redistribution or numerical effects; without node-head and exchange-flow time series they cannot be attributed to surcharge. If the research aim is a conservative peak-reduction proxy, sink-only is reasonable. If the aim is to reproduce the paired coupled water-depth field, signed residuals are required. The paper therefore reports both instead of selecting one by a single average metric.

### 4.4 Relation to LarNO

DrainLite was designed as a post-processor that can accept a LarNO water-depth field, but the present experiments use Itzï surface-only depth. No LarNO checkpoint was run in this validation, and no claim is made about zero-shot resolution generalisation, LarNO accuracy or end-to-end inference speed. The next model-level test is straightforward: run the released LarNO checkpoint on the same 20 m events, replace `h_s` with LarNO output, and measure the combined error to both MIKE and the revised coupled labels. Until that experiment is complete, LarNO belongs in the application discussion rather than in the title claim.

### 4.5 Remaining sensitivities and data limitations

The event pool is small and comes from one fixed spatial window. Leave-one-event-out validation tests rainfall-event transfer, not transfer to a new network or catchment. The bootstrap interval is based on five events and should not be treated as a population confidence interval. Additional coupled events should be generated before model complexity is increased.

Building rainfall is redistributed uniformly to active cells. This conserves volume but changes where runoff enters the surface system. A no-redistribution and a local-edge redistribution sensitivity have not yet been run. Likewise, the effective-loss analysis uses a constant rate rather than a soil or land-cover model. These omissions should be resolved before the recession mismatch with MIKE is interpreted physically.

Finally, the current features are static. Event-level SWMM summaries do not describe the changing network state. Node hydraulic head, link flow, capacity ratio, node flooding and signed inlet exchange should be rasterised or encoded as graph-to-grid dynamic features. Their value can then be tested against a static model and a shuffled dynamic-state control.

## 5. Conclusions

Paired ITZI-SWMM simulations were used to examine a lightweight route for adding drainage effects to an existing urban-flood prediction. After correcting the static coordinate mapping and removing row and column coordinates from the primary model, all static drainage fields reduced five-event leave-one-event-out MAE from 3.190 to {float(main['mae_mm']):.3f} mm relative to a non-network residual baseline. Jointly shuffling the drainage fields removed this gain, providing direct evidence that their spatial alignment matters.

The result is not a simple endorsement of signed residual learning. A sink-only model had slightly lower global MAE, while the signed model better represented positive residual cells, deep inundation, root-mean-square error and final volume. Model ranking also changed when comparison shifted from the coupled label to MIKE. The appropriate target therefore depends on whether the objective is conservative depth reduction, complete paired-label emulation or agreement with an external hydraulic reference.

The hydraulic audit imposes a firm boundary on the claim. High Dynamic Wave nonconvergence, zero-slope and parallel conduits, synthetic free outfalls, closed surface boundaries and the absence of a surveyed sewer inventory prevent the coupled simulations from serving as validated municipal drainage truth. At present, DrainLite is a reproducible proof of concept for emulating a specified road-aligned conceptual network. A stability-improved network, additional events, building-rainfall sensitivity, dynamic network-state output and actual LarNO inference are required before the method can support an operational drainage-aware urban-flood forecast.

## Data and code availability

The public LarNO code and released 20 m benchmark are available from the project repository and dataset archive. The major-revision experiment scripts, aligned conceptual-network rasters, event lists, trained fold models, full predictions, metrics, figures and evidence manifest are stored in the local study package generated with this manuscript. The conceptual network is derived from OpenStreetMap and is not an official Shenzhen drainage inventory. Repository and archival identifiers for the new derived dataset remain **to be assigned before submission**.

## References

Bates, P.D., Horritt, M.S., Fewtrell, T.J., 2010. A simple inertial formulation of the shallow water equations for efficient two-dimensional flood inundation modelling. Journal of Hydrology 387, 33-45.

Cao, X., Yao, Y., Wang, Z., Zhao, Z., Borthwick, A.G.L., Qin, H., 2026. Large-scale urban flood modeling and zero-shot high-resolution generalization with LarNO. Journal of Hydrology, 135686. https://doi.org/10.1016/j.jhydrol.2026.135686.

Cao, X., Qin, H., 2025. Benchmark dataset of “Large-scale urban flood modeling and zero-shot high-resolution generalization with LarNO”. https://doi.org/10.6084/m9.figshare.30529031.v4.

EPA, 2022. Storm Water Management Model User's Manual Version 5.2. United States Environmental Protection Agency.

Itzï developers, 2026. Itzï documentation: configuration and experimental SWMM drainage coupling. https://itzi.readthedocs.io/.

Kovachki, N., Li, Z., Liu, B., Azizzadenesheli, K., Bhattacharya, K., Stuart, A., Anandkumar, A., 2023. Neural operator: learning maps between function spaces with applications to PDEs. Journal of Machine Learning Research 24, 1-97.

Li, Z., Kovachki, N., Azizzadenesheli, K., Liu, B., Bhattacharya, K., Stuart, A., Anandkumar, A., 2021. Fourier neural operator for parametric partial differential equations. International Conference on Learning Representations.

Rossman, L.A., Simon, M.A., 2022. Storm Water Management Model User's Manual Version 5.2. U.S. Environmental Protection Agency.

University of Manchester, 2026. Academic Phrasebank: reporting results, discussing findings and writing conclusions. https://www.phrasebank.manchester.ac.uk/.
"""


def report() -> str:
    summary = {row["model"]: row for row in read_csv(METRICS / "ablation_summary.csv")}
    main = summary["no_xy_all_static"]
    return f"""# 基于成对 ITZI-SWMM 模拟的轻量排水残差学习研究报告

## 报告说明

本报告与英文论文草稿使用同一批代码、数组、指标和图件。报告的任务不是把结果简单罗列出来，而是说明每一步为什么做、图表应该怎样阅读、哪些结论由数据直接支持，以及哪些内容仍然不能下结论。当前工作是概念网络条件下的探索性研究，不是深圳真实市政管网的校准模型。

## 摘要

公开 LarNO 数据集提供降雨、地形和 MIKE 水深，但没有公开真实排水管网。为了研究“如果显式给出一套管网，轻量模型能否学习其对地表积水的影响”，本研究在 20 m 网格上构建道路对齐概念管网，使用 Itzï 地表动力学求解器和其原生 SWMM 双向交换接口，分别计算无管网和带管网情景。模型把两者之差作为残差学习目标，不重训 LarNO 主干。

审稿复核促使本轮完成了三项关键修正。第一，敏感性脚本不再把 event68 水深与 event1 的 SWMM 元数据拼接。第二，静态管网栅格改用与耦合程序完全一致的 `int()` 坐标映射，1169 个耦合节点现在对应 1169 个唯一像元。第三，基础模型中的行列坐标被单独拆出，避免把固定位置记忆误称为管网水力贡献。

修正后的五事件留一验证表明：surface-only 相对耦合标签的平均绝对误差为 4.932 mm；不含坐标、也不含管网的基础残差模型为 3.190 mm；加入全部静态管网特征后为 {float(main['mae_mm']):.3f} mm。将管网字段整体打乱后误差回到 3.195 mm，说明提升依赖正确空间位置。与此同时，SWMM 报告显示 59.49%-87.40% 的 routing steps not converging，网络还包含零坡管、平行管和自由合成排放口。因此，目前可以证明“模型能逼近这套概念耦合计算”，但不能证明它代表真实城市管网。

## 目录

1. 研究背景与目的
2. 数据、物理模型与概念管网
3. 标签质量和参数筛选
4. DrainLite 方法与验证设计
5. 模型结果
6. MIKE 对比的正确解释
7. 主要结论
8. 不足与下一阶段验收条件
9. 文件与可复现性

## 1. 研究背景与目的

城市暴雨积水不仅由降雨和地形决定。道路低点控制汇流，建筑改变流路，雨水口把地表水送入地下，管道储水和输水，满管后还可能发生反向交换。若训练标签由含管网的 MIKE 模型产生，而机器学习输入中不含管网，模型只能在固定区域中隐式记住平均排水效应，无法方便地表示“换一套管网会怎样”。

本研究不尝试在本机上重训完整 LarNO。目标被收窄为一个可验证问题：给定一幅已有的地表水深场，能否用轻量模型预测“加入指定概念管网后，水深应作何修正”？这样既保留 LarNO 未来作为上游预测器的可能性，又能在 4 GB 显存条件下完成完整交叉验证和消融实验。

{fig(1, 'fig01_provenance_workflow.png', '研究数据链和验证边界。')}

**图 1 如何阅读。** 上方实线流程是本研究真正参与训练的数据链：降雨、地形和概念管网进入成对 Itzï 计算，得到 surface-only 和 ITZI-SWMM 两套水深，再形成有符号残差，最后由 DrainLite 学习。下方紫色虚线表示 MIKE 的不同角色。MIKE 没有作为机器学习标签，也没有进入 early stopping，但它参与了 1 和 2 mm h−1 有效损失率的有限配置筛选；因此后面的 MIKE 对比不能称为完全独立验证。把这条边界画出来，是为了防止“没有直接训练 MIKE”被误写成“MIKE 完全没有参与模型选择”。

## 2. 数据、物理模型与概念管网

### 2.1 计算区域和事件

计算窗口为 200 x 280 个 20 m 网格，即 4.0 x 5.6 km。每个事件有 72 个五分钟时刻，总时长 6 h。43,606 个像元参与评价，其余为建筑或墙体。8 个事件都有物理模拟结果，其中 5 个进入正式留一事件验证，3 个仅用于诊断。

**表 1. 降雨特征、SWMM 数值诊断与事件用途。**

{markdown_table(['事件', '平均 6 h 降雨 (mm)', '局地最大 6 h 降雨 (mm)', '最大 5 min 降雨 (mm)', '连续性误差 (%)', '不收敛步 (%)', '水量筛选', '用途'], rainfall_rows())}

表中的“continuity-pass”只说明 SWMM 水量连续性误差绝对值不超过 2%，“continuity-warning”表示 2%-8%，“continuity-fail”表示超过 8%。这些阈值是本研究的数据筛选规则，不是 EPA 官方验收标准。最后一列明确哪些事件真正进入机器学习结果，防止把 8 个诊断事件和 5 个正式事件混为一谈。

### 2.2 Itzï 与 SWMM 的耦合方式

地表计算使用 Itzï 25.4 的 SurfaceFlowSimulation，而不是洼地填充或静态平衡模型。地表时间步受 CFL 条件控制，最大 1 s；活动像元 Manning 系数为 0.015，建筑像元设置为高程 50 m 和高阻力。初始水深为 0，二维矩形边界封闭。

管网计算使用 EPA SWMM 5.2.4 的 Dynamic Wave，PySWMM 负责调用。Itzï 的 DrainageSimulation 在每个耦合节点计算地表与管网之间的双向交换。这里的“原生耦合”指交换公式和时间推进来自 Itzï 已安装代码，不是另写一个独立 sink 再把水扣掉。当前 Itzï 文档把该功能标为 experimental，这一点与本轮发现的大量不收敛步相互印证。

### 2.3 管网如何构建和对齐

没有真实管网资产数据，因此管网沿 OpenStreetMap 道路图概化。它包含 1169 个 junction、12 个自由合成 outfall 和 2887 条 conduit。管径、埋深和坡度来自概念规则，不是实测值。所有 12 个 outfall 位于二维窗口之外，不参与地表耦合。

{fig(2, 'fig02_aligned_network_dem.png', '概念管网与地形的对齐结果。')}

**图 2 如何阅读。** 大图先看整体：灰色是建筑或墙体，彩色底图是可流动地形，蓝线是概念管道，红点是与地表交换的节点。右侧两个放大图用于检查管道是否落在城市通道和建筑间隙，而不是落到山体或建筑内部。该图证明“代码里的节点和静态特征使用同一位置”，不证明道路管网等于真实地下管网。图中没有 outfall 点，是因为它们被有意布置在当前二维窗口之外。

**表 2. 概念管网的拓扑、坡度、管径与栅格化审计。**

{network_table()}

拓扑表暴露了两个需要后续修正的问题。其一，所有 junction 在无向意义下都能连到 outfall，但按当前 from-to 方向只有 89 个节点能到达 outfall。Dynamic Wave 允许反向流，因此这不等于其余节点完全不排水，但说明管段方向与坡度组织不理想。其二，156 条管段为零坡，271 组端点存在平行管，2940 个管线像元叠加多条管段。这些结构很可能加剧数值不稳定。

## 3. 标签质量和参数筛选

{fig(3, 'fig03_swmm_stability.png', '连续性误差与动态波不收敛步比例。')}

**图 3 如何阅读。** 上图看水量平衡：柱越靠近 0 越好，虚线和点线是本研究的 ±2% 与 ±8% 分级。下图看数值迭代：紫色柱表示有多少 routing step 没有达到 SWMM 的收敛条件。最关键的读图结论是，上图“通过”不等于下图“稳定”。例如 event68、69、70 的连续性误差小于 2%，但不收敛步仍约 86%。所以本报告把这些事件称为 continuity-screened exploratory labels，而不是 validated hydraulic truth。

{fig(4, 'fig04_five_event_hydrographs.png', '五事件最大水深和地表水量过程。')}

**图 4 如何阅读。** 每一行对应一个事件。左列是全域单个最深像元的水深，容易受边界低点或局部洼地控制；右列是所有活动像元水深乘以面积后的地表水体积，更能反映整体蓄水和退水。黑线是 MIKE，蓝线是 Itzï surface-only，红线是 ITZI-SWMM。圆点表示 6 h 以内找到峰值；向右三角表示最大值出现在最后一帧，只能说峰值在 6 h 或更晚。图中 MIKE 大多在 2-3 h 后退水，而 Itzï 退水偏慢。这个差异不能仅归因于管网，还与二维封闭边界、统一有效损失、建筑降雨重分配和 MIKE 未公开边界条件有关。

{fig(5, 'fig05_effective_loss_sensitivity.png', 'event68 有效损失率敏感性。')}

**图 5 如何阅读。** 左图看最大水深，中图看水量，右图看连续性误差。损失率增大时，整体蓄水通常降低，但 SWMM 连续性误差逐渐向负值移动。修正后的 1-5 mm h−1 连续性误差依次为 +1.534、-1.961、-5.850、-8.913 和 -10.663%。这里不能把 1 mm h−1 解释为真实入渗率；它是用同一批 MIKE 结果辅助筛出的 lumped effective loss。原敏感性表曾把 event1 的 -4.213% 错放到 event68，现已修复。

## 4. DrainLite 方法与验证设计

DrainLite 预测 `r = h_coupled - h_surface`。最终水深为 `max(h_surface + r_pred, 0)`。主模型不使用 row/column，避免记忆固定像元。每折用 4 个事件训练、1 个事件测试，循环 5 次。所有模型和负对照使用完全相同的样本位置，因此差异来自特征，而不是抽样位置。

**表 3. DrainLite 消融、空间负对照和目标形式的实验矩阵。**

{experiment_table()}

空间负对照有两种。shift 把全部管网字段向东、向南各移一个 20 m 像元；shuffle 用同一个随机排列联合打乱所有管网字段，保留每个字段的数值分布和字段间对应关系。两者都不改变训练标签和样本位置。如果原始管网位置真的有用，它们应比 aligned model 更差。

## 5. 模型结果

{fig(6, 'fig09_ablation_event_dots.png', '坐标、管网特征、空间负对照和目标形式。')}

**图 6 如何阅读。** 左图柱高是 5 个留出事件平均绝对误差（mean absolute error，表示预测水深与耦合标签逐像元差值绝对值的平均）的平均值，黑点是单个事件。Surface-only 为 4.932 mm；无坐标 base 为 3.190 mm；无坐标 all static 为 {float(main['mae_mm']):.3f} mm。加入 XY 后还会下降，说明固定位置本身有信息，但这种信息可能是空间记忆。右图直接连接同一事件：绿色 aligned signed 始终优于橙色 20 m shift 和紫色 shuffle，说明管网的正确空间位置有贡献。灰色 sink-only 的总体 MAE 更低，这不是绘图错误，而是目标不对称造成的结果。

**表 4. 五事件留一验证的总体性能。**

{markdown_table(['配置', 'MAE (mm)', 'RMSE (mm)', 'CSI@0.15 m', '绝对峰值误差 (mm)', '绝对最终体积误差 (10³ m³)'], ablation_rows())}

表格说明“最好”取决于指标。含坐标的模型 MAE 最低，但不适合作为空间泛化证据。sink-only 的全域 MAE 较低，但 RMSE、深水 CSI 和体积误差较差。无坐标 signed all-static 是本报告的主模型，因为它既避免坐标泄漏，又保留正负残差能力。

{fig(7, 'fig07_event68_target_conditioned_spatial.png', 'event68 目标时刻空间分布。')}

**图 7 如何阅读。** 上排四图使用同一水深色标，比较 surface-only、耦合标签、DrainLite 和同一时刻 MIKE。不要只凭整体颜色相似就判断模型正确，因为大部分像元水浅。下排把差异放大：蓝色为耦合后更浅，红色为耦合后更深。真实残差和预测残差的主要蓝色区域位置一致，误差图大部分接近白色，但局部深点和干湿边界仍有误差。这里的正残差不能直接称为 surcharge，因为没有逐节点水头和交换流证据。

{fig(8, 'fig08_five_event_signed_residual_maps.png', '五事件真实残差、预测残差和误差。')}

**图 8 如何阅读。** 每行一个事件，三列依次是真实残差、预测残差、预测误差。每行色标按该事件单独设置，因此适合比较同一行的空间位置，不适合用不同事件颜色深浅直接比较数值。五个事件都出现相似的残差骨架，说明静态地形和管网位置决定了相当一部分响应；这也解释了为什么 row/column 会带来额外提升，并提醒我们当前只完成了事件泛化，没有完成跨区域泛化。

{fig(9, 'fig10_conditional_performance.png', '不同物理子集上的条件误差。')}

**图 9 如何阅读。** 横轴不是不同事件，而是不同像元条件。靠近管网、真实残差大于 1 mm、目标水深大于 30 mm 时，signed static 的绿色柱明显低于 surface-only。正残差子集中，sink-only 橙色柱最高，因为它的定义禁止增加水深。距管网超过 140 m 时，surface-only 反而最好，说明残差模型在远场引入了小误差。后续可增加“远离管网则收缩为 0”的门控。

**表 5. 不同空间和水深条件下的平均绝对误差（mm）。**

{markdown_table(['条件', 'Surface-only', 'Base 无坐标', 'Signed static 无坐标', 'Sink-only 无坐标'], conditional_rows())}

{fig(10, 'fig11_cross_fold_importance.png', '跨折 permutation importance。')}

**图 10 如何阅读。** 横轴表示把某个特征随机打乱后，留出事件 MAE 增加多少。条越长，模型越依赖该特征。五个黑点对应五折，误差线表示折间波动。surface depth、elevation 和局部平均水深最重要；管网相关特征中，distance to outfall 和 pipe density 较高。由于特征彼此相关，不能把条长解释为因果贡献，更不能说某个管径参数“导致”了多少削峰。

## 6. MIKE 对比的正确解释

{fig(11, 'fig06_mike_metric_dependent_comparison.png', '五事件 MIKE 描述性对比。')}

**图 11 如何阅读。** 三个子图都越低越好，柱是五事件平均，黑点是事件。左图显示 surface-only 的全时序 MAE 最低；中图显示某些残差模型的绝对峰值误差略小；右图显示 surface-only 的最终水量误差远小于耦合标签和 DrainLite。由此不能说 DrainLite “总体上更接近 MIKE”。DrainLite 的主任务是逼近 ITZI-SWMM 耦合标签，而 MIKE 是参与过配置筛选的参考。两个目标不相同。

**表 6. 五事件与 MIKE 的描述性比较。**

{markdown_table(['模型', 'MAE (mm)', 'CSI@0.15 m', '有符号峰值偏差 (mm)', '绝对峰值误差 (mm)', '有符号最终体积误差 (10³ m³)', '绝对最终体积误差 (10³ m³)'], mike_summary_rows())}

## 7. 主要结论

1. 修正坐标映射后，1169 个耦合 junction 与 1169 个入口像元一一对应，旧稿中 866 个像元的说法已经失效。
2. 不含坐标的静态管网模型把耦合标签 MAE 从 base 的 3.190 mm 降至 {float(main['mae_mm']):.3f} mm，五个留出事件全部改善。
3. 管网字段 shuffle 后 MAE 回到 3.195 mm，说明正确空间位置是增益来源之一。
4. sink-only 在全域 MAE 上略好，但 signed 模型在正残差、RMSE、深水 CSI 和体积误差上更合理。二者是不同用途的模型，不应只选一个指标宣布绝对优胜。
5. DrainLite 没有改善 MIKE 全时序 MAE；MIKE 还参与过有效损失配置筛选，因此不是独立验证集。
6. 当前最严重的问题是 SWMM 动态波不收敛，而不是机器学习误差。现有结果只能作为概念耦合模型的探索性标签。

## 8. 不足与下一阶段验收条件

### 8.1 必须完成的物理模型修正

- 清理相同端点、相同属性的重复 conduit，并说明保留的平行管是否代表多管并行。
- 为每个 component 建立明确的主干方向和非零设计坡度，重新检查 cover depth 与地面冲突。
- 为 outfall 指定物理水位或尾水情景，不再只用 FREE outfall。
- 至少重跑 3 个事件，同时满足预先规定的 continuity 和 nonconverging-step 条件。
- 保存逐 routing step 的 node head、link flow、capacity ratio、signed exchange flow 和 outfall discharge。

### 8.2 必须完成的敏感性

- 比较建筑降雨不重分配、均匀重分配和局部边缘重分配。
- 比较二维封闭边界与合理开放/河流水位边界。
- 将 0、1、2 mm h−1 的有效损失对残差标签和 DrainLite 结果的影响放入同一正式表。

### 8.3 必须完成的模型验证

- 增加正式耦合事件，避免五事件统计过弱。
- 运行真实 LarNO checkpoint，把 LarNO 水深作为 DrainLite 上游输入，再评价端到端误差。
- 完成跨空间窗口或跨管网测试，而不只是同一网格上的跨降雨事件测试。
- 增加远场零残差门控，并用独立事件验证其是否消除距管网 140 m 以外的误差增加。

## 9. 文件与可复现性

本报告对应的训练脚本为 `extended_study/run_drainlite_major_revision_experiments.py`，管网审计脚本为 `extended_study/audit_connected_swmm_network_major_revision.py`，图件脚本为 `extended_study/generate_major_revision_publication_figures.py`。全部核心数组、模型、CSV、图和文档位于 `extended_study/output/larno_drainlite_major_revision/`。详细哈希和证据边界见同目录 `scientific_integrity_audit.md`。
"""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def evidence_manifest() -> dict[str, object]:
    key_files = [
        ROOT / "extended_study" / "run_drainlite_major_revision_experiments.py",
        ROOT / "extended_study" / "audit_connected_swmm_network_major_revision.py",
        ROOT / "extended_study" / "generate_major_revision_publication_figures.py",
        ROOT / "extended_study" / "generate_major_revision_documents.py",
        ROOT / "extended_study" / "render_major_revision_documents.py",
        ROOT / "extended_study" / "validate_major_revision_package.py",
        ROOT / "extended_study" / "summarize_infiltration_sensitivity.py",
        ROOT / "extended_study" / "build_connected_swmm_drainage_dataset.py",
        METRICS / "event_metrics.csv",
        METRICS / "ablation_summary.csv",
        METRICS / "conditional_metrics.csv",
        METRICS / "mike_configuration_screening_comparison.csv",
        OUT / "network_audit" / "connected_swmm_major_revision_audit.json",
        ROOT / "extended_study" / "output" / "infiltration_sensitivity" / "event68_infiltration_sensitivity_metrics.csv",
        ROOT / "extended_study" / "output" / "infiltration_sensitivity" / "event68_infiltration_sensitivity_summary.json",
        ROOT / "extended_study" / "output" / "connected_swmm_dataset_v2_inf1mmh" / "swmm_connected_sub.inp",
    ]
    for event in EVENTS:
        for name in ["rainfall.npy", "h_itzi_surface.npy", "h_itzi_swmm_connected.npy", "h_mike_ref.npy"]:
            key_files.append(FLOOD / event / name)
        key_files.append(OUT / "predictions" / event / "h_no_xy_all_static.npy")
        key_files.append(
            ROOT
            / "extended_study"
            / "output"
            / "connected_itzi_swmm_inf_1mmh"
            / f"{event}_swmm_connected.rpt"
        )
    records = []
    for path in key_files:
        records.append({"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": sha256(path)})
    return {
        "generated_for": "DrainLite major revision",
        "dataset": "region1_20m_connected_swmm_v2_inf1mmh",
        "formal_events": EVENTS,
        "diagnostic_events": [event for event in ALL_EVENTS if event not in EVENTS],
        "records": records,
    }


def audit_document(manifest: dict[str, object]) -> str:
    records = manifest["records"]
    record_table = markdown_table(
        ["Path", "Bytes", "SHA-256"],
        [[record["path"], record["bytes"], record["sha256"]] for record in records],
    )
    return f"""# Scientific integrity and reproducibility audit

## 1. Purpose

This document records how the numerical values, plots and claims in the major-revision manuscript were checked. It is not a narrative summary. Its function is to make unsupported statements visible before submission and to distinguish generated evidence from assumptions.

## 2. Evidence hierarchy

### Directly supported

- Array dimensions, units and event identities are read from the files listed in the manifest.
- DrainLite metrics are recomputed from complete 72 x 200 x 280 predictions for five held events.
- Static-feature ablations use identical sampled event-time-pixel indices within each fold.
- The shifted and shuffled controls alter only drainage fields; labels and sample indices remain fixed.
- The SWMM continuity and nonconverging-step values are parsed from separate event report files.
- The event68 sensitivity continuity sequence was corrected by selecting the row whose `event` field equals `event68`.

### Supported only within the conceptual model

- DrainLite learns the residual generated by the specified road-aligned ITZI-SWMM configuration.
- Static drainage alignment contributes predictive information on the fixed study grid.
- Signed and sink-only residuals have different metric trade-offs.

### Not established

- The conceptual network is not a surveyed Shenzhen municipal sewer network.
- Positive residuals are not proof of sewer surcharge because node-head and signed exchange-flow time series were not saved.
- Continuity-screened labels are not numerically validated; routing nonconvergence remains high.
- MIKE comparison is not independent because MIKE informed effective-loss configuration screening.
- LarNO was not run in the current validation, so no LarNO accuracy, speed or zero-shot claim is made.
- No spatial generalisation claim is supported because all folds use the same grid and network.

## 3. Corrected errors

1. **event68 metadata mismatch.** The previous sensitivity script returned the first row of a multi-event CSV. This attached event1 continuity data to event68 water-depth arrays. The script now requires exactly one matching event row. Correct event68 continuity errors at 1-5 mm h−1 are +1.534, -1.961, -5.850, -8.913 and -10.663%.
2. **Static coordinate mapping.** The earlier static builder used round-to-nearest while the coupling runner used integer truncation. The builder now uses the runner's exact mapping and does not clip exterior outfalls onto the boundary. Coupled junction count and unique inlet cells are both 1,169.
3. **Coordinate leakage.** Earlier `base` features contained normalised row and column. Revised experiments separate `base_no_xy` and `base_xy`. Claims about static drainage contribution use the coordinate-free comparison.
4. **Multiplicity.** New rasters store inlet count and pipe-segment count. There are no multi-junction cells under the corrected mapping, but 2,940 pipe cells contain multiple segments. Count features did not improve held-event MAE.
5. **Target-form claim.** The earlier preference for signed residuals was not supported by a same-data sink-only comparison. The revised experiment shows sink-only has lower global MAE but worse RMSE, positive-residual error and final-volume error.
6. **MIKE wording.** MIKE is now described as a configuration-informed descriptive reference, not independent validation or ground truth.

## 4. Numerical and model checks

- All formal water-depth arrays have shape 72 x 200 x 280 and unit metres.
- All saved predictions are finite and nonnegative.
- Sink-only predictions are constrained not to exceed surface-only depth.
- Signed predictions may be above or below surface-only depth.
- Five leave-one-event-out folds use event1, event67, event68, event69 and event70 once each as the held event.
- The primary coordinate-free static model improves all five held events relative to surface-only and base_no_xy.
- Exact five-pair sign-flip p-values cannot be below 0.0625; the manuscript does not label these results statistically significant.

## 5. Hydraulic audit and submission blockers

- Dynamic Wave routing steps not converging range from 59.49% to 87.40% across the eight events.
- The network contains 156 zero-slope conduits, 271 parallel endpoint groups and 12 free synthetic outfalls.
- Only 89 of 1,169 junctions have a directed path to an outfall under current link orientation, although all have undirected connectivity and Dynamic Wave permits reversal.
- A stability-improved network rerun has not been completed. This is a submission blocker for any claim that coupled labels form validated hydraulic reference data.
- Building-rainfall redistribution sensitivity has not been completed.
- Open-boundary and outfall-tailwater sensitivities have not been completed.
- Actual LarNO inference with the revised adapter has not been completed.

## 6. Reviewer-action matrix

| Review issue | Action | Status |
| --- | --- | --- |
| Table 1/Table 2 event68 conflict | Event-keyed CSV lookup; sensitivity regenerated | Completed |
| High SWMM nonconvergence hidden by continuity grading | Separate panel and explicit exploratory-label wording | Completed for disclosure; stable rerun pending |
| Row/column memorisation | Coordinate-free primary model and XY ablation | Completed |
| MIKE presented as untouched external validation | Provenance diagram and revised terminology | Completed |
| LarNO in title without LarNO inference | Removed from title; retained as future compatible use | Completed |
| Network topology insufficiently audited | Graph, duplicate, slope, outfall and raster audit | Completed |
| Junction multiplicity lost | Count rasters added; aligned mapping shows zero multi-junction cells | Completed |
| SWMM and Itzï configuration incomplete | Versions, solver options and coupling coefficients reported | Completed |
| Building rainfall redistribution sensitivity | Listed as required physical sensitivity | Pending |
| Effective-loss sensitivity too narrow | Corrected 1-5 mm h−1 event68 and 1/2 mm h−1 package discussion | Completed descriptively |
| Boundary and synthetic outfall effects | Explicit limitation and rerun requirement | Completed for disclosure; simulation pending |
| Five-event uncertainty | Event dots, paired bootstrap and exact-test limit | Completed |
| Conditional metrics | Wet, severe, residual-sign and network-distance subsets | Completed |
| Dynamic-state overclaim | Static-only wording; dynamic state listed as future requirement | Completed |
| Signed versus sink-only | Same-fold comparison | Completed |
| Positive residual labelled surcharge | Causal surcharge language removed | Completed |
| Metric-dependent ranking | Coupled-label and MIKE metrics shown separately | Completed |
| Signed MIKE peak bias | Absolute peak error and event dots added | Completed |

## 7. File manifest

{record_table}
"""


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    manuscript_path = OUT / "manuscript_major_revision.md"
    report_path = OUT / "report.md"
    audit_path = OUT / "scientific_integrity_audit.md"
    manuscript_path.write_text(manuscript(), encoding="utf-8")
    report_path.write_text(report(), encoding="utf-8")
    manifest = evidence_manifest()
    (OUT / "evidence_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    audit_path.write_text(audit_document(manifest), encoding="utf-8")
    print(manuscript_path)
    print(report_path)
    print(audit_path)
    print(OUT / "evidence_manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
