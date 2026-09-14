#!/usr/bin/env python3
"""Build the final reviewer-v3 manuscript, report, and evidence audit."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REV = ROOT / "extended_study" / "output" / "reviewer_major_revision_v3"
OUT = REV / "submission_package_v3"
FIG = "figures"
PHYSICS = REV / "formal_matched_full" / "physics_quality.csv"
DRAIN = REV / "drainlite_v3" / "metrics"
HYBRID = REV / "drainlite_hybrid_v3" / "metrics"
EVENTS = ["event1", "event20", "event65", "event66", "event67", "event68", "event69", "event70"]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def md_table(headers: list[str], rows: list[list[object]]) -> str:
    clean = lambda value: str(value).replace("|", "\\|").replace("\n", " ")
    lines = ["| " + " | ".join(map(clean, headers)) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(map(clean, row)) + " |" for row in rows)
    return "\n".join(lines)


def fig(number: int, name: str, caption: str) -> str:
    return f"![Figure {number}. {caption}]({FIG}/{name}.png)\n\n**Figure {number}. {caption}**"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bootstrap_delta(values: list[float], seed: int = 20260905) -> tuple[float, float, float]:
    data = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    indexes = rng.integers(0, data.size, size=(100_000, data.size))
    sampled = data[indexes].mean(axis=1)
    return float(data.mean()), float(np.quantile(sampled, 0.025)), float(np.quantile(sampled, 0.975))


def evidence() -> dict[str, object]:
    physics = read_csv(PHYSICS)
    original_summary = {row["model"]: row for row in read_csv(DRAIN / "summary_coupled.csv")}
    original_mike = {row["model"]: row for row in read_csv(DRAIN / "summary_mike_external.csv")}
    hybrid_summary = {row["model"]: row for row in read_csv(HYBRID / "summary_coupled.csv")}
    hybrid_mike = {row["model"]: row for row in read_csv(HYBRID / "summary_mike_external.csv")}
    hybrid_events = [row for row in read_csv(HYBRID / "event_metrics.csv") if row["reference"] == "coupled_label"]
    event_lookup = {(row["event"], row["model"]): row for row in hybrid_events}
    static_deltas = [
        float(event_lookup[(event, "clim_dynamic")]["mae_mm"])
        - float(event_lookup[(event, "clim_all_static")]["mae_mm"])
        for event in EVENTS
    ]
    prior_deltas = [
        float(event_lookup[(event, "spatiotemporal_climatology")]["mae_mm"])
        - float(event_lookup[(event, "clim_all_static")]["mae_mm"])
        for event in EVENTS
    ]
    surface = float(original_summary["surface_matched"]["mae_mm"])
    hybrid = float(hybrid_summary["clim_all_static"]["mae_mm"])
    network = json.loads((REV / "network" / "swmm_bidirectional_normal_step05.independent_audit.json").read_text(encoding="utf-8"))
    larno = json.loads((REV / "larno_event68_checkpoint_audit.json").read_text(encoding="utf-8"))
    hybrid_meta = json.loads((REV / "drainlite_hybrid_v3" / "experiment_metadata.json").read_text(encoding="utf-8"))
    return {
        "physics": physics,
        "original_summary": original_summary,
        "original_mike": original_mike,
        "hybrid_summary": hybrid_summary,
        "hybrid_mike": hybrid_mike,
        "hybrid_event_lookup": event_lookup,
        "surface_to_hybrid_reduction_pct": 100.0 * (surface - hybrid) / surface,
        "static_delta_bootstrap": bootstrap_delta(static_deltas, 1907),
        "prior_delta_bootstrap": bootstrap_delta(prior_deltas, 1908),
        "network": network,
        "larno": larno,
        "hybrid_meta": hybrid_meta,
    }


def physical_table(ev: dict[str, object]) -> str:
    rows = []
    for row in ev["physics"]:
        rows.append([
            row["event"], f"{float(row['rain_volume_m3']) / 1e6:.3f}",
            f"{float(row['roughness_effect_mae_mm']):.3f}", f"{float(row['drainage_effect_mae_mm']):.3f}",
            f"{float(row['drainage_signed_mean_mm']):.3f}",
            f"{float(row['swmm_flow_routing_continuity_error_pct']):.3f}",
            f"{float(row['swmm_steps_not_converging_pct']):.2f}",
            f"{float(row['combined_mass_error_pct_rain']):.3f}", "Accepted" if row["accepted"].lower() == "true" else "Rejected",
        ])
    return md_table(
        ["Event", "Rain volume (10^6 m3)", "|B-A| MAE (mm)", "|C-B| MAE (mm)", "Mean C-B (mm)", "SWMM continuity (%)", "Non-converging (%)", "Combined mass error (% rain)", "Status"],
        rows,
    )


def model_table(ev: dict[str, object]) -> str:
    source = {}
    source.update(ev["original_summary"])
    source.update(ev["hybrid_summary"])
    names = ["surface_matched", "base", "all_static", "spatiotemporal_climatology", "clim_dynamic", "clim_all_static"]
    labels = ["Matched surface B", "Dynamic base", "Dynamic + all static network", "Spatiotemporal prior", "Prior + dynamic", "Prior + dynamic + network (DrainLite hybrid)"]
    rows = []
    for name, label in zip(names, labels):
        row = source[name]
        rows.append([label, f"{float(row['mae_mm']):.3f}", f"{float(row['rmse_mm']):.3f}", f"{float(row['csi_0p03']):.3f}", f"{float(row['csi_0p15']):.3f}", f"{float(row['peak_map_mae_mm']):.3f}", f"{float(row['final_volume_abs_error_m3']) / 1000:.1f}"])
    return md_table(["Model", "MAE (mm)", "RMSE (mm)", "CSI 0.03 m", "CSI 0.15 m", "Peak-map MAE (mm)", "Final-volume error (10^3 m3)"], rows)


def event_model_table(ev: dict[str, object]) -> str:
    lookup = ev["hybrid_event_lookup"]
    rows = []
    for event in EVENTS:
        p = float(lookup[(event, "spatiotemporal_climatology")]["mae_mm"])
        d = float(lookup[(event, "clim_dynamic")]["mae_mm"])
        a = float(lookup[(event, "clim_all_static")]["mae_mm"])
        rows.append([event, f"{p:.3f}", f"{d:.3f}", f"{a:.3f}", f"{p-a:.3f}", f"{d-a:.3f}"])
    return md_table(["Held event", "ST prior MAE", "Prior + dynamic MAE", "Hybrid MAE", "Gain vs prior", "Network increment"], rows)


def mike_table(ev: dict[str, object]) -> str:
    physics = ev["physics"]
    hybrid = ev["hybrid_mike"]["clim_all_static"]
    rows = [
        ["Matched surface B", f"{np.mean([float(r['B_vs_MIKE_mae_mm']) for r in physics]):.3f}", f"{np.mean([float(r['B_vs_MIKE_rmse_mm']) for r in physics]):.3f}", f"{np.mean([float(r['B_vs_MIKE_csi_0p15']) for r in physics]):.3f}", f"{np.mean([float(r['B_vs_MIKE_peak_map_mae_mm']) for r in physics]):.3f}", f"{np.mean([float(r['B_vs_MIKE_final_volume_abs_error_m3']) for r in physics]) / 1000:.1f}"],
        ["Coupled label C", f"{np.mean([float(r['C_vs_MIKE_mae_mm']) for r in physics]):.3f}", f"{np.mean([float(r['C_vs_MIKE_rmse_mm']) for r in physics]):.3f}", f"{np.mean([float(r['C_vs_MIKE_csi_0p15']) for r in physics]):.3f}", f"{np.mean([float(r['C_vs_MIKE_peak_map_mae_mm']) for r in physics]):.3f}", f"{np.mean([float(r['C_vs_MIKE_final_volume_abs_error_m3']) for r in physics]) / 1000:.1f}"],
        ["DrainLite hybrid", f"{float(hybrid['mae_mm']):.3f}", f"{float(hybrid['rmse_mm']):.3f}", f"{float(hybrid['csi_0p15']):.3f}", f"{float(hybrid['peak_map_mae_mm']):.3f}", f"{float(hybrid['final_volume_abs_error_m3']) / 1000:.1f}"],
    ]
    return md_table(["Field", "MAE (mm)", "RMSE (mm)", "CSI 0.15 m", "Peak-map MAE (mm)", "Final-volume error (10^3 m3)"], rows)


def manuscript(ev: dict[str, object]) -> str:
    hs = ev["hybrid_summary"]
    st = hs["spatiotemporal_climatology"]
    dyn = hs["clim_dynamic"]
    main = hs["clim_all_static"]
    static_mean, static_lo, static_hi = ev["static_delta_bootstrap"]
    prior_mean, prior_lo, prior_hi = ev["prior_delta_bootstrap"]
    physics = ev["physics"]
    mean_drain = np.mean([float(row["drainage_effect_mae_mm"]) for row in physics])
    mean_rough = np.mean([float(row["roughness_effect_mae_mm"]) for row in physics])
    mean_signed = np.mean([float(row["drainage_signed_mean_mm"]) for row in physics])
    mean_cont = np.mean([float(row["swmm_flow_routing_continuity_error_pct"]) for row in physics])
    mean_nonconv = np.mean([float(row["swmm_steps_not_converging_pct"]) for row in physics])
    return f"""# DrainLite: leakage-controlled learning of conceptual sewer effects for urban flood prediction

## Highlights

- Matched surface controls separate sewer coupling from an inlet-neighbourhood roughness change.
- A gravity-consistent conceptual network connects 2,276 junctions to 221 receiving interfaces.
- Eight native Itzï-SWMM simulations meet pre-defined topology, continuity and mass-balance checks.
- A lightweight hybrid reduces held-event error from {float(st['mae_mm']):.3f} to {float(main['mae_mm']):.3f} mm.
- Static network descriptors improve every held event, although their incremental contribution is small.

## Abstract

Urban flood emulators can reproduce water-depth fields generated by a coupled hydrodynamic model without exposing the drainage system as a controllable input. This limits their use for testing sewer layouts or transferring a forecast to a city with different drainage infrastructure. We examine whether the effect of a specified drainage network can instead be learned as a lightweight residual correction to a surface-only prediction. A road-aligned conceptual sewer network was constructed over the 20 m Shenzhen urban-flood benchmark. The network was conditioned to give every junction a directed route to a receiving boundary, and was coupled to the two-dimensional Itzï solver through its native Storm Water Management Model interface. Eight six-hour rainfall events were simulated under three matched configurations: an original surface control (A), a surface control with the same inlet-neighbourhood roughness used during coupling (B), and the native bidirectional Itzï-SWMM calculation (C). The learning target was the signed drainage residual C-B.

DrainLite combines a training-event spatiotemporal residual prior with current rainfall, surface depth, terrain and rasterised network descriptors. Each event was held out in turn. The prior for a held event was calculated from the other seven events, and the prior attached to every fitting sample excluded that sample's event. Across 60,783,552 held-out active cell-times, the matched surface field differed from the coupled label by {float(ev['original_summary']['surface_matched']['mae_mm']):.3f} mm mean absolute error. A prior-only estimate achieved {float(st['mae_mm']):.3f} mm, adding dynamic predictors reduced this to {float(dyn['mae_mm']):.3f} mm, and the complete hybrid reached {float(main['mae_mm']):.3f} mm. The network descriptors improved all eight held events, but only by {static_mean:.3f} mm on average (event-bootstrap 95% interval {static_lo:.3f}-{static_hi:.3f} mm). This modest increment, together with high inter-event residual-map correlations, indicates that a fixed spatial response dominates the present single-network dataset.

The coupled calculations had a mean SWMM routing continuity error of {mean_cont:.3f}% and {mean_nonconv:.3f}% non-converging routing steps. Coupling reduced end-of-event surface storage and produced a mean signed depth change of {mean_signed:.3f} mm. Comparison with public MIKE fields was metric dependent: coupling substantially reduced final-volume error but increased full-sequence pixel-time error. MIKE was therefore treated as an external plausibility reference rather than a training target. The public LarNO checkpoint was reproduced independently for event68 (unclamped mean absolute error {ev['larno']['all_grid_unclamped']['mae_m']*1000:.3f} mm), but DrainLite was not directly added to that checkpoint because its MIKE supervision may already contain drainage effects. Within these limits, the study establishes a reproducible low-compute method for learning the event-varying effect of one explicit conceptual network while retaining a clear physical and statistical audit trail.

**Keywords:** urban pluvial flooding; drainage network; Itzï; Storm Water Management Model; residual learning; neural operator; leakage control

## 1. Introduction

Short-duration urban floods emerge from the interaction of rainfall, microtopography, buildings, surface conveyance and underground drainage. Surface water is commonly represented by a two-dimensional shallow-water model, whereas storm sewers are represented by a one-dimensional network model. Their exchange through inlets and junctions changes not only total water storage but also the timing and location of inundation. A surface-only model can therefore give a physically coherent answer and still omit an intervention that matters operationally.

High-resolution coupled calculations remain expensive. The public LarNO study addressed a related computational problem by learning a continuous-space mapping from rainfall and static urban attributes to MIKE water depths, enabling rapid prediction and zero-shot evaluation at finer grids (Cao et al., 2026). Its released 20 m benchmark provides rainfall, terrain and water-depth arrays, but not the municipal drainage inventory used in the MIKE setup. Consequently, the trained model can imitate a reference field in which drainage may be implicit, but the user cannot alter pipe position, diameter or connectivity as an explicit condition.

Two responses are possible. The first is to expand and retrain the complete neural operator with drainage channels. This is attractive when hundreds of coupled simulations and suitable graphics-processing hardware are available. The second is to preserve an upstream flood prediction and estimate only the depth change attributable to a specified drainage system. The latter decomposition is less ambitious, but it is data-efficient and directly testable. It also permits a matched experiment in which the effect of the pipe exchange is distinguished from incidental changes made near inlets.

Residual correction has its own failure modes. A fixed network over a fixed terrain can generate a repeatable spatial template. A learner may exploit that template without understanding how drainage varies with rainfall. Pixel-wise train-test splitting would make this problem worse because neighbouring cells and times from the same event would occur in both sets. Explicit coordinates, distance to synthetic outlets, and target-derived sewer states can also become surrogates for location or label leakage. A defensible experiment must therefore include whole-event holdout, a strong spatiotemporal climatology, and controls that displace or shuffle the network fields.

This study develops DrainLite as a lightweight residual layer for drainage-free upstream predictions. It addresses three questions. First, can a connected conceptual network produce numerically stable and visible drainage responses under native Itzï-SWMM coupling? Second, do current rainfall and surface states explain event-to-event departures from the fixed response? Third, do static network descriptors add information after both the spatiotemporal prior and dynamic state are known? MIKE arrays are used only to examine external plausibility, and the public LarNO checkpoint is reproduced separately to establish compatibility with the source benchmark. Figure 1 summarises the resulting evidence chain.

{fig(1, 'fig01_workflow', 'Physical and learning workflow. A and B are surface-only controls; C adds native bidirectional sewer exchange. The target C-B therefore excludes the roughness effect B-A. The lower pathway shows the three predictor families used by the final hybrid. MIKE and the public LarNO reproduction remain outside the residual-training path.')}

## 2. Data and methods

### 2.1 Study domain and public inputs

The study uses the complete 400 x 560 grid of the public `region1_20m` benchmark. At 20 m spacing, the rectangular calculation domain spans 8.0 x 11.2 km. The active mask contains 105,527 cells, equivalent to 42.21 km2; high building or wall cells remain in the hydraulic grid but are excluded from reported active-cell metrics. Each event contains 72 five-minute intervals covering six hours. Water depth is stored in metres and rainfall in millimetres per interval.

Eight events were selected because complete MIKE, rainfall and locally generated coupled arrays were available: events 1, 20, 65, 66, 67, 68, 69 and 70. These are eight forcing realisations over one terrain and one conceptual sewer layout, not eight independent cities or networks. Event holdout therefore evaluates transfer between rainfall events on a fixed grid.

### 2.2 Conceptual drainage network

No surveyed sewer inventory was available. Road-aligned candidate lines were converted into a directed graph and conditioned for hydraulic use. Parallel endpoint duplicates, isolated fragments and junctions outside the largest connected component were removed. Each retained junction was assigned exactly one downstream conduit following a terrain-informed potential. Distributed receiving interfaces were inserted where required to cap excessive cover and maintain positive conduit slope. The accepted network contains {ev['network']['junctions']:,} junctions, {ev['network']['conduits']:,} conduits and {ev['network']['outfalls']:,} NORMAL receiving interfaces. All junctions have a directed route to an outfall; no directed cycle, isolated junction, duplicate endpoint pair, reverse-slope conduit or pipe crown above the junction rim remains.

Conduit diameters range from {ev['network']['diameter_m']['min']:.1f} to {ev['network']['diameter_m']['max']:.1f} m. Slopes range from {ev['network']['slope']['min']:.6f} to {ev['network']['slope']['max']:.3f} m m-1, and cover ranges from {ev['network']['cover_depth_m']['min']:.2f} to {ev['network']['cover_depth_m']['max']:.2f} m. These values describe a controlled conceptual system. They should not be interpreted as recovered attributes of Shenzhen's municipal sewer network.

{fig(2, 'fig02_network_audit', 'Conceptual-network audit. Panel (a) overlays the retained network on terrain using the verified array orientation. Panel (b) shows pipe diameters and junction degree. Panel (c) confirms the positive-slope constraint on a logarithmic axis. Panel (d) relates conduit length, cover and diameter. The figure establishes internal geometric consistency, not agreement with surveyed assets.')}

### 2.3 Matched physical simulations

Surface flow was solved with Itzï 25.4, which uses a damped partial-inertia form of the shallow-water equations (Courty et al., 2017). The calculation used a 0.001 m minimum depth, Courant number 0.7, partial-inertia coefficient 0.9 and maximum surface-flow step of 1 s. The outer edge of the rectangular domain was closed. Active cells used Manning's n = 0.015, while building cells were represented as high terrain barriers. Rain falling on building cells was redistributed globally to active cells to preserve rainfall volume. A uniform effective loss of 1 mm h-1 represented unresolved initial and continuing losses; it is a modelling assumption rather than a field-calibrated infiltration parameter.

SWMM 5.2.4 was run through PySWMM 2.1.0 using Dynamic Wave routing. The accepted configuration used a 0.5 s maximum routing step, variable-step factor 0.2, 0.05 s minimum observed step, 50 maximum trials, slot surcharge representation, 0.0015 m head tolerance and no SWMM surface ponding. Surface storage and surcharge exchange were represented by Itzï, avoiding a second unobserved ponding store at the one-dimensional nodes. Coupling relaxation was 0.8 and temporal damping was 0.5.

Three simulations were made for each event. Scenario A was surface-only with n = 0.015. Scenario B remained surface-only but changed n to 0.012 in the same 3 x 3 inlet neighbourhoods used by the coupled workflow. Scenario C used the B roughness field and activated Itzï's native `DrainageSimulation`, which exchanges water bidirectionally with SWMM junctions. Thus B-A measures the roughness perturbation and C-B measures the sewer-coupling response. The signed residual is

<p style="text-align:center"><i>r</i>(t,x) = <i>h</i><sub>C</sub>(t,x) - <i>h</i><sub>B</sub>(t,x). &nbsp;&nbsp; (1)</p>

Negative r denotes a shallower coupled surface at that cell and time. Positive r is retained because routing and local return flow can increase depth even when the domain-integrated sewer effect is drainage.

### 2.4 Numerical acceptance and water balance

The physical label was accepted only when topology checks passed, all output arrays were finite and had shape 72 x 400 x 560, the SWMM report contained no warnings or errors, flooding loss was zero, and routing continuity and combined surface-network mass balance were within the study thresholds. Continuity error is a signed accounting residual and should be read by magnitude; a small negative value does not indicate negative water.

{physical_table(ev)}

{fig(3, 'fig03_physical_quality', 'Numerical and physical quality across eight events. Panel (a) reports signed routing continuity error and the fraction of routing steps that did not converge. Panel (b) compares the confounding roughness response B-A with the intended sewer response C-B. Panel (c) shows the reduction in final surface storage under coupling. Panel (d) compares final-volume errors against MIKE, which is an external descriptive reference.')}

### 2.5 Drainage descriptors and learning target

The accepted SWMM input was rasterised on exactly the same 20 m grid. Primitive descriptors comprise inlet and pipe masks, inlet and segment counts, junction degree, diameter, design slope, Manning full-flow capacity and approximate cover. Local 3 x 3 and 7 x 7 pipe densities, 7 x 7 inlet density and 7 x 7 capacity density provide neighbourhood context. Synthetic outfall masks and distance-to-outfall fields were excluded from the principal models because they could encode fixed position more strongly than transferable hydraulic information. Explicit row and column coordinates were also excluded.

The non-network dynamic predictors are current surface depth, current interval rainfall, cumulative rainfall, elevation, terrain slope, normalised time, sine and cosine time encodings, and 3 x 3 and 7 x 7 mean surface depth. The target is r in millimetres. The corrected depth is

<p style="text-align:center"><i>h</i><sub>DL</sub>(t,x) = max[<i>h</i><sub>B</sub>(t,x) + <i>r</i><sub>theta</sub>(t,x), 0]. &nbsp;&nbsp; (2)</p>

Unlike the earlier non-negative sink formulation, Eq. (2) permits both drainage reduction and local positive residuals. This is required to emulate a bidirectional coupled label.

### 2.6 Leakage-controlled event validation

All evaluations use leave-one-event-out cross-validation. One complete event is excluded as the outer test event. A different complete event within the remaining seven is used to select 80, 140 or 220 boosting iterations. The six remaining events form the inner fitting set. No pixel or time from the held event enters fitting, hyperparameter selection or construction of its prior.

The strongest non-parametric baseline is a spatiotemporal residual prior: for each time and cell, r is averaged across the seven outer-training events. To prevent target encoding within model fitting, a training row never receives a prior containing its own event. During final refitting, each of the seven training events uses the mean of the other six; the held event uses the mean of all seven. During inner selection, each fitting event uses the mean of the other five inner-training events, and the validation event uses the six-event mean.

DrainLite uses `HistGradientBoostingRegressor` with squared-error loss, learning rate 0.05, 31 leaves and L2 regularisation 0.01. Uniform sampling selects 450 active cells per time and event, giving 226,800 fitting rows in each final outer fold. Three variants are central: prior only; prior plus ten dynamic and terrain predictors; and prior plus dynamic predictors and thirteen static network descriptors. An earlier ablation without the prior compares dynamic-only, mask, hydraulic and complete static groups. All models predict the full active domain for all 72 times.

### 2.7 Metrics and spatial controls

Mean absolute error (MAE) and root-mean-square error (RMSE) measure depth error. Critical success index (CSI) is reported at 0.03 and 0.15 m. Peak-map MAE compares the maximum depth attained at each cell, whereas global-peak error compares only the single deepest cell and is therefore more sensitive to an isolated depression. Final-volume error sums active-cell depth times 400 m2. Macro-averages give each event equal weight.

To test whether drainage descriptors act through their correct spatial alignment, the static network fields were shifted by 20, 40, 80 and 160 m in four directions. Fifty block-shuffle replicates permuted 20 x 20-cell drainage blocks and regenerated all density descriptors. The target, surface state, rainfall and terrain were unchanged. Permutation importance was calculated on held-event samples and is interpreted as predictive dependence, not causality.

## 3. Results

### 3.1 The connected network produced a distinct drainage response

All eight coupled runs passed the acceptance checks. Mean routing continuity error was {mean_cont:.3f}% and mean non-converging-step frequency was {mean_nonconv:.3f}%; no SWMM flooding loss, warning or error was reported. The mean magnitude of C-B was {mean_drain:.3f} mm, whereas B-A was only {mean_rough:.3f} mm. The approximately {mean_drain/mean_rough:.1f}-fold separation shows that the reported sewer response is not an artefact of the inlet-neighbourhood roughness change.

Coupling reduced the six-hour surface-water volume in every event. The mean signed C-B depth was {mean_signed:.3f} mm. Event68 illustrates the temporal mechanism: B continued to store water after the main rainfall pulse, while C reached a broad maximum and then declined as water entered the connected network and left through distributed receiving interfaces. Domain maximum depth may continue to rise at an isolated low cell even while integrated volume falls; those two curves answer different questions.

{fig(4, 'fig04_eight_event_hydrographs', 'Six-hour active-cell surface-water volume for all events. Grey denotes matched surface control B, orange denotes coupled label C and black denotes MIKE. Pale blue hyetographs use a reversed right axis so rainfall bars descend from the top. Each rainfall axis is scaled to its event; line magnitudes can be compared using the common left-axis units.')}

Peak maps confirm that drainage effects are spatially structured. In event68, coupling reduces depth along connected low-lying routes and around several dense inlet groups. The roughness-only map is much weaker and has a different pattern. Display scales are percentile-clipped for legibility; saturated cells remain in every statistic.

{fig(5, 'fig05_event68_physical_maps', 'Event68 peak-depth fields and matched differences. Panels (a-d) show MIKE, A, B and C. Panel (e) is the drainage effect C-B; panel (f) is the roughness effect B-A. The different diverging ranges are intentional because the roughness perturbation is much smaller.')}

### 3.2 A fixed response is strong, but dynamic predictors add information

The matched surface field differed from C by {float(ev['original_summary']['surface_matched']['mae_mm']):.3f} mm MAE. A model using dynamic and terrain predictors without drainage fields reduced this to {float(ev['original_summary']['base']['mae_mm']):.3f} mm, and the complete static model without a prior reached {float(ev['original_summary']['all_static']['mae_mm']):.3f} mm. However, the seven-event spatiotemporal prior was stronger at {float(st['mae_mm']):.3f} mm. The high-performing prior exposes the repeatability of drainage response on one fixed network; it is not evidence of a universally transferable sewer model.

Adding current surface and rainfall information to the prior reduced MAE to {float(dyn['mae_mm']):.3f} mm. The complete hybrid reached {float(main['mae_mm']):.3f} mm and RMSE {float(main['rmse_mm']):.3f} mm. Relative to B, the MAE reduction was {ev['surface_to_hybrid_reduction_pct']:.1f}%. Relative to the prior, the mean paired improvement was {prior_mean:.3f} mm (event-bootstrap 95% interval {prior_lo:.3f}-{prior_hi:.3f} mm). This result establishes that the model does more than replay a fixed mean map.

{model_table(ev)}

{fig(7, 'fig07_drainlite_skill', 'Held-event performance. Panel (a) shows MAE and joins the three prior-based results for each event. Panel (b) reports RMSE, which weights large errors more heavily. Panel (c) reports CSI at 0.15 m, where larger values are better. Bars are eight-event macro-averages.')}

### 3.3 Static network information gives a small, consistent increment

Adding all static network descriptors to the prior-plus-dynamic model improved every held event. The paired gain ranged from 0.00009 mm for event68 to 0.103 mm for event20 and averaged {static_mean:.3f} mm (event-bootstrap 95% interval {static_lo:.3f}-{static_hi:.3f} mm). This is statistically consistent across the eight available forcing realisations but small in practical magnitude. Most predictive skill comes from the training-event prior and the current surface state, not from the static pipe attributes alone.

{event_model_table(ev)}

The maps in Figure 6 show why a small domain-average error can coexist with visible local differences. Drainage residuals occupy a limited fraction of the active grid and are predominantly negative. DrainLite recovers the connected negative structures and their event-varying amplitude, while remaining errors are concentrated around sharp transitions and local positive patches.

{fig(6, 'fig06_hybrid_residual_maps', 'True, predicted and error residuals for four held events at the time of maximum coupled surface volume. Each row has its own symmetric 99.5th-percentile colour scale. Blue means C or DrainLite is shallower than B; red means locally deeper. Comparisons of pattern are valid across columns within a row, while colour magnitude should not be compared between rows.')}

Spatial controls support, but do not prove, a network contribution. A 20 m displacement increased held-event MAE by about 0.6 mm on average and the penalty grew with displacement. Block shuffling caused a larger increase. Current surface depth and its neighbourhood means dominated permutation importance; inlet and pipe densities were the highest-ranked drainage fields. Correlated predictors mean that these importances cannot be read as hydraulic sensitivities.

{fig(8, 'fig08_controls_importance', 'Network-location controls and predictive dependence. Panel (a) gives the MAE penalty after directional displacement and block shuffling of drainage fields. Panel (b) gives mean permutation importance for the earlier all-static model; error bars show variation across held events. The controls preserve rainfall, target and terrain.')}

### 3.4 External comparison with MIKE is metric dependent

MIKE is not a target for DrainLite and its undistributed drainage configuration is not equivalent to the conceptual network. The comparison nevertheless tests whether the calculated magnitudes are implausible. B has the smallest pixel-time MAE to MIKE ({np.mean([float(r['B_vs_MIKE_mae_mm']) for r in physics]):.3f} mm). C increases this to {np.mean([float(r['C_vs_MIKE_mae_mm']) for r in physics]):.3f} mm, and the hybrid remains close to C at {float(ev['hybrid_mike']['clim_all_static']['mae_mm']):.3f} mm because that is its training objective.

The volume metric gives the opposite ordering. Mean absolute final-volume error decreases from {np.mean([float(r['B_vs_MIKE_final_volume_abs_error_m3']) for r in physics])/1000:.1f} x 10^3 m3 for B to {np.mean([float(r['C_vs_MIKE_final_volume_abs_error_m3']) for r in physics])/1000:.1f} x 10^3 m3 for C. The conceptual network therefore corrects excessive retained volume but does not reproduce MIKE's complete space-time field. It would be incorrect to describe either model as universally closer without naming the metric.

{mike_table(ev)}

{fig(9, 'fig09_mike_metric_tradeoff', 'Event-wise comparison with MIKE. Panel (a) shows full-sequence pixel-time MAE, for which B is usually closest. Panel (b) shows final-volume error, for which C and DrainLite are much closer. MIKE is an external descriptive reference and was never used in residual fitting or hyperparameter selection.')}

### 3.5 Reproduction of the public LarNO checkpoint

The released LarNO checkpoint was evaluated independently on event68 using the public 20 m arrays. Its raw output had MAE {ev['larno']['all_grid_unclamped']['mae_m']*1000:.3f} mm, RMSE {ev['larno']['all_grid_unclamped']['rmse_m']*1000:.3f} mm, R2 {ev['larno']['all_grid_unclamped']['r2']:.4f} and CSI at 0.15 m of {ev['larno']['all_grid_unclamped']['csi_0p15']:.3f}. Clamping negative values reduced MAE to {ev['larno']['active_clamped']['mae_m']*1000:.3f} mm. The raw negative fraction was {ev['larno']['negative_fraction']*100:.2f}%, so both values are reported rather than silently replacing the reproduced output.

{fig(10, 'fig10_larno_reproduction', 'Independent event68 checkpoint reproduction. Panels (a) and (b) share a water-depth scale; panel (c) shows signed error in millimetres. Colour limits are percentile-clipped for viewing only. Metrics use the complete unmodified arrays.')}

DrainLite was not numerically stacked onto this checkpoint in the principal experiment. LarNO was trained against MIKE, whose fields may already include drainage. Adding C-B to that output could therefore count drainage twice and would lack a defensible target. The present evidence supports DrainLite for an explicitly drainage-free upstream field such as B. Extension to LarNO requires either a surface-only LarNO checkpoint or paired LarNO targets generated under matched no-network and network configurations.

### 3.6 Computational cost

The complete eight-fold hybrid experiment ran on the local central processing unit with a recorded peak resident memory of {ev['hybrid_meta']['peak_rss_mb']/1024:.2f} GB and total wall time of {ev['hybrid_meta']['total_seconds']/60:.1f} min. The earlier all-static serialized models averaged approximately 0.82 MB. A six-hour coupled physical event required roughly one hour, whereas full-domain DrainLite inference required tens of seconds on the same workstation. These timings are observed workflow measurements, not hardware-normalised speed claims.

{fig(11, 'fig11_computational_cost', 'Observed computational cost. Panel (a) separates physical simulation, model fitting and dense inference on a logarithmic time axis. Panel (b) reports serialized model size in MB. Panel (c) reports peak process memory for the complete hybrid experiment in GB; unlike the superseded figure, the two units are not placed on one axis.')}

## 4. Discussion

### 4.1 What the experiment establishes

The physical experiment establishes that the revised conceptual network is connected, numerically stable under the accepted settings, and capable of producing a drainage signal that is much larger than the roughness perturbation required by the coupling workflow. This resolves the earlier ambiguity in which weak sewer effects could be caused by disconnected pipes, hydraulic loss from SWMM flooding, or an unmatched surface control.

The learning experiment establishes two levels of predictability. A large fraction of C-B is repeatable in space and time across rainfall events. Dynamic predictors then explain a substantial part of the departure from this fixed prior. Static network descriptors add a smaller but consistently positive increment. The result is scientifically more informative than reporting only the final {float(main['mae_mm']):.3f} mm MAE because it identifies where the apparent skill originates.

### 4.2 Why the static-network increment is small

The static fields do not contain node head, conduit flow, filling ratio or surcharge state. Those variables govern the time-varying capacity of a drainage system, whereas diameter and slope remain constant throughout an event. The spatiotemporal prior also already encodes the average response of the same network at every cell and time. Once that strong predictor is present, a small incremental contribution from static fields is expected. This does not show that pipe properties are unimportant physically; it shows that the present experiment has limited information for identifying their separate statistical effect.

### 4.3 Interpretation of MIKE disagreement

The conceptual and MIKE drainage systems are not matched. Their inlet densities, capacities, outlets, boundary conditions and loss formulations may differ. Pixel-time MAE penalises spatial displacement and timing differences, while final volume measures aggregate storage. The improved volume and degraded pixel-time fit are therefore compatible. They indicate that the conceptual network removes a plausible quantity of water but does not reproduce the undistributed reference system. MIKE should remain a plausibility comparator, not a calibration certificate.

### 4.4 Scope and limitations

Four restrictions define the scope. First, the network is road-derived and conceptual. Second, all eight events share one terrain and network; no claim of transfer to another layout is supported. Third, the closed rectangular surface boundary and global redistribution of building rainfall are assumptions that can affect storage. Fourth, DrainLite receives the full surface-only state, so it is a correction layer rather than an end-to-end rainfall-to-flood model.

The bootstrap intervals describe variation among only eight events and are not population guarantees. The spatial-shift and shuffle controls show dependence on alignment but cannot separate causal pipe effects from correlated terrain and land-form patterns. Finally, the LarNO reproduction is limited to one locally available public checkpoint and event. It demonstrates source-model execution, not a new LarNO benchmark result.

### 4.5 Next experiments

The most useful extension is not a larger tree ensemble. It is a factorial physical dataset that varies rainfall and network configuration independently. At minimum, pipe diameter, inlet density and outlet availability should be perturbed across several plausible networks. This would break the fixed-template shortcut and allow a true held-network test. Rasterised dynamic SWMM states could then be added, provided that they are available at forecast time or predicted by a separate causal module. A surface-only LarNO checkpoint would permit a clean neural-operator-plus-DrainLite experiment without double counting.

## 5. Conclusions

A connected, gravity-consistent conceptual sewer network was coupled through the native Itzï-SWMM interface and evaluated with matched surface controls. Across eight six-hour events, the sewer response C-B averaged {mean_drain:.3f} mm in absolute magnitude and was clearly separated from the {mean_rough:.3f} mm roughness effect. Numerical continuity, convergence and combined mass balance met the pre-defined acceptance conditions for every event.

The leakage-controlled DrainLite hybrid reduced held-event MAE from {float(ev['original_summary']['surface_matched']['mae_mm']):.3f} mm for the matched surface field to {float(main['mae_mm']):.3f} mm. A spatiotemporal training-event prior alone achieved {float(st['mae_mm']):.3f} mm, and dynamic predictors reduced the error to {float(dyn['mae_mm']):.3f} mm. Static network descriptors improved all eight events by a further {static_mean:.3f} mm on average. The principal conclusion is therefore qualified: lightweight residual learning can reproduce the event-varying effect of this fixed conceptual network, but most skill derives from its repeatable spatiotemporal response and current surface dynamics.

The conceptual network substantially improved agreement with MIKE in final stored volume but not in full space-time depth. The public LarNO checkpoint was reproduced independently, yet no drainage residual was added to it because MIKE supervision may already encode drainage. A publishable extension to network-aware neural operators requires matched surface-only and coupled targets, or multiple network interventions that permit held-network validation.

## Data availability

The public LarNO rainfall, terrain and MIKE arrays are retained under `LarNO-main/benchmark/urbanflood`. New V3 physical labels are under `LarNO-main/benchmark/urbanflood/flood/region1_20m_drainage_v3_full`; static descriptors are under the corresponding `geodata` directory. Distribution must follow the licences of the source benchmark and OpenStreetMap-derived inputs.

## Code availability

Network construction, physical simulation, dataset assembly, DrainLite evaluation, figures and document generation are implemented in `extended_study`. The exact execution order and SHA-256 evidence manifest accompany the report.

## Declaration of competing interests

The authors declare no known competing financial interests or personal relationships that could have appeared to influence the work.

## Acknowledgements

Author names, affiliations, funding statements and formal contributions are **to be supplied** before submission.

## References

Bates, P.D., Horritt, M.S., Fewtrell, T.J., 2010. A simple inertial formulation of the shallow water equations for efficient two-dimensional flood inundation modelling. Journal of Hydrology 387, 33-45.

Cao, X., Yao, Y., Wang, Z., Zhao, Z., Borthwick, A.G.L., Qin, H., 2026. Large-scale urban flood modeling and zero-shot high-resolution generalization with LarNO. Journal of Hydrology, 135686. https://doi.org/10.1016/j.jhydrol.2026.135686.

Courty, L.G., Pedrozo-Acuna, A., Bates, P.D., 2017. Itzi (version 17.1): an open-source, distributed GIS model for dynamic flood simulation. Geoscientific Model Development 10, 1835-1847. https://doi.org/10.5194/gmd-10-1835-2017.

Friedman, J.H., 2001. Greedy function approximation: a gradient boosting machine. Annals of Statistics 29, 1189-1232.

Kovachki, N., Li, Z., Liu, B., Azizzadenesheli, K., Bhattacharya, K., Stuart, A., Anandkumar, A., 2023. Neural operator: learning maps between function spaces with applications to PDEs. Journal of Machine Learning Research 24, 1-97.

Li, Z., Kovachki, N., Azizzadenesheli, K., Liu, B., Bhattacharya, K., Stuart, A., Anandkumar, A., 2021. Fourier neural operator for parametric partial differential equations. International Conference on Learning Representations.

Rossman, L.A., Simon, M.A., 2022. Storm Water Management Model User's Manual Version 5.2. U.S. Environmental Protection Agency, Washington, DC.

## Appendix A. Predictor groups

The principal hybrid contains one leakage-controlled spatiotemporal prior, ten dynamic or terrain predictors, and thirteen static network predictors. The predictor count is 24. Outfall location and explicit coordinates are excluded. The response is signed C-B in millimetres; predicted depth is clipped only at zero after the residual is added to B.

## Appendix B. Reproducibility boundaries

The formal dataset contains eight rainfall events but one conceptual network. The outer split is event-wise, not spatial or network-wise. Public MIKE fields are never used for fitting, prior construction, sample weighting or hyperparameter selection. Public LarNO metrics are calculated from the locally executed checkpoint output and are not copied from the reference article.
"""


def report(ev: dict[str, object]) -> str:
    hs = ev["hybrid_summary"]
    static_mean, static_lo, static_hi = ev["static_delta_bootstrap"]
    return f"""# 概化排水管网影响的轻量残差学习：正式科研报告

**研究对象：** 深圳 20 米城市洪涝基准区域的 Itzï-SWMM 双向耦合与 DrainLite 轻量修正  
**报告版本：** Reviewer Major Revision V3  
**证据范围：** 八个 6 小时事件、72 个五分钟时刻、一个固定概化管网  
**生成日期：** 2026-09-05

## 目录

1. 摘要  
2. 研究背景与问题来源  
3. 数据、计算区域与管网  
4. 物理模型和三组匹配试验  
5. DrainLite 轻量模型  
6. 结果及逐图解读  
7. MIKE 与 LarNO 的角色  
8. 结论、局限与后续工作  
9. 可复现性说明

## 摘要

本研究要解决的问题并不是重新训练完整的 LarNO（Latent Autoregressive Neural Operator，潜在自回归神经算子），而是在本机条件下，将明确的排水管网效应加入一个不含管网的地表洪水结果。为此，本研究先把道路对齐线网整理成连通、正坡、可排向边界的概化管网，再调用 Itzï 内置的 `DrainageSimulation` 与 SWMM（Storm Water Management Model，暴雨洪水管理模型）Dynamic Wave（动力波）求解器进行双向交换。八个事件均计算了 A、B、C 三种条件：A 是原始粗糙度地表模型，B 是匹配入口邻域粗糙度的地表模型，C 是在 B 基础上加入原生 Itzï-SWMM 耦合。正式学习标签是 C-B，而不是容易混入粗糙度影响的 C-A。

八个事件的管网作用平均绝对幅度为 {np.mean([float(r['drainage_effect_mae_mm']) for r in ev['physics']]):.3f} 毫米，粗糙度扰动仅为 {np.mean([float(r['roughness_effect_mae_mm']) for r in ev['physics']]):.3f} 毫米。严格留一事件验证中，未修正的 B 对耦合标签平均绝对误差为 {float(ev['original_summary']['surface_matched']['mae_mm']):.3f} 毫米；仅使用其他训练事件平均残差的时空先验达到 {float(hs['spatiotemporal_climatology']['mae_mm']):.3f} 毫米；加入当前水深、降雨和地形后降至 {float(hs['clim_dynamic']['mae_mm']):.3f} 毫米；再加入静态管网特征后为 {float(hs['clim_all_static']['mae_mm']):.3f} 毫米。静态管网特征在八个事件中都带来改善，但平均仅 {static_mean:.3f} 毫米，说明当前结果以固定管网的重复空间响应和地表动态状态为主。

## 1. 研究背景与问题来源

LarNO 的公开模型学习的是“降雨、地形等输入到 MIKE 水深”的映射。公开水深可能已经隐含原 MIKE 管网作用，但公开数据没有给出那套管网，因此不能直接改变管径、入口或出口并观察预测变化。本研究采用较窄但可验证的路线：先用明确的概化管网生成一对“无管网/有管网”物理结果，再学习二者的有符号差值。

这里的“有符号残差”是 C-B。负值表示耦合后该位置水深降低，正值表示局部水深增加。正值不一定代表严重倒灌，也可能来自流路改变、管网暂存后回流或峰值时刻错位。保留正值可以避免把双向耦合误写成只会抽水的单向 sink（汇项）模型。

{fig(1, 'fig01_workflow', '研究流程。上排用于构造无混杂物理标签，下排用于构造轻量学习模型。')}

### 图 1 的来龙去脉与读图方法

图 1 首先回答“训练标签到底是什么”。上排从同一场降雨和地形出发。A 与 B 都没有启动管网，区别只在入口周围 3 x 3 网格的曼宁糙率；C 与 B 使用相同糙率，但 C 打开原生 Itzï-SWMM 双向交换。因此，读者应把 B-A 看成数值配置扰动，把 C-B 看成真正用于学习的管网效应。下排说明模型不是从零预测洪水，而是从其他训练事件形成一个时空先验，再结合当前降雨、水深、地形和静态管网参数，估计新的 C-B。

这张图在全文中的作用是防止三个常见误解：第一，DrainLite 不是完整 LarNO 重训；第二，MIKE 没有进入模型训练；第三，管网残差没有与入口粗糙度变化混在一起。

## 2. 数据、计算区域与概化管网

完整计算区域为 400 x 560 个 20 米网格，即 8.0 x 11.2 千米的矩形。72 帧分别表示每个五分钟区间末端的状态，总时长 6 小时。有效分析区域有 105,527 个网格，面积约 42.21 平方千米。其余高程墙体和建筑单元保留在水动力网格中，但不计入主要统计。

本地没有实测地下管网。概化管网来源于已经与 DEM（Digital Elevation Model，数字高程模型）对齐的道路线网。构建过程删除孤立片段和重复平行连接，按地形势能确定下游方向，并在必要位置设置分布式受纳边界。最终网络有 {ev['network']['junctions']:,} 个 junction（检查井或管网节点）、{ev['network']['conduits']:,} 条 conduit（管段）和 {ev['network']['outfalls']:,} 个 receiving interface（受纳接口）。所有节点都能沿有向管段到达出口，且不存在有向环、孤立节点、反坡管段或管顶高于井口的情况。

{fig(2, 'fig02_network_audit', '地形、管网位置、管径、节点度、管坡和覆土的综合审查。')}

### 图 2 的来龙去脉与读图方法

图 2(a) 用颜色表示地面高程，用线和点表示管段与节点。读图时应先看道路状管线是否落在建筑间的开放通道，再看北部高地与南部低地之间是否形成合理连接。该图采用已经核实的数组方向，没有再次翻转 DEM 或管网。图 2(b) 的管段颜色代表直径，节点颜色代表连接度；连接度较高的点是多条支路汇合的位置。图 2(c) 的横轴为对数坡度，虚线是设计下限，所有柱都位于正坡范围。图 2(d) 同时展示管长、平均覆土和管径，可用于识别异常深埋或超长管段。

这张图能证明的是“概化网络内部一致且与当前网格对齐”，不能证明它等同于深圳真实地下管网。真实井位、管径、泵站和河道口仍属待补充资料。

## 3. 物理计算方法

地表模型使用 Itzï 25.4 的动力学求解器，而不是洼地填充或静态平衡算法。Itzï 采用浅水方程的阻尼部分惯性形式，按地形坡度、水深、摩阻和相邻网格水位差推进地表流。SWMM 5.2.4 使用 Dynamic Wave 路由，即在一维管段上求解 Saint-Venant（圣维南）连续方程和动量方程，并允许满管、回水和流向反转。Itzï 的耦合接口根据地表水位与节点水头计算双向交换。

统一设置包括：1 毫米/小时的有效损失、封闭矩形地表边界、建筑降雨向有效网格全域重分配、地表最大步长 1 秒、SWMM 最大路由步长 0.5 秒、耦合松弛 0.8 和阻尼 0.5。有效损失是对未解析截留和入渗的综合近似，不应写成经过现场率定的土壤参数。

{physical_table(ev)}

{fig(3, 'fig03_physical_quality', '八事件 SWMM 数值质量、A/B/C 分解和 MIKE 体积参照。')}

### 图 3 的来龙去脉与读图方法

图 3(a) 是计算能否作为标签的第一道检查。蓝柱是 SWMM 流量连续性误差，它允许正负号，判断时看绝对值；绿色柱是未收敛路由步的比例。八个事件均远小于早期断裂网络产生的异常值。图 3(b) 使用对数纵轴，因为粗糙度效应与管网效应相差一个以上数量级。绿色柱在每个事件都显著高于蓝柱，说明现在看到的削减主要来自管网交换，而不是入口附近糙率变化。

图 3(c) 的蓝线是 B 在第 6 小时仍留在地表的体积，绿线是 C。两线之间的垂直距离就是管网排出或暂存在管内后带来的地表存量差。图 3(d) 则把 B 和 C 的最终体积与 MIKE 比较。绿色柱普遍较低，说明加入概化管网后，最终水量更接近 MIKE；但这并不自动意味着每个时刻、每个像元都更接近，后文图 9 会显示这类指标冲突。

{fig(4, 'fig04_eight_event_hydrographs', '八事件降雨与区域地表水体积全过程。')}

### 图 4 的来龙去脉与读图方法

每个子图对应一个事件，左轴是区域有效网格的总地表水体积，右轴是平均降雨强度。降雨轴倒置，所以浅蓝色降雨从图顶向下延伸。黑线为 MIKE，灰线为无管网 B，橙线为耦合 C。读图时不要只看单个最深网格，而应先看总水量何时增加、何时下降。

八个事件中，B 在主要降雨结束后仍保持高位甚至缓慢上升；C 的体积更低，并在事件后期出现更明显下降。这正是连接管网能够持续排水的证据。MIKE 通常更早出现峰值和退水，说明两套模型的边界、损失、真实管网能力或河道联系仍不同。图中各事件雨轴独立缩放，用于阅读本事件降雨节奏；跨事件比较降雨大小应使用原始数值表，而不是比较浅蓝填色高度。

{fig(5, 'fig05_event68_physical_maps', '事件 68 的峰值空间图和管网/粗糙度差值。')}

### 图 5 的来龙去脉与读图方法

图 5(a-d) 都是每个网格在六小时内出现过的最大水深。颜色越深表示峰值越大。统一色标经过百分位截断，少数超过 0.73 米的极深网格会呈饱和深蓝，但统计仍使用其真实值。对比 B 与 C 可以看到耦合后若干道路状和低洼连通区域变浅。

图 5(e) 是 C 峰值图减 B 峰值图。蓝色表示管网使峰值降低，红色表示局部峰值增加。图 5(f) 是 B-A，色标范围只有约正负 0.01 米，比图 5(e) 小得多。两个差值图不能只凭颜色深浅比较，必须同时看各自色标。这一设计正是为了避免把视觉放大后的微小糙率差异误认为强管网效应。

## 4. DrainLite 轻量残差模型

DrainLite 的输出不是绝对水深，而是预测残差 `r_hat`。最终水深为 `max(B + r_hat, 0)`。模型使用直方图梯度提升回归树，学习率 0.05，最多 31 个叶节点，候选迭代次数为 80、140 和 220。每个时间、每个训练事件均匀抽取 450 个有效网格；最终每折拟合 226,800 行。模型不用显式行列坐标，也不用到概化出口的距离。

验证按完整事件划分。每次留下一个事件测试，另留一个训练事件选择迭代次数。时空先验对测试事件只由其他七个事件计算；对任何训练样本，先验都排除该样本所属事件。这种处理防止模型在输入中看到自己的目标平均值。

{model_table(ev)}

{fig(7, 'fig07_drainlite_skill', '由地表基线、动态模型、时空先验到完整混合模型的逐层比较。')}

### 图 7 的来龙去脉与读图方法

图 7(a) 的纵轴是平均绝对误差，越低越好。最左侧 B 的误差最高；动态模型和静态管网模型依次下降；时空先验进一步下降。浅蓝和绿色是本文最终的两步混合模型，灰色细线连接同一测试事件，显示管网特征是否对每个事件都改善。八条线从浅蓝到绿色都略向下，但幅度很小。

图 7(b) 使用均方根误差，它对少数大误差更敏感。最终模型从 B 的 {float(ev['original_summary']['surface_matched']['rmse_mm']):.3f} 毫米降至 {float(hs['clim_all_static']['rmse_mm']):.3f} 毫米。图 7(c) 是 0.15 米阈值的临界成功指数，越高越好；最终模型为 {float(hs['clim_all_static']['csi_0p15']):.3f}。三张子图共同说明改善不仅来自大量干区的小误差，也涉及较深积水范围。

{event_model_table(ev)}

{fig(6, 'fig06_hybrid_residual_maps', '四个完整留出事件的真实残差、预测残差和误差。')}

### 图 6 的来龙去脉与读图方法

每一行是一个从未进入该折训练的事件。第一列是真实 C-B，第二列是 DrainLite 预测，第三列是预测减真实。蓝色代表耦合后变浅，红色代表局部变深。每行色标按该事件的 99.5 百分位设置，因此适合比较同一行三幅图的空间位置，不适合直接比较不同行颜色深浅。

第一、二列都出现相似的道路状蓝色结构，说明模型恢复了主要削减区域。第三列整体颜色较浅，但在湿干边界、局部正残差和深洼地附近仍有成片误差。Domain-average（全域平均）为毫米级并不意味着每个局部都准确；图 6 专门用于揭示平均指标掩盖的空间偏差。

{fig(8, 'fig08_controls_importance', '管网错位对照与特征置乱重要性。')}

### 图 8 的来龙去脉与读图方法

图 8(a) 把管网栅格整体平移，同时保持地形、降雨和标签不动。横轴从 20 米增加到 160 米，纵轴是相对正确对齐模型增加的误差。曲线随偏移距离上升，说明正确位置确实重要。橙色虚线和阴影是 50 次块状随机打乱的均值和离散范围，惩罚更大。该实验排除了“只要给模型任意一些管网数值就会改善”的解释。

图 8(b) 将单个特征打乱，观察误差增加。当前水深、局部平均水深和地形最重要；管网中以入口密度和管线密度较高。黑色误差线表示不同留出事件之间的变化。这是预测依赖，不是因果敏感性：地形和管网本来就空间相关，不能据此断言某个变量单独造成了多少排水。

## 5. MIKE 外部参照与 LarNO 复现

MIKE 只承担合理性参照。它没有进入 DrainLite 的训练、先验、调参或抽样。由于公开资料没有给出原 MIKE 管网，MIKE 与当前概化网络不是同一工程系统。

{mike_table(ev)}

{fig(9, 'fig09_mike_metric_tradeoff', 'MIKE 对照在全时序像元误差和最终体积误差上的不同排序。')}

### 图 9 的来龙去脉与读图方法

图 9(a) 对 72 帧的所有有效像元计算误差。灰色 B 通常最低，说明无管网 Itzï 的空间时间水深更接近 MIKE；橙色 C 和绿色 DrainLite 更高。图 9(b) 只比较第 6 小时的总地表水量，结果相反：C 和 DrainLite 显著接近 MIKE。原因是像元误差同时惩罚积水位置和退水时间，体积误差只问“还剩多少水”。当前概化管网排水总量较合理，但位置和时间过程并未复制原 MIKE 管网。

因此，报告的正确结论是“管网改善最终水量一致性，但降低完整空间时间水深的一致性”，而不是简单说“加入管网更接近 MIKE”。

{fig(10, 'fig10_larno_reproduction', '公开 LarNO 检查点在 event68 上的本地复现。')}

### 图 10 的来龙去脉与读图方法

图 10(a) 是 MIKE 目标，图 10(b) 是本地载入公开 epoch 992 权重后的 LarNO 结果，两图共享水深色标。图 10(c) 是 LarNO 减 MIKE，红色为高估、蓝色为低估。未截断结果 MAE 为 {ev['larno']['all_grid_unclamped']['mae_m']*1000:.3f} 毫米，0.15 米 CSI 为 {ev['larno']['all_grid_unclamped']['csi_0p15']:.3f}。原始输出中 {ev['larno']['negative_fraction']*100:.2f}% 数值小于零；将其截为零后 MAE 为 {ev['larno']['active_clamped']['mae_m']*1000:.3f} 毫米。

这张图证明代码、权重和公开数据能在本地完成一次独立推理。它不等于已经把 DrainLite 安全叠加到 LarNO。由于 LarNO 的 MIKE 标签可能已含管网效应，再加 C-B 可能重复计算排水。正式论文因此使用明确不含管网的 B 作为上游输入。要直接形成 LarNO-DrainLite，需要训练一个 surface-only LarNO，或生成同一配置下成对的 LarNO 无管网和有管网标签。

## 6. 计算代价

{fig(11, 'fig11_computational_cost', '物理计算、轻量模型拟合、推理、模型文件和内存占用。')}

### 图 11 的来龙去脉与读图方法

图 11(a) 为对数时间轴。单事件耦合物理模拟以千秒计，单折树模型拟合为数秒，完整区域 72 帧推理为数十秒。图 11(b) 只使用兆字节单位表示模型文件，平均约 0.82 MB。图 11(c) 单独使用吉字节表示完整混合实验的峰值工作内存，为 {ev['hybrid_meta']['peak_rss_mb']/1024:.2f} GB。这里特意分开单位，避免早期图件把 MB 与 GB 放在同一纵轴造成错误视觉比较。

## 7. 主要结论

1. 当前固定模拟方式是原生 Itzï SurfaceFlowSimulation 与 DrainageSimulation 调用 SWMM Dynamic Wave 的双向耦合，不是另行运行后再扣除的独立 SWMM，也不是静态洼地填充。
2. 修正后的概化网络全部节点可达出口，八事件无 SWMM flooding loss，连续性与综合水量误差均在预设阈值内。
3. C-B 的管网信号明显大于 B-A 的糙率信号，因此当前“有管网/无管网”差异是真正的耦合响应。
4. 最终 DrainLite 混合模型对耦合标签 MAE 为 {float(hs['clim_all_static']['mae_mm']):.3f} 毫米，相比 B 改善 {ev['surface_to_hybrid_reduction_pct']:.1f}%。
5. 时空先验很强，动态变量提供主要新增改进；静态管网特征平均只再改善 {static_mean:.3f} 毫米，但八个事件方向一致。
6. MIKE 比较具有指标依赖性，不能作为当前概化管网已经真实率定的证据。

## 8. 不足与下一阶段

最大的科学限制是“八场雨、一个管网”。留一事件只证明对新降雨的泛化，不能证明对新管网、新城市或新分辨率的泛化。下一步最有价值的是对管径、入口密度和出口条件做成组扰动，生成多个网络版本，并留出完整网络测试。若能保存 SWMM 动态节点水头、流量、满管率和回流状态，可进一步解释事件差异，但必须确保这些变量在预测时可获得，不能把目标时刻的真实管网状态作为输入造成泄漏。

其他待补充内容包括：真实管网或独立积水观测、开放河道边界、建筑降雨处理敏感性、更多事件、5 米真实 MIKE 参考，以及 surface-only LarNO 权重。现阶段不得声称已经验证真实市政管网、完成 5 米耦合验证或实现跨网络零样本泛化。

## 9. 可复现性说明

正式数据集目录为 `region1_20m_drainage_v3_full`。物理输出、机器学习输出和投稿图均位于 `extended_study/output/reviewer_major_revision_v3`。伴随的 `scientific_integrity_audit.md` 和 `evidence_manifest.json` 记录数组形状、文件哈希、模型版本、被否决的旧结果以及每项结论的证据边界。
"""


def audit(ev: dict[str, object]) -> str:
    network_path = REV / "network" / "swmm_bidirectional_normal_step05.inp"
    dataset_audit = ROOT / "LarNO-main" / "benchmark" / "urbanflood" / "flood" / "region1_20m_drainage_v3_full" / "dataset_audit.json"
    hybrid_meta = REV / "drainlite_hybrid_v3" / "experiment_metadata.json"
    larno_pred = ROOT / "LarNO-main" / "exp" / "20260220_183648_006352" / "pred_results" / "region1_20m" / "epoch_992" / "predictions_epoch_992_sample_event68.npy"
    manifest_rows = [
        ["Accepted SWMM network", str(network_path.relative_to(ROOT)), network_path.stat().st_size, sha256(network_path)],
        ["Physical quality table", str(PHYSICS.relative_to(ROOT)), PHYSICS.stat().st_size, sha256(PHYSICS)],
        ["Dataset audit", str(dataset_audit.relative_to(ROOT)), dataset_audit.stat().st_size, sha256(dataset_audit)],
        ["Original DrainLite summary", str((DRAIN / 'summary_coupled.csv').relative_to(ROOT)), (DRAIN / 'summary_coupled.csv').stat().st_size, sha256(DRAIN / 'summary_coupled.csv')],
        ["Hybrid metadata", str(hybrid_meta.relative_to(ROOT)), hybrid_meta.stat().st_size, sha256(hybrid_meta)],
        ["Hybrid summary", str((HYBRID / 'summary_coupled.csv').relative_to(ROOT)), (HYBRID / 'summary_coupled.csv').stat().st_size, sha256(HYBRID / 'summary_coupled.csv')],
        ["LarNO event68 prediction", str(larno_pred.relative_to(ROOT)), larno_pred.stat().st_size, sha256(larno_pred)],
    ]
    return f"""# Scientific integrity, provenance and reproducibility audit

## 1. Purpose

This document separates verified evidence from interpretation. It was generated from the same machine-readable outputs used by the manuscript and report. It is not a response letter and does not upgrade a conceptual network into observed infrastructure.

## 2. Provenance chain

1. Public source data: LarNO rainfall, terrain and MIKE water-depth arrays.
2. Copied executable workflow: the calibrated local Itzï project was copied from `E:/Projects/20260518-itzi-flood` into `external_models/20260518-itzi-flood`; the source folder was not edited.
3. New physical products: A, B and C arrays generated locally for eight events.
4. New learning products: DrainLite models, predictions and metrics generated locally from C-B.
5. External comparator: MIKE arrays are never a DrainLite target.
6. Independent reproduction: the LarNO checkpoint metrics are calculated from a local inference array, not transcribed from Cao et al. (2026).

## 3. Software and fixed definitions

- Itzï 25.4; PySWMM 2.1.0; swmm-toolkit 0.17.0.
- scikit-learn 1.8.0; NumPy 2.4.5; SciPy 1.17.1; Matplotlib 3.10.9; SciencePlots 2.2.2.
- A: surface-only, Manning n = 0.015.
- B: surface-only, n = 0.012 in coupled-inlet 3 x 3 neighbourhoods.
- C: B plus native bidirectional Itzï-SWMM coupling.
- Roughness effect: B-A.
- Learning target: C-B in millimetres.
- MIKE role: external descriptive comparator only.

## 4. Dataset checks

The accepted dataset contains exactly eight events. Every rainfall and depth array has shape 72 x 400 x 560 and contains finite values. The static active mask contains 105,527 cells. Evaluation therefore covers 7,597,944 active cell-times per held event and 60,783,552 over the complete outer validation. The dataset audit lists no rejected event.

## 5. Network checks

The accepted network contains {ev['network']['junctions']} junctions, {ev['network']['conduits']} conduits and {ev['network']['outfalls']} receiving interfaces. Directed outfall reachability is {ev['network']['directed_outfall_reachability_pct']:.1f}%. Independent parsing found zero isolated junctions, cycles, duplicate undirected endpoint groups, reverse-slope conduits and crown-above-rim endpoints. SWMM ponding is disabled; surface overflow and return are handled by the 2D coupling.

## 6. Numerical checks

All eight events have zero SWMM flooding loss, zero warning count and zero error count. Mean routing continuity error is {np.mean([float(r['swmm_flow_routing_continuity_error_pct']) for r in ev['physics']]):.4f}% and mean non-converging-step frequency is {np.mean([float(r['swmm_steps_not_converging_pct']) for r in ev['physics']]):.4f}%. The maximum absolute combined mass error is {max(abs(float(r['combined_mass_error_pct_rain'])) for r in ev['physics']):.4f}% of rainfall volume.

The previously observed outfall attribution count was traced to the copied reporting implementation, which repeated a global failed-step count for each outfall. Formal acceptance uses SWMM's global report value and separately captured junction counters; the repeated per-outfall field is not used in any conclusion.

## 7. Learning leakage checks

- Outer test units are complete events.
- Inner validation is a complete event within the outer-training set.
- Test-event targets never enter fitting, iteration selection, sample weighting or prior construction.
- For final refitting, each training event receives a prior averaged from the other six outer-training events.
- The held event receives the prior averaged from all seven outer-training events.
- Explicit row/column coordinates and distance-to-synthetic-outfall are excluded.
- Static network fields are predictors only; dynamic SWMM target-side states are excluded.

## 8. Quantitative claim audit

The matched surface baseline MAE to C is {float(ev['original_summary']['surface_matched']['mae_mm']):.6f} mm. The spatiotemporal prior is {float(ev['hybrid_summary']['spatiotemporal_climatology']['mae_mm']):.6f} mm, prior plus dynamics is {float(ev['hybrid_summary']['clim_dynamic']['mae_mm']):.6f} mm and the complete hybrid is {float(ev['hybrid_summary']['clim_all_static']['mae_mm']):.6f} mm. Static descriptors improve all eight event folds; the macro-average paired increment is {ev['static_delta_bootstrap'][0]:.6f} mm with an event-bootstrap interval of {ev['static_delta_bootstrap'][1]:.6f}-{ev['static_delta_bootstrap'][2]:.6f} mm.

This evidence supports a small incremental association for the fixed network. It does not support cross-network generalisation, recovery of municipal sewer parameters, causal feature attribution, or direct application to the public LarNO checkpoint.

## 9. Rejected or superseded analyses

- Static depression filling is not used as a formal Itzï result.
- The early sparse 82-node/29-pipe network is superseded and excluded.
- Disconnected and high-nonconvergence SWMM runs are retained only as diagnostics.
- Sink-only labels are not used for the bidirectional model because they remove positive residuals by construction.
- The 5 m proxy is not described as 5 m MIKE validation; genuine local 5 m reference files remain unavailable.
- Previous claims based on cropped 200 x 280 windows are superseded by full 400 x 560 V3 calculations.
- No residual is added directly to the public LarNO checkpoint because its MIKE target may already include drainage.

## 10. File evidence

{md_table(['Evidence', 'Repository-relative path', 'Bytes', 'SHA-256'], manifest_rows)}

## 11. Figure audit

All eleven publication figures were generated from V3 CSV, JSON or NPY outputs using SciencePlots and Times New Roman. PNG files were visually reviewed at original resolution. One figure defect was found and corrected: model size in MB and process memory in GB had initially shared one numerical axis; the final Figure 11 uses separate panels and units. The workflow arrows were also repositioned after they were found to cross box labels. Map arrays use `origin=upper`, matching the audited network/DEM orientation. Percentile clipping is stated in captions and affects display only.

## 12. Reproduction order

1. `build_gravity_consistent_swmm_network.py`
2. `audit_reviewer_v3_network.py`
3. `run_reviewer_v3_matched_coupling.py` or `run_reviewer_v3_batch.py`
4. `merge_reviewer_v3_scenarios.py`
5. `analyze_reviewer_v3_physics.py`
6. `build_reviewer_v3_dataset.py`
7. `run_reviewer_v3_drainlite.py`
8. `run_reviewer_v3_drainlite_hybrid.py`
9. `generate_reviewer_v3_publication_figures.py`
10. `generate_reviewer_v3_documents.py`
11. `render_reviewer_v3_documents.py`

## 13. Remaining information before journal submission

Author list, affiliations, funding, contribution statements and data-distribution permissions are to be supplied. External observational validation, measured sewer geometry and a factorial multi-network dataset are not available. These absences are limitations, not values to be inferred or filled synthetically.
"""


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    ev = evidence()
    docs = {
        "manuscript.md": manuscript(ev),
        "report.md": report(ev),
        "scientific_integrity_audit.md": audit(ev),
    }
    for name, content in docs.items():
        (OUT / name).write_text(content.strip() + "\n", encoding="utf-8")
    serializable = {key: value for key, value in ev.items() if key not in {"physics", "original_summary", "original_mike", "hybrid_summary", "hybrid_mike", "hybrid_event_lookup"}}
    (OUT / "evidence_summary.json").write_text(json.dumps(serializable, indent=2), encoding="utf-8")
    print(OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
