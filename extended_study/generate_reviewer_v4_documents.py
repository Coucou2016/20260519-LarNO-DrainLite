#!/usr/bin/env python3
"""Build the V4 manuscript, Chinese report and evidence audit from calculated files."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
V3 = ROOT / "extended_study" / "output" / "reviewer_major_revision_v3"
V4 = ROOT / "extended_study" / "output" / "reviewer_major_revision_v4"
EXP = V4 / "final_hybrid_controls"
OUT = V4 / "submission_package_v4"
PHYS = V3 / "formal_matched_full" / "physics_quality.csv"
NETWORK = V3 / "network" / "swmm_bidirectional_normal_step05.independent_audit.json"
LARNO = V3 / "larno_event68_checkpoint_audit.json"
EVENTS = ["event1", "event20", "event65", "event66", "event67", "event68", "event69", "event70"]
CANONICAL_SEED = 1907


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def table(headers: list[str], values: list[list[object]]) -> str:
    clean = lambda value: str(value).replace("|", "\\|").replace("\n", " ")
    lines = ["| " + " | ".join(map(clean, headers)) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(map(clean, row)) + " |" for row in values)
    return "\n".join(lines)


def figure(number: int, stem: str, caption: str) -> str:
    return f"![Figure {number}. {caption}](figures/{stem}.png)\n\n**Figure {number}. {caption}**"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_evidence() -> dict[str, object]:
    metrics = read_csv(EXP / "metrics" / "event_metrics_all_seeds.csv")
    summaries = read_csv(EXP / "metrics" / "model_summary_mean_sd.csv")
    controls = read_csv(EXP / "metrics" / "final_hybrid_network_controls.csv")
    conditional = read_csv(EXP / "metrics" / "conditional_metrics_canonical_seed.csv")
    runtime = read_csv(EXP / "metrics" / "runtime_uncached_canonical.csv")
    clipping = read_csv(EXP / "metrics" / "clipping_sensitivity.csv")
    inventory = read_csv(EXP / "metrics" / "event_selection_inventory.csv")
    summary = {(row["reference"], row["model"]): row for row in summaries}

    def value(model: str, metric: str, reference: str = "coupled_label") -> float:
        return float(summary[(reference, model)][f"{metric}_mean"])

    def seed_event(model: str, reference: str = "coupled_label") -> list[dict[str, str]]:
        return [row for row in metrics if row["model"] == model and row["reference"] == reference]

    seed_increments = []
    for seed in sorted({int(row["seed"]) for row in metrics}):
        dynamic = [float(row["mae_mm"]) for row in metrics if int(row["seed"]) == seed and row["model"] == "hybrid_dynamic" and row["reference"] == "coupled_label"]
        network = [float(row["mae_mm"]) for row in metrics if int(row["seed"]) == seed and row["model"] == "hybrid_all" and row["reference"] == "coupled_label"]
        seed_increments.append(float(np.mean(dynamic) - np.mean(network)))

    control_groups: dict[str, list[float]] = defaultdict(list)
    shift_groups: dict[int, list[float]] = defaultdict(list)
    for row in controls:
        name = row["control"]
        delta = float(row["delta_mae_vs_aligned_mm"])
        control_groups[name].append(delta)
        if name.startswith("shift_"):
            shift_groups[int(name.split("_")[1].replace("m", ""))].append(delta)

    condition_groups: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in conditional:
        condition_groups[(row["model"], row["condition"])].append(float(row["mae_mm"]))

    return {
        "physics": read_csv(PHYS),
        "metrics": metrics,
        "summary": summary,
        "value": value,
        "seed_increments": seed_increments,
        "control_groups": control_groups,
        "shift_groups": shift_groups,
        "condition_groups": condition_groups,
        "runtime": runtime,
        "clipping": clipping,
        "inventory": inventory,
        "network": json.loads(NETWORK.read_text(encoding="utf-8")),
        "larno": json.loads(LARNO.read_text(encoding="utf-8")),
        "metadata": json.loads((EXP / "experiment_metadata.json").read_text(encoding="utf-8")),
        "seed_event": seed_event,
    }


def physical_table(ev: dict[str, object]) -> str:
    values = []
    for row in ev["physics"]:
        values.append([
            row["event"], f"{float(row['rain_volume_m3']) / 1e6:.3f}",
            f"{float(row['roughness_effect_mae_mm']):.3f}", f"{float(row['drainage_effect_mae_mm']):.3f}",
            f"{float(row['drainage_signed_mean_mm']):.3f}", f"{float(row['swmm_flow_routing_continuity_error_pct']):.3f}",
            f"{float(row['swmm_steps_not_converging_pct']):.3f}", f"{float(row['combined_mass_error_pct_rain']):.3f}",
            "Accepted" if row["accepted"].lower() == "true" else "Rejected",
        ])
    return table(["Event", "Rain volume (10^6 m3)", "|B-A| MAE (mm)", "|C-B| MAE (mm)", "Mean C-B (mm)", "Routing continuity (%)", "Non-converging (%)", "Combined mass error (% rain)", "Status"], values)


def performance_table(ev: dict[str, object]) -> str:
    value = ev["value"]
    order = [
        ("surface_matched", "Matched surface B"), ("prior_only", "Spatiotemporal prior"),
        ("dynamic_no_prior", "Dynamic predictors, no prior"), ("hybrid_dynamic", "Prior + dynamic"),
        ("hybrid_mask", "Prior + dynamic + masks"), ("hybrid_hydraulic", "Prior + dynamic + hydraulics"),
        ("hybrid_all", "Prior + dynamic + all network fields"),
    ]
    values = []
    for name, label in order:
        row = ev["summary"][("coupled_label", name)]
        values.append([
            label, int(row["n_seeds"]), f"{value(name, 'mae_mm'):.3f}", (f"{float(row['mae_mm_sd']):.3f}" if int(row["n_seeds"]) > 1 else "Not estimated"),
            f"{value(name, 'rmse_mm'):.3f}", f"{value(name, 'csi_0p03'):.3f}", f"{value(name, 'csi_0p15'):.3f}",
            f"{value(name, 'peak_map_mae_mm'):.3f}", f"{value(name, 'final_volume_abs_error_m3') / 1000:.1f}",
        ])
    return table(["Model", "Seeds", "MAE (mm)", "Seed SD", "RMSE (mm)", "CSI 0.03 m", "CSI 0.15 m", "Peak-map MAE (mm)", "Final-volume error (10^3 m3)"], values)


def event_table(ev: dict[str, object]) -> str:
    metrics = ev["metrics"]
    values = []
    for event in EVENTS:
        def get(model: str) -> float:
            return float(next(row["mae_mm"] for row in metrics if row["event"] == event and int(row["seed"]) == CANONICAL_SEED and row["model"] == model and row["reference"] == "coupled_label"))
        prior, dynamic, all_network = get("prior_only"), get("hybrid_dynamic"), get("hybrid_all")
        values.append([event, f"{prior:.3f}", f"{dynamic:.3f}", f"{all_network:.3f}", f"{prior-all_network:.3f}", f"{dynamic-all_network:.6f}"])
    return table(["Held event", "Prior MAE", "Prior + dynamic", "Full hybrid", "Gain over prior", "Network increment"], values)


def conditional_table(ev: dict[str, object]) -> str:
    groups = ev["condition_groups"]
    conditions = [
        ("all_active", "All active cell-times"), ("target_wet_0p03m", "Coupled depth >= 0.03 m"),
        ("effect_abs_over_5mm", "|C-B| > 5 mm"), ("net_drainage_over_5mm", "C-B < -5 mm"),
        ("positive_residual_over_5mm", "C-B > 5 mm"), ("near_inlet_0_20m", "Within 20 m of inlet"),
        ("near_pipe_0_20m", "Within 20 m of pipe"),
    ]
    values = []
    for condition, label in conditions:
        values.append([
            label,
            f"{np.mean(groups[('surface_matched', condition)]):.3f}",
            f"{np.mean(groups[('prior_only', condition)]):.3f}",
            f"{np.mean(groups[('hybrid_dynamic', condition)]):.3f}",
            f"{np.mean(groups[('hybrid_all', condition)]):.3f}",
        ])
    return table(["Evaluation subset", "Surface B", "ST prior", "Prior + dynamic", "Full hybrid"], values)


def mike_table(ev: dict[str, object]) -> str:
    physics = ev["physics"]
    value = ev["value"]
    specifications = [
        ("Matched surface B", "B"), ("Coupled label C", "C"), ("DrainLite full hybrid", "DL"),
    ]
    values = []
    for label, key in specifications:
        if key == "DL":
            row = [value("hybrid_all", metric, "mike_external") for metric in ["mae_mm", "rmse_mm", "csi_0p15", "peak_map_mae_mm", "final_volume_abs_error_m3"]]
        else:
            row = [np.mean([float(item[f"{key}_vs_MIKE_{metric}"]) for item in physics]) for metric in ["mae_mm", "rmse_mm", "csi_0p15", "peak_map_mae_mm", "final_volume_abs_error_m3"]]
        values.append([label, f"{row[0]:.3f}", f"{row[1]:.3f}", f"{row[2]:.3f}", f"{row[3]:.3f}", f"{row[4]/1000:.1f}"])
    return table(["Field", "MAE (mm)", "RMSE (mm)", "CSI 0.15 m", "Peak-map MAE (mm)", "Final-volume error (10^3 m3)"], values)


def event_selection_table(ev: dict[str, object], only_used: bool = False) -> str:
    rows = [row for row in ev["inventory"] if row["public_arrays_valid"].lower() == "true"]
    if only_used:
        rows = [row for row in rows if row["used_in_paired_v3"].lower() == "true"]
    values = [[row["event"], row["used_in_paired_v3"], f"{float(row['mean_6h_rainfall_mm_active']):.2f}", f"{float(row['max_local_6h_rainfall_mm_active']):.2f}", f"{float(row['mike_global_peak_m_active']):.3f}"] for row in rows]
    return table(["Event", "Paired A/B/C", "Mean 6 h rain (mm)", "Maximum local 6 h rain (mm)", "MIKE global peak (m)"], values)


def manuscript(ev: dict[str, object]) -> str:
    physics = ev["physics"]
    value = ev["value"]
    network = ev["network"]
    larno = ev["larno"]
    mean_drain = np.mean([float(row["drainage_effect_mae_mm"]) for row in physics])
    mean_rough = np.mean([float(row["roughness_effect_mae_mm"]) for row in physics])
    mean_signed = np.mean([float(row["drainage_signed_mean_mm"]) for row in physics])
    mean_continuity = np.mean([float(row["swmm_flow_routing_continuity_error_pct"]) for row in physics])
    mean_nonconv = np.mean([float(row["swmm_steps_not_converging_pct"]) for row in physics])
    network_increment = np.asarray(ev["seed_increments"])
    control = ev["control_groups"]
    runtime = [row for row in ev["runtime"] if row["model"] == "hybrid_all" and int(row["seed"]) == CANONICAL_SEED]
    correction_s = np.mean([float(row["total_correction_seconds"]) for row in runtime])
    total_s = np.mean([float(row["surface_plus_correction_seconds"]) for row in runtime])
    coupled_s = np.mean([float(row["coupled_runtime_s"]) for row in runtime])
    speedup = np.mean([float(row["speedup_vs_coupled"]) for row in runtime])
    used = sum(row["used_in_paired_v3"].lower() == "true" for row in ev["inventory"])
    valid = sum(row["public_arrays_valid"].lower() == "true" for row in ev["inventory"])
    return f"""# DrainLite: leakage-controlled residual emulation of a fixed conceptual sewer network for urban flood modelling

## Highlights

- Matched surface controls isolate sewer exchange from inlet-neighbourhood roughness.
- A road-aligned conceptual network connects 2,276 junctions to 221 receiving interfaces.
- Whole-event validation reduces coupled-label depth error from {mean_drain:.3f} to {value('hybrid_all', 'mae_mm'):.3f} mm.
- A training-event prior explains most of the repeatable fixed-network response.
- Static network fields add {network_increment.mean():.3f} +/- {network_increment.std(ddof=1):.3f} mm MAE improvement across five sampling seeds.

## Abstract

Urban flood surrogates commonly reproduce the output of a coupled hydrodynamic model without exposing the drainage system as an explicit condition. This study considers a narrower problem: whether the surface-depth change caused by one specified sewer network can be emulated as a lightweight correction to a drainage-free hydrodynamic state. A road-aligned conceptual network was constructed for the 20 m Shenzhen benchmark and coupled bidirectionally to the two-dimensional Itzï solver through the Storm Water Management Model. Eight six-hour rainfall events were calculated under an original surface configuration (A), a surface configuration with coupling-neighbourhood roughness but no pipe exchange (B), and native Itzï-SWMM coupling (C). The signed learning target was C-B.

DrainLite combines an event-excluded spatiotemporal residual prior with current rainfall, the contemporaneous B depth field, terrain and rasterised network attributes. Complete events were held out in turn. Five independent spatial sampling seeds were used for the principal dynamic and full-network models, while model complexity was fixed before fitting. Across 60,783,552 held-out active cell-times, B differed from C by {mean_drain:.3f} mm mean absolute error. The spatiotemporal prior attained {value('prior_only', 'mae_mm'):.3f} mm, the prior-plus-dynamic model attained {value('hybrid_dynamic', 'mae_mm'):.3f} mm, and the full hybrid attained {value('hybrid_all', 'mae_mm'):.3f} mm. The static-network increment averaged {network_increment.mean():.3f} mm across sampling seeds and was smaller than the contribution of the prior and dynamic state. Displacing, block-shuffling or jointly permuting the network fields increased the error of the final hybrid, indicating that the small increment depended on spatial alignment.

The coupled calculations met the predefined topology, continuity and mass-balance conditions. Their mean routing continuity error was {mean_continuity:.3f}%, and their mean signed depth change was {mean_signed:.3f} mm. Comparison with public MIKE fields was metric dependent: coupling reduced final-volume error but increased full space-time depth error. MIKE was therefore retained as a descriptive external comparator, not a training label or independent validation dataset. The resulting method is a fixed-network residual emulator that requires the contemporaneous surface-only field; it is not an end-to-end rainfall-to-flood predictor and does not establish transfer to an unseen sewer layout.

**Keywords:** urban pluvial flooding; conceptual sewer network; Itzï; Storm Water Management Model; residual emulation; whole-event validation

## 1. Introduction

Short-duration urban flooding reflects the joint influence of rainfall, buildings, surface conveyance, local storage and underground drainage. Two-dimensional surface models resolve overland propagation, whereas one-dimensional network models describe flow through junctions and pipes. Their exchange through inlets can reduce surface storage, redistribute water and return surcharge to the street. Omitting this exchange does not make a surface calculation dynamically invalid, but it changes the system being represented.

The computational cost of coupled simulation remains a practical barrier to event ensembles and rapid forecasting. Neural operators offer a complementary route by learning mappings between forcing fields and hydrodynamic solutions. The Large-scale Latent Autoregressive Neural Operator (LarNO) was developed for large-area urban flood prediction and demonstrated zero-shot evaluation across grid resolutions using a public Shenzhen benchmark (Cao et al., 2026). Its released inputs and checkpoint permit reproduction of the reference mapping, but the municipal drainage inventory represented in the MIKE calculations is not released as an independently changeable model input.

One response would be to retrain the complete neural operator with additional pipe channels. That strategy requires many matched network simulations and substantially more hardware than is available in the present study. A residual formulation provides a lower-cost alternative. If a drainage-free model supplies a surface state, a second model can estimate only the depth difference induced by a prescribed sewer system. This decomposition is useful only if the target is physically identifiable and the validation prevents information from the tested event entering the predictor.

Fixed terrain and a fixed network create a further difficulty. Their drainage response may recur at the same cells and times across events. A model can therefore appear skilful by recovering a climatological template, even if it makes little use of rainfall or pipe attributes. Random cell-wise splitting compounds this problem, because adjacent samples from one event enter both fitting and evaluation. A suitable test must retain entire rainfall events, compare against an event-excluded spatiotemporal prior, and perturb the network fields in the final model rather than in an earlier surrogate.

Here, DrainLite is formulated as a correction layer for a fixed conceptual network. Three questions are examined. First, does a connected road-aligned network produce a numerically controlled response that can be separated from ancillary roughness changes? Second, how much of this response is explained by a cross-event prior and by the current surface-rainfall state? Third, after those predictors are known, is the remaining contribution of correctly aligned static network information detectable? MIKE and the public LarNO checkpoint are analysed outside the residual-fitting pathway to clarify what the experiment does, and does not, establish.

{figure(1, 'fig01_workflow', 'Experimental design. Scenarios B and C share their surface parameterisation, so C-B isolates the bidirectional sewer exchange. DrainLite receives the contemporaneous B field and predicts a signed residual. The held event is excluded from fitting and prior construction. MIKE and LarNO are external to the principal learning pathway.')}

## 2. Data and methods

### 2.1 Study domain and event inclusion

The public 20 m benchmark comprises a 400 x 560 rectangular grid, corresponding to 8.0 x 11.2 km. The reporting mask contains 105,527 active cells (42.21 km2). Each event contains 72 five-minute fields spanning six hours. Water depth is stored in metres and rainfall as millimetres per five-minute interval. High building cells remain as hydraulic barriers but are excluded from active-cell performance statistics.

Paired local A/B/C simulations were available for events 1, 20 and 65-70. These eight events were defined by the availability of complete locally generated matched calculations, not by their DrainLite error. Of {valid} public event directories with readable 20 m rainfall and MIKE arrays, {used} therefore entered the paired experiment. Event 79 was retained in the inventory as unreadable and excluded. Figure 11d and Appendix B place the selected events within the rainfall and reference-severity distribution. The design evaluates event transfer on one fixed terrain and network, not transfer between cities or sewer layouts.

### 2.2 Construction and audit of the conceptual network

Surveyed sewer records were unavailable. Road-aligned candidates derived from OpenStreetMap geometry were converted to a directed network and conditioned for hydraulic use. Disconnected fragments and duplicate endpoint pairs were removed. Terrain-informed invert levels were assigned while maintaining cover, positive conduit slope and a downstream path to a receiving interface. The accepted network contains {network['junctions']:,} junctions, {network['conduits']:,} conduits and {network['outfalls']:,} NORMAL receiving interfaces. Pipe diameters range from {network['diameter_m']['min']:.1f} to {network['diameter_m']['max']:.1f} m; slopes range from {network['slope']['min']:.6f} to {network['slope']['max']:.3f} m m-1.

Every junction reaches an outfall in the directed graph. The independent parser found no isolated junction, cycle, duplicate endpoint pair, reverse-slope conduit or pipe crown above a junction rim. These checks establish internal topological and geometric consistency. They do not establish correspondence with Shenzhen's surveyed municipal network. OpenStreetMap-derived geometry is attributed to OpenStreetMap contributors under the Open Database License.

{figure(2, 'fig02_network_audit', 'Conceptual-network geometry and topology. Panel (a) places the road-aligned network over the digital elevation model. Panel (b) shows diameter and degree. Panel (c) verifies the positive design-slope distribution. Panel (d) relates length, cover and diameter. The panels test internal consistency; they are not a validation against surveyed pipes.')}

### 2.3 Matched physical calculations

Surface flow was solved with Itzï 25.4 using its damped partial-inertia formulation. The minimum water depth was 0.001 m, the Courant number 0.7, the partial-inertia coefficient 0.9 and the maximum surface step 1 s. The rectangular outer boundary was closed. Active cells used Manning's n = 0.015 except where stated below. Building cells were elevated barriers. Rain falling on those cells was globally redistributed over active cells to conserve rainfall volume. A uniform 1 mm h-1 effective loss represented unresolved interception, infiltration and other continuing losses; it was not fitted as a measured soil parameter.

The one-dimensional network was solved by SWMM 5.2.4 through PySWMM 2.1.0. Dynamic-wave routing used a 0.5 s maximum step, variable-step factor 0.2, 50 trials, slot surcharge representation and 0.0015 m head tolerance. SWMM node ponding was disabled because surface storage and surcharge exchange were represented by Itzï. The coupling relaxation and damping factors were 0.8 and 0.5. The executable entry point now rejects unreviewed Itzï, PySWMM and SWMM-toolkit versions before opening the native engine.

Scenario A used surface flow with Manning's n = 0.015. Scenario B remained surface-only but applied n = 0.012 in the same 3 x 3 inlet neighbourhoods used during coupling. Scenario C retained the B roughness field and activated Itzï's native DrainageSimulation interface to exchange water bidirectionally with SWMM. Hence B-A measures the local roughness perturbation, while C-B measures the pipe-coupling response:

<p style="text-align:center"><i>r</i>(t,x) = <i>h</i><sub>C</sub>(t,x) - <i>h</i><sub>B</sub>(t,x). &nbsp;&nbsp; (1)</p>

Negative r denotes local drainage relative to B; positive r is retained because network routing and surcharge may locally increase surface depth.

### 2.4 Physical-run acceptance criteria

The repository applies the following acceptance conditions; their historical pre-registration has not been independently established. Each array must be finite and have shape 72 x 400 x 560. Every junction requires a directed outfall path. Absolute SWMM flow-routing continuity error must be at most 2%, non-converging routing steps at most 2%, and absolute combined surface-network mass error at most 0.5% of event rainfall. SWMM flooding loss, warning count and error count must be zero. A signed continuity value is an accounting residual; acceptance uses its magnitude.

{physical_table(ev)}

{figure(3, 'fig03_physical_quality', 'Physical-run quality. Panel (a) reports routing continuity and non-converging steps against the stated numerical criteria. Panel (b) compares the roughness effect B-A with the sewer effect C-B. Panel (c) shows final surface storage under B and C. Panel (d) gives final-volume discrepancy relative to MIKE. The logarithmic scale in panel (b) is required because the two effects differ by nearly two orders of magnitude.')}

### 2.5 Predictor fields and signed residual

The network was rasterised on the 20 m model grid. Eight mask and topology predictors describe inlet presence, pipe presence, inlet count, segment count, degree and local pipe/inlet densities. Five hydraulic predictors describe diameter, slope, full-flow capacity, cover and local capacity density. No row or column coordinate, target-side SWMM head, target-side pipe flow, synthetic outfall mask or distance-to-outfall field entered the principal models.

Ten non-network predictors describe the current B depth, current and cumulative rainfall, elevation, terrain slope, normalised time, sine and cosine time encodings, and 3 x 3 and 7 x 7 local depth means. The local means are mask-aware, so building and inactive cells do not depress neighbourhood values. The target is r in millimetres, and the corrected field is

<p style="text-align:center"><i>h</i><sub>DL</sub>(t,x) = max[<i>h</i><sub>B</sub>(t,x) + <i>r</i><sub>theta,mm</sub>(t,x)/1000, 0]. &nbsp;&nbsp; (2)</p>

No residual clipping is used in the primary experiment. The final maximum enforces non-negative water depth; the affected cell-time fraction is reported with the clipping sensitivity rather than being applied silently.

### 2.6 Whole-event validation and sampling repeats

Eight outer folds were formed by withholding one complete event. A spatiotemporal prior was calculated at every time and cell from the other seven events. For fitting rows, the prior also excluded the row's own event, so no target from that event contributed to its prior. MIKE arrays were never used in fitting, weighting, feature construction or model selection.

HistGradientBoostingRegressor used squared-error loss, learning rate 0.05, 31 leaves, L2 regularisation 0.01 and a pre-specified 140 boosting iterations. The fixed iteration count removes sensitivity to a single inner validation event. Each training event contributed 450 uniformly selected active cells per time step. The principal prior-plus-dynamic and full-network models were repeated for five pre-specified sampling seeds. Subgroup models and spatial controls used seed {CANONICAL_SEED} to limit redundant full-domain inference. Every held event was then predicted over all active cells and all 72 times.

### 2.7 Network controls and conditional evaluation

The final prior-plus-dynamic-plus-network model was subjected to three destructive controls. Primitive network fields were shifted by 20, 40, 80 and 160 m in each cardinal direction; 20 x 20-cell blocks were jointly shuffled in ten replicates; and all primitive network values were jointly permuted among active cells in five replicates. Derived density fields were regenerated after each transformation. A zero-static-fields control removed the explicit network fields but retained the event-excluded fixed-network prior. The latter is therefore a test of added static fields, not a claim that the complete model is network-free.

Errors were calculated over the full active domain, wet cells, cell-times with |C-B| above 1, 5 and 10 mm, negative and positive residual subsets, and bands within 20, 20-60, 60-100 and more than 100 m from inlets and pipes. This prevents dry or weak-effect cells from dominating interpretation.

### 2.8 Performance and computational metrics

Mean absolute error (MAE), root-mean-square error (RMSE), bias and critical success index (CSI) at 0.03 and 0.15 m were calculated on complete held events. Peak-map MAE compares the maximum depth at each cell. Global-peak depth and timing concern only the single largest depth. Surface volume and inundated-area errors were evaluated at every time and at six hours. Metrics were macro-averaged over events.

Runtime measurement includes uncached dynamic neighbourhood computation, feature construction, input assembly, estimator prediction and non-negativity post-processing. Static descriptors, the fitted model and the training-derived prior are prepared beforehand; input/output disk access is excluded. The workflow comparison sums this in-memory correction time and the recorded historical B simulation time and contrasts the sum with the recorded C time. It is not a newly timed integrated execution of both components.

## 3. Results

### 3.1 Coupling produced a resolved and numerically controlled response

All eight events met the physical acceptance conditions. Mean routing continuity error was {mean_continuity:.3f}%, and mean non-converging-step frequency was {mean_nonconv:.3f}%. The mean |C-B| response was {mean_drain:.3f} mm, compared with {mean_rough:.3f} mm for |B-A|. The {mean_drain/mean_rough:.1f}-fold difference shows that the learning target is dominated by sewer exchange rather than the inlet-neighbourhood roughness adjustment. Coupling reduced six-hour surface storage in every event, with a mean signed C-B depth of {mean_signed:.3f} mm.

{figure(4, 'fig04_eight_event_hydrographs', 'Active-domain surface-water volume for eight events. Grey is matched surface B, orange is coupled C and black is MIKE. Pale blue rainfall uses the right axis. The left axis is common in units but each right rainfall axis is event-scaled. A declining C volume after rainfall indicates drainage from the surface even where a single-cell maximum continues to rise.')}

{figure(5, 'fig05_event68_physical_maps', 'Event 68 peak-depth comparison. Panels (a-d) show MIKE, A, B and C. Panel (e) gives C-B, and panel (f) gives B-A. The separate residual scales reflect their different magnitudes. Display clipping affects colour only; all cells remain in the statistics.')}

### 3.2 The fixed-network prior accounts for most of the predictable structure

The matched surface field had {value('surface_matched', 'mae_mm'):.3f} mm MAE relative to C. Dynamic predictors without a prior reduced the error to {value('dynamic_no_prior', 'mae_mm'):.3f} mm. The event-excluded spatiotemporal prior was stronger, at {value('prior_only', 'mae_mm'):.3f} mm, showing that much of the drainage response recurred at the same cells and times on this fixed system. Adding the current surface-rainfall state to the prior reduced MAE to {value('hybrid_dynamic', 'mae_mm'):.3f} mm. The full model reached {value('hybrid_all', 'mae_mm'):.3f} mm and RMSE {value('hybrid_all', 'rmse_mm'):.3f} mm.

{performance_table(ev)}

{figure(7, 'fig07_skill_and_sampling_seeds', 'Whole-event performance. Panels (a-c) report MAE, RMSE and CSI at 0.15 m. Lines in panel (a) connect prior-based results for individual events under the canonical seed. Error bars show sampling-seed standard deviation where five repeats were run; subgroup models were intentionally evaluated only under the canonical seed and therefore have zero repeat error bars.')}

### 3.3 Static network information is detectable but secondary

Across five sampling seeds, adding all static network fields to the prior-plus-dynamic model changed MAE by {network_increment.mean():.3f} +/- {network_increment.std(ddof=1):.3f} mm. This increment is small relative to the gains from the prior and dynamic state. It is also event dependent, as shown by the canonical-seed values below.

{event_table(ev)}

The final-model controls nevertheless show that the increment is tied to spatial organisation. A 20 m shift increased MAE by {np.mean(ev['shift_groups'][20]):.3f} mm on average, and the penalty at 160 m was {np.mean(ev['shift_groups'][160]):.3f} mm. Zeroing the static fields increased MAE by {np.mean(control['zero_static_network_fields']):.3f} mm; block shuffling and group permutation increased it by {np.mean(control['block_shuffle']):.3f} and {np.mean(control['group_permutation']):.3f} mm, respectively. These controls do not demonstrate transfer to another sewer layout, because the spatiotemporal prior still encodes the average response of the original network.

{figure(8, 'fig08_final_hybrid_network_controls', 'Controls applied to the final hybrid. Panel (a) shows the MAE penalty as correctly aligned fields are displaced. Panel (b) shows zero-field, block-shuffle and joint-permutation penalties. Panel (c) compares mask and hydraulic groups. Panel (d) gives the paired static-network increment for five sampling seeds. Positive values mean the aligned full-network model is more accurate.')}

{figure(6, 'fig06_fixed_network_residual_maps', 'Held-event residual fields at the time of maximum coupled surface volume. Column 1 is the contemporaneous B input; columns 2 and 3 are true and predicted C-B; column 4 is the resulting depth error. Each row uses its own percentile-based residual range, so spatial pattern should be compared within rows rather than by colour intensity between events.')}

### 3.4 Improvements persist where drainage is active

Full-domain means can be dominated by dry or weak-effect cells. Conditional evaluation gives a more demanding view. In the canonical-seed experiment, the full hybrid reduced MAE not only over all active cell-times but also for wet cells, locations with |C-B| above 5 mm, net-drainage cells, positive residual cells and the immediate inlet and pipe neighbourhoods. Errors remain larger in the strong-effect subsets, which identifies the residual transitions and local surcharge patches as the main unresolved structures.

{conditional_table(ev)}

{figure(9, 'fig09_conditioned_performance', 'Conditional MAE. Panel (a) separates the full domain from wet cells. Panel (b) progressively restricts evaluation to stronger drainage effects. Panel (c) groups cells by distance to an inlet. The change in vertical scale between panels is intentional: strong-effect and near-inlet subsets are harder than the domain average.')}

### 3.5 MIKE comparison depends on the quantity being evaluated

MIKE was not generated with the same disclosed conceptual network and is not an independent validation target for C. The comparison is therefore descriptive. Matched surface B has the lowest full space-time MAE to MIKE, whereas C and DrainLite substantially reduce six-hour volume discrepancy. The opposite ordering of these metrics means that coupling removes a more plausible aggregate water volume without reproducing MIKE's complete spatial field.

{mike_table(ev)}

{figure(10, 'fig10_mike_metric_tradeoff', 'Metric-dependent comparison with MIKE. Panels (a-d) show full-field MAE, peak-map MAE, final-volume error and CSI at 0.15 m. Smaller values are better in panels (a-c); larger values are better in panel (d). No single field dominates all four quantities, so the comparison cannot be reduced to a single statement of agreement.')}

### 3.6 End-to-end timing and post-processing

Feature construction, input assembly, model prediction and post-processing together required {correction_s:.1f} s per event on average for the canonical full hybrid. The observed B-plus-DrainLite workflow required {total_s:.1f} s, compared with {coupled_s:.1f} s for C, giving a mean speed ratio of {speedup:.1f}. This is a workflow measurement on one workstation rather than a hardware-normalised benchmark. It also shows why estimator-only timing would overstate acceleration.

The primary result uses no residual clipping. Sensitivity calculations at +/-500 and +/-1200 mm quantify the effect of optional bounds, and the recorded non-negativity count shows how often Eq. (2) changes an otherwise negative predicted depth. Event selection is shown alongside these computational diagnostics to keep performance claims connected to the available forcing set.

{figure(11, 'fig11_runtime_clipping_event_selection', 'Computational and inclusion diagnostics. Panel (a) decomposes the complete correction time. Panel (b) compares B, B plus DrainLite and C on a logarithmic axis. Panel (c) reports residual-clipping sensitivity; the primary result is the unbounded residual followed only by the physical non-negativity constraint. Panel (d) places paired events within the readable public rainfall-reference inventory.')}

### 3.7 Relation to the public LarNO checkpoint

The public LarNO checkpoint was executed independently for event 68. Its unclamped MAE was {larno['all_grid_unclamped']['mae_m']*1000:.3f} mm, RMSE {larno['all_grid_unclamped']['rmse_m']*1000:.3f} mm and CSI at 0.15 m {larno['all_grid_unclamped']['csi_0p15']:.3f}. Negative raw predictions comprised {larno['negative_fraction']*100:.2f}% of cells, so both raw and clamped statistics are retained.

{figure(12, 'fig12_larno_reproduction', 'Independent public-checkpoint reproduction for event 68. Reference and prediction share a depth scale; the third panel shows signed error. Percentile limits are used only for display. This calculation verifies execution of the upstream checkpoint and is not part of DrainLite fitting.')}

DrainLite was not appended to the public checkpoint. That checkpoint was trained against MIKE fields that may already contain drainage effects; applying C-B to it could count drainage twice. A defensible end-to-end LarNO-DrainLite calculation requires a LarNO model trained to drainage-free targets or paired neural-operator targets under matched network interventions.

## 4. Discussion

### 4.1 What is learned on a fixed network

The results separate three sources of predictive skill. The event-excluded prior captures the repeatable cell-time response of one terrain-network system. Current surface depth and rainfall account for much of the event-specific departure from that response. Static pipe fields provide a smaller final adjustment. This hierarchy explains why a low-compute model can emulate C accurately while still offering limited evidence for network-conditioned generalisation.

The final-hybrid displacement and shuffle controls strengthen the interpretation of the static increment. Unlike earlier controls applied to a no-prior model, they perturb the exact estimator used for the main result. Their penalties show that correctly aligned fields are preferable to displaced or destroyed fields. Yet the prior remains fixed during those tests. The experiment therefore supports aligned static information within the original system, not counterfactual prediction after redesigning that system.

### 4.2 Error structure and physical meaning

Conditioned metrics show why the 2 mm-scale domain average should not be read as uniform accuracy. Where |C-B| exceeds 5 or 10 mm, the residual is sharper and less frequent, and the error rises. Immediate inlet areas are similarly difficult because the exchange is concentrated within a small number of cells. Positive C-B patches are not necessarily numerical failures: bidirectional routing can return water or displace storage locally while total surface volume falls. Keeping a signed target is therefore essential.

### 4.3 External reference and model scope

The MIKE comparison constrains interpretation rather than certifying calibration. The conceptual network and undisclosed MIKE drainage representation differ in topology, inlet density, capacity and boundary treatment. Better final-volume agreement under C suggests that surface-only storage is excessive, whereas poorer pixel-time MAE shows that the spatial redistribution is not the same. Both observations can be true.

Four boundaries remain. First, the sewer is road-derived rather than surveyed. Second, eight rainfall events share one terrain and network. Third, the surface edge is closed, building rainfall is globally redistributed and the 1 mm h-1 loss is an effective assumption. The event68 paired sensitivity experiment below quantifies their influence for one event; it does not establish robustness across events or retrained surrogates. Fourth, DrainLite requires the contemporaneous B field and should be described as a residual emulator, not an independent forecast model.

### 4.4 Implications for further development

The next physical dataset should vary network configuration as well as rainfall. Holding out a pipe-capacity, inlet-density or outlet-layout scenario would test whether the static fields act as intervention variables. Such an experiment should omit any prior derived from the target network. Dynamic node head, conduit flow, fullness and surcharge could then be rasterised as causal inputs if they are available at inference, or predicted by a separate network-state model. Only after a drainage-free LarNO upstream model is available should the two components be assessed as an end-to-end rainfall-to-flood system.

## 5. Conclusions

A connected conceptual sewer network was coupled through Itzï's native SWMM interface and evaluated using matched surface controls. Across eight events, |C-B| averaged {mean_drain:.3f} mm, compared with {mean_rough:.3f} mm for the roughness-only difference. All events met the predefined topology, continuity, convergence and combined mass-balance criteria.

Whole-event validation showed that the event-excluded spatiotemporal prior reduced MAE to {value('prior_only', 'mae_mm'):.3f} mm. Adding current surface and rainfall information reduced it to {value('hybrid_dynamic', 'mae_mm'):.3f} mm, and adding all static network fields yielded {value('hybrid_all', 'mae_mm'):.3f} mm. The static increment was {network_increment.mean():.3f} +/- {network_increment.std(ddof=1):.3f} mm across five sampling seeds. Final-model displacement and destructive controls confirm that this small increment depends on correct network alignment, while conditional metrics show that improvements persist in wet, near-inlet and strong-effect cells.

The evidence supports rapid residual emulation for rainfall events on this fixed conceptual network. It does not establish generalisation to another pipe layout or an end-to-end LarNO-DrainLite predictor. That distinction is central to both the scientific contribution and the next experimental step.

## Data availability

Public LarNO rainfall, terrain, MIKE arrays and the upstream checkpoint retain their original provenance. Locally calculated A/B/C labels and static descriptors are provided in the V3 canonical dataset. Road-derived geometry requires attribution to © OpenStreetMap contributors and is subject to the Open Database License. The repository licence map identifies materials whose downstream redistribution terms remain to be confirmed with the source authors.

## Code availability

Network construction, physical simulation, residual fitting, controls, figures and read-only verification are available at https://github.com/Coucou2016/20260519-LarNO-DrainLite. The repository records exact package versions, Git LFS objects, SHA-256 manifests and the distinction between regeneration and verification.

## Declaration of competing interests

The authors declare no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

## Acknowledgements and author information

Author names, affiliations, contributions and funding information are **to be supplied before submission**.

## References

Bates, P.D., Horritt, M.S., Fewtrell, T.J., 2010. A simple inertial formulation of the shallow water equations for efficient two-dimensional flood inundation modelling. Journal of Hydrology 387, 33-45.

Cao, X., Yao, Y., Wang, Z., Zhao, Z., Borthwick, A.G.L., Qin, H., 2026. Large-scale urban flood modeling and zero-shot high-resolution generalization with LarNO. Journal of Hydrology, 135686. https://doi.org/10.1016/j.jhydrol.2026.135686.

Courty, L.G., Pedrozo-Acuna, A., Bates, P.D., 2017. Itzi (version 17.1): an open-source, distributed GIS model for dynamic flood simulation. Geoscientific Model Development 10, 1835-1847. https://doi.org/10.5194/gmd-10-1835-2017.

Friedman, J.H., 2001. Greedy function approximation: a gradient boosting machine. Annals of Statistics 29, 1189-1232.

Kovachki, N., Li, Z., Liu, B., Azizzadenesheli, K., Bhattacharya, K., Stuart, A., Anandkumar, A., 2023. Neural operator: learning maps between function spaces with applications to PDEs. Journal of Machine Learning Research 24, 1-97.

Li, Z., Kovachki, N., Azizzadenesheli, K., Liu, B., Bhattacharya, K., Stuart, A., Anandkumar, A., 2021. Fourier neural operator for parametric partial differential equations. International Conference on Learning Representations.

Rossman, L.A., Simon, M.A., 2022. Storm Water Management Model User's Manual Version 5.2. U.S. Environmental Protection Agency, Washington, DC.

## Appendix A. Predictor groups

The full hybrid comprises one event-excluded spatiotemporal prior, ten dynamic or terrain predictors and thirteen static network predictors. Explicit coordinates, synthetic outfall fields, distance-to-outfall fields and target-side SWMM state are excluded. The primary residual is not clipped; only the reconstructed water depth is constrained to be non-negative.

## Appendix B. Public-event inventory

{event_selection_table(ev)}
"""


def report(ev: dict[str, object]) -> str:
    value = ev["value"]
    network_increment = np.asarray(ev["seed_increments"])
    control = ev["control_groups"]
    physics = ev["physics"]
    mean_drain = np.mean([float(row["drainage_effect_mae_mm"]) for row in physics])
    mean_rough = np.mean([float(row["roughness_effect_mae_mm"]) for row in physics])
    return f"""# 固定概化排水管网影响的轻量残差模拟科研报告

## 目录

1. 摘要  
2. 研究背景与问题界定  
3. 数据、物理模型与质量控制  
4. DrainLite 方法与验证设计  
5. 结果及逐图解读  
6. 讨论、结论与下一步工作  

## 摘要

本报告记录一条可审计的研究链：先用匹配的地表对照和 Itzï-SWMM 双向耦合计算得到“同一场降雨、同一地形下，管网使水深发生了多少变化”，再用轻量梯度提升模型模拟这一有符号变化。这里的 DrainLite 不是从降雨直接生成洪水图的独立预报器；它接收当前时刻的无管网 Itzï 水深 B，并预测耦合水深 C 与 B 的差值。八场六小时事件均在 20 米、72 时步、400 x 560 全域网格上计算。

物理结果表明，管网效应 |C-B| 平均为 {mean_drain:.3f} 毫米，而为匹配耦合设置引入的局部糙率效应 |B-A| 仅为 {mean_rough:.3f} 毫米。留一事件验证中，B 相对 C 的平均绝对误差为 {value('surface_matched', 'mae_mm'):.3f} 毫米；事件排除的时空先验将其降到 {value('prior_only', 'mae_mm'):.3f} 毫米，加入当前水深和降雨后降到 {value('hybrid_dynamic', 'mae_mm'):.3f} 毫米，完整静态管网模型为 {value('hybrid_all', 'mae_mm'):.3f} 毫米。五个空间采样种子下，静态管网字段的平均增量为 {network_increment.mean():.3f} +/- {network_increment.std(ddof=1):.3f} 毫米。该增量较小，但最终模型的错位、块打乱和整组置换均使误差上升，说明正确空间位置确实提供了信息。

## 1. 研究背景与问题界定

城市暴雨积水由地表汇流和地下排水共同决定。地表二维模型回答“雨水如何沿地形和道路传播”，一维管网模型回答“雨水如何经检查井和管道输送并排出”。原 LarNO 工作解决的是大范围、高分辨率洪水场的快速学习问题；本研究关注更具体的缺口：当上游模型没有把可修改的管网作为显式条件时，能否用一个本机可训练的小模型补充指定管网的影响。

这一问题必须严格限定。当前模型使用一个由道路对齐、地形约束和水力规则构造的概化网络，而不是真实城市管网；八场事件共享同一网络；DrainLite 还需要同一时刻的 B 水深。因此本研究证明的是“固定网络上的跨降雨事件残差模拟”，不是“对任意城市管网都能泛化”，也不是已经完成的“降雨到洪水的端到端 LarNO-DrainLite”。

{figure(1, 'fig01_workflow', '研究流程。上排先生成匹配的物理对照：B 和 C 的地表参数相同，二者只相差是否开启 Itzï 原生的 SWMM 双向交换。下排把训练事件先验、当前地表状态和静态管网字段依次加入残差模型。读图时应注意，MIKE 和公共 LarNO 复现没有进入模型训练。')}

**图 1 逐项解释。** 左上输入框代表降雨、地形和建筑物。B 框是没有地下管网交换的地表动力学结果；C 框在相同地表设置下开启管网，因此 C-B 才能解释为管网效应。下方紫色框是由其余事件计算的平均时空响应，蓝色框代表当前事件已经发生到该时刻的降雨和地表水深，绿色框代表管网位置、密度、管径、坡度、能力和覆土等静态信息。最后的水深不允许小于零。图中没有从 MIKE 或测试事件把答案直接送入模型的箭头，这是泄漏控制的核心。

## 2. 数据、物理模型与质量控制

### 2.1 计算区域和事件

完整区域为 400 x 560 个 20 米网格，即约 8.0 x 11.2 千米。评价掩膜含 105,527 个活动网格，面积约 42.21 平方千米。每场事件有 72 幅五分钟水深图，总时长六小时。八场配对事件为 event1、event20 和 event65-event70。选择依据是本地存在完整的 A/B/C 匹配计算，而不是根据 DrainLite 表现挑选。完整公共事件严重度背景见图 11d 和附表。

### 2.2 概化网络

网络沿 OpenStreetMap 道路几何布置，并使用地形高程设置管底和坡向。最终网络含 {ev['network']['junctions']:,} 个节点、{ev['network']['conduits']:,} 条管段和 {ev['network']['outfalls']:,} 个接收边界。所有节点都有有向路径通往排放口；没有孤立节点、回路、重复端点管段、反坡管段或管顶高于节点地面的情况。这些检查说明网络内部可运行，但不能把它写成深圳真实管网。

{figure(2, 'fig02_network_audit', '概化网络审计。子图 (a) 检查管网与地形的相对位置和方向；子图 (b) 观察管径和节点连接度；子图 (c) 在对数坐标中检查管坡是否为正；子图 (d) 同时查看管长、覆土和管径，防止几何参数互相矛盾。道路几何来源应标注“© OpenStreetMap contributors”。')}

**图 2 逐项解释。** 子图 (a) 的底图是高程，线和点是管线及节点。它用于回答早期最容易出错的问题：管网是否上下翻转、整体偏移或落到建筑墙体上。子图 (b) 用颜色或尺寸显示管径和连接度，主干位置应形成连续结构，而不能只剩零散点。子图 (c) 的横轴和纵轴用于检查坡度分布；对数轴把很小的正坡度展开，所有值位于零以上。子图 (d) 检查较长管段是否出现不合理的覆土或直径组合。图只能证明规则一致，不能证明参数经过现场校准。

### 2.3 A/B/C 物理对照

A 是原始地表方案；B 在入口附近采用与耦合运行相同的局部糙率，但不开启管网；C 在 B 的基础上启用 Itzï 25.4 内置的 DrainageSimulation，并由 SWMM 5.2.4 动力波求解管网。这样，B-A 是糙率改变造成的差异，C-B 才是地下管网交换造成的差异。C-B 允许为正，因为满管或局部回流可能让某些地表网格变深。

### 2.4 验收阈值

每场运行必须同时满足：数组为 72 x 400 x 560 且无 NaN/Inf；所有节点可达排放口；SWMM 流量连续性误差绝对值不超过 2%；不收敛步比例不超过 2%；地表-管网合并水量误差绝对值不超过降雨量的 0.5%；SWMM flooding loss、warning 和 error 均为零。

{physical_table(ev)}

{figure(3, 'fig03_physical_quality', '八场物理标签质量。子图 (a) 是数值连续性和不收敛步；子图 (b) 比较管网效应与糙率效应；子图 (c) 比较六小时地表存水量；子图 (d) 把最终水量与 MIKE 参照比较。')}

**图 3 逐项解释。** 子图 (a) 的柱越接近零越好；连续性误差有正负号，但验收看绝对值。子图 (b) 用对数纵轴，管网效应柱普遍远高于糙率效应柱，说明结果不是靠入口附近降低糙率“制造”出来的。子图 (c) 中 C 低于 B 表示管网把一部分水从地表系统中输送出去。子图 (d) 只比较最终总体水量，不能替代空间水深检验。

{figure(4, 'fig04_eight_event_hydrographs', '八场事件的地表水量过程。灰线为 B，橙线为 C，黑线为 MIKE，蓝色阴影为降雨。')}

**图 4 逐项解释。** 每个子图是一场独立降雨，横轴为小时，左轴为活动网格上的地表水量。灰线和橙线之间的垂直距离就是管网对总体存水的影响。降雨停止后，如果橙线下降而灰线继续维持或上升，说明管网正在排空地表。黑线来自 MIKE，但其管网和边界细节不公开，因此只用于判断量级和峰现时间是否明显异常。单个最低点的最大水深可能仍上升，这与全域水量下降并不矛盾。

{figure(5, 'fig05_event68_physical_maps', 'Event68 的峰值空间图及差值图。前四幅是 MIKE、A、B、C，后两幅分别是 C-B 和 B-A。')}

**图 5 逐项解释。** 前四个子图要看积水热点是否沿相似低地出现，而不是只看最大值。C-B 图中蓝色表示耦合后变浅，红色表示局部变深；连续的蓝色带通常对应入口密集区和排水路径。B-A 使用更窄的色标，因为糙率效应远小于管网效应。色标采用百分位裁剪只是为了看清纹理，统计量仍使用完整值。

## 3. DrainLite 方法与验证设计

模型预测的是有符号残差 r=C-B，再把它加回 B。时空先验由训练事件逐时逐格平均得到；测试事件从先验中完全排除，训练样本自己的事件也从该样本先验中排除。动态特征包括当前 B 水深、降雨、累积降雨、地形、时间和掩膜感知的 3 x 3、7 x 7 邻域水深。静态特征包括入口、管线、密度、管径、坡度、满流能力和覆土。坐标、排放口距离和目标侧 SWMM 动态状态均不使用。

验证采用八折留一事件法。完整模型与“先验+动态”模型分别用五个随机空间采样种子训练，以检查 0.1 毫米量级的管网增量是否只是抽样偶然性。树数量固定为 140，不再用某一个内部事件挑选。最终模型还要接受网络平移、块打乱、整组置换和静态字段清零。

## 4. 结果及逐图解读

### 4.1 总体精度和模型组成

{performance_table(ev)}

{figure(7, 'fig07_skill_and_sampling_seeds', '完整事件留出结果。三个子图依次是平均绝对误差、均方根误差和 0.15 米阈值的临界成功指数。')}

**图 7 逐项解释。** 子图 (a) 柱越低越好。Surface B 是完全不修正的起点；ST prior 表示只重复其他事件的平均响应；Dynamic (no prior) 检验没有固定模板时动态变量能学到多少；后续三组逐步加入管网信息。子图 (b) 对少量大误差更敏感，因此数值通常高于子图 (a)。子图 (c) 越高越好，表示超过 0.15 米的淹没区域是否重合。误差棒只出现在做了五随机种子的关键模型上，不能把规范种子单次消融误读成无随机波动。

先验从 {value('surface_matched', 'mae_mm'):.3f} 毫米降到 {value('prior_only', 'mae_mm'):.3f} 毫米，说明同一地形和网络产生了很强的重复空间模板。动态变量进一步降到 {value('hybrid_dynamic', 'mae_mm'):.3f} 毫米，说明事件降雨强度和当时地表状态仍然必要。完整模型为 {value('hybrid_all', 'mae_mm'):.3f} 毫米；因此管网静态字段有贡献，但不是主要精度来源。

### 4.2 管网信息是否真的被使用

{event_table(ev)}

{figure(8, 'fig08_final_hybrid_network_controls', '最终 hybrid 的管网空间对照。子图 (a) 为平移距离，(b) 为清零、块打乱和整组置换，(c) 为管网特征分组，(d) 为五随机种子的静态增量。')}

**图 8 逐项解释。** 子图 (a) 的纵轴是“错位模型 MAE 减去正确对齐模型 MAE”，高于零表示错位有害。随距离增大而上升，比单独一个 20 米对照更能说明模型依赖正确位置。子图 (b) 中，清零测试的是显式静态字段的增量，但时空先验仍保留原网络平均效应；块打乱保留局部数值分布却破坏大片空间组织；整组置换同时破坏位置。子图 (c) 比较入口/管线掩膜与管径/坡度等水力组。子图 (d) 每根柱对应一个采样种子，若都在零以上，说明小增量不是单一种子的偶然结果；若接近或跨过零，则必须降低结论强度。

正确静态字段相对先验+动态的五种子增量为 {network_increment.mean():.3f} +/- {network_increment.std(ddof=1):.3f} 毫米。清零、块打乱和整组置换的平均惩罚分别为 {np.mean(control['zero_static_network_fields']):.3f}、{np.mean(control['block_shuffle']):.3f} 和 {np.mean(control['group_permutation']):.3f} 毫米。它们支持“正确对齐有用”，但由于先验仍编码固定网络，不能据此声称可泛化到新管网。

{figure(6, 'fig06_fixed_network_residual_maps', '四场留出事件的空间残差。每行依次为 B 输入、真实 C-B、预测残差和最终深度误差。')}

**图 6 逐项解释。** 第一列帮助读者把残差放回原积水背景：深水区不一定是排水变化最大的区。第二列蓝色为真实削减、红色为真实局部增水。第三列若能恢复第二列的连续条带和热点，说明模型不仅降低了平均误差，也恢复了位置。第四列接近白色代表预测与 C 接近；蓝红边缘说明模型对陡峭残差边界存在平滑或错位。每行色标按本事件设置，不能拿不同事件的颜色深浅直接比较数值。

### 4.3 排水真正发生的区域

{conditional_table(ev)}

{figure(9, 'fig09_conditioned_performance', '条件误差：湿区、强管网效应区和不同入口距离。')}

**图 9 逐项解释。** 子图 (a) 说明把评价限制到 3 厘米或 15 厘米以上积水后，误差会高于全域平均，因为干网格不再稀释结果。子图 (b) 逐步筛选 |C-B| 大于 1、5、10 毫米的网格，阈值越高表示管网改变越强，也越难预测。子图 (c) 按入口距离分组，0-20 米是交换最集中的区域。关键不是完整模型在每组都只有 2 毫米，而是它相对 B 是否持续降低误差，以及在哪些组仍然偏高。

### 4.4 与 MIKE 的外部对照

{mike_table(ev)}

{figure(10, 'fig10_mike_metric_tradeoff', 'B、C 和 DrainLite 相对 MIKE 的四类指标。')}

**图 10 逐项解释。** 四个子图采用相同的模型行顺序和颜色，圆点的横坐标表示指标数值，旁边直接标注该数值；浅灰引导线仅帮助读数。子图 (a) 和 (b) 衡量逐格逐时水深与逐格峰值误差，圆点越靠左越好；B 在这些指标上可能更接近 MIKE。子图 (c) 衡量六小时后总体存水量误差，圆点越靠左越好，C 和 DrainLite 通常更接近，说明排水纠正了过多存水。子图 (d) 是 0.15 米淹没范围重合度，圆点越靠右越好。四图排序不一致不是统计错误，而是说明概化管网修正了总体水量，却没有复制 MIKE 中未公开的真实管网空间分配。报告因此不把 MIKE 称为独立验证真值。

### 4.5 时间、裁剪和事件覆盖

{figure(11, 'fig11_runtime_clipping_event_selection', '运行时间、后处理敏感性和事件覆盖。')}

**图 11 逐项解释。** 子图 (a) 用细横条把特征生成、矩阵组装、树模型预测和非负处理分别计时，横条长度表示秒数，防止只报告最快的 `predict()`。子图 (b) 用圆点在对数横轴上比较 B、B+DrainLite 和 C，越靠左表示耗时越短，真正部署时间是 B 与完整修正之和。子图 (c) 用横向点图比较不裁剪、正负 500 毫米和正负 1200 毫米残差，横坐标为平均绝对误差，越靠左越好；主结果不做残差裁剪。数值标注用于辨认很小的差异，不放大坐标范围来夸大差别。子图 (d) 横轴为平均六小时降雨，纵轴为 MIKE 全域峰值，橙点是有完整 A/B/C 的八场事件，灰点是只有公共强迫和参照的事件。它直观展示样本覆盖，而不是把八场事件说成全部公开数据。

### 4.6 与 LarNO 的关系

{figure(12, 'fig12_larno_reproduction', '公共 LarNO checkpoint 的 event68 独立复现。')}

**图 12 逐项解释。** 第一幅是真实 MIKE 水深，第二幅是 LarNO 输出，二者共用色标；第三幅是有符号误差。原始输出有 {ev['larno']['negative_fraction']*100:.2f}% 的负值，因此报告同时保留未截断和截断指标。DrainLite 没有直接叠加到这个 checkpoint 上，因为该 LarNO 已用可能包含排水的 MIKE 标签训练，再加 C-B 可能重复计算排水。当前论文的主结果来自 B 到 C，而不是 LarNO 到 C。

## 5. 讨论、结论与下一步工作

本研究最稳妥的结论是：在一个固定、已检查连通性的概化管网上，管网效应可以用低算力残差模型跨降雨事件模拟；主要信息来自其他训练事件的固定时空响应和当前地表状态，静态管网字段提供小而可检测的附加信息。条件指标证明改进并非只来自大量干网格，最终模型空间破坏对照证明这部分小增量依赖正确位置。

限制同样明确。网络不是真实管网，只有八场事件和一个网络；闭边界、建筑降雨全域重分配和 1 毫米每小时有效损失都是假设；现有物理敏感性还不足以证明 C-B 对这些假设完全稳健；模型需要当前 B 场。下一阶段最有价值的工作不是继续堆叠静态特征，而是构造多套管径、入口密度和排放口布局，做“留一管网”验证，并在没有目标网络时空先验的条件下检验可干预性。随后才能使用无排水标签训练 LarNO 上游模型，形成真正端到端的 LarNO-DrainLite。

## 附录：公共事件覆盖表

{event_selection_table(ev)}
"""


def integrity_audit(ev: dict[str, object]) -> str:
    evidence_files = [
        PHYS, NETWORK, LARNO, EXP / "experiment_metadata.json",
        EXP / "metrics" / "event_metrics_all_seeds.csv",
        EXP / "metrics" / "final_hybrid_network_controls.csv",
        EXP / "metrics" / "conditional_metrics_canonical_seed.csv",
        EXP / "metrics" / "runtime_end_to_end.csv",
        EXP / "metrics" / "runtime_uncached_canonical.csv",
        V4 / "physical_sensitivity_event68/physical_sensitivity_metrics.csv",
    ]
    rows = [[str(path.relative_to(ROOT)), path.stat().st_size, sha256(path)] for path in evidence_files]
    return f"""# Scientific integrity and reproducibility audit

## 1. Audit purpose

This document identifies the evidence behind the manuscript and separates calculated results from interpretation. It is not a response letter. The audit can be checked independently using the read-only repository verifier.

## 2. Provenance boundary

- Public LarNO rainfall, terrain, MIKE fields and checkpoint are upstream artifacts and are not presented as original calculations.
- The conceptual network, matched A/B/C Itzï-SWMM arrays, DrainLite estimators, predictions, controls, statistics and V4 figures were generated within this project.
- MIKE is never a DrainLite target, fitting input, sample weight or hyperparameter-selection variable.
- The public LarNO event68 output is reproduced separately and is not corrected by the principal model.
- OpenStreetMap-derived road geometry is attributed to © OpenStreetMap contributors and remains subject to source terms.

## 3. Scientific claims tied to evidence

The physical claim uses `physics_quality.csv` and the canonical V3 arrays. The learning claim uses complete V4 held-event predictions and `event_metrics_all_seeds.csv`. The static-network claim is limited to the measured increment between `hybrid_dynamic` and `hybrid_all` across five sampling seeds and to controls applied to the final hybrid. It is not described as unseen-network generalisation. Runtime starts before feature construction. The primary prediction does not clip residuals; non-negative reconstructed depth and optional clipping are reported separately.

## 4. Leakage controls

The outer unit is a complete rainfall event. The held event is absent from model fitting and its spatiotemporal prior. A fitting event is also removed from the prior attached to its own sampled rows. Explicit coordinates, synthetic outfall distance and target-side SWMM dynamics are excluded. Model complexity is fixed at 140 iterations before fitting. Unit tests exercise the masked neighbourhood, own-event prior exclusion and zero-network control.

## 5. Physical quality conditions

The declared thresholds are absolute routing continuity <= 2%, non-converging routing steps <= 2%, absolute combined mass error <= 0.5% of rainfall, zero SWMM flooding loss, zero warnings/errors, finite 72 x 400 x 560 arrays and complete directed outfall reachability. All eight canonical events pass. These are project acceptance criteria, not field calibration.

## 6. Evidence hashes

{table(['Evidence file', 'Bytes', 'SHA-256'], rows)}

## 7. Known unresolved evidence

No surveyed sewer inventory is available. No held-network experiment has been completed. Events outside the eight paired calculations do not have matched A/B/C labels; event79 is unreadable locally. Physical sensitivity to building-rainfall routing, effective loss and Manning roughness has not been repeated as a complete multi-event coupled factorial experiment. Author identities, affiliations, contributions, funding and target-journal metadata remain to be supplied. The upstream LarNO snapshot states MIT in its README but lacks a root licence file at the audited commit; dataset and checkpoint redistribution rights therefore require confirmation.

## 8. Verification commands

```bash
python -m unittest discover -s tests -v
python scripts/verify_repository.py
```

Verification is read-only. Evidence regeneration is a separate explicit command and changes committed hashes.
"""


def main() -> int:
    ev = load_evidence()
    OUT.mkdir(parents=True, exist_ok=True)
    documents = {
        "manuscript.md": manuscript(ev),
        "report.md": report(ev),
        "scientific_integrity_audit.md": integrity_audit(ev),
    }
    sensitivity = read_csv(V4 / "physical_sensitivity_event68/physical_sensitivity_metrics.csv")
    sensitivity_table = table(
        ["Case", "Mean |C-B| (mm)", "Final B-C volume (m3)", "C vs MIKE MAE (mm)", "Routing error (%)"],
        [[row["case"], f"{float(row['drainage_effect_mae_mm']):.3f}",
          f"{float(row['final_surface_reduction_m3']):.0f}",
          f"{float(row['C_vs_MIKE_mae_mm']):.3f}", row["routing_continuity_error_pct"]]
         for row in sensitivity])
    sensitivity_english = """### 3.8 Paired physical sensitivities for event 68

Five alternatives were calculated with paired B and C runs, changing one setting at a time: effective loss of 0 or 2 mm h-1, nearest-active-cell redistribution of building rainfall, exclusion of building rainfall, or inlet-neighbourhood Manning coefficient 0.015. The baseline retained 1 mm h-1 loss, global redistribution and coefficient 0.012. All alternatives produced finite 72-frame fields. Routing continuity magnitudes were below 0.2%; these diagnostics alone do not establish physical validity.

The mean drainage effect was 11.914 mm at baseline, 12.993 mm without effective loss and 10.874 mm at 2 mm h-1. Nearest-cell redistribution reduced it to 7.799 mm, and exclusion to 3.465 mm; the latter also removes rainfall volume and cannot isolate spatial routing alone. Changing inlet roughness gave 11.894 mm. Thus the existence of a drainage response persisted, while its magnitude depended substantially on rainfall treatment. C-to-MIKE MAE increased from 24.023 to 35.080 mm under nearest-cell redistribution, showing that local redistribution does not automatically improve agreement. This single-event experiment supports a sensitivity statement, not calibration or robustness of the trained residual model across physical configurations.

""" + sensitivity_table + "\n\n" + figure(13, "fig13_physical_sensitivity", "Paired event68 physical sensitivities. Panels (a-d) show mean absolute drainage response, final surface-volume reduction, coupled-field discrepancy from MIKE and signed SWMM routing continuity. Loss values are in mm per hour. Exclude removes building rainfall volume; nearest changes its spatial allocation. Numerical continuity is not evidence of surveyed-network accuracy.") + "\n\n"
    documents["manuscript.md"] = documents["manuscript.md"].replace("## 4. Discussion", sensitivity_english + "## 4. Discussion")
    sensitivity_chinese = """### 4.7 物理假设的配对敏感性

这一补充实验回答“管网效应是否依赖降雨施加和局部糙率设置”。在 event68 中分别改变有效损失、建筑降雨处理或入口糙率，每种设置都重新计算 B 和 C。八事件训练数据保持原设置，本实验没有重新训练模型，因此不能据此推断模型可适应新的物理参数。

""" + sensitivity_table + "\n\n" + figure(13, "fig13_physical_sensitivity", "Event68 配对物理敏感性：管网响应、水量削减、MIKE 差异和管网连续性。") + """

**图 13 逐项解释。** 子图 (a) 的柱高表示整个事件中耦合造成的平均绝对水深变化，越高表示管网影响越强，不代表预测越准确。子图 (b) 是六小时末 B 减 C 的存水体积，正值表示耦合后地表存水更少。子图 (c) 表示 C 与 MIKE 的逐格逐时差异，越低越接近外部参照。子图 (d) 是 SWMM 水量记账误差，接近零说明数值连续性较好，但不能证明真实管网被准确重建。

基准平均管网效应为 11.914 毫米；有效损失改为 0 或 2 毫米每小时后分别为 12.993 和 10.874 毫米。最近活动网格接收建筑降雨时为 7.799 毫米，排除建筑降雨时为 3.465 毫米；后者同时减少总降雨输入，不能解释为单纯空间分配差异。入口糙率改成 0.015 后为 11.894 毫米，变化很小。最近邻方案相对 MIKE 的误差升至 35.080 毫米，说明更局地的施雨方式不一定在当前概化地形上更接近 MIKE。这些结果揭示了不确定性来源，不能用来反向挑选参数并声称已经完成校准。

"""
    documents["report.md"] = documents["report.md"].replace("## 5. 讨论、结论与下一步工作", sensitivity_chinese + "## 5. 讨论、结论与下一步工作")
    # Place residual maps before the aggregate skill figures in reading order.
    for name in ("manuscript.md", "report.md"):
        content = documents[name]
        start = content.index("![Figure 6.")
        end_marker = "### 3.4" if name == "manuscript.md" else "### 4.3"
        end = content.index(end_marker, start)
        block = content[start:end]
        content = content[:start] + content[end:]
        target = "![Figure 7."
        content = content.replace(target, block + target, 1)
        complete = []
        for (reference, model), row in ev["summary"].items():
            for metric in row:
                if metric.endswith("_mean"):
                    sd = row.get(metric.removesuffix("_mean") + "_sd", "")
                    complete.append([reference, model, metric.removesuffix("_mean"),
                        f"{float(row[metric]):.6g}", f"{float(sd):.6g}" if int(row["n_seeds"]) > 1 else "Not estimated"])
        content += "\n\n## Supplement. Complete metric disclosure\n\nAll metrics use equal event weights. Repeated-seed variability is not an event-population confidence interval. Deterministic B and prior baselines are repeated identically across seeds. Single-seed subgroup results cannot establish a seed-averaged ranking.\n\n" + table(["Reference", "Model", "Metric (unit in name)", "Mean", "Seed SD"], complete)
        documents[name] = content
    for name, content in documents.items():
        captions = {
            "| Event | Rain volume": "Physical quality of the eight paired events",
            "| Model | Seeds": "Whole-event model comparison; uncertainties refer to sampling seeds",
            "| Held event |": "Canonical-seed held-event errors and network increments (mm)",
            "| Evaluation subset |": "Conditional depth MAE (mm), macro-averaged over events",
            "| Field | MAE": "External comparison with MIKE",
            "| Case | Mean": "Event68 paired physical sensitivities",
            "| Event | Paired": "Locally readable public event inventory",
            "| Evidence file |": "Evidence files and content hashes",
        }
        lines, numbered, previous_table = content.splitlines(), [], False
        table_number = 0
        for line in lines:
            is_table = line.startswith("| ")
            if is_table and not previous_table:
                table_number += 1
                title = next((v for k, v in captions.items() if line.startswith(k)), "Complete calculated metrics")
                numbered.extend([f"**Table {table_number}. {title}.**", ""])
            numbered.append(line)
            previous_table = is_table
        content = "\n".join(numbered)
        (OUT / name).write_text(content.strip() + "\n", encoding="utf-8")
        print(OUT / name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
