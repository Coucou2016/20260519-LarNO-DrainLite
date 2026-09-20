# Scientific integrity, provenance and reproducibility audit

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

The accepted network contains 2276 junctions, 2276 conduits and 221 receiving interfaces. Directed outfall reachability is 100.0%. Independent parsing found zero isolated junctions, cycles, duplicate undirected endpoint groups, reverse-slope conduits and crown-above-rim endpoints. SWMM ponding is disabled; surface overflow and return are handled by the 2D coupling.

## 6. Numerical checks

All eight events have zero SWMM flooding loss, zero warning count and zero error count. Mean routing continuity error is -0.0912% and mean non-converging-step frequency is 0.0113%. The maximum absolute combined mass error is 0.1107% of rainfall volume.

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

The matched surface baseline MAE to C is 9.265926 mm. The spatiotemporal prior is 3.225513 mm, prior plus dynamics is 2.183922 mm and the complete hybrid is 2.129864 mm. Static descriptors improve all eight event folds; the macro-average paired increment is 0.054058 mm with an event-bootstrap interval of 0.025694-0.082733 mm.

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

| Evidence | Repository-relative path | Bytes | SHA-256 |
| --- | --- | --- | --- |
| Accepted SWMM network | extended_study\output\reviewer_major_revision_v3\network\swmm_bidirectional_normal_step05.inp | 510702 | 965978312ffa3959c21826943aa5526b90e0834b08f7698dc6455aba5f990be2 |
| Physical quality table | extended_study\output\reviewer_major_revision_v3\formal_matched_full\physics_quality.csv | 12539 | abd4736235f314b1117d04a066126e43087f0b4cbf2a7f525017183c775c40c1 |
| Dataset audit | LarNO-main\benchmark\urbanflood\flood\region1_20m_drainage_v3_full\dataset_audit.json | 464 | 30b08a0f252daedcb05bb4ea8d6c333f798d06d9c470233570c9028eddabaa16 |
| Original DrainLite summary | extended_study\output\reviewer_major_revision_v3\drainlite_v3\metrics\summary_coupled.csv | 1953 | e5773bedc62bc8bb290fbd0903592f93c44dbe2a68b03dbec6f269f2d1bb64f9 |
| Hybrid metadata | extended_study\output\reviewer_major_revision_v3\drainlite_hybrid_v3\experiment_metadata.json | 1714 | af0b5d1b2e11ce94af884b4a8c4a1be4e42ca095906dc5ef751753cb9e8b8c79 |
| Hybrid summary | extended_study\output\reviewer_major_revision_v3\drainlite_hybrid_v3\metrics\summary_coupled.csv | 702 | bcc6ebd647f2bf6b3c21a7e7087856dcbd2c979b37b41d30cfcd75e81d9fb097 |
| LarNO event68 prediction | LarNO-main\exp\20260220_183648_006352\pred_results\region1_20m\epoch_992\predictions_epoch_992_sample_event68.npy | 64512128 | b33c9b84facd93e0da1847f8139fab554be2f937b339b22334832a899e650fef |

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
