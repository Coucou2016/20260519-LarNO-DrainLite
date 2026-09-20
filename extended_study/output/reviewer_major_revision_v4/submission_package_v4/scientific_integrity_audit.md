# Scientific integrity and reproducibility audit

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

**Table 1. Evidence files and content hashes.**

| Evidence file | Bytes | SHA-256 |
| --- | --- | --- |
| extended_study\output\reviewer_major_revision_v3\formal_matched_full\physics_quality.csv | 12539 | abd4736235f314b1117d04a066126e43087f0b4cbf2a7f525017183c775c40c1 |
| extended_study\output\reviewer_major_revision_v3\network\swmm_bidirectional_normal_step05.independent_audit.json | 2216 | 9b4d76d0b9fdcd21c23424334a597aca66a2b0ec65c63c69d5229595a95e50c1 |
| extended_study\output\reviewer_major_revision_v3\larno_event68_checkpoint_audit.json | 1280 | 030af0f33d876b0ee8e6c79990402fe4d3056571f54185cbea9e5b3dfc66a538 |
| extended_study\output\reviewer_major_revision_v4\final_hybrid_controls\experiment_metadata.json | 2300 | 203add1005ffe769f1e9bab15c653b6c50d4d4f86ed05997661e9c94661841d8 |
| extended_study\output\reviewer_major_revision_v4\final_hybrid_controls\metrics\event_metrics_all_seeds.csv | 137771 | 75e177a7281a3140348173ae892a7d25192695a2483c5921f22fca69a4d3f12b |
| extended_study\output\reviewer_major_revision_v4\final_hybrid_controls\metrics\final_hybrid_network_controls.csv | 145064 | f51d2906061ee162aefd1e7adb015eba763a4bb4a444fa68c3fa007820aaa55a |
| extended_study\output\reviewer_major_revision_v4\final_hybrid_controls\metrics\conditional_metrics_canonical_seed.csv | 212168 | e1a7842b5366a6dbbbee9310724d2410f6d899d80574cb6ec81ef9e07e57228f |
| extended_study\output\reviewer_major_revision_v4\final_hybrid_controls\metrics\runtime_end_to_end.csv | 24790 | f36bd02fb9f81d6ebb3f3e705865ce099b36b806bd43e12a579a353e33ddc9da |
| extended_study\output\reviewer_major_revision_v4\final_hybrid_controls\metrics\runtime_uncached_canonical.csv | 2461 | b0637096a0eade804a043ef6f170f8c8529c49819624ec50c0051a846c5818f3 |
| extended_study\output\reviewer_major_revision_v4\physical_sensitivity_event68\physical_sensitivity_metrics.csv | 1523 | d0a2cf7d8e63284bb323d564e0b60fc4a2887509dd0d92da615497a2f37cd6c1 |

## 7. Known unresolved evidence

No surveyed sewer inventory is available. No held-network experiment has been completed. Events outside the eight paired calculations do not have matched A/B/C labels; event79 is unreadable locally. Physical sensitivity to building-rainfall routing, effective loss and Manning roughness has not been repeated as a complete multi-event coupled factorial experiment. Author identities, affiliations, contributions, funding and target-journal metadata remain to be supplied. The upstream LarNO snapshot states MIT in its README but lacks a root licence file at the audited commit; dataset and checkpoint redistribution rights therefore require confirmation.

## 8. Verification commands

```bash
python -m unittest discover -s tests -v
python scripts/verify_repository.py
```

Verification is read-only. Evidence regeneration is a separate explicit command and changes committed hashes.
