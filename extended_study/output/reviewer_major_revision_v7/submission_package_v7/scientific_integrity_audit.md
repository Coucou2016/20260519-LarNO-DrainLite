# V5 scientific integrity and reviewer implementation audit

## Evidence boundary

Frozen source commit: `d5fefaa0cf7a31cf7d29830ed5b070e1d2dac3a2`. Current physical arrays, held-event predictions and training labels were not altered. MIKE rainfall/terrain/reference arrays and the public LarNO checkpoint are upstream materials, clearly attributed; A/B/C coupled simulations, fitted residual estimators and their diagnostics are this project's calculations. The manuscript does not claim all data are original observations.

## Verified findings

The raw rainfall sum and active-cell mean use different masks. Geometric slope inspection finds no zero/negative conduit slope and 626 values near the design floor. No outfall has its own coordinate record. Spatial maps mark connecting terminals only. Peak-map differences are not instantaneous residuals. The held-event network increment is 0.047773 mm, event SD 0.037333 mm and sampling-seed macro SD 0.001819 mm. All are computed, not borrowed from the LarNO paper.

## Implementation status

| Review item | Status | Evidence or limitation |
| --- | --- | --- |
| Rainfall mask and volume | Recomputed | 576 intervals; verified numerical redistribution, not surveyed roof mask |
| Peak-map meaning | Corrected | max(C)-max(B) explicitly separated from C(t)-B(t) |
| Slope zeros and outfalls | Recomputed | 0 nonpositive; 626 near floor; outfall coordinates absent |
| Event and seed uncertainty | Recomputed | 40 event-seed gains; event bootstrap conditional on exchangeability |
| Inference perturbations | Reframed | OOD sensitivity separated from retrained feature benefit |
| Native exchange formula/sign | Unit check only | Prescribed-head test double; no routed pipe benchmark |
| Full joint internal-step water ledger | Pending | 30-minute cumulative records and final report only |
| Broader paired events / weather independence | Pending | Nine readable cases unpaired; dates/families unverified |
| Multi-event rainfall treatment / network capacities / tailwater | Pending | Only existing single-event setting sensitivities completed |
| Coarse prior / shifted rain / cross-scenario retraining | Pending | No new predictive claims |
| Unclamped depth error changes and extreme LarNO tails | Partially addressed | LarNO full-range maps and extremes rebuilt; clamp counts retained; full raw-depth change diagnostics still need inference replay |
| Authors and redistribution rights | Author action | Do not invent author declarations or upstream permission |

## Exact numerical evidence

| File | SHA256 |
| --- | --- |
| extended_study/output/reviewer_major_revision_v5/diagnostics/event_seed_gain.csv | 38a470bd1594e725bdce7cfa3799018a417986646b2ea70a983ac0e83a6397a0 |
| extended_study/output/reviewer_major_revision_v5/diagnostics/hydrograph_diagnostics.csv | d650c81cee3961e409672859b3a2eae4e95139ed320ab504b7763b93a59cf2c8 |
| extended_study/output/reviewer_major_revision_v5/diagnostics/larno_extremes.json | 1593c5d449e64c7b9bec5fbad13e9310b1bd3a1a4eb1f1796bb2214d3e482428 |
| extended_study/output/reviewer_major_revision_v5/diagnostics/map_display_audit.csv | 8d9d7b3d1190d4f5b1727e4274f56af29a4800472f184104d406a709a2d80ebe |
| extended_study/output/reviewer_major_revision_v5/diagnostics/native_exchange_unit_check.json | ef1f6529a83bec2b1b036952b12c66d62c718779d178d379db5a21281eb73f5c |
| extended_study/output/reviewer_major_revision_v5/diagnostics/network_geometry.json | 4f694b48565d4e5f3a045afdefc8650d5ffe4faa85cca9756dbbfde62e0f46f8 |
| extended_study/output/reviewer_major_revision_v5/diagnostics/rainfall_event.csv | 9fafcafc913163ff5342fe24622daae99a62dbba229939cce0b828921af6cec5 |
| extended_study/output/reviewer_major_revision_v5/diagnostics/rainfall_steps.csv | 927a98aa697bf07c0d02e5d4f9aaa008c4e20f2a4b00c5a616f662eea01f00bf |
| extended_study/output/reviewer_major_revision_v5/diagnostics/signed_residual_diagnostics.csv | e17a7c27b821c13d73f4ac5cc8a288e1ef2f6fb38242c703cf1c7d649ef9af50 |
| extended_study/output/reviewer_major_revision_v5/diagnostics/summary.json | 3953ac835be6b93713e6efc661dbcea8432a2f7794c452848a3b17b40fa6f711 |
| extended_study/output/reviewer_major_revision_v5/diagnostics/water_ledger_saved_times.csv | f694e6605b502d64038af47257aea863c4637c10e8d388fed95dd9176598a6b6 |

## Reproduction commands

```bash
python extended_study/audit_reviewer_v5_evidence.py
python extended_study/test_v5_native_exchange.py
python extended_study/generate_reviewer_v5_figures.py
python extended_study/generate_reviewer_v5_documents.py
python extended_study/render_reviewer_v5_documents.py
```

Primary training remains `extended_study/run_final_hybrid_major_revision.py` (fixed 140 iterations, no residual clipping). Historical V3 tuning is not a reproduction command for the V4/V5 primary results. Figures 1/2/6/7/9/10/11 and supplementary maps are rebuilt by the V5 figure script; unchanged physical-quality, hydrograph and peak-map illustrations originate from the frozen V4 package. Figure 5's data formula is audited in `plot_reviewer_v3_physics.py`. Full CSV evidence remains alongside concise manuscript tables; removing duplicate plots does not remove inconvenient results.

## Writing sources and source verification

The revision follows concrete, evidence-led prose and distinguishes methods, results and interpretation. It does not manufacture an author voice, field observations or stronger conclusions to evade AI detection. Sources consulted: Manchester Academic Phrasebank (https://www.phrasebank.manchester.ac.uk/), KKKKhazix/human-writing (https://github.com/KKKKhazix/human-writing), petergyang/no-ai-slop (https://github.com/petergyang/no-ai-slop). These are editorial guidance, not scientific evidence.

MIKE's coupled origin is established in Section 4.1 of the supplied Cao et al. paper, DOI 10.1016/j.jhydrol.2026.135686. The current public model card documents 13 released input channels and the 64/16 Futian split (https://github.com/holmescao/LarNO/blob/main/huggingface/model_card.md). Current Itzi stable documentation resolves to 26.6, while the frozen calculations use 25.4; the current documentation version must not be represented as the executed engine. Normal-depth outfalls are conceptual hydraulic boundary conditions, not measurements of receiving-water levels.

## Submission readiness

This revision improves presentation and evidence transparency. It does not close the pending physical validation and scenario-generalisation requests. Numerical consistency checks alone are not grounds to declare the study calibrated or ready for unconditional submission.


## V6 editorial change record

The physical arrays and learned predictions are unchanged. Figure 2 is copied byte-for-byte; Figures 3-5 retain data and layouts with terminology-only label changes. Figure 6 retains its fields, time selection and colour bounds. The skill plot is removed because Table 2 already reports its results. The event scatter is replaced by a full supplementary inventory. Sensitivity time series are recomputed from the six existing paired simulations, and their temporal mean absolute residual and final-volume differences are checked against the original summary CSV (tolerances 0.001 mm and 1 m3). Figure labels use LarNO; implementation provenance remains in the supplement.

Terminology source: Fan et al. (2017), https://doi.org/10.1155/2017/2819308. The established term coupled 1D-2D hydrodynamic model motivates the consistent 1D/2D wording; it does not imply independent validation of this implementation. Event68 rationale describes measured characteristics retrospectively, not a fabricated prospective selection rule. Pending physical experiments remain pending.


## V7 correction to prior quality claims

# Numerical quality assessment

Global routing continuity errors range from -0.178% to -0.060%, and reported nonconverging steps from 0.00% to 0.04%. Continuity error is a numerical water-budget residual, not a disconnected-pipe count. A nonzero residual is not automatically a model failure, but small global values do not establish local accuracy. In event68 the reported node N01520 has a 63.01% inflow-normalised local imbalance. Its rounded cumulative inflow is about 114 m3, corresponding to approximately 71.8 m3 of imbalance; this is not 63% of whole-network water. The same node's inflow table reports 170.331% because that table uses outflow as its denominator. These are different normalisations of the same budget discrepancy, not conflicting results. The existing aggregate screening therefore does not close the numerical validation. The reported local errors must be checked against node-level exchanges and shorter-step results before the labels can be described as locally validated.

| Event | Global balance (%) | Nonconverging steps (%) | Top local node | Local balance (% inflow) | Assessment |
| --- | --- | --- | --- | --- | --- |
| event1 | -0.083 | 0.02 | N00184 | 47.07 | Local review required |
| event20 | -0.178 | 0.0 | N00184 | 69.6 | Local review required |
| event65 | -0.111 | 0.0 | N00184 | 65.45 | Local review required |
| event66 | -0.101 | 0.0 | N00184 | 58.75 | Local review required |
| event67 | -0.066 | 0.01 | N01520 | 65.04 | Local review required |
| event68 | -0.06 | 0.04 | N01520 | 63.01 | Local review required |
| event69 | -0.068 | 0.01 | N01520 | 52.93 | Local review required |
| event70 | -0.063 | 0.01 | N01520 | 56.86 | Local review required |

## Interpretation and required follow-up

The EPA manual (Section 8.5) treats continuity as a numerical diagnostic, calls for investigation of excessive local errors at relevant nodes, and recommends time series at one-minute or shorter resolution for initial instability screening. Its example of a 10% excessive global error is not a universal safe threshold. The repository limits of 2% routing/nonconvergence and 0.5% combined error are project screens, not proof of physical validity. A printed 0.00% is rounded and does not prove zero failed iterations.

A nonconverging-step percentage measures steps that fail the iteration stopping criterion; it is neither a percent of wrong depths nor an estimate of mass loss. The current report prints two decimal places and cannot recover exact failed-step counts or their timing.

Next acceptance: instrument inflow, outflow, storage and signed exchange at the flagged nodes; repeat event68 and the event with the largest local discrepancy with reduced maximum routing steps (0.5, 0.25, 0.125 s) and reviewed head tolerance/iteration settings; compare node hydrographs, local budgets and surface labels. Do not alter the physical network merely to improve a diagnostic. No convergence rerun was executed by this audit, and no label or learned prediction has been replaced.

## Sources

- [EPA manual, Sections 8.5 and 9.1](https://www.epa.gov/system/files/documents/2022-04/swmm-users-manual-version-5.2.pdf)
- [Engine 5.2.4 local ranking formula](https://raw.githubusercontent.com/USEPA/Stormwater-Management-Model/v5.2.4/src/solver/stats.c)
- [Engine 5.2.4 inflow-table formula](https://raw.githubusercontent.com/USEPA/Stormwater-Management-Model/v5.2.4/src/solver/statsrpt.c)
