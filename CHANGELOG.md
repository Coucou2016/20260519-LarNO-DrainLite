# Changelog

This file records scientific revisions rather than every exploratory command.
Superseded exploratory outputs remain local but are not treated as canonical
evidence.

## v4.1 - 2026-09-21

- Restyled Figures 10 and 11 only: aligned horizontal dot comparisons,
  slim runtime component bars, direct value labels and compact panels.
- Updated figure-reading guidance and embedded figures in manuscript/report
  exports. Numerical inputs, aggregation, experiments and conclusions unchanged.

## v4.0 - 2026-09-19

- Repositioned DrainLite as a fixed-network drainage residual emulator rather
  than an end-to-end rainfall-to-flood or unseen-network generalization model.
- Replaced single-seed final evaluation with five-seed, eight-fold whole-event
  validation using a fixed iteration count and leakage-controlled priors.
- Added final-hybrid network shift, block-shuffle, group-permutation, mask-only,
  hydraulic-only, and zero-network-field controls.
- Added wet-cell, near-network, strong-effect, drainage, and surcharge
  conditioned metrics, together with complete peak, volume, area, CSI, and
  timing metrics.
- Changed neighbourhood summaries to active-mask-aware filtering and removed
  residual clipping from the primary prediction path; retained clipping only as
  an explicit sensitivity analysis.
- Split feature construction, estimator prediction, postprocessing, and total
  correction runtime, and added the surface-plus-correction comparison against
  the coupled simulation.
- Separated evidence generation from read-only verification, expanded Git LFS
  checks to all tracked objects, and added fresh SWMM-input topology auditing.
- Added path-scoped licensing, citation metadata, an environment lock, portable
  path handling, unit tests, and a lightweight continuous-integration workflow.
- Completed 40 event/seed folds. Full-hybrid MAE is 2.169 +/- 0.008 mm;
  the paired static-network increment is 0.048 +/- 0.002 mm.
- Completed five paired event68 physical sensitivities. Building-rainfall
  treatment materially changes drainage magnitude; inlet roughness has a much
  smaller effect. These are sensitivity experiments, not field calibration.
- Remeasured inference including uncached dynamic neighbourhood computation;
  checked reconstructed arrays against saved predictions with no difference.
- Rebuilt 13 figures and the manuscript, report and integrity audit, including
  complete metric disclosure and six-decimal small-effect reporting.

## v3.0 - 2026-09-14

- Changed `Coucou2016/20260519-LarNO-DrainLite` to public visibility at the
  repository owner's request and updated the publication audit accordingly.
- Prepared the complete GitHub research snapshot for independent Agent review.
- Added repository-level provenance, reproduction, data-manifest, third-party,
  environment, and integrity-verification documents.
- Selected one canonical copy of each large scientific artifact and excluded
  byte-duplicate arrays, installers, browser profiles, and temporary builds.
- Added Git LFS tracking for arrays, estimators, and neural-network weights.
- Retained the final 107-check acceptance record and per-file SHA-256 manifest.

## v2.3 - final reviewer revision

- Rewrote the manuscript in journal form without changing calculated results.
- Rebuilt all 11 figures with SciencePlots and Times New Roman and completed
  multiple visual inspections of maps, scales, legends, labels, and layouts.
- Generated Markdown, standalone Base64 HTML, and PDF versions of the
  manuscript, Chinese research report, and scientific-integrity audit.
- Independently reproduced LarNO checkpoint predictions for event 68 and
  documented negative raw predictions and clipping sensitivity.

## v2.2 - leakage-controlled DrainLite

- Replaced random row validation with complete-event inner validation.
- Implemented eight-fold leave-one-event-out evaluation.
- Removed row/column coordinates and target-side dynamic SWMM states from model
  inputs.
- Fitted each spatiotemporal prior without the held-out event.
- Added a spatiotemporal-climatology null model and a dynamic-only ablation.

## v2.1 - formal matched physical experiment

- Defined cases A, B, and C so that `C - B` isolates drainage exchange and
  `B - A` identifies the inlet-neighbourhood roughness effect.
- Added a 1 mm/h effective rainfall loss consistently across cases.
- Recomputed all eight 6 h events at 20 m and 5 min output intervals.
- Promoted the coupled outputs to the canonical `drainage_v3_full` dataset.

## v2.0 - connected conceptual sewer

- Rebuilt the road-aligned network around a connected 2,276-node main graph.
- Added 2,276 conduits, one receiving outfall per junction, and 221 surface
  exchange interfaces.
- Removed isolated nodes, disconnected links, duplicate parallel links,
  reverse slopes, and crown-elevation violations.
- Fixed SWMM dynamic-wave and Itzi exchange parameters after stability testing.

## v1.x - exploratory development

- Compared depression filling, surface-only Itzi, conceptual sink drainage,
  standalone SWMM, and early coupled prototypes.
- Diagnosed DEM/road orientation and translation issues and regenerated the
  network from the accepted road alignment.
- Produced preliminary MIKE comparisons, reports, and DrainLite prototypes.
- These outputs informed v2 but are not used as final quantitative evidence.
