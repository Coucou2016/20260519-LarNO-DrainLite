# DrainLite: fixed-network drainage residual emulation

This repository is the auditable research snapshot for DrainLite, a lightweight
correction layer that emulates the effect of one fixed, road-aligned conceptual
sewer network on surface-only urban-flood simulations. The principal experiment
uses contemporaneous surface-only Itzi states and does not constitute an
end-to-end rainfall-to-flood LarNO-DrainLite system. It contains the project
code, the configured and quality-controlled Itzi-SWMM workflow copied into this
workspace, the eight-event canonical dataset, leakage-controlled whole-event
predictions, statistics, publication figures, manuscript, standalone report,
and scientific-integrity audit.

The upstream LarNO project and its public benchmark remain credited to the
original authors. The new results in this repository were calculated locally
from the workflows recorded here; they are not copied from the reference paper.

## Headline result

The formal comparison separates three simulations:

- **A, surface-only:** Itzi surface flow with Manning's coefficient 0.015.
- **B, matched surface:** no sewer exchange, with Manning's coefficient 0.012
  only in the same inlet neighbourhoods used by the coupled run.
- **C, coupled:** case B plus native Itzi surface exchange with a connected
  SWMM dynamic-wave network.

The drainage target is the signed residual `C - B`, which avoids attributing
the local roughness change to the sewer. Across eight events, the mean absolute
drainage effect is 9.266 mm. Five-seed whole-event evaluation gives a hybrid
mean absolute error of 2.169 +/- 0.008 mm. Explicit network fields improve on
the prior-plus-dynamic model by 0.048 +/- 0.002 mm across seeds. The principal
gain comes from the event-excluded prior and current surface state. Final-model
shifts, shuffles and conditional metrics are supplied. The event68 paired
physical sensitivities show that rainfall treatment materially changes the
drainage response; these are sensitivity results, not field calibration.

MIKE is used only as an external plausibility reference. It is not a DrainLite
training target. The public LarNO checkpoint is also audited independently and
is not directly corrected by `C - B`, because its MIKE labels were generated
with 1D-2D drainage coupling; the underlying sewer details are not public.

## V5 evidence and presentation revision

The current paper, report, supplement and integrity audit are in
`extended_study/output/reviewer_major_revision_v5/submission_package_v5/`, each
in Markdown, self-contained HTML and PDF. The main paper has 11 figures and
5 tables. Training results remain the frozen V4 five-seed experiment; V5
adds rainfall-mask reconciliation, signed-residual diagnostics, event-level
uncertainty, full-range LarNO maps and explicit peak-map definitions.

The 0.048 mm network increment has event-level SD 0.037 mm, distinct from
the 0.002 mm sampling-seed SD. There are no nonpositive pipe slopes, but
626 of 2,276 pipes lie near the imposed minimum. The barrier mask is a DEM
threshold proxy, not a verified building inventory. Numerical closure does
not establish field calibration. The routed single-pipe benchmark, complete
internal-step joint ledger and multi-event scenario-transfer experiments
remain pending and are listed in the integrity audit.

Read-only verification: `python scripts/verify_reviewer_v5.py`.
Artifact freezing is a separate explicit generation step:
`python scripts/freeze_reviewer_v5.py`. It must not be used to make a failed
verification pass without reviewing and documenting the underlying changes.

## Canonical contents

| Path | Purpose |
|---|---|
| `extended_study/` | Source code for network construction, coupled simulations, DrainLite, figures, documents, and acceptance tests. |
| `external_models/20260518-itzi-flood/` | Non-destructive copy of the formal Itzi-SWMM workflow and Shenzhen configuration. |
| `LarNO-main/code/` | Upstream LarNO source used for checkpoint reproduction. |
| `LarNO-main/benchmark/urbanflood/flood/region1_20m_drainage_v3_full/` | Canonical eight-event dynamic dataset, 72 frames at 20 m. |
| `LarNO-main/benchmark/urbanflood/geodata/region1_20m_drainage_v3_full/` | DEM, masks, and eight rasterized drainage descriptors. |
| `extended_study/output/reviewer_major_revision_v4/final_hybrid_controls/` | Five-seed whole-event models, complete canonical predictions, final-hybrid network controls, conditioned metrics, and end-to-end runtime records. |
| `extended_study/output/reviewer_major_revision_v3/formal_matched_full/` | SWMM reports, dynamic-state diagnostics, event metadata, and physical-quality table. Large duplicate depth arrays are represented once in the canonical dataset. |
| `extended_study/output/reviewer_major_revision_v4/submission_package_v4/` | Current manuscript, report, audit, figures, acceptance JSON, and SHA-256 manifest. |
| `REPRODUCE.md` | Exact verification and regeneration sequence. |
| `DATA_MANIFEST.md` | Data provenance, inclusion policy, and known limits. |
| `CHANGELOG.md` | Research and repository revision history. |

## Clone and verify

Git LFS is required because the canonical arrays and model weights are binary
research artifacts.

```bash
git lfs install
git clone https://github.com/Coucou2016/20260519-LarNO-DrainLite.git
cd 20260519-LarNO-DrainLite
git lfs pull
python scripts/verify_repository.py
```

The verifier checks every Git LFS object, recomputes the network audit directly
from the accepted SWMM input file, verifies committed manifests without
rewriting them, and runs the repository consistency and acceptance audits. A
valid snapshot reports `PASS`.

## Final documents

- Manuscript: `extended_study/output/reviewer_major_revision_v4/submission_package_v4/manuscript.pdf`
- Standalone report: `extended_study/output/reviewer_major_revision_v4/submission_package_v4/report.html`
- Integrity audit: `extended_study/output/reviewer_major_revision_v4/submission_package_v4/scientific_integrity_audit.pdf`

## Scope and limitations

The sewer is a road-aligned conceptual network because surveyed municipal pipe
records were unavailable. The evidence covers eight rainfall events, one study
area, and one network realization. The coupled labels passed the project-level
mass-balance and topology checks, but they are not presented as observations of
the real Shenzhen sewer system. The evidence supports rainfall-event transfer
for this fixed network, not generalization to unseen sewer layouts. Static
drainage features have shown limited marginal skill; dynamic sewer states,
additional events, and held-out network realizations remain separate future
tests. The public LarNO checkpoint is reproduced independently, but it is not
used as the upstream field in the principal DrainLite experiment.

## Third-party material

No new blanket license is asserted over upstream LarNO, Itzi, SWMM, MIKE
reference data, or the supplied reference article. Their original terms and
citations continue to apply. See `THIRD_PARTY_NOTICES.md` before redistribution.
