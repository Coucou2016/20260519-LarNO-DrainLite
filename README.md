# LarNO-DrainLite: drainage-aware urban flood prediction

This repository is the auditable research snapshot for a lightweight extension
of LarNO that represents urban drainage effects without retraining the full
neural-operator backbone. It contains the project code, the calibrated
Itzi-SWMM workflow copied into this workspace, the eight-event canonical
dataset, leakage-controlled leave-one-event-out predictions, statistics,
publication figures, manuscript, standalone report, and scientific-integrity
audit.

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
drainage effect is 9.266 mm. Leakage-controlled leave-one-event-out validation
reduces the coupled-label mean absolute error from 9.266 mm for B to 2.130 mm
for the DrainLite hybrid. Static pipe descriptors add a small but consistent
0.054 mm improvement over the dynamic-only model; this limited increment is a
reported result rather than being hidden.

MIKE is used only as an external plausibility reference. It is not a DrainLite
training target. The public LarNO checkpoint is also audited independently and
is not directly corrected by `C - B`, because its MIKE labels may already
contain drainage effects.

## Canonical contents

| Path | Purpose |
|---|---|
| `extended_study/` | Source code for network construction, coupled simulations, DrainLite, figures, documents, and acceptance tests. |
| `external_models/20260518-itzi-flood/` | Non-destructive copy of the formal Itzi-SWMM workflow and Shenzhen configuration. |
| `LarNO-main/code/` | Upstream LarNO source used for checkpoint reproduction. |
| `LarNO-main/benchmark/urbanflood/flood/region1_20m_drainage_v3_full/` | Canonical eight-event dynamic dataset, 72 frames at 20 m. |
| `LarNO-main/benchmark/urbanflood/geodata/region1_20m_drainage_v3_full/` | DEM, masks, and eight rasterized drainage descriptors. |
| `extended_study/output/reviewer_major_revision_v3/drainlite_hybrid_v3/` | Fold-specific estimators, complete predictions, metrics, and runtime records. |
| `extended_study/output/reviewer_major_revision_v3/formal_matched_full/` | SWMM reports, dynamic-state diagnostics, event metadata, and physical-quality table. Large duplicate depth arrays are represented once in the canonical dataset. |
| `extended_study/output/reviewer_major_revision_v3/submission_package_v3/` | Final manuscript, report, audit, figures, acceptance JSON, and SHA-256 manifest. |
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

The verifier checks that LFS objects are materialized and then executes the
107-item independent acceptance audit. A valid snapshot reports `PASS`.

## Final documents

- Manuscript: `extended_study/output/reviewer_major_revision_v3/submission_package_v3/manuscript.pdf`
- Standalone report: `extended_study/output/reviewer_major_revision_v3/submission_package_v3/report.html`
- Integrity audit: `extended_study/output/reviewer_major_revision_v3/submission_package_v3/scientific_integrity_audit.pdf`

## Scope and limitations

The sewer is a road-aligned conceptual network because surveyed municipal pipe
records were unavailable. The evidence covers eight rainfall events, one study
area, and one network realization. The coupled labels passed the project-level
mass-balance and topology checks, but they are not presented as observations of
the real Shenzhen sewer system. Static drainage features show limited marginal
skill; dynamic sewer states and independent network realizations remain the
main next tests.

## Third-party material

No new blanket license is asserted over upstream LarNO, Itzi, SWMM, MIKE
reference data, or the supplied reference article. Their original terms and
citations continue to apply. See `THIRD_PARTY_NOTICES.md` before redistribution.
