# Data manifest and provenance

## Canonical dataset

The publishable dataset identifier is `region1_20m_drainage_v3_full`.

- Events: `event1`, `event20`, `event65`, `event66`, `event67`, `event68`,
  `event69`, and `event70`.
- Dynamic shape: `72 x 400 x 560`.
- Temporal interval: 5 min; event duration: 6 h.
- Spatial resolution: 20 m.
- Depth unit: m; rainfall follows the upstream LarNO array convention recorded
  by the preprocessing code.
- Active cells: 105,527.

Each event contains the rainfall forcing, MIKE external reference,
surface-only case A, matched-surface case B, coupled case C, and the training
alias `h.npy = h_itzi_swmm.npy`. Event metadata records model settings and
quality diagnostics. The geodata directory contains the DEM, active/building
masks, inlet/outfall/pipe masks, pipe diameter, slope, capacity, cover depth,
and distance-to-outfall fields.

## Sources and roles

| Source | Role in this study | Training status |
|---|---|---|
| Public LarNO rainfall and MIKE arrays | Rainfall forcing and external plausibility reference | MIKE is not a DrainLite target |
| Copied formal Itzi-SWMM workflow | Generates A/B/C depth sequences | C is the physical target; B is the baseline |
| Road-aligned conceptual network | Supplies exchange locations and static pipe descriptors | Static model inputs |
| SWMM dynamic-state diagnostics | Physical diagnosis and future-feature evidence | Excluded from current predictive inputs |
| Public LarNO checkpoint | Independent reproduction only | Not fine-tuned and not residual-corrected |

## GitHub inclusion policy

All unique, research-critical artifacts needed to inspect the reported results
are included. Large binary arrays and fitted estimators use Git LFS.

The following local material is intentionally not uploaded:

1. `cache/`, `tmp/`, Python caches, and Chrome profiles: machine-local files;
   browser profiles can contain authentication databases.
2. `WinGRASS-8.4.2-1-Setup.exe`: third-party 938 MB installer, obtainable from
   its publisher and unrelated to result provenance.
3. `main (7).md` and `main (7).pdf`: supplied reference article; the repository
   records its DOI instead of redistributing the full text.
4. Original public `region1_20m` benchmark copies: superseded locally by the
   canonical eight-event package for this study; the complete public benchmark
   remains available from the LarNO dataset page.
5. Exploratory and superseded output arrays: many are exact or semantic
   duplicates of final A/B/C arrays. Their final metrics, accepted code path,
   and canonical replacements are retained.
6. Duplicate `itzi-flood/` workspace: the accepted non-destructive copy is
   `external_models/20260518-itzi-flood/`.

This policy preserves reproducibility while preventing a 43 GB working
directory, dominated by repeated arrays and installers, from being mislabeled
as a 43 GB scientific dataset.

## Integrity

- Submission artifacts are hashed in
  `extended_study/output/reviewer_major_revision_v3/submission_package_v3/sha256_manifest.csv`.
- Repository-level canonical artifacts are hashed by
  `scripts/generate_data_manifest.py` into `canonical_data_sha256.csv`.
- Numerical and structural consistency is checked by
  `extended_study/final_reviewer_v3_acceptance.py`.

## Known scientific limits

- No surveyed sewer inventory was available; the network is conceptual.
- Only one network realization and eight rainfall events are evaluated.
- MIKE represents an external reference with different and incompletely known
  drainage details.
- The public LarNO checkpoint may already encode MIKE drainage effects.
- No valid claim of clean LarNO-plus-drainage stacking can be made until a
  surface-only LarNO baseline is trained or obtained.

