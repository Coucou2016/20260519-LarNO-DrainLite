# Reproduction and audit sequence

Run commands from the repository root. The exact historical Python environment
is summarized in `requirements-research.txt`; Itzi/SWMM execution also requires
the copied model environment documented under
`external_models/20260518-itzi-flood/`.

## 1. Materialize and verify the published snapshot

```bash
git lfs install
git lfs pull
python scripts/verify_repository.py
```

Expected terminal summary:

```text
status: PASS
checks: 107/107
```

This is the fastest route for cross-review because it independently recalculates
array shapes, finite values, target aliases, physical residual statistics,
DrainLite metrics, network acceptance, HTML embedding, PDF readability, and
figure dimensions.

## 2. Rebuild and audit the conceptual SWMM network

```bash
python extended_study/audit_reviewer_v3_network.py
python extended_study/plot_reviewer_v3_network.py
```

The accepted network is:

```text
extended_study/output/reviewer_major_revision_v3/network/
  swmm_bidirectional_normal_step05.inp
```

Do not substitute the earlier sparse 82-node/29-pipe prototype.

## 3. Recalculate formal physical scenarios

The final simulation path is implemented by:

```bash
python extended_study/run_reviewer_v3_batch.py
python extended_study/analyze_reviewer_v3_physics.py
python extended_study/build_reviewer_v3_dataset.py
```

These are expensive CPU simulations. Cases A, B, and C must retain their
documented Manning coefficients, rainfall loss, building redistribution,
boundary treatment, coupling relaxation, and SWMM settings. The canonical
published arrays are already supplied, so cross-review does not require rerunning
the full batch unless the physics itself is being challenged.

## 4. Refit DrainLite with event-level separation

```bash
python extended_study/run_reviewer_v3_drainlite_hybrid.py
```

Required controls are: eight outer leave-one-event-out folds, one complete inner
validation event, held-event exclusion from the climatological prior, no row or
column coordinates, and no target-side SWMM dynamic states.

## 5. Rebuild figures and documents

```bash
python extended_study/generate_reviewer_v3_publication_figures.py
python extended_study/generate_reviewer_v3_documents.py
python extended_study/render_reviewer_v3_documents.py
python extended_study/final_reviewer_v3_acceptance.py
```

The final HTML files are standalone and contain their images as Base64 data.
The 11 figures are also retained separately as PNG, PDF, and SVG.

## 6. Regenerate repository hashes

```bash
python scripts/generate_data_manifest.py
```

Run this only after intentional canonical-data changes. A hash change without a
corresponding changelog entry should be treated as an audit failure.

