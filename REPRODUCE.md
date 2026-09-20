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

This is the fastest route for cross-review. It is read-only: it checks all Git
LFS objects, committed hashes, array shapes, finite values, target aliases,
physical residual statistics, DrainLite metrics, the SWMM input topology, HTML
embedding, PDF readability, and figure dimensions without regenerating the
expected evidence.

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

## 4. Refit the major-revision DrainLite experiment

```bash
python extended_study/run_final_hybrid_major_revision.py
python extended_study/benchmark_v4_correction_runtime.py
```

The revision uses eight outer leave-one-event-out folds and five sampling seeds.
The iteration count is fixed before the outer evaluation. Every climatological
prior excludes the held event and, for sampled training rows, excludes the row's
own event. Coordinates and target-side SWMM states remain excluded. The script
also evaluates final-hybrid network shifts, block shuffles, feature-group
permutations, conditioned subsets, clipping sensitivity, and complete
correction-layer runtime. Interrupted runs can be resumed from per-fold
checkpoints.

## 5. Rebuild figures and documents

The event68 physical sensitivities are generated separately:

```bash
python extended_study/run_v4_physical_sensitivity.py --workers 2 --resume
python extended_study/analyze_v4_physical_sensitivity.py
```

```bash
python extended_study/generate_reviewer_v4_publication_figures.py
python extended_study/generate_reviewer_v4_documents.py
python extended_study/render_reviewer_v4_documents.py
python extended_study/final_reviewer_v4_acceptance.py
```

The final HTML files are standalone and contain their images as Base64 data.
Publication figures are retained separately as PNG, PDF, and SVG.

## 6. Regenerate acceptance evidence after an intentional change

```bash
python scripts/regenerate_acceptance_evidence.py
```

This command is deliberately separate from verification. Run it only after an
intentional, reviewed artifact change, then rerun
`python scripts/verify_repository.py`. A hash change without a corresponding
changelog entry should be treated as an audit failure.
