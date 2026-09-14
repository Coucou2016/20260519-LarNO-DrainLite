# Demo cases notice

The following directories contain useful code experiments, but they are not the
final real-data reproduction requested by the user:

- `test_cases/synthetic_urban`
- `test_cases/urban_district`
- `test_cases/urban_drainage`
- `benchmark_data/ea_test5`
- `benchmark_data/wellington`

Reasons:

- some terrain/buildings were synthetic,
- some drainage networks were manually designed,
- some benchmarks were reconstructed from published parameters rather than run
  from authoritative original input files,
- the early drainage comparison included simplified / partial coupling behavior.

The corrected workflow is in:

- `benchmark_data/WELLINGTON_REAL_WORKFLOW.md`
- `benchmark_data/wellington_real_pipeline.py`
- `benchmark_data/wellington_real_audit.py`
- `benchmark_data/wellington_real_build_swmm.py`
- `benchmark_data/wellington_real_prepare_rasters.py`
- `benchmark_data/wellington_real_run_itzi_swmm.py`
- `benchmark_data/wellington_real_export_itzi.py`
- `benchmark_data/wellington_real_compare.py`
- `benchmark_data/wellington_real_validate.py`

Completion must be judged by `wellington_real_validate.py` and by inspecting the
comparison metrics against WCC official flood-depth layers.

