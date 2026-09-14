# Wellington external execution package

The current Codex shell is failing before commands start with:

```text
windows sandbox: setup refresh failed with status exit code: 1
```

Until that is fixed, run the strict preparation workflow from a normal
PowerShell, WSL Ubuntu shell, or Docker shell that can access:

```text
E:\Projects\20260518-itzi-flood
```

## Required inputs

- A downloaded/clipped LINZ Wellington 1 m DEM GeoTIFF.
- Rainfall forcing CSV. Exploratory forcing is enough for Level 1/2 only;
  Level 3 requires the official Wellington Water/WCC hyetograph.
- Outfall/tailwater stage CSV. Exploratory boundary is enough for Level 1/2
  only; Level 3 requires official Wellington Water/WCC boundary evidence.
- Optional but recommended: LINZ API key for building outlines.
- GRASS GIS / ITZI / Python geospatial environment for the final model run.

## PowerShell preparation

Preferred wrapper:

```powershell
cd E:\Projects\20260518-itzi-flood
$env:LINZ_API_KEY="your_linz_key"
.\run_wellington_strict_reproduction.ps1 -Dem path\to\wellington_dem.tif -KeepGoing
```

To also launch the GRASS/ITZI shell runner when `bash` is available:

```powershell
.\run_wellington_strict_reproduction.ps1 -Dem path\to\wellington_dem.tif -KeepGoing -RunGrass
```

Manual preparation command:

```powershell
cd E:\Projects\20260518-itzi-flood
$env:LINZ_API_KEY="your_linz_key"
python benchmark_data\wellington_strict_reproduction_driver.py --dem path\to\wellington_dem.tif --keep-going
```

If you do not have a LINZ API key yet:

```powershell
python benchmark_data\wellington_strict_reproduction_driver.py --dem path\to\wellington_dem.tif --skip-buildings --keep-going
```

That is acceptable only for a first exploratory surface setup. Exact urban CBD
reproduction should document building treatment.

## WSL preparation

Preferred wrapper:

```bash
cd /mnt/e/Projects/20260518-itzi-flood
export LINZ_API_KEY="your_linz_key"
bash run_wellington_strict_reproduction.sh --dem /mnt/e/path/to/wellington_dem.tif --keep-going
```

To also launch the GRASS/ITZI runner:

```bash
bash run_wellington_strict_reproduction.sh --dem /mnt/e/path/to/wellington_dem.tif --keep-going --run-grass
```

Manual preparation command:

```bash
cd /mnt/e/Projects/20260518-itzi-flood
export LINZ_API_KEY="your_linz_key"
python benchmark_data/wellington_strict_reproduction_driver.py --dem /mnt/e/path/to/wellington_dem.tif --keep-going
```

## Key reports

The strict driver writes:

```text
wellington_real/reports/strict_reproduction_driver.json
```

The PowerShell wrapper also writes logs under:

```text
wellington_real/reports/external_run_logs/
```

The reports that matter most are:

```text
wellington_real/reports/objective_completion_audit.json
wellington_real/reports/wellington_report_summary.json
wellington_real/reports/static_script_audit.json
wellington_real/reports/runtime_dependency_audit.json
wellington_real/reports/source_provenance_audit_v2.json
wellington_real/reports/domain_coverage_audit.json
wellington_real/reports/building_integration_audit.json
wellington_real/reports/rainfall_forcing_audit.json
wellington_real/reports/outfall_tailwater_audit_v2.json
wellington_real/reports/scenario_consistency_audit.json
wellington_real/reports/acceptance_metrics_audit.json
wellington_real/reports/grass_runtime_inputs_audit.json
wellington_real/reports/level3_evidence_audit.json
wellington_real/reports/build_swmm_from_gis.json
wellington_real/reports/build_itzi_config.json
wellington_real/reports/no_shortcut_audit.json
wellington_real/reports/official_target_not_input_audit.json
```

If `runtime_dependency_audit.json` fails, fix the environment before interpreting
model results. The required runtime checks include Python `numpy`, `rasterio`,
`matplotlib`, and command-line `grass` and `itzi`.

Register rainfall before interpreting a run:

```bash
python benchmark_data/wellington_register_rainfall_forcing.py --csv path/to/rainfall.csv --status exploratory --forcing-id exploratory_test_storm
python benchmark_data/wellington_rainfall_forcing_audit.py
```

For official reproduction, use `--status official` and provide owner/source/AEP
metadata as described in `WELLINGTON_RAINFALL_FORCING.md`.

Register outfall/tailwater before interpreting a dynamic-wave run:

```bash
python benchmark_data/wellington_register_outfall_tailwater.py --csv path/to/tailwater.csv --status exploratory --boundary-id exploratory_tailwater --outfalls OUT1,OUT2
python benchmark_data/wellington_outfall_tailwater_audit_v2.py
```

For official reproduction, use `--status official` and provide owner/source/datum
metadata as described in `WELLINGTON_OUTFALL_TAILWATER.md`.

The Level 3 evidence audit reads both dedicated forcing audits:

```bash
python benchmark_data/wellington_rainfall_forcing_audit.py
python benchmark_data/wellington_outfall_tailwater_audit_v2.py
python benchmark_data/wellington_scenario_consistency_audit.py
python benchmark_data/wellington_acceptance_metrics_audit.py
python benchmark_data/wellington_grass_runtime_inputs_audit.py
python benchmark_data/wellington_forcing_integration_audit.py
python benchmark_data/wellington_level3_evidence_audit.py
```

## Final ITZI/GRASS run

Only after the preparation gates are good enough for Level 1:

```bash
bash benchmark_data/run_wellington_in_grass.sh
```

After exporting the model maximum water-depth raster, compare it against the
official/public target:

```bash
python benchmark_data/wellington_official_target_field_discovery.py
python benchmark_data/wellington_compare_model_to_official.py --model-max-depth wellington_real/outputs/max_water_depth.tif
python benchmark_data/wellington_model_run_artifact_audit.py
python benchmark_data/wellington_animation_sequence_audit.py
python benchmark_data/wellington_result_package_audit.py
python benchmark_data/wellington_objective_completion_audit.py
```

Use the official gate only after official rainfall, boundary, datum/freeboard,
schematisation, and acceptance evidence are supplied:

```bash
bash benchmark_data/run_wellington_in_grass.sh --official-gate
```

## What counts as success

Level 1 success:

- LINZ DEM registered and audited.
- Wellington stormwater node/pipe data downloaded.
- SWMM `.inp` generated from GIS assets with source provenance.
- SWMM `FLOW_ROUTING DYNWAVE` passes integrity/runtime checks.
- ITZI config uses `[drainage]` with `swmm_inp`.
- No shortcut/surface-removal substitute is present.
- Time-varying ITZI outputs and drainage exchange diagnostics are exported.

Level 2 success:

- Level 1 success plus comparison against WCC/Wellington Water flood-depth target.

Level 3 success:

- Level 2 success plus official rainfall hyetograph, outfall/tailwater boundary,
  vertical datum/freeboard/scenario metadata, official schematisation evidence,
  and acceptance metrics.

If `objective_completion_audit.json` fails, the full user objective is not
complete, even if individual scripts ran successfully.

Failure triage:

```text
WELLINGTON_RUN_FAILURE_TRIAGE.md
```
