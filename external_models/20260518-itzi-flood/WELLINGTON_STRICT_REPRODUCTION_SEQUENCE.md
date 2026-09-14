# Wellington strict coupled reproduction sequence

This sequence is for a real Wellington CBD ITZI + SWMM coupled run. It is not a
surface-only run and it is not a drainage-capacity shortcut.

The guarded driver for the preparation sequence is:

```bash
python benchmark_data/wellington_strict_reproduction_driver.py --dem path/to/wellington_dem.tif --linz-api-key your_linz_key --keep-going
```

The driver prepares data and audits. It does not run GRASS/ITZI itself.
The LINZ key is passed to child processes through the environment so it is not
written into the recorded command list.

On Windows PowerShell, you can also set the key first:

```powershell
$env:LINZ_API_KEY="your_linz_key"
python benchmark_data/wellington_strict_reproduction_driver.py --dem path\to\wellington_dem.tif --keep-going
```

## 1. Download public Wellington ArcGIS target and stormwater assets

For a first pass, download only the official/public target layers:

```bash
python benchmark_data/wellington_download_official_targets_only.py
```

The flood-depth layers are comparison targets only.

## 2. Derive the CBD model domain

```bash
python benchmark_data/wellington_derive_domain_bbox.py --buffer-m 250
```

Use `bbox_argument` from:

```text
wellington_real/metadata/wellington_cbd_domain_bbox.json
```

for subsequent clipped downloads.

## 3. Redownload/clipped public ArcGIS data with the derived bbox

```bash
python benchmark_data/wellington_download_public_arcgis_sources.py --bbox xmin,ymin,xmax,ymax
python benchmark_data/wellington_source_manifest_from_arcgis_downloads.py
python benchmark_data/wellington_canonicalize_raw_files.py
```

## 4. Register real terrain and buildings

Register the LINZ Wellington 1 m DEM after downloading/clipping it from LINZ:

```bash
python benchmark_data/wellington_register_linz_dem.py --dem path/to/wellington_dem.tif --copy-to-raw
```

Download LINZ building outlines for the same bbox:

```bash
set LINZ_API_KEY=your_linz_key
python benchmark_data/wellington_download_linz_buildings_wfs.py --bbox xmin,ymin,xmax,ymax
```

## 5. Confirm fields and build the SWMM dynamic-wave network

```bash
python benchmark_data/wellington_arcgis_field_discovery.py
copy benchmark_data\wellington_swmm_field_map.template.json wellington_real\metadata\wellington_swmm_field_map.json
```

Edit the copied field map using `wellington_real/reports/arcgis_field_discovery.json`,
then run:

```bash
python benchmark_data/wellington_build_swmm_from_gis.py
python benchmark_data/wellington_swmm_provenance_audit.py
python benchmark_data/wellington_swmm_topology_audit.py
python benchmark_data/swmm_dynwave_integrity_audit.py
```

Strict conversion must pass for an official reproduction claim. Do not use
`--allow-estimated-hydraulics` for official reproduction.

## 6. Build ITZI drainage coupling config

```bash
python benchmark_data/wellington_build_itzi_config.py
python benchmark_data/wellington_itzi_official_config_audit.py
python benchmark_data/wellington_drainage_exchange_output_audit.py
python benchmark_data/wellington_no_shortcut_audit.py
python benchmark_data/wellington_official_target_not_input_audit.py
```

## 7. Run evidence gates before model execution

```bash
python benchmark_data/wellington_terrain_source_gate.py
python benchmark_data/wellington_crs_datum_consistency_audit.py
python benchmark_data/wellington_full_source_manifest.py
python benchmark_data/wellington_manifest_role_audit.py
python benchmark_data/source_provenance_audit_v2.py
python benchmark_data/wellington_level3_evidence_audit.py
python benchmark_data/multi_case_readiness_audit.py
python benchmark_data/wellington_workflow_order_audit.py
python benchmark_data/wellington_objective_completion_audit.py
python benchmark_data/wellington_level1_real_pipeline.py --keep-going
```

`wellington_level3_evidence_audit.py` is expected to fail until official rainfall,
outfall/tailwater, datum/freeboard, schematisation, and acceptance evidence are
provided.

## 8. Run ITZI/GRASS and export visual results

After the gates above pass for Level 1:

```bash
bash benchmark_data/run_wellington_in_grass.sh
```

The final package must include:

- time-varying ITZI depth frames;
- drainage exchange outputs such as `mean_drainage_flow`;
- GIF or MP4 animation;
- representative PNGs;
- official target comparison metrics;
- provenance and no-shortcut reports.

Audit the package with:

```bash
python benchmark_data/wellington_compare_model_to_official.py --model-max-depth wellington_real/outputs/max_water_depth.tif
python benchmark_data/wellington_animation_sequence_audit.py
python benchmark_data/wellington_result_package_audit.py
```

For exploratory runs only:

```bash
python benchmark_data/wellington_result_package_audit.py --allow-exploratory
```

## Official reproduction rule

Do not claim exact official reproduction unless:

```bash
python benchmark_data/wellington_level3_evidence_audit.py
python benchmark_data/wellington_result_package_audit.py
```

both pass without exploratory allowances.
