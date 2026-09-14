# Wellington run failure triage

Use this when the external preparation or GRASS/ITZI run fails.

## First report to inspect

```text
wellington_real/reports/wellington_report_summary.json
```

Then inspect the first failed item reported in:

```text
wellington_real/reports/strict_reproduction_driver.json
wellington_real/reports/objective_completion_audit.json
```

## Common failure classes

### Runtime dependencies missing

Relevant report:

```text
wellington_real/reports/runtime_dependency_audit.json
```

Required:

- Python modules: `numpy`, `rasterio`, `matplotlib`;
- command-line tools on PATH: `grass`, `itzi`.

Optional but useful:

- `pyswmm`;
- `gdalinfo` / `ogrinfo`;
- `swmm5`.

### DEM missing or rejected

Relevant reports:

```text
wellington_real/reports/linz_dem_registration.json
wellington_real/reports/terrain_source_gate.json
wellington_real/reports/crs_datum_consistency_audit.json
wellington_real/reports/domain_coverage_audit.json
wellington_real/reports/building_integration_audit.json
```

Expected condition:

- real LINZ DEM file exists;
- CRS is NZTM / EPSG:2193;
- resolution is near 1 m;
- vertical datum evidence is documented.
- DEM covers the model domain and vector layers intersect the same domain.
- CBD buildings are represented or explicitly documented as exploratory-only.

### ArcGIS downloads empty

Relevant report:

```text
wellington_real/reports/public_arcgis_downloads.json
```

Required non-empty layers:

- Stormwater Node;
- Stormwater Pipe;
- North CBD official/public target;
- Southern CBD official/public target.

Pumpstations, open channels, and connection pipes may be empty in a clipped CBD
domain and should be warnings, not hard failures.

### SWMM network cannot be built

Relevant reports:

```text
wellington_real/reports/arcgis_field_discovery.json
wellington_real/reports/build_swmm_from_gis.json
wellington_real/reports/swmm_gis_trace.json
wellington_real/reports/swmm_topology_audit.json
wellington_real/reports/swmm_inflow_coupling_audit.json
wellington_real/reports/swmm_pump_control_audit.json
wellington_real/reports/swmm_hydraulic_parameter_audit.json
```

Likely causes:

- field map does not match actual ArcGIS attributes;
- pipe diameter or invert fields are absent;
- upstream/downstream node fields are absent and geometry snapping exceeds the
  allowed distance;
- outfalls cannot be identified.
- SWMM has no rainfall-runoff/direct inflow/ITZI exchange evidence.
- pumpstation assets exist but SWMM has no pump/curve/control representation.
- hydraulic parameters are missing, invalid, or marked exploratory/default.

### ITZI coupling rejected

Relevant reports:

```text
wellington_real/reports/build_itzi_config.json
wellington_real/reports/itzi_official_config_audit.json
wellington_real/reports/grass_runtime_inputs_audit.json
wellington_real/reports/drainage_exchange_output_audit.json
wellington_real/reports/no_shortcut_audit.json
```

Expected condition:

- config has `[drainage]`;
- `swmm_inp` points to the generated SWMM file;
- required GRASS maps/time series are registered as existing;
- `mean_drainage_flow` is requested;
- no surface-removal shortcut is present.

### Official exact reproduction rejected

Relevant report:

```text
wellington_real/reports/level3_evidence_audit.json
wellington_real/reports/rainfall_forcing_audit.json
wellington_real/reports/outfall_tailwater_audit_v2.json
wellington_real/reports/scenario_consistency_audit.json
wellington_real/reports/acceptance_metrics_audit.json
```

This is expected to fail until official Wellington Water/WCC rainfall,
tailwater/outfall, datum/freeboard, schematisation, and acceptance metrics are
available.

### Animation/result package rejected

Relevant reports:

```text
wellington_real/reports/model_run_artifact_audit.json
wellington_real/reports/animation_sequence_audit.json
wellington_real/reports/result_package_audit.json
```

Expected condition:

- at least three time frames;
- GIF/MP4/WebM animation;
- representative PNGs;
- comparison metrics.
