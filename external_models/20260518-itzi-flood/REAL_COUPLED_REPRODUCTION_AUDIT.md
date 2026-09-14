# Real coupled reproduction audit

This project must not drift into a surface-only or visually plausible demo. The
target is a real, traceable ITZI + SWMM dynamic-wave reproduction workflow.

## Correct standard

A case is acceptable only when it can prove all of the following:

- Real DEM/DTM source, local file, checksum, CRS, vertical datum, resolution, and
  domain coverage.
- Real stormwater asset source, local node/link files, checksums, connectivity,
  and audited field mapping into SWMM.
- SWMM dynamic-wave routing, with `FLOW_ROUTING DYNWAVE`, valid junction,
  conduit, and outfall sections, and a runtime-open test.
- ITZI `[drainage]` coupling with `swmm_inp`, not a surface sink/capacity
  approximation.
- Model output exported as time-varying depth maps and drainage exchange
  diagnostics.
- Comparison against an official/public target for the same named scenario.

An exact official reproduction additionally needs official rainfall, outfall or
tailwater boundary, vertical datum transformation, freeboard/scenario handling,
and acceptance thresholds.

## Current Wellington judgement

Wellington CBD remains the best candidate because public sources expose all
three major data classes:

- Wellington stormwater nodes/pipes/channels/pumpstations via Wellington Water
  ArcGIS REST.
- Wellington flood-depth target layers including North CBD and Southern CBD.
- LINZ Wellington LiDAR DEM.

But the honest status is still:

- Suitable target for a Level 1 real-data coupled run once local downloads and
  GRASS/ITZI execution succeed.
- Suitable target for a Level 2 public target comparison once the public depth
  polygons are converted into a comparison raster/band map.
- Not yet a Level 3 exact official reproduction unless official forcing,
  boundary, datum/freeboard, and acceptance evidence are obtained.

## Candidate registry

The registry is stored at:

```text
benchmark_data/real_case_candidate_registry.json
```

The audit script is:

```bash
python benchmark_data/real_case_candidate_registry_audit.py
```

It writes:

```text
wellington_real/reports/real_case_candidate_registry_audit.json
```

The no-shortcut model audit is:

```bash
python benchmark_data/wellington_no_shortcut_audit.py
```

It writes:

```text
wellington_real/reports/no_shortcut_audit.json
```

This fails active model/config files that use drainage-capacity or sink
approximations instead of ITZI drainage coupling to a SWMM DYNWAVE network.

The official-target separation audit is:

```bash
python benchmark_data/wellington_official_target_not_input_audit.py
```

It ensures the WCC flood-depth polygons are used only as comparison/validation
targets, never as model inputs.

The SWMM provenance audit is:

```bash
python benchmark_data/wellington_swmm_provenance_audit.py
```

It requires SWMM junctions, conduits, and outfalls to carry source layer/ObjectID
or asset provenance markers, so the underground network can be traced back to
the downloaded public GIS features.

The GIS-to-SWMM converter is:

```bash
python benchmark_data/wellington_build_swmm_from_gis.py
```

It is strict by default and fails rather than guessing missing hydraulic fields.
The optional exploratory mode is explicitly labelled `PASS_EXPLORATORY` and is
not acceptable for official reproduction claims.

The ITZI config builder is:

```bash
python benchmark_data/wellington_build_itzi_config.py
```

It writes a config with `[drainage]`, `swmm_inp`, drainage vector output, and
`mean_drainage_flow` so the surface-pipe exchange is visible in the results.

The result package audit is:

```bash
python benchmark_data/wellington_result_package_audit.py
```

It checks that the final deliverables include model-output frames, animation
media, representative PNGs, comparison metrics, provenance reports, no-shortcut
checks, and the Level 3 evidence audit.

The terrain source and datum audits are:

```bash
python benchmark_data/wellington_register_linz_dem.py --dem path/to/wellington_dem.tif --copy-to-raw
python benchmark_data/wellington_terrain_source_gate.py
python benchmark_data/wellington_crs_datum_consistency_audit.py
```

They prevent the public flood-depth target from being mistaken for terrain and
require the DEM/asset/target datum story to be documented.

The Level 3 official reproduction evidence audit is:

```bash
python benchmark_data/wellington_level3_evidence_audit.py
```

It intentionally fails until official rainfall, outfall/tailwater, vertical
datum/freeboard, official model schematisation, and acceptance metrics are all
documented locally.

Generate a request package for the missing official evidence with:

```bash
python benchmark_data/wellington_generate_official_data_request.py
```

It writes:

```text
wellington_real/reports/wellington_level3_official_data_request.md
```

The current registry intentionally separates:

- Wellington CBD: strongest candidate, public target available, official exact
  reproduction still missing forcing/boundary evidence.
- Hutt City and Hamilton: real stormwater network candidates, not benchmarks
  until target flood results are identified.
- Christchurch/Styx: strong follow-up candidate because StormWater asset layers,
  drainage data, model reports, and ECan adopted depth imagery exist, but the
  exact open raster/export, SWMM field mapping, and scenario forcing still need
  verification.
- Auckland: discovery candidate; Flood_Plains has a queryable FeatureServer, but
  the stormwater network export endpoint and exact scenario forcing still need
  confirmation.
- Waimakariri: open stormwater asset candidate, not a benchmark until a matching
  official/public flood-depth target is found.
- UK EA 2D benchmark: useful 2D benchmark data, but not automatically a
  stormwater-pipe-coupled ITZI+SWMM case.

## Windows/ITZI correction

ITZI documentation now says Windows 11 has been tested and the installation
steps are the same as GNU/Linux after installing `uv`. ITZI also depends on
GRASS GIS. This project still keeps WSL/Docker runbooks because large GRASS/GDAL
batch processing is usually more reproducible there, but Windows-native ITZI is
not ruled out by the current official documentation.
