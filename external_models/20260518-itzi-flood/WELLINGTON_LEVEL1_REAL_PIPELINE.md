# Wellington Level 1 real-data pipeline

This is the runnable preparation chain for the Wellington CBD real-data ITZI +
SWMM case. It is deliberately stricter than a demo and deliberately weaker than
an official exact reproduction:

- It requires real local DEM, real stormwater network data, SWMM dynamic-wave
  configuration, source provenance, and surface-to-drainage coupling evidence.
- It does not treat missing official rainfall, tailwater, freeboard, or datum
  transformation evidence as solved.
- It does not create synthetic pipes, synthetic terrain, or drainage-capacity
  sinks.

Run it from the repository root after the raw Wellington/LINZ/Wellington Water
downloads are present:

```bash
python benchmark_data/wellington_level1_real_pipeline.py --keep-going
```

The wrapper writes:

```text
wellington_real/reports/level1_real_pipeline.json
```

It also runs the raw-input gate:

```bash
python benchmark_data/wellington_raw_input_gate.py
```

That gate writes:

```text
wellington_real/reports/raw_input_gate.json
```

The first required data step is:

```bash
python benchmark_data/wellington_canonicalize_raw_files.py
```

That step copies acceptable raw download names into the canonical names expected
by the rest of the workflow:

```text
wellington_real/raw/stormwater_node.geojson
wellington_real/raw/stormwater_pipe.geojson
wellington_real/raw/wcc_flood_depth.geojson
```

This matters because ArcGIS and manual exports often produce names such as
`Stormwater_Node.geojson`, `stormwater_nodes.geojson`, or
`WCC100yrCC2025FloodDepths_NCBD_FB.geojson`. The pipeline should normalize those
before building `source_manifest.json`, otherwise later audits may fail for a
filename mismatch instead of a real data problem.

After the Level 1 pipeline passes, the Linux/WSL/GRASS execution path remains:

```bash
bash benchmark_data/run_wellington_in_grass.sh
```

Use the final official gate only when the missing WCC-specific rainfall,
boundary, vertical-datum, and comparison evidence has been supplied:

```bash
bash benchmark_data/run_wellington_in_grass.sh --official-gate
```

At the current state, the honest status is:

- Level 1 target: real-data coupled ITZI + SWMM run, suitable for local
  engineering exploration once the downloaded inputs are in place.
- Level 3 target: official WCC scenario reproduction, still dependent on
  official forcing/boundary/datum evidence.
