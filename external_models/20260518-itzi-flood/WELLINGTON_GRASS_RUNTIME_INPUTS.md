# Wellington GRASS runtime inputs

ITZI config names are not enough. The required maps/time series must exist in
the GRASS runtime before the coupled run is credible.

Template:

```text
benchmark_data/wellington_grass_runtime_inputs.template.json
```

Registration command:

```bash
python benchmark_data/wellington_register_grass_runtime_inputs.py \
  --grass-location wellington_nztm \
  --mapset PERMANENT \
  --dem wellington_dem \
  --friction wellington_manning_n \
  --rain wellington_rain \
  --initial-water-depth wellington_initial_depth
```

Audit:

```bash
python benchmark_data/wellington_grass_runtime_inputs_audit.py
```

Outputs:

```text
wellington_real/metadata/grass_runtime_inputs.json
wellington_real/reports/grass_runtime_inputs_audit.json
```

Best practice is to generate this evidence from the GRASS runner after successful
`g.findfile` / `t.list` checks, rather than filling it manually.
