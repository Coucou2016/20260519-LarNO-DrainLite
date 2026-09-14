# Wellington ITZI coupled config

Generate the strict coupled ITZI configuration with:

```bash
python benchmark_data/wellington_build_itzi_config.py --path-style relative
```

Default output:

```text
wellington_real/configs/wellington_cbd_itzi.ini
wellington_real/reports/build_itzi_config.json
```

The generated config includes:

- `[drainage]`;
- `swmm_inp = wellington_real/swmm/wellington_cbd_from_gis.inp`;
- drainage vector output `wellington_cbd_drainage`;
- `mean_drainage_flow` in ITZI output values;
- water-depth and water-surface-elevation outputs.

Audit the exchange-output requirements with:

```bash
python benchmark_data/wellington_itzi_official_config_audit.py
python benchmark_data/wellington_drainage_exchange_output_audit.py
```

This is important because a valid coupled run must show how water moves between
the 2D surface and 1D drainage network, not just produce a final flood-depth
picture.

The official-config audit checks the fields documented by ITZI, including
`[drainage] swmm_inp`, drainage `output`, and the `mean_drainage_flow` output
value.

When running from WSL against files originally created on Windows, generate WSL
paths explicitly:

```bash
python benchmark_data/wellington_build_itzi_config.py --path-style wsl
```
