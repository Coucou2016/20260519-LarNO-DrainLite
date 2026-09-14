# Wellington scenario consistency

The official/public target, rainfall, tailwater, freeboard, climate-change
allowance, and comparison thresholds must describe the same scenario.

Register scenario metadata:

```bash
python benchmark_data/wellington_register_official_scenario.py \
  --status public_metadata_only \
  --scenario-id WCC100yrCC2025FloodDepths_FB \
  --return-period-or-aep "100 year ARI" \
  --climate-change-allowance CC2025 \
  --freeboard-included true
```

For Level 3, use `--status official` and provide source/depth-band evidence:

```bash
python benchmark_data/wellington_register_official_scenario.py \
  --status official \
  --source-url-or-correspondence "official-source-or-correspondence-reference" \
  --depth-band-definition "official depth-band/freeboard definition"
```

Audit:

```bash
python benchmark_data/wellington_scenario_consistency_audit.py
```

Important rule:

```text
public_metadata_only supports Level 2 comparison only. Level 3 requires official scenario metadata.
```
