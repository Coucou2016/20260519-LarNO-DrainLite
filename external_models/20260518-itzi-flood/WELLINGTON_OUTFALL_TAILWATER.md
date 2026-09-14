# Wellington outfall/tailwater boundary

SWMM dynamic-wave results depend on outfall/tailwater stage. For Wellington CBD,
this is especially important because the drainage system discharges near the
harbour.

Copy the template if documenting manually:

```text
benchmark_data/wellington_outfall_tailwater.template.json
```

Or register a CSV stage time series:

```bash
python benchmark_data/wellington_register_outfall_tailwater.py \
  --csv path/to/tailwater.csv \
  --status exploratory \
  --boundary-id exploratory_tailwater \
  --outfalls OUT1,OUT2
```

Official Level 3 evidence requires source metadata:

```bash
python benchmark_data/wellington_register_outfall_tailwater.py \
  --csv path/to/official_tailwater.csv \
  --status official \
  --source-url "official-source-or-correspondence-reference" \
  --owner "Wellington Water / WCC" \
  --vertical-datum NZVD2016 \
  --outfalls OUT1,OUT2
```

Audit:

```bash
python benchmark_data/wellington_outfall_tailwater_audit_v2.py
python benchmark_data/wellington_forcing_integration_audit.py
python benchmark_data/wellington_level3_evidence_audit.py
```

Important rule:

```text
status=exploratory cannot support an exact official reproduction claim.
```

`wellington_level3_evidence_audit.py` reads
`outfall_tailwater_audit_v2.json` and requires
`level3_boundary_verdict=PASS` before the boundary evidence can satisfy Level 3.

For official tailwater, the SWMM `[OUTFALLS]` records must not remain `FREE`.
They must use an appropriate fixed/tidal/time-series boundary, and the referenced
outfall IDs must match `applies_to_outfalls`.
