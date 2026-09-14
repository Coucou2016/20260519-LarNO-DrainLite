# Wellington rainfall forcing

Rainfall forcing must be traceable. Without the official Wellington Water/WCC
hyetograph, the run can be exploratory only.

Copy the template if documenting manually:

```text
benchmark_data/wellington_rainfall_forcing.template.json
```

Or register a CSV hyetograph:

```bash
python benchmark_data/wellington_register_rainfall_forcing.py \
  --csv path/to/rainfall.csv \
  --status exploratory \
  --forcing-id exploratory_test_storm
```

Official Level 3 evidence requires source metadata:

```bash
python benchmark_data/wellington_register_rainfall_forcing.py \
  --csv path/to/official_hyetograph.csv \
  --status official \
  --source-url "official-source-or-correspondence-reference" \
  --owner "Wellington Water / WCC" \
  --return-period-or-aep "100 year ARI"
```

Audit:

```bash
python benchmark_data/wellington_rainfall_forcing_audit.py
python benchmark_data/wellington_forcing_integration_audit.py
python benchmark_data/wellington_level3_evidence_audit.py
```

The audit writes:

```text
wellington_real/reports/rainfall_forcing_audit.json
wellington_real/metadata/rainfall_forcing.json
```

Important rule:

```text
status=exploratory cannot support an exact official reproduction claim.
```

`wellington_level3_evidence_audit.py` reads
`rainfall_forcing_audit.json` and requires
`level3_forcing_verdict=PASS` before the rainfall evidence can satisfy Level 3.

The integration audit also checks that ITZI `[input] rain` exists. A separate
GRASS import report is still needed to prove that the registered hyetograph has
actually been imported into the model runtime.
