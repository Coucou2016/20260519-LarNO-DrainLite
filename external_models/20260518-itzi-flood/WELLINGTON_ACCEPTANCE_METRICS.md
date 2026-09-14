# Wellington acceptance metrics

Level 3 exact reproduction needs official acceptance criteria from Wellington
Water/WCC. Proposed thresholds are useful for exploratory reporting but cannot
support an official reproduction claim.

Template:

```text
benchmark_data/wellington_acceptance_metrics.template.json
```

Register proposed thresholds:

```bash
python benchmark_data/wellington_register_acceptance_metrics.py \
  --status proposed_only \
  --minimum-iou 0.45 \
  --minimum-hit-rate 0.65 \
  --maximum-false-alarm-ratio 0.45 \
  --maximum-depth-mae-m 0.25 \
  --maximum-depth-rmse-m 0.35 \
  --maximum-bias-m 0.20
```

Register official thresholds:

```bash
python benchmark_data/wellington_register_acceptance_metrics.py \
  --status official \
  --source-url-or-correspondence "official-source-or-correspondence-reference" \
  --minimum-iou ... \
  --minimum-hit-rate ... \
  --maximum-false-alarm-ratio ... \
  --maximum-depth-mae-m ... \
  --maximum-depth-rmse-m ... \
  --maximum-bias-m ... \
  --critical-location-level-tolerance-m ...
```

Audit:

```bash
python benchmark_data/wellington_acceptance_metrics_audit.py
python benchmark_data/wellington_level3_evidence_audit.py
```

Important rule:

```text
status=proposed_only cannot support Level 3.
```
