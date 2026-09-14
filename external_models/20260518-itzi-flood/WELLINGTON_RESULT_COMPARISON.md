# Wellington result comparison

A Wellington run is not validated by images alone. Compare the model maximum
water-depth raster against the official/public WCC flood-depth target:

```bash
python benchmark_data/wellington_official_target_field_discovery.py
python benchmark_data/wellington_compare_model_to_official.py --model-max-depth wellington_real/outputs/max_water_depth.tif
```

Outputs:

```text
wellington_real/reports/comparison_quality_gate.json
wellington_real/visualizations/wellington_model_vs_official_comparison.png
```

The default thresholds are stored in:

```text
benchmark_data/wellington_comparison_thresholds.template.json
```

Level 2 uses pragmatic public-target comparison thresholds. Level 3 exact
official reproduction requires Wellington Water/WCC official acceptance
criteria, so the Level 3 thresholds are intentionally `null` until those are
provided.

If the official target has no parseable depth attribute, the comparison script
fails by default. For extent-only checking, run explicitly:

```bash
python benchmark_data/wellington_compare_model_to_official.py --model-max-depth wellington_real/outputs/max_water_depth.tif --allow-binary-target
```

In that mode only IoU, hit rate, and false-alarm ratio are valid; depth
MAE/RMSE/bias are not valid evidence.
