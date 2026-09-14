# Wellington visualization deliverables

The Wellington result must show flood evolution through time. A single image is
not enough.

Required deliverables:

- at least three time frames;
- a GIF, MP4, or WebM animation;
- representative timestep/comparison PNGs;
- model-vs-official comparison image after `wellington_compare_model_to_official.py`.

Audit with:

```bash
python benchmark_data/wellington_animation_sequence_audit.py
python benchmark_data/wellington_result_package_audit.py
python benchmark_data/wellington_objective_completion_audit.py
```

The animation audit writes:

```text
wellington_real/reports/animation_sequence_audit.json
```

Expected output locations:

```text
wellington_real/outputs/
wellington_real/visualizations/
```
