"""Remeasure canonical inference including uncached dynamic neighbourhoods.

Static fields, fitted estimators and training-derived priors are prepared once.
Input/output file access is excluded and this is stated in the output metadata.
"""
import json
import os
os.environ.setdefault("OMP_NUM_THREADS", "8")
from pathlib import Path
import joblib
import numpy as np
import run_final_hybrid_major_revision as experiment

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "extended_study/output/reviewer_major_revision_v4/final_hybrid_controls"

def main():
    flood = ROOT / "LarNO-main/benchmark/urbanflood/flood" / experiment.DEFAULT_DATASET
    geo = ROOT / "LarNO-main/benchmark/urbanflood/geodata" / experiment.DEFAULT_DATASET
    meta = json.loads((OUT / "experiment_metadata.json").read_text())
    events = meta["events"]
    static = experiment.dl.load_static(geo)
    arrays = {event: experiment.dl.load_event(flood, event) for event in events}
    physics = experiment.load_physics_runtime(ROOT)
    records = []
    for event in events:
        model = joblib.load(OUT / "models/seed_1907" / f"fold_{event}_hybrid_all.joblib")
        prior = experiment.mean_residual(arrays, [e for e in events if e != event])
        prediction, timing = experiment.dense_prediction(
            model, experiment.dl.SUPERSET, arrays[event], static, prior, 50000)
        saved = np.load(OUT / "predictions" / event / "h_hybrid_all.npy", mmap_mode="r")
        difference = float(np.max(np.abs(prediction - saved)))
        if difference > 1e-6:
            raise ValueError(f"Prediction changed for {event}: {difference}")
        run = physics[event]
        total = run["surface_runtime_s"] + timing["total_correction_seconds"]
        records.append(dict(seed=1907, model="hybrid_all", test_event=event,
            **timing, **run, surface_plus_correction_seconds=total,
            speedup_vs_coupled=run["coupled_runtime_s"] / total,
            maximum_difference_from_saved_m=difference))
        print(event, timing["total_correction_seconds"], flush=True)
    experiment.write_csv(OUT / "metrics/runtime_uncached_canonical.csv", records)
    (OUT / "runtime_scope.json").write_text(json.dumps({
        "scope": "in-memory correction including uncached dynamic neighbourhoods, feature assembly, prediction and non-negativity",
        "excluded": "one-time static field and estimator loading, offline training-event prior preparation, input/output disk access",
        "physical_time_source": "historical physics_quality.csv; workflow total is summed, not a fresh integrated wall-clock run",
        "threads": os.environ["OMP_NUM_THREADS"]}, indent=2))

if __name__ == "__main__":
    main()
