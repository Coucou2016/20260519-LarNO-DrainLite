#!/usr/bin/env python3
"""Validate drainage dataset v1 and LarNO-D input plumbing."""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "LarNO-main" / "code" / "urbanflood_larfno"
BENCH = ROOT / "LarNO-main" / "benchmark" / "urbanflood"
LOCATION = "region1_20m_drainage_v1"
OUT = ROOT / "extended_study" / "output" / "drainage_dataset_v1"

sys.path.insert(0, str(CODE))
os.chdir(CODE)

from configmypy import ArgparseConfig, ConfigPipeline, YamlConfig
from neuralop import get_model
from neuralop.data.datasets.Dynamic2DFlood import Dynamic2DFlood
from neuralop.models import DrainageAdapter
from neuralop.models.fno import preprocess_inputs


FEATURES = [
    "drain_inlet_mask",
    "drain_outfall_mask",
    "pipe_mask",
    "pipe_diameter",
    "pipe_slope",
    "pipe_capacity",
    "pipe_cover_depth",
    "distance_to_outfall",
]


def check_array(path: Path, expected_shape: tuple[int, ...] | None = None) -> dict[str, object]:
    arr = np.load(path)
    row = {
        "file": str(path.relative_to(ROOT)),
        "shape": "x".join(map(str, arr.shape)),
        "dtype": str(arr.dtype),
        "nan_count": int(np.isnan(arr).sum()) if np.issubdtype(arr.dtype, np.floating) else 0,
        "inf_count": int(np.isinf(arr).sum()) if np.issubdtype(arr.dtype, np.floating) else 0,
        "min": float(np.nanmin(arr)) if arr.size and np.issubdtype(arr.dtype, np.number) else "",
        "max": float(np.nanmax(arr)) if arr.size and np.issubdtype(arr.dtype, np.number) else "",
        "status": "ok",
    }
    if expected_shape is not None and tuple(arr.shape) != tuple(expected_shape):
        row["status"] = f"shape_mismatch_expected_{expected_shape}"
    if row["nan_count"] or row["inf_count"]:
        row["status"] = "nan_or_inf"
    return row


def main() -> int:
    geodata = BENCH / "geodata" / LOCATION
    flood = BENCH / "flood" / LOCATION
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    rows.append(check_array(geodata / "dem.npy", (200, 280)))
    for feature in FEATURES:
        rows.append(check_array(geodata / f"{feature}.npy", (200, 280)))
    for event_dir in sorted(p for p in flood.iterdir() if p.is_dir()):
        for name in ["rainfall.npy", "h.npy", "h_mike_ref.npy", "h_itzi_surface.npy", "h_itzi_sink.npy"]:
            rows.append(check_array(event_dir / name, (72, 200, 280)))

    with (OUT / "validation_arrays.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    dataset = Dynamic2DFlood(
        data_root=str(BENCH),
        split="train",
        location=LOCATION,
        train_list="region1_drainage_train.txt",
        test_list="region1_drainage_test.txt",
        wall_height=50,
        drainage_features=FEATURES,
    )
    inputs, target, event_name = dataset[0]
    input_batch = {k: v.unsqueeze(0) if torch.is_tensor(v) else v for k, v in inputs.items()}
    target_batch = target.unsqueeze(0)
    processed = preprocess_inputs(0, input_batch, torch.device("cpu"))
    adapter = DrainageAdapter(drainage_channels=len(FEATURES), hidden_channels=8)
    base_stub = torch.zeros((1, 1, target_batch.shape[2], target_batch.shape[3], 2), dtype=torch.float32)
    adapted = adapter(base_stub, {k: v[:, :, :, :, :2] if k in {"rainfall", "cumsum_rainfall"} else v for k, v in input_batch.items()})

    pipe = ConfigPipeline([
        YamlConfig("region1_drainage_scratch.yaml", config_name="default", config_folder="./configs"),
        ArgparseConfig(infer_types=True, config_name=None, config_file=None),
        YamlConfig(config_folder="../configs"),
    ])
    config = pipe.read_conf()
    model = get_model(config)
    with torch.no_grad():
        out = model(0, input_batch, torch.device("cpu"), init_shape=(1, target_batch.shape[2], target_batch.shape[3], 1), cache_key=None)

    summary = {
        "dataset_len": len(dataset),
        "first_event": event_name,
        "target_shape": tuple(target_batch.shape),
        "processed_input_shape": tuple(processed.shape),
        "adapter_output_shape": tuple(adapted.shape),
        "larno_d_forward_shape": tuple(out.shape),
        "array_status": "ok" if all(row["status"] == "ok" for row in rows) else "failed",
    }
    (OUT / "validation_summary.txt").write_text(
        "\n".join(f"{k}: {v}" for k, v in summary.items()),
        encoding="utf-8",
    )
    print(summary)
    if summary["array_status"] != "ok":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
