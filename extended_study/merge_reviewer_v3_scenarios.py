#!/usr/bin/env python3
"""Merge independently computed A/B/C scenario folders without changing arrays."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("inputs", nargs="+", type=Path)
    args = parser.parse_args()

    event_sources: dict[str, list[Path]] = {}
    for root in args.inputs:
        for event_dir in sorted(root.glob("event*")):
            if (event_dir / "metadata.json").exists():
                event_sources.setdefault(event_dir.name, []).append(event_dir)

    if not event_sources:
        raise SystemExit("No event metadata found in input directories")

    args.output.mkdir(parents=True, exist_ok=True)
    for event, sources in sorted(event_sources.items()):
        destination = args.output / event
        destination.mkdir(parents=True, exist_ok=True)
        merged = None
        source_manifest: list[dict[str, object]] = []

        for source in sources:
            metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
            if merged is None:
                merged = {key: value for key, value in metadata.items() if key not in {"records", "swmm_quality"}}
                merged["records"] = {}
            else:
                for key in ("event", "domain", "shape", "cell_size_m", "infiltration_mmh", "building_rainfall_mode"):
                    if metadata.get(key) != merged.get(key):
                        raise ValueError(f"{event}: incompatible {key}: {metadata.get(key)!r} != {merged.get(key)!r}")

            overlap = set(merged["records"]) & set(metadata.get("records", {}))
            if overlap:
                raise ValueError(f"{event}: duplicate scenario records {sorted(overlap)}")
            merged["records"].update(metadata.get("records", {}))
            if "swmm_quality" in metadata:
                merged["swmm_quality"] = metadata["swmm_quality"]

            copied: list[dict[str, str]] = []
            for source_file in sorted(source.iterdir()):
                if not source_file.is_file() or source_file.name == "metadata.json":
                    continue
                destination_file = destination / source_file.name
                source_hash = sha256(source_file)
                if destination_file.exists():
                    if sha256(destination_file) != source_hash:
                        raise ValueError(f"{event}: conflicting file {source_file.name}")
                else:
                    shutil.copy2(source_file, destination_file)
                copied.append({"file": source_file.name, "sha256": source_hash})
            source_manifest.append({"source": str(source.resolve()), "files": copied})

        assert merged is not None
        if set(merged["records"]) >= {"A", "B", "C"}:
            a = np.load(destination / "h_A_surface_n015.npy")
            b = np.load(destination / "h_B_surface_inlet_n012.npy")
            c = np.load(destination / "h_C_itzi_swmm.npy")
            if a.shape != b.shape or a.shape != c.shape:
                raise ValueError(f"{event}: A/B/C array shapes differ")
            np.save(destination / "residual_roughness_B_minus_A.npy", (b - a).astype(np.float32))
            np.save(destination / "residual_drainage_C_minus_B.npy", (c - b).astype(np.float32))

        merged["scenario_merge"] = {
            "method": "byte-preserving copy with SHA-256 conflict checks",
            "sources": source_manifest,
        }
        (destination / "metadata.json").write_text(json.dumps(merged, indent=2), encoding="utf-8")
        print(f"{event}: merged scenarios {sorted(merged['records'])} -> {destination}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
