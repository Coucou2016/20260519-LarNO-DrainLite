#!/usr/bin/env python3
"""Hash canonical data and model artifacts for independent provenance checks."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "canonical_data_sha256.csv"
TARGETS = [
    ROOT / "LarNO-main" / "benchmark" / "urbanflood" / "flood" / "region1_20m_drainage_v3_full",
    ROOT / "LarNO-main" / "benchmark" / "urbanflood" / "geodata" / "region1_20m_drainage_v3_full",
    ROOT / "LarNO-main" / "exp" / "20260220_183648_006352",
    ROOT / "extended_study" / "output" / "reviewer_major_revision_v3" / "drainlite_hybrid_v3",
    ROOT / "extended_study" / "output" / "reviewer_major_revision_v3" / "formal_matched_full",
    ROOT / "extended_study" / "output" / "reviewer_major_revision_v3" / "network",
    ROOT / "extended_study" / "output" / "reviewer_major_revision_v3" / "submission_package_v3",
    ROOT / "extended_study" / "output" / "reviewer_major_revision_v4" / "final_hybrid_controls",
    ROOT / "extended_study" / "output" / "reviewer_major_revision_v4" / "submission_package_v4",
    ROOT / "extended_study" / "output" / "reviewer_major_revision_v4" / "physical_sensitivity_event68",
]


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def included(path: Path) -> bool:
    relative = path.relative_to(ROOT).as_posix()
    if "/formal_matched_full/" in f"/{relative}" and path.suffix == ".npy":
        return False
    return not any(part in {"__pycache__", "chrome-profile", ".browser_pdf_profile"} for part in path.parts)


def main() -> None:
    files = sorted(
        {
            path
            for target in TARGETS
            if target.exists()
            for path in target.rglob("*")
            if path.is_file() and included(path)
        },
        key=lambda path: path.as_posix(),
    )
    with OUTPUT.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "bytes", "sha256"])
        writer.writeheader()
        for path in files:
            writer.writerow(
                {
                    "path": path.relative_to(ROOT).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": digest(path),
                }
            )
    print(f"Wrote {len(files)} entries to {OUTPUT}")


if __name__ == "__main__":
    main()
