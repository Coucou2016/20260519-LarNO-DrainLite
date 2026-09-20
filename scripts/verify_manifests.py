#!/usr/bin/env python3
"""Read-only verification of committed SHA-256 manifests."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFESTS = [
    (ROOT / "canonical_data_sha256.csv", ROOT),
    (
        ROOT
        / "extended_study"
        / "output"
        / "reviewer_major_revision_v3"
        / "submission_package_v3"
        / "sha256_manifest.csv",
        ROOT
        / "extended_study"
        / "output"
        / "reviewer_major_revision_v3"
        / "submission_package_v3",
    ),
    (
        ROOT
        / "extended_study"
        / "output"
        / "reviewer_major_revision_v4"
        / "submission_package_v4"
        / "sha256_manifest.csv",
        ROOT
        / "extended_study"
        / "output"
        / "reviewer_major_revision_v4"
        / "submission_package_v4",
    ),
]


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def verify(manifest: Path, base: Path) -> dict[str, object]:
    failures: list[str] = []
    checked = 0
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            checked += 1
            path = base / row["path"]
            if not path.is_file():
                failures.append(f"missing: {path}")
                continue
            expected_size = int(row["bytes"])
            if path.stat().st_size != expected_size:
                failures.append(
                    f"size: {path} expected {expected_size}, got {path.stat().st_size}"
                )
                continue
            actual = digest(path)
            if actual.lower() != row["sha256"].lower():
                failures.append(f"sha256: {path}")
    return {
        "manifest": str(manifest.relative_to(ROOT)),
        "records": checked,
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        action="append",
        type=Path,
        help="Additional manifest path; its parent is used as the path base.",
    )
    args = parser.parse_args()
    specifications = list(DEFAULT_MANIFESTS)
    for manifest in args.manifest or []:
        resolved = manifest.resolve()
        specifications.append((resolved, resolved.parent))
    results = []
    for manifest, base in specifications:
        if not manifest.is_file():
            results.append(
                {
                    "manifest": str(manifest),
                    "records": 0,
                    "status": "FAIL",
                    "failures": ["manifest missing"],
                }
            )
        else:
            results.append(verify(manifest, base))
    status = "PASS" if all(row["status"] == "PASS" for row in results) else "FAIL"
    print(json.dumps({"status": status, "results": results}, indent=2))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
