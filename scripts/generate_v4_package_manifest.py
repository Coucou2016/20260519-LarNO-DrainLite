#!/usr/bin/env python3
"""Generate the V4 package manifest; this command intentionally writes evidence."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "extended_study" / "output" / "reviewer_major_revision_v4" / "submission_package_v4"
OUTPUT = PACKAGE / "sha256_manifest.csv"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> int:
    files = sorted(
        path for path in PACKAGE.rglob("*")
        if path.is_file() and path != OUTPUT and "__pycache__" not in path.parts
    )
    with OUTPUT.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "bytes", "sha256"])
        writer.writeheader()
        for path in files:
            writer.writerow({"path": path.relative_to(PACKAGE).as_posix(), "bytes": path.stat().st_size, "sha256": digest(path)})
    print(f"Wrote {len(files)} records to {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
