#!/usr/bin/env python3
"""Read-only verification of every Git LFS path and local object."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def lfs_executable() -> str:
    discovered = shutil.which("git-lfs")
    if discovered:
        return discovered
    bundled = (
        ROOT
        / "cache"
        / "tools"
        / "git-lfs-v3.8.0"
        / "git-lfs-3.8.0"
        / "git-lfs.exe"
    )
    if bundled.is_file():
        return str(bundled)
    raise FileNotFoundError("git-lfs is neither on PATH nor in the bundled cache")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> int:
    executable = lfs_executable()
    listed = subprocess.run(
        [executable, "ls-files", "--long"],
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    if listed.returncode:
        print(json.dumps({"status": "FAIL", "stderr": listed.stderr}, indent=2))
        return 1
    failures: list[str] = []
    records = []
    for line in listed.stdout.splitlines():
        if not line.strip():
            continue
        oid, marker, relative = line.split(maxsplit=2)
        path = ROOT / relative
        if not path.is_file():
            failures.append(f"missing: {relative}")
            continue
        if path.stat().st_size <= 1024 and path.read_bytes().startswith(
            b"version https://git-lfs.github.com/spec/v1"
        ):
            failures.append(f"unresolved pointer: {relative}")
            continue
        actual = digest(path)
        ok = actual.lower() == oid.lower()
        if not ok:
            failures.append(f"object hash mismatch: {relative}")
        records.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "expected_oid": oid,
                "actual_sha256": actual,
                "match": ok,
                "lfs_marker": marker,
            }
        )
    fsck = subprocess.run(
        [executable, "fsck", "--objects", "--dry-run"],
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    if fsck.returncode:
        failures.append(f"git lfs fsck failed: {fsck.stderr.strip()}")
    result = {
        "status": "PASS" if not failures else "FAIL",
        "tracked_paths": len(records),
        "materialized_paths": sum(bool(row["match"]) for row in records),
        "fsck_returncode": fsck.returncode,
        "failures": failures,
    }
    print(json.dumps(result, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
