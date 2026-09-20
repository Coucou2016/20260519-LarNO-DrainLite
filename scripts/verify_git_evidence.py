#!/usr/bin/env python3
"""Check that non-LFS manifested evidence is staged byte-for-byte."""

import csv
import hashlib
import json
import subprocess

from verify_manifests import DEFAULT_MANIFESTS, ROOT


def main():
    listing = subprocess.check_output(
        ["git", "ls-files", "--stage", "-z"], cwd=ROOT
    ).decode("utf-8")
    indexed = {}
    for entry in listing.split("\0"):
        if entry:
            metadata, path = entry.split("\t", 1)
            indexed[path] = metadata.split()[1]
    failures = []
    checked = set()
    lfs_extensions = {".npy", ".npz", ".joblib", ".pt", ".pth", ".ckpt"}
    for manifest, base in DEFAULT_MANIFESTS:
        with manifest.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                path = base / row["path"]
                relative = path.relative_to(ROOT).as_posix()
                if path.suffix in lfs_extensions or relative in checked:
                    continue
                checked.add(relative)
                content = path.read_bytes()
                oid = hashlib.sha1(
                    f"blob {len(content)}\0".encode() + content
                ).hexdigest()
                if indexed.get(relative) != oid:
                    failures.append(relative)
    print(json.dumps({"status": "FAIL" if failures else "PASS",
                      "checked": len(checked), "failures": failures}, indent=2))
    return bool(failures)


if __name__ == "__main__":
    raise SystemExit(main())
