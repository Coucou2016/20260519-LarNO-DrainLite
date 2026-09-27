"""Explicit V6 artifact manifest generation, separate from read-only verification."""
from pathlib import Path
import hashlib
import json

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'extended_study/output/reviewer_major_revision_v6'

if __name__=='__main__':
    paths=sorted(p for name in ['diagnostics','submission_package_v6'] for p in (OUT/name).rglob('*') if p.is_file())
    paths+=sorted((ROOT/'extended_study').glob('*v6*.py'))
    manifest={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    (OUT/'artifact_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print(f'Frozen {len(paths)} files')
