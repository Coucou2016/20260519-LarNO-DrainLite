"""Explicit artifact-generation step; never called by the read-only verifier."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V5 = ROOT / 'extended_study/output/reviewer_major_revision_v5'

if __name__ == '__main__':
    paths = sorted(p for folder in ['diagnostics', 'submission_package_v5'] for p in (V5 / folder).rglob('*') if p.is_file())
    paths += sorted((ROOT / 'extended_study').glob('*v5*.py'))
    manifest = {'scope': 'V5 presentation and diagnostics; raw V3/V4 sources retain their existing manifests',
                'sha256': {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    (V5 / 'artifact_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(f'Frozen {len(paths)} files')
