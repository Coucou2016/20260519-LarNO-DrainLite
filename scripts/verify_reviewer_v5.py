"""Read-only checks of V5 document structure and independently derived evidence."""
import csv
import hashlib
import json
import re
from pathlib import Path

from bs4 import BeautifulSoup
import pymupdf

ROOT = Path(__file__).resolve().parents[1]
V5 = ROOT / 'extended_study/output/reviewer_major_revision_v5'
PACKAGE = V5 / 'submission_package_v5'


def rows(name):
    with (V5 / 'diagnostics' / name).open(newline='', encoding='utf-8-sig') as stream:
        return list(csv.DictReader(stream))


def main():
    checks = []
    def check(name, passed):
        if not passed:
            raise AssertionError(name)
        checks.append(name)

    text = (PACKAGE / 'manuscript.md').read_text(encoding='utf-8')
    check('11 sequential figure captions', re.findall(r'^\*\*Figure (\d+)\.', text, re.M) == [str(i) for i in range(1, 12)])
    check('5 sequential main tables', re.findall(r'^\*\*Table (\d+)\.', text, re.M) == [str(i) for i in range(1, 6)])
    for stem in ['manuscript', 'report', 'scientific_integrity_audit', 'evidence_supplement']:
        html = (PACKAGE / f'{stem}.html').read_text(encoding='utf-8')
        soup = BeautifulSoup(html, 'html.parser')
        check(f'{stem}: complete HTML', all(soup.find(tag) is not None for tag in ['html', 'head', 'style', 'body']))
        check(f'{stem}: embedded images', all(img.get('src', '').startswith('data:image/') for img in soup.find_all('img')))
        check(f'{stem}: no external scripts/styles', not soup.select('script[src], link[rel=stylesheet]'))
        with pymupdf.open(PACKAGE / f'{stem}.pdf') as doc:
            check(f'{stem}: nonempty PDF pages', len(doc) > 0 and all(len(p.get_text().strip()) > 10 or len(p.get_images()) > 0 for p in doc))
        check(f'{stem}: all Markdown images exist', all((PACKAGE / path).is_file() for path in re.findall(r'!\[[^\n]*\]\(([^\n]+)\)', (PACKAGE / f'{stem}.md').read_text(encoding='utf-8'))))

    rain = rows('rainfall_steps.csv')
    check('576 rainfall intervals', len(rain) == 8 * 72)
    check('rainfall partition closure < 0.001 m3', all(abs(float(r['raw_m3'])-float(r['active_direct_m3'])-float(r['inactive_reassigned_m3'])) < .001 for r in rain))
    check('float32 redistribution difference < 0.1 m3/interval', all(abs(float(r['roundoff_m3'])) < .1 for r in rain))
    summary = json.loads((V5 / 'diagnostics/summary.json').read_text())
    check('no nonpositive conduit slopes', summary['network']['zero_or_negative_slopes'] == 0)
    check('event uncertainty separately disclosed', summary['event_gain_sd_mm'] > summary['seed_macro_gain_sd_mm'])
    check('full joint ledger not falsely claimed', summary['full_joint_step_ledger_available'] is False)
    manifest_path = V5 / 'artifact_manifest.json'
    check('frozen manifest present', manifest_path.is_file())
    manifest = json.loads(manifest_path.read_text())
    for relative, expected in manifest['sha256'].items():
        path = ROOT / relative
        check(f'hash: {relative}', path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == expected)
    print(json.dumps({'passed': len(checks), 'checks': checks, 'scope': 'Consistency checks, not additional physical experiments'}, indent=2))


if __name__ == '__main__':
    main()
