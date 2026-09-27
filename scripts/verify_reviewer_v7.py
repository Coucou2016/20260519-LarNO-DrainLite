"""Read-only V7 audit verification; does not declare hydraulic validation complete."""
import csv
import hashlib
import re
from pathlib import Path
from bs4 import BeautifulSoup
import pymupdf

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'extended_study/output/reviewer_major_revision_v7'
PACKAGE=OUT/'submission_package_v7'

if __name__=='__main__':
    for stem in ['manuscript','report','evidence_supplement','scientific_integrity_audit']:
        md=(PACKAGE/f'{stem}.md').read_text(encoding='utf-8')
        soup=BeautifulSoup((PACKAGE/f'{stem}.html').read_text(encoding='utf-8'),'html.parser')
        assert all(soup.find(tag) for tag in ['html','head','style','body'])
        assert all(i.get('src','').startswith('data:image/') for i in soup.find_all('img'))
        assert not soup.select('script[src],link[rel=stylesheet]')
        with pymupdf.open(PACKAGE/f'{stem}.pdf') as doc:
            assert all(p.get_text().strip() or p.get_images() for p in doc)
        if stem in ['manuscript','report']:
            assert re.findall(r'^\*\*Table (\d+)\.',md,re.M)==list(map(str,range(1,5)))
            assert re.findall(r'^\*\*Figure (\d+)\.',md,re.M)==list(map(str,range(1,10)))
    with (OUT/'diagnostics/routing_report_audit.csv').open(newline='') as stream:
        rows=list(csv.DictReader(stream))
    assert len(rows)==8
    for r in rows:
        path=ROOT/'extended_study/output/reviewer_major_revision_v3/formal_matched_full'/r['event']/'swmm_C.rpt'
        assert hashlib.sha256(path.read_bytes()).hexdigest()==r['report_sha256']
        assert r['assessment']=='Global screen passes; local hydraulic adequacy unresolved'
    text=(PACKAGE/'manuscript.md').read_text(encoding='utf-8')
    assert '63.01%' in text and 'local node-balance errors remain unresolved' in text
    assert 'All eight events met the physical acceptance conditions' not in text
    print('PASS: document structure, embedded images, raw-report hashes and qualified conclusions. Local hydraulic adequacy remains unresolved.')
