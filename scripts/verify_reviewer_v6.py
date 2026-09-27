"""Read-only editorial checks: figures, sources, tables and standalone exports."""
import csv
import hashlib
import json
import re
from pathlib import Path
from bs4 import BeautifulSoup
import pymupdf

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'extended_study/output/reviewer_major_revision_v6'
PACKAGE=OUT/'submission_package_v6'
OLD=ROOT/'extended_study/output/reviewer_major_revision_v5/submission_package_v5'


def main():
    checks=[]
    def check(name,condition):
        if not condition:raise AssertionError(name)
        checks.append(name)
    for stem in ['manuscript','report','evidence_supplement','scientific_integrity_audit']:
        md=(PACKAGE/f'{stem}.md').read_text(encoding='utf-8')
        html=(PACKAGE/f'{stem}.html').read_text(encoding='utf-8')
        soup=BeautifulSoup(html,'html.parser')
        check(stem+': complete HTML',all(soup.find(tag) for tag in ['html','head','style','body']))
        check(stem+': self-contained images',all(i.get('src','').startswith('data:image/') for i in soup.find_all('img')))
        check(stem+': no external runtime',not soup.select('script[src],link[rel=stylesheet]'))
        check(stem+': image files exist',all((PACKAGE/p).is_file() for p in re.findall(r'!\[[^\n]*\]\(([^\n]+)\)',md)))
        with pymupdf.open(PACKAGE/f'{stem}.pdf') as doc:
            check(stem+': no empty pages',all(p.get_text().strip() or p.get_images() for p in doc))
        if stem in ['manuscript','report']:
            check(stem+': nine ordered figures',re.findall(r'^\*\*Figure (\d+)\.',md,re.M)==list(map(str,range(1,10))))
            check(stem+': five ordered tables',re.findall(r'^\*\*Table (\d+)\.',md,re.M)==list(map(str,range(1,6))))
    md=(PACKAGE/'manuscript.md').read_text(encoding='utf-8')
    body=md.split('## References')[0]
    check('main terminology unified',not re.search(r'Itz[iï]|SWMM|Public LarNO|public LarNO',body))
    for suffix in ['png','pdf','svg']:
        name=f'figures/fig02_network_audit.{suffix}'
        check('Figure 2 unchanged '+suffix,(PACKAGE/name).read_bytes()==(OLD/name).read_bytes())
    table_pattern=r'\*\*Table 2\.[^\n]+\*\*\s*\n(?:\s*\n)?((?:\|[^\n]*\n)+)'
    check('Table 2 values unchanged',re.search(table_pattern,md)[1]==re.search(table_pattern,(OLD/'manuscript.md').read_text(encoding='utf-8'))[1])
    with (OUT/'diagnostics/sensitivity_time_series.csv').open(newline='') as stream:rows=list(csv.DictReader(stream))
    check('six scenarios x 72 frames',len(rows)==432)
    metrics=ROOT/'extended_study/output/reviewer_major_revision_v4/physical_sensitivity_event68/physical_sensitivity_metrics.csv'
    with metrics.open(newline='',encoding='utf-8-sig') as stream:
        for row in csv.DictReader(stream):
            rr=[r for r in rows if r['case']==row['case']]
            check(row['case']+': same mean residual',abs(sum(float(r['mean_absolute_residual_mm']) for r in rr)/72-float(row['drainage_effect_mae_mm']))<.001)
            check(row['case']+': same final volume',abs(float(rr[-1]['surface_volume_difference_million_m3'])*1e6-float(row['final_surface_reduction_m3']))<1)
    manifest=json.loads((OUT/'artifact_manifest.json').read_text(encoding='utf-8'))
    for name,expected in manifest.items():
        path=ROOT/name
        check('hash: '+name,path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest()==expected)
    print(json.dumps({'passed':len(checks),'checks':checks},indent=2))


if __name__=='__main__':main()
