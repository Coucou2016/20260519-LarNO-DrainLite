"""Render the quality-qualified manuscript and report."""
import shutil
import render_reviewer_v4_documents as renderer
from review_v7_numerical_quality import PACKAGE,ROOT

if __name__=='__main__':
    renderer.PACKAGE=PACKAGE
    pairs=[renderer.render_one('manuscript','DrainLite manuscript V7','en'),
           renderer.render_one('report','DrainLite 科研报告 V7','zh-CN',report=True),
           renderer.render_one('scientific_integrity_audit','DrainLite numerical evidence audit V7','en',audit=True),
           renderer.render_one('evidence_supplement','DrainLite evidence supplement V7','en',audit=True)]
    renderer.render_pdfs(pairs,renderer.chrome_path(None))
    for ext in ['md','html','pdf']:
        try:shutil.copy2(PACKAGE/f'report.{ext}',ROOT/f'report.{ext}')
        except OSError as exc:
            if getattr(exc,'winerror',None) not in (32,1224):raise
            print(f'Root report.{ext} locked; current version remains in {PACKAGE}')
    print(PACKAGE)
