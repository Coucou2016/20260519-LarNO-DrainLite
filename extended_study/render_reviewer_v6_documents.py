"""Render the editorial V6 package without altering prior evidence packages."""
import shutil
import render_reviewer_v4_documents as renderer
from revise_v6_presentation import PACKAGE, ROOT


if __name__=='__main__':
    renderer.PACKAGE=PACKAGE
    pairs=[renderer.render_one('manuscript','DrainLite manuscript V6','en'),
           renderer.render_one('report','DrainLite 科研报告 V6','zh-CN',report=True),
           renderer.render_one('scientific_integrity_audit','DrainLite evidence audit V6','en',audit=True),
           renderer.render_one('evidence_supplement','DrainLite evidence supplement V6','en',audit=True)]
    renderer.render_pdfs(pairs,renderer.chrome_path(None))
    for ext in ['md','html','pdf']:
        try:
            shutil.copy2(PACKAGE/f'report.{ext}',ROOT/f'report.{ext}')
        except OSError as exc:
            if getattr(exc,'winerror',None) not in (32,1224):raise
            print(f'Root report.{ext} is locked; current file is {PACKAGE / f"report.{ext}"}')
    print(PACKAGE)
