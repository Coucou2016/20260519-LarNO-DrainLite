"""Export the V5 manuscript, report, supplement and audit to standalone HTML/PDF."""
import shutil
import render_reviewer_v4_documents as renderer
from generate_reviewer_v5_documents import PACKAGE, ROOT


def main():
    renderer.PACKAGE=PACKAGE
    pairs=[renderer.render_one("manuscript","DrainLite manuscript V5","en"),
           renderer.render_one("report","DrainLite 科研报告 V5","zh-CN",report=True),
           renderer.render_one("scientific_integrity_audit","DrainLite evidence audit V5","en",audit=True),
           renderer.render_one("evidence_supplement","DrainLite evidence supplement V5","en",audit=True)]
    renderer.render_pdfs(pairs,renderer.chrome_path(None))
    for ext in ["md","html","pdf"]:
        try:
            shutil.copy2(PACKAGE/f"report.{ext}",ROOT/f"report.{ext}")
        except OSError as exc:
            if getattr(exc,"winerror",None) not in (32,1224):
                raise
            print(f"Root report.{ext} is open in another application; current export remains at {PACKAGE / f'report.{ext}'}")
    print(PACKAGE)


if __name__=="__main__":main()
