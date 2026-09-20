"""Render PDF contact sheets and record page-level layout diagnostics."""
import json
from pathlib import Path
import pymupdf
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "extended_study/output/reviewer_major_revision_v4/submission_package_v4"
OUT = ROOT / "tmp/v4_pdf_review"

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    diagnostics = {}
    for stem in ("manuscript", "report", "scientific_integrity_audit"):
        doc = pymupdf.open(PACKAGE / f"{stem}.pdf")
        pages = []
        for page in doc:
            pix = page.get_pixmap(matrix=pymupdf.Matrix(0.65, 0.65))
            thumb = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            tile = Image.new("RGB", (400, 580), "#dddddd")
            tile.paste(thumb, (4, 25))
            ImageDraw.Draw(tile).text((8, 5), f"{stem} p{page.number+1}", fill="black")
            pages.append(tile)
        for start in range(0, len(pages), 12):
            selected = pages[start:start+12]
            sheet = Image.new("RGB", (1600, 580*((len(selected)+3)//4)), "white")
            for i, tile in enumerate(selected):
                sheet.paste(tile, ((i%4)*400,(i//4)*580))
            sheet.save(OUT / f"{stem}_{start+1}.png")
        diagnostics[stem] = {"pages": len(doc), "page_text_characters": [len(p.get_text()) for p in doc]}
    (OUT / "page_diagnostics.json").write_text(json.dumps(diagnostics, indent=2))
    print(json.dumps(diagnostics, indent=2))

if __name__ == "__main__":
    main()
