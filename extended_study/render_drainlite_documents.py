#!/usr/bin/env python3
"""Render the DrainLite manuscript, report, and integrity audit deliverables."""

from __future__ import annotations

import argparse
import base64
import mimetypes
import shutil
from pathlib import Path

import markdown
from bs4 import BeautifulSoup
from markdownify import markdownify as to_markdown
from playwright.sync_api import sync_playwright


ROOT = Path(__file__).resolve().parents[1]
CHROME_CANDIDATES = (
    Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
    Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
    Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
)


def document_html(markdown_text: str, title: str, language: str = "en") -> str:
    body = markdown.markdown(markdown_text, extensions=["tables", "fenced_code", "sane_lists"])
    return f"""<!DOCTYPE html>
<html lang="{language}">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
  @page {{ size: A4; margin: 18mm 17mm 19mm; }}
  body {{ max-width: 178mm; margin: 0 auto; color: #111827; font-family: "Times New Roman", "SimSun", "Songti SC", serif; font-size: 10.5pt; line-height: 1.48; }}
  h1 {{ font-size: 20pt; line-height: 1.18; color: #111827; margin: 0 0 16px; text-align: center; letter-spacing: 0; }}
  h2 {{ font-size: 14.5pt; color: #111827; border-bottom: 0.7pt solid #9ca3af; padding-bottom: 4px; margin: 25px 0 9px; letter-spacing: 0; }}
  h3 {{ font-size: 11.5pt; color: #111827; margin: 18px 0 6px; letter-spacing: 0; }}
  p {{ text-align: justify; margin: 6px 0; orphans: 3; widows: 3; }}
  ul, ol {{ margin: 6px 0 8px 22px; padding: 0; }}
  li {{ margin: 3px 0; }}
  table {{ width: 100%; border-collapse: collapse; margin: 9px 0 15px; font-size: 8.2pt; break-inside: avoid; page-break-inside: avoid; }}
  th, td {{ border: 0.55pt solid #6b7280; padding: 4px 5px; vertical-align: top; }}
  th {{ background: #eef1f4; text-align: left; font-weight: 700; }}
  img {{ display: block; max-width: 100%; height: auto; margin: 9px auto 5px; break-inside: avoid; page-break-inside: avoid; }}
  p:has(> img) {{ break-inside: avoid; page-break-inside: avoid; }}
  strong {{ font-weight: 700; }}
  code {{ font-family: Consolas, "Courier New", monospace; background: #f3f4f6; padding: 0 2px; overflow-wrap: anywhere; }}
  pre {{ white-space: pre-wrap; background: #f3f4f6; padding: 7px; font-size: 8.5pt; }}
  blockquote {{ margin: 8px 0 8px 18px; padding-left: 10px; border-left: 2px solid #9ca3af; color: #374151; }}
  h2, h3 {{ break-after: avoid; page-break-after: avoid; }}
</style>
</head>
<body>{body}</body>
</html>"""


def embed_local_images(html_text: str, base_dir: Path) -> str:
    soup = BeautifulSoup(html_text, "html.parser")
    for image in soup.find_all("img"):
        src = image.get("src", "")
        if not src or src.startswith("data:") or "://" in src:
            continue
        image_path = (base_dir / src).resolve()
        if not image_path.exists():
            raise FileNotFoundError(f"Image referenced by document is missing: {image_path}")
        mime = mimetypes.guess_type(image_path.name)[0] or "image/png"
        payload = base64.b64encode(image_path.read_bytes()).decode("ascii")
        image["src"] = f"data:{mime};base64,{payload}"
    return str(soup)


def report_markdown(html_text: str, output_path: Path) -> None:
    soup = BeautifulSoup(html_text, "html.parser")
    for tag in soup.find_all(["style", "script", "nav"]):
        tag.decompose()
    image_dir = output_path.parent / "report_markdown_figures"
    if image_dir.exists():
        shutil.rmtree(image_dir)
    image_dir.mkdir(parents=True, exist_ok=True)
    for index, image in enumerate(soup.find_all("img"), start=1):
        src = image.get("src", "")
        if not src.startswith("data:image/") or "," not in src:
            continue
        header, payload = src.split(",", 1)
        extension = header.split("/", 1)[1].split(";", 1)[0].replace("jpeg", "jpg")
        filename = f"figure_{index:02d}.{extension}"
        (image_dir / filename).write_bytes(base64.b64decode(payload))
        image["src"] = f"report_markdown_figures/{filename}"
    content = soup.find("main") or soup.find("body") or soup
    converted = to_markdown(str(content), heading_style="ATX", bullets="-")
    output_path.write_text(converted.strip() + "\n", encoding="utf-8")


def chrome_path(explicit: str | None) -> Path:
    if explicit:
        candidate = Path(explicit)
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"Chrome/Edge executable not found: {candidate}")
    for candidate in CHROME_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("No Chrome or Edge executable was found")


def render_pdfs(pairs: list[tuple[Path, Path]], executable: Path) -> None:
    footer = """
    <div style="width:100%;font-family:'Times New Roman',serif;font-size:8px;color:#6b7280;text-align:center;">
      <span class="pageNumber"></span>
    </div>
    """
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            headless=True,
            executable_path=str(executable),
            args=["--allow-file-access-from-files"],
        )
        try:
            page = browser.new_page(viewport={"width": 1440, "height": 1000})
            for html_path, pdf_path in pairs:
                page.goto(html_path.resolve().as_uri(), wait_until="load", timeout=120_000)
                page.emulate_media(media="print")
                page.pdf(
                    path=str(pdf_path),
                    format="A4",
                    print_background=True,
                    prefer_css_page_size=True,
                    display_header_footer=True,
                    header_template="<div></div>",
                    footer_template=footer,
                )
        finally:
            browser.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--package-dir",
        default="extended_study/output/larno_drainlite_v2_inf1mmh_package",
    )
    parser.add_argument(
        "--report-html",
        default="extended_study/output/drainlite_connected_residual_v2_inf1mmh/report.html",
    )
    parser.add_argument("--chrome", default=None)
    args = parser.parse_args()

    package = ROOT / args.package_dir
    manuscript_md = package / "manuscript_larno_drainlite_v2_inf1mmh.md"
    manuscript_html = package / "manuscript_larno_drainlite_v2_inf1mmh.html"
    manuscript_pdf = package / "manuscript_larno_drainlite_v2_inf1mmh.pdf"
    audit_md = package / "scientific_integrity_audit.md"
    audit_html = package / "scientific_integrity_audit.html"
    audit_pdf = package / "scientific_integrity_audit.pdf"
    report_html = ROOT / args.report_html
    report_md = report_html.with_name("report.md")
    report_pdf = report_html.with_name("report.pdf")

    manuscript_rendered = document_html(
        manuscript_md.read_text(encoding="utf-8"),
        "Drainage-aware residual correction for LarNO-compatible urban flood forecasting",
    )
    manuscript_html.write_text(
        embed_local_images(manuscript_rendered, manuscript_md.parent), encoding="utf-8"
    )
    audit_rendered = document_html(
        audit_md.read_text(encoding="utf-8"),
        "Scientific integrity and reproducibility audit",
    )
    audit_html.write_text(embed_local_images(audit_rendered, audit_md.parent), encoding="utf-8")
    report_text = report_html.read_text(encoding="utf-8")
    report_markdown(report_text, report_md)

    render_pdfs(
        [
            (manuscript_html, manuscript_pdf),
            (report_html, report_pdf),
            (audit_html, audit_pdf),
        ],
        chrome_path(args.chrome),
    )

    for path in (manuscript_html, manuscript_pdf, report_md, report_pdf, audit_html, audit_pdf):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
