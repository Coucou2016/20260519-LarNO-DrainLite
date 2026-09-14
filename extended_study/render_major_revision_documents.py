#!/usr/bin/env python3
"""Render the major-revision manuscript, report and evidence audit."""

from __future__ import annotations

import argparse
from pathlib import Path

from bs4 import BeautifulSoup

from render_drainlite_documents import (
    ROOT,
    chrome_path,
    document_html,
    embed_local_images,
    render_pdfs,
)


def add_report_cover(html_text: str) -> str:
    soup = BeautifulSoup(html_text, "html.parser")
    first_heading = soup.body.find("h1") if soup.body else None
    if first_heading is None:
        return html_text
    cover = soup.new_tag("section")
    cover["class"] = "report-cover"
    first_heading.extract()
    cover.append(first_heading)
    subtitle = soup.new_tag("p")
    subtitle["class"] = "cover-subtitle"
    subtitle.string = "独立科研报告"
    cover.append(subtitle)
    scope = soup.new_tag("p")
    scope["class"] = "cover-scope"
    scope.string = "20 m grid | paired ITZI-SWMM simulations | five-event leave-one-event-out evaluation"
    cover.append(scope)
    note = soup.new_tag("p")
    note["class"] = "cover-note"
    note.string = "依据本地数组、代码、指标与图件生成；数据来源和证据边界均在正文中说明。"
    cover.append(note)
    soup.body.insert(0, cover)
    style = soup.find("style")
    if style is not None:
        style.append(
            """
            .report-cover { min-height: 230mm; display: flex; flex-direction: column;
              justify-content: center; border-top: 5px solid #1f4e79;
              border-bottom: 1px solid #9ca3af; page-break-after: always; }
            .report-cover h1 { font-size: 25pt; line-height: 1.35; margin: 0 0 28px;
              color: #17365d; }
            .cover-subtitle { text-align: center; font-size: 15pt; color: #334155;
              margin: 0 0 34px; }
            .cover-scope { text-align: center; font-size: 10.5pt; color: #475569;
              margin: 0 0 10px; }
            .cover-note { text-align: center; font-size: 9pt; color: #64748b;
              max-width: 135mm; margin: 0 auto; }
            """
        )
    return str(soup)


def allow_table_page_breaks(html_text: str) -> str:
    soup = BeautifulSoup(html_text, "html.parser")
    style = soup.find("style")
    if style is not None:
        style.append(
            """
            table { break-inside: auto !important; page-break-inside: auto !important; }
            tr { break-inside: avoid; page-break-inside: avoid; }
            td { overflow-wrap: anywhere; word-break: break-all; }
            """
        )
    return str(soup)


def render_markdown(
    markdown_path: Path,
    title: str,
    language: str,
    cover: bool = False,
    split_tables: bool = False,
) -> Path:
    html_path = markdown_path.with_suffix(".html")
    rendered = document_html(markdown_path.read_text(encoding="utf-8"), title, language)
    if cover:
        rendered = add_report_cover(rendered)
    if split_tables:
        rendered = allow_table_page_breaks(rendered)
    html_path.write_text(embed_local_images(rendered, markdown_path.parent), encoding="utf-8")
    return html_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--package-dir",
        default="extended_study/output/larno_drainlite_major_revision",
    )
    parser.add_argument("--chrome", default=None)
    args = parser.parse_args()

    package = ROOT / args.package_dir
    manuscript_md = package / "manuscript_major_revision.md"
    report_md = package / "report.md"
    audit_md = package / "scientific_integrity_audit.md"

    manuscript_html = render_markdown(
        manuscript_md,
        "DrainLite: lightweight learning of drainage-induced residuals for urban flood prediction",
        "en",
    )
    report_html = render_markdown(
        report_md,
        "基于成对 ITZI-SWMM 模拟的轻量排水残差学习研究报告",
        "zh-CN",
        cover=True,
    )
    audit_html = render_markdown(
        audit_md,
        "Scientific integrity and reproducibility audit",
        "en",
        split_tables=True,
    )

    pairs = []
    for html_path in (manuscript_html, report_html, audit_html):
        pairs.append((html_path, html_path.with_suffix(".pdf")))
    render_pdfs(pairs, chrome_path(args.chrome))

    for html_path, pdf_path in pairs:
        print(html_path)
        print(pdf_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
