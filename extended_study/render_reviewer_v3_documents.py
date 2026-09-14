#!/usr/bin/env python3
"""Render reviewer-v3 Markdown documents to standalone HTML and PDF."""

from __future__ import annotations

import shutil
from pathlib import Path

from bs4 import BeautifulSoup

from render_drainlite_documents import ROOT, chrome_path, document_html, embed_local_images, render_pdfs


PACKAGE = ROOT / "extended_study" / "output" / "reviewer_major_revision_v3" / "submission_package_v3"


def polish(html_text: str, *, report: bool = False, audit: bool = False) -> str:
    soup = BeautifulSoup(html_text, "html.parser")
    style = soup.find("style")
    if style is not None:
        style.append(
            """
            body { color: #171717; line-height: 1.55; }
            table { font-size: 7.25pt; break-inside: auto; page-break-inside: auto; }
            tr { break-inside: avoid; page-break-inside: avoid; }
            th { background: #edf2f6; }
            img { width: 100%; max-height: 238mm; object-fit: contain; }
            p:has(> img) { margin-top: 12px; }
            p > strong:first-child { color: #273746; }
            h2 { color: #17365d; border-bottom-color: #7893aa; }
            h3 { color: #274c68; }
            a { color: #1f4e79; text-decoration: none; }
            """
        )
    if report:
        first = soup.body.find("h1")
        cover = soup.new_tag("section")
        cover["class"] = "report-cover"
        if first is not None:
            first.extract()
            cover.append(first)
        subtitle = soup.new_tag("p")
        subtitle["class"] = "cover-subtitle"
        subtitle.string = "独立、自包含科研报告"
        cover.append(subtitle)
        scope = soup.new_tag("p")
        scope["class"] = "cover-scope"
        scope.string = "20 米全域网格 | 八场降雨事件 | 原生 Itzï-SWMM 双向耦合 | 严格留一事件验证"
        cover.append(scope)
        provenance = soup.new_tag("p")
        provenance["class"] = "cover-note"
        provenance.string = "全部数值由本地正式 V3 数组、CSV 和 JSON 自动生成；MIKE 为外部参照，不是学习标签。"
        cover.append(provenance)
        soup.body.insert(0, cover)
        if style is not None:
            style.append(
                """
                .report-cover { min-height: 245mm; display: flex; flex-direction: column;
                  justify-content: center; border-top: 5px solid #17365d;
                  border-bottom: 1px solid #94a3b8; page-break-after: always; }
                .report-cover h1 { font-size: 25pt; line-height: 1.35; color: #17365d; }
                .cover-subtitle { text-align: center; font-size: 15pt; color: #334155; margin: 20px 0 32px; }
                .cover-scope { text-align: center; font-size: 10.5pt; color: #475569; }
                .cover-note { text-align: center; font-size: 9pt; color: #64748b; max-width: 140mm; margin: 10px auto; }
                """
            )
    if audit and style is not None:
        style.append("table { font-size: 6.7pt; } td { overflow-wrap: anywhere; }")
    return "<!DOCTYPE html>\n" + str(soup)


def render_one(stem: str, title: str, language: str, **kwargs: bool) -> tuple[Path, Path]:
    md = PACKAGE / f"{stem}.md"
    html = PACKAGE / f"{stem}.html"
    pdf = PACKAGE / f"{stem}.pdf"
    raw = document_html(md.read_text(encoding="utf-8"), title, language)
    standalone = embed_local_images(polish(raw, **kwargs), PACKAGE)
    html.write_text(standalone, encoding="utf-8")
    return html, pdf


def main() -> int:
    pairs = [
        render_one("manuscript", "DrainLite: leakage-controlled learning of conceptual sewer effects", "en"),
        render_one("report", "概化排水管网影响的轻量残差学习科研报告", "zh-CN", report=True),
        render_one("scientific_integrity_audit", "Scientific integrity and reproducibility audit", "en", audit=True),
    ]
    render_pdfs(pairs, chrome_path(None))
    shutil.copy2(PACKAGE / "report.html", ROOT / "report.html")
    for html, pdf in pairs:
        print(html)
        print(pdf)
    print(ROOT / "report.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
