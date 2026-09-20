#!/usr/bin/env python3
"""Render V4 Markdown documents as standalone HTML and paged PDF."""

from __future__ import annotations

import shutil
from pathlib import Path

from bs4 import BeautifulSoup

from render_drainlite_documents import ROOT, chrome_path, document_html, embed_local_images, render_pdfs


PACKAGE = ROOT / "extended_study" / "output" / "reviewer_major_revision_v4" / "submission_package_v4"


def polish(html_text: str, *, report: bool = False, audit: bool = False) -> str:
    soup = BeautifulSoup(html_text, "html.parser")
    style = soup.find("style")
    if style is not None:
        style.append(
            """
            body { color: #171717; line-height: 1.56; }
            table { font-size: 7.15pt; break-inside: auto; page-break-inside: auto; }
            tr { break-inside: avoid; page-break-inside: avoid; }
            th { background: #edf2f6; }
            img { width: 100%; max-height: 236mm; object-fit: contain; }
            p:has(> img) { margin-top: 12px; }
            p > strong:first-child { color: #273746; }
            h2 { color: #17365d; border-bottom-color: #7893aa; }
            h3 { color: #274c68; }
            a { color: #1f4e79; text-decoration: none; }
            code { overflow-wrap: anywhere; }
            figure { margin: 12px 0; break-inside: avoid; page-break-inside: avoid; }
            figure img { max-height: 208mm; }
            figcaption { font-size: 9pt; line-height: 1.4; }
            .short-table { break-inside: avoid; page-break-inside: avoid; }
            """
        )
    for paragraph in list(soup.find_all("p")):
        if paragraph.find("img", recursive=False) is None:
            continue
        caption = paragraph.find_next_sibling()
        if caption is None or caption.name != "p" or not caption.get_text().startswith("Figure "):
            continue
        figure = soup.new_tag("figure")
        paragraph.insert_before(figure)
        figure.append(paragraph.extract())
        caption.name = "figcaption"
        figure.append(caption.extract())
    for table in soup.find_all("table"):
        if len(table.find_all("tr")) <= 12:
            table["class"] = ["short-table"]
    if report:
        first = soup.body.find("h1")
        cover = soup.new_tag("section")
        cover["class"] = "report-cover"
        if first is not None:
            first.extract()
            cover.append(first)
        for css_class, value in [
            ("cover-subtitle", "独立、自包含科研报告"),
            ("cover-scope", "20 米全域网格 | 八场降雨 | 原生 Itzï-SWMM 双向耦合 | 五随机种子复验"),
            ("cover-note", "全部统计由正式 V3 物理数组和 V4 留一事件预测自动生成；MIKE 为描述性外部参照。"),
        ]:
            paragraph = soup.new_tag("p")
            paragraph["class"] = css_class
            paragraph.string = value
            cover.append(paragraph)
        soup.body.insert(0, cover)
        if style is not None:
            style.append(
                """
                .report-cover { min-height: 245mm; display: flex; flex-direction: column;
                  justify-content: center; border-top: 5px solid #17365d;
                  border-bottom: 1px solid #94a3b8; page-break-after: always; }
                .report-cover h1 { font-size: 24pt; line-height: 1.35; color: #17365d; }
                .cover-subtitle { text-align: center; font-size: 15pt; color: #334155; margin: 20px 0 32px; }
                .cover-scope { text-align: center; font-size: 10.5pt; color: #475569; }
                .cover-note { text-align: center; font-size: 9pt; color: #64748b; max-width: 145mm; margin: 10px auto; }
                """
            )
    if audit and style is not None:
        style.append("table { font-size: 6.6pt; } td { overflow-wrap: anywhere; }")
    return "<!DOCTYPE html>\n" + str(soup)


def render_one(stem: str, title: str, language: str, **kwargs: bool) -> tuple[Path, Path]:
    markdown = PACKAGE / f"{stem}.md"
    html = PACKAGE / f"{stem}.html"
    pdf = PACKAGE / f"{stem}.pdf"
    raw = document_html(markdown.read_text(encoding="utf-8"), title, language)
    html.write_text(embed_local_images(polish(raw, **kwargs), PACKAGE), encoding="utf-8")
    return html, pdf


def main() -> int:
    pairs = [
        render_one("manuscript", "DrainLite: fixed-network drainage-residual emulation", "en"),
        render_one("report", "固定概化排水管网影响的轻量残差模拟科研报告", "zh-CN", report=True),
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
