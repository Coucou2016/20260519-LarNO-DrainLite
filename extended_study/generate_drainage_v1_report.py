#!/usr/bin/env python3
"""Generate a standalone HTML report for the LarNO drainage v1 implementation."""

from __future__ import annotations

import base64
import csv
import html
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "extended_study" / "output" / "drainage_dataset_v1"
REPORT = OUT / "larno_drainage_v1_implementation_report.html"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def img_uri(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def table(rows: list[dict[str, object]], columns: list[str]) -> str:
    head = "".join(f"<th>{html.escape(col)}</th>" for col in columns)
    body = []
    for row in rows:
        body.append("<tr>" + "".join(f"<td>{html.escape(str(row.get(col, '')))}</td>" for col in columns) + "</tr>")
    return f"<div class='table-wrap'><table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table></div>"


def main() -> int:
    quality = read_csv(OUT / "dataset_quality.csv")
    validation = (OUT / "validation_summary.txt").read_text(encoding="utf-8") if (OUT / "validation_summary.txt").exists() else "待补充"
    arrays = read_csv(OUT / "validation_arrays.csv") if (OUT / "validation_arrays.csv").exists() else []
    qa_images = sorted(OUT.glob("*_qa.png"))
    qa_html = "".join(
        f"<figure><figcaption>{html.escape(path.stem)}</figcaption><img src='{img_uri(path)}' alt='{html.escape(path.name)}'></figure>"
        for path in qa_images
    )
    columns = [
        "event", "rainfall_shape", "h_shape", "mike_peak_m", "itzi_surface_peak_m",
        "itzi_sink_peak_m", "sink_minus_surface_peak_m", "mean_sink_reduction_m", "swmm_status",
    ]
    array_columns = ["file", "shape", "dtype", "nan_count", "inf_count", "min", "max", "status"]
    css = """
    body { margin:0; background:#eef2f5; color:#17202a; font-family:'Microsoft YaHei',Arial,sans-serif; line-height:1.65; }
    .page { max-width:1180px; margin:0 auto; background:white; padding:42px 58px; box-shadow:0 18px 60px rgba(0,0,0,.16); }
    h1 { font-size:30px; margin:0 0 12px; }
    h2 { margin-top:32px; border-bottom:2px solid #d6dbdf; padding-bottom:8px; }
    .lead { background:#f4f9fb; border-left:4px solid #1f618d; padding:12px 14px; }
    .cards { display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin:18px 0; }
    .card { border:1px solid #d6dbdf; border-radius:8px; padding:12px; }
    .label { color:#5d6d7e; font-size:13px; }
    .value { font-size:24px; font-weight:700; color:#1f618d; }
    table { width:100%; border-collapse:collapse; font-size:13px; }
    th,td { border-bottom:1px solid #e5e8e8; padding:7px 8px; white-space:nowrap; }
    th { background:#f8f9f9; text-align:left; }
    .table-wrap { overflow:auto; border:1px solid #d6dbdf; border-radius:8px; margin:12px 0 20px; }
    figure { border:1px solid #d6dbdf; border-radius:8px; overflow:hidden; margin:18px 0; }
    figcaption { font-weight:700; background:#eef3f6; padding:10px 12px; }
    img { display:block; width:100%; height:auto; }
    code { background:#f4f6f7; padding:1px 4px; border:1px solid #e5e8e8; border-radius:4px; }
    pre { background:#17202a; color:#f8f9f9; padding:14px; overflow:auto; border-radius:8px; }
    @media(max-width:900px){ .page{padding:24px;} .cards{grid-template-columns:1fr;} th,td{white-space:normal;} }
    """
    html_text = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>LarNO Drainage v1 Implementation Report</title>
  <style>{css}</style>
</head>
<body>
<main class="page">
  <h1>LarNO-D / DrainAdapter 城市排水管网增强实现报告</h1>
  <p class="lead">本报告记录本轮实现结果：构建 <code>region1_20m_drainage_v1</code> 数据集，新增 8 个管网栅格特征，生成 72 步 ITZI surface/sink 监督标签，扩展 LarNO-D 21 通道输入，并实现兼容预训练权重的 DrainAdapter 微调路线。</p>
  <div class="cards">
    <div class="card"><div class="label">事件数</div><div class="value">{len(quality)}</div></div>
    <div class="card"><div class="label">训练/测试拆分</div><div class="value">6 / 2</div></div>
    <div class="card"><div class="label">输入通道</div><div class="value">21</div></div>
    <div class="card"><div class="label">标签时间步</div><div class="value">72</div></div>
  </div>
  <h2>1. 生成的数据集</h2>
  <p>数据集写入 <code>LarNO-main/benchmark/urbanflood/geodata/region1_20m_drainage_v1</code> 和 <code>LarNO-main/benchmark/urbanflood/flood/region1_20m_drainage_v1</code>。默认 <code>h.npy</code> 为 ITZI + 概念性道路对齐入口 sink 标签；原 MIKE reference 保存在 <code>h_mike_ref.npy</code>。</p>
  {table(quality, columns)}
  <h2>2. 数组完整性验证</h2>
  <pre>{html.escape(validation)}</pre>
  {table(arrays[:32], array_columns)}
  <h2>3. 代码入口</h2>
  <ul>
    <li>生成 ITZI 72 步标签：<code>python extended_study/run_itzi_drainage_timeseries.py</code></li>
    <li>构建 drainage v1 数据集：<code>python extended_study/build_drainage_dataset_v1.py --overwrite</code></li>
    <li>验证数据和 LarNO-D 输入：<code>python extended_study/validate_drainage_dataset_v1.py</code></li>
    <li>训练 LarNO-D：<code>python train.py --config region1_drainage_finetune.yaml --device cpu</code></li>
    <li>DrainAdapter dry-run：<code>python train_drain_adapter.py --config region1_drainage_adapter.yaml --device cpu --dry_run</code></li>
  </ul>
  <h2>4. QA 图</h2>
  {qa_html}
  <h2>5. 当前边界</h2>
  <p>v1 标签来自 ITZI + 概念性入口 sink，SWMM 目前作为 standalone dynamic-wave 管网指标进入数据集，不作为完整地表-管网双向耦合标签。真实 <code>region1_5m</code> MIKE reference 仍未在本地发现。</p>
</main>
</body>
</html>"""
    REPORT.write_text(html_text, encoding="utf-8")
    print(f"Wrote {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
