#!/usr/bin/env python3
"""Generate a standalone HTML report for the DrainLite residual study."""

from __future__ import annotations

import base64
import csv
import html
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "extended_study" / "output" / "drainlite_residual"
DATASET_OUT = ROOT / "extended_study" / "output" / "drainage_dataset_v1"
SWMM_CSV = ROOT / "extended_study" / "output" / "swmm_network" / "swmm_summary_metrics.csv"
REPORT = OUT / "drainlite_standalone_report.html"


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def img_uri(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def fmt(value: object, digits: int = 3) -> str:
    try:
        number = float(value)
    except Exception:
        return html.escape(str(value))
    if math.isnan(number):
        return "NA"
    return f"{number:.{digits}f}"


def table(rows: list[dict[str, object]], columns: list[str], max_rows: int | None = None) -> str:
    shown = rows[:max_rows] if max_rows else rows
    head = "".join(f"<th>{html.escape(col)}</th>" for col in columns)
    body = []
    for row in shown:
        body.append("<tr>" + "".join(f"<td>{html.escape(str(row.get(col, '')))}</td>" for col in columns) + "</tr>")
    note = ""
    if max_rows and len(rows) > max_rows:
        note = f"<p class='note'>仅展示前 {max_rows} 行，共 {len(rows)} 行；完整数据见同目录 CSV。</p>"
    return f"<div class='table-wrap'><table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table></div>{note}"


def metric_lookup(summary: list[dict[str, str]], model: str, key: str) -> float:
    for row in summary:
        if row.get("model") == model:
            try:
                return float(row[key])
            except Exception:
                return math.nan
    return math.nan


def render_figure(number: int, title: str, path: Path, caption: str) -> str:
    if not path.exists():
        return f"<p class='missing'>图 {number} {html.escape(title)}：待补充，未找到 {html.escape(str(path))}</p>"
    return (
        f"<figure id='fig{number}'>"
        f"<figcaption>图 {number}. {html.escape(title)}<span>{html.escape(caption)}</span></figcaption>"
        f"<img src='{img_uri(path)}' alt='{html.escape(title)}'>"
        f"</figure>"
    )


def main() -> int:
    metrics = read_csv(OUT / "metrics" / "drainlite_event_metrics.csv")
    summary = read_csv(OUT / "metrics" / "drainlite_ablation_summary.csv")
    swmm = read_csv(SWMM_CSV)
    quality = read_csv(DATASET_OUT / "dataset_quality.csv")
    metadata = json.loads((OUT / "drainlite_run_metadata.json").read_text(encoding="utf-8"))
    imp_all = read_csv(OUT / "metrics" / "feature_importance_all_static.csv")
    imp_swmm = read_csv(OUT / "metrics" / "feature_importance_swmm_assisted.csv")

    all_mae = metric_lookup(summary, "all_static", "mae_mm")
    surface_mae = metric_lookup(summary, "surface_only", "mae_mm")
    base_mae = metric_lookup(summary, "base", "mae_mm")
    all_vs_surface = metric_lookup(summary, "all_static", "mae_improvement_vs_surface_pct")
    all_vs_base = metric_lookup(summary, "all_static", "mae_improvement_vs_base_pct")
    best_model = next((row["model"] for row in summary if row["model"] != "itzi_sink_label"), "待补充")
    mean_swmm_cont = sum(float(r["routing_continuity_error_pct"]) for r in swmm) / max(len(swmm), 1)

    summary_cols = [
        "model", "target", "n_events", "mae_mm", "rmse_mm", "csi_0p03", "csi_0p15",
        "peak_error_mm", "reduction_mae_mm", "mae_improvement_vs_surface_pct", "mae_improvement_vs_base_pct",
    ]
    metric_cols = [
        "event", "model", "target", "mae_m", "rmse_m", "csi_0p03", "csi_0p15",
        "peak_error_m", "reduction_mae_m", "reduction_capture_pct",
    ]
    swmm_cols = [
        "event", "precip_mm", "routing_inflow_m3", "external_outflow_m3",
        "routing_final_storage_m3", "routing_continuity_error_pct",
        "outfall_volume_m3", "flooded_node_count", "node_flooding_volume_m3",
    ]
    quality_cols = [
        "event", "mike_peak_m", "itzi_surface_peak_m", "itzi_sink_peak_m",
        "sink_minus_surface_peak_m", "mean_sink_reduction_m", "swmm_status",
    ]
    imp_cols = ["model", "feature", "importance_mean", "importance_std"]

    figures = []
    fig_no = 1
    figures.append(render_figure(fig_no, "DEM 与道路对齐概化管网叠加", OUT / "figures" / "static_dem_network.png", "黑线为管段，蓝点为入口，红点为排放口；底图为 20 m DEM。"))
    fig_no += 1
    figures.append(render_figure(fig_no, "SWMM standalone 动态波指标汇总", OUT / "figures" / "swmm_summary.png", "展示 17 个事件的 outfall 出流、节点 ponding 体量和 routing continuity error。"))
    fig_no += 1
    figures.append(render_figure(fig_no, "DrainLite all_static 特征重要性", OUT / "figures" / "feature_importance_all_static.png", "基于测试样本 permutation importance，数值表示打乱特征后 MAE 增量。"))
    fig_no += 1
    figures.append(render_figure(fig_no, "DrainLite swmm_assisted 特征重要性", OUT / "figures" / "feature_importance_swmm_assisted.png", "用于判断 SWMM 事件级指标是否给残差学习提供额外信息。"))
    fig_no += 1
    for event in metadata["test_events"]:
        figures.append(render_figure(fig_no, f"{event} 峰值空间对比", OUT / "figures" / f"{event}_peak_maps.png", "依次为 MIKE、ITZI surface-only、ITZI+sink、DrainLite、DrainLite-sink 误差和预测削峰量。"))
        fig_no += 1

    css = """
    :root { --ink:#18212b; --muted:#617182; --line:#d8dee6; --band:#f4f7fa; --brand:#205b8f; --accent:#a23b2a; }
    * { box-sizing:border-box; }
    body { margin:0; background:#e9eef3; color:var(--ink); font-family:"Microsoft YaHei", "Segoe UI", Arial, sans-serif; line-height:1.68; }
    .page { width:min(1180px, calc(100% - 36px)); margin:24px auto; background:white; padding:42px 56px; box-shadow:0 16px 50px rgba(20,35,50,.15); }
    .cover { border-bottom:3px solid var(--brand); padding-bottom:24px; margin-bottom:28px; }
    h1 { font-size:31px; margin:0 0 10px; letter-spacing:0; }
    h2 { font-size:22px; margin:34px 0 12px; padding-bottom:7px; border-bottom:1px solid var(--line); }
    h3 { font-size:17px; margin:22px 0 8px; }
    p { margin:8px 0 12px; }
    .lead { font-size:16px; background:#f3f8fc; border-left:4px solid var(--brand); padding:12px 14px; }
    .meta { color:var(--muted); font-size:13px; }
    .cards { display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin:20px 0; }
    .card { border:1px solid var(--line); background:#fbfcfd; border-radius:8px; padding:12px 14px; min-height:88px; }
    .label { color:var(--muted); font-size:13px; }
    .value { color:var(--brand); font-weight:700; font-size:24px; margin-top:4px; }
    .note { color:var(--muted); font-size:13px; margin-top:-8px; }
    .warn { background:#fff8ef; border-left:4px solid #d9822b; padding:10px 12px; }
    code { background:#f3f5f7; border:1px solid #e2e7ed; border-radius:4px; padding:1px 5px; }
    pre { background:#17202a; color:#f8f9f9; padding:14px; border-radius:8px; overflow:auto; }
    .table-wrap { overflow:auto; border:1px solid var(--line); border-radius:8px; margin:12px 0 16px; }
    table { width:100%; border-collapse:collapse; font-size:12.5px; }
    th, td { border-bottom:1px solid #e7ebef; padding:7px 8px; white-space:nowrap; text-align:left; }
    th { background:var(--band); color:#263542; }
    figure { margin:20px 0; border:1px solid var(--line); border-radius:8px; overflow:hidden; background:#fff; }
    figcaption { background:var(--band); padding:10px 12px; font-weight:700; }
    figcaption span { display:block; color:var(--muted); font-weight:400; font-size:13px; margin-top:3px; }
    img { display:block; width:100%; height:auto; }
    .toc { columns:2; padding:12px 18px; background:#fbfcfd; border:1px solid var(--line); border-radius:8px; }
    .toc a { color:var(--brand); text-decoration:none; }
    .missing { color:var(--accent); }
    @media(max-width:900px){ .page{padding:24px;} .cards{grid-template-columns:1fr 1fr;} .toc{columns:1;} th,td{white-space:normal;} }
    @media(max-width:560px){ .cards{grid-template-columns:1fr;} }
    """

    html_text = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>LarNO-DrainLite 本机轻量化管网增强研究报告</title>
  <style>{css}</style>
</head>
<body>
<main class="page">
  <section class="cover">
    <h1>LarNO-DrainLite 本机轻量化管网增强研究报告</h1>
    <p class="meta">项目路径：{html.escape(str(ROOT))}<br>输出路径：{html.escape(str(OUT))}</p>
    <p class="lead">本报告在不进行全量 LarNO-D 训练、不依赖服务器 GPU 的前提下，构建并验证一种管网特征驱动的轻量排水残差模型。模型学习 <code>ITZI surface-only</code> 到 <code>ITZI+sink</code> 的削峰残差，并用 MIKE reference 与 SWMM standalone 指标进行外部参照和物理解释。</p>
  </section>

  <section>
    <h2>目录</h2>
    <ol class="toc">
      <li><a href="#abstract">摘要</a></li>
      <li><a href="#background">研究背景与目的</a></li>
      <li><a href="#data">数据与方法</a></li>
      <li><a href="#model">DrainLite 模型</a></li>
      <li><a href="#results">结果展示</a></li>
      <li><a href="#discussion">分析与讨论</a></li>
      <li><a href="#conclusion">结论与展望</a></li>
    </ol>
  </section>

  <section id="abstract">
    <h2>摘要</h2>
    <p>当前本机硬件为低显存 GPU 环境，不适合全量训练 LarNO-D。为在现有条件下形成可完成的优化创新，本研究将任务改写为“排水残差学习”：使用 17 个 20 m 事件、72 个时间步、8 个道路对齐管网特征，训练 CPU 友好的 HistGradientBoostingRegressor，对无管网 ITZI 地表水深进行非负削峰修正。</p>
    <div class="cards">
      <div class="card"><div class="label">有效事件</div><div class="value">{len(quality)}</div></div>
      <div class="card"><div class="label">训练 / 测试</div><div class="value">{len(metadata["train_events"])} / {len(metadata["test_events"])}</div></div>
      <div class="card"><div class="label">all_static MAE</div><div class="value">{fmt(all_mae)} mm</div></div>
      <div class="card"><div class="label">相对 surface-only</div><div class="value">{fmt(all_vs_surface, 1)}%</div></div>
    </div>
    <p>在 5 个测试事件上，<code>all_static</code> 模型相对 <code>surface_only</code> 的 ITZI+sink 标签 MAE 改善约 {fmt(all_vs_surface, 1)}%，相对不含管网特征的 <code>base</code> 改善约 {fmt(all_vs_base, 1)}%。这说明道路对齐管网空间先验和水力参数可以在轻量模型中被有效利用。</p>
  </section>

  <section id="background">
    <h2>研究背景与目的</h2>
    <p>原 LarNO 研究重点是从降雨、DEM 等输入学习 MIKE reference 产生的水深时空响应，但当前公开训练流程未显式纳入城市排水管网。前期工作已经完成道路/DEM 对齐、概化管网生成、ITZI 地表动力学模拟、ITZI+sink 标签以及 SWMM standalone 管网指标。本阶段目标不是重训大模型，而是在本机条件下完成一种轻量、可解释、可消融的管网增强方法。</p>
    <p class="warn">边界说明：本报告中的 SWMM 是独立 1D dynamic-wave 管网路由指标，不是 ITZI-SWMM 双向在线耦合；MIKE reference 仅作为外部参照，不作为 DrainLite 主训练标签；本地仍无真实 <code>region1_5m</code> MIKE reference。</p>
  </section>

  <section id="data">
    <h2>数据与方法</h2>
    <p>数据集为 <code>region1_20m_drainage_v1</code>，空间窗口为 200 × 280，时间长度为 72 步。默认 <code>h.npy</code> 与 <code>h_itzi_sink.npy</code> 表示 ITZI + 概念性道路排水入口 sink 结果；<code>h_itzi_surface.npy</code> 表示无管网地表动力学结果；<code>h_mike_ref.npy</code> 为 MIKE reference。</p>
    <h3>表 1. 数据集质量摘要</h3>
    {table(quality, quality_cols)}
    {figures[0]}
    {figures[1]}
    <h3>表 2. SWMM standalone 指标</h3>
    <p>17 个事件均完成 SWMM standalone 解析，平均 routing continuity error 为 {fmt(mean_swmm_cont)}%。该指标用于物理解释和 <code>swmm_assisted</code> 对照。</p>
    {table(swmm, swmm_cols)}
  </section>

  <section id="model">
    <h2>DrainLite 模型</h2>
    <p>DrainLite 学习目标为：</p>
    <pre>target_reduction_mm = 1000 * max(h_itzi_surface - h_itzi_sink, 0)
h_drainlite_m = max(h_itzi_surface_m - pred_reduction_mm / 1000, 0)</pre>
    <p>该结构保证预测水深非负，并且不会高于无管网 surface-only 结果。模型采用 HistGradientBoostingRegressor，本次实际运行参数为 <code>pixels_per_step={metadata["pixels_per_step"]}</code>、<code>max_iter={metadata["max_iter"]}</code>、<code>learning_rate={metadata["learning_rate"]}</code>。脚本保留完整默认参数，可在同一机器上加大采样量复跑。</p>
    <h3>表 3. 消融实验平均结果</h3>
    <p class="note"><code>itzi_sink_label</code> 是监督标签自身，用作理论下界/核对项，不是可预测模型；模型比较重点为 <code>surface_only</code>、<code>base</code>、<code>mask_only</code>、<code>hydraulic_only</code>、<code>all_static</code> 和 <code>swmm_assisted</code>。</p>
    {table(summary, summary_cols)}
    {figures[2]}
    {figures[3]}
    <h3>表 4. all_static 特征重要性</h3>
    {table(imp_all, imp_cols, max_rows=20)}
    <h3>表 5. swmm_assisted 特征重要性</h3>
    {table(imp_swmm, imp_cols, max_rows=20)}
  </section>

  <section id="results">
    <h2>结果展示</h2>
    <p>下列峰值空间图展示 5 个测试事件的 MIKE reference、ITZI surface-only、ITZI+sink、DrainLite all_static、DrainLite 误差和预测削峰量。DrainLite 的主要作用不是重建 MIKE，而是学习概念性管网 sink 相对 surface-only 的削峰空间分布。</p>
    {''.join(figures[4:])}
    <h3>表 6. 测试事件逐项指标</h3>
    {table(metrics, metric_cols)}
  </section>

  <section id="discussion">
    <h2>分析与讨论</h2>
    <p>消融结果显示，<code>all_static</code> 在 ITZI+sink 标签上的 MAE 为 {fmt(all_mae)} mm，显著低于 <code>surface_only</code> 的 {fmt(surface_mae)} mm，也低于不含管网特征的 <code>base</code> 的 {fmt(base_mae)} mm。<code>mask_only</code> 与 <code>all_static</code> 接近，说明 inlet/outfall/pipe 位置及 distance-to-outfall 是本轻量模型中最直接有效的管网先验。</p>
    <p><code>swmm_assisted</code> 在本次测试中没有优于 <code>all_static</code>。这并不否定 SWMM 指标的物理意义，而是说明当前事件级 SWMM 汇总量较粗，尚未形成对每个像元、每个时刻都有效的空间条件。后续若继续轻量路线，优先考虑把节点水位、管段流量或 outfall 距离路径场转化为空间/时序特征，而不是只使用事件级汇总值。</p>
  </section>

  <section id="conclusion">
    <h2>主要结论、不足与展望</h2>
    <p>本机轻量化目标已经形成闭环：完成 17 事件数据集、SWMM standalone 指标、DrainLite 残差模型、五组消融、特征重要性和空间可视化。结果支持“道路对齐概化管网特征可以作为轻量排水先验，改善无管网地表模拟向带管网标签的修正”。</p>
    <p>不足在于：当前监督标签仍为概念性 sink，不是完整真实管网耦合；SWMM 仍为 standalone；DrainLite 是残差修正模型，不是端到端 LarNO-D 大模型。后续在不改变轻量定位的前提下，可继续优化管网特征，例如引入网络距离、上游汇水面积、节点服务区、水力瓶颈指数和管网连通性指标。</p>
  </section>
</main>
</body>
</html>"""
    REPORT.write_text(html_text, encoding="utf-8")
    print(f"Wrote {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
