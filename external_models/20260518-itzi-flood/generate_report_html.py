#!/usr/bin/env python3
"""Generate self-contained HTML research report."""
import os
import base64
import json
import numpy as np
from datetime import date

ROOT = r"e:\Projects\20260518-itzi-flood"
OUT = os.path.join(ROOT, "report.html")

# Figures: (id, title, path, caption)
FIGURES = [
    ("fig1", "合成城市案例洪水演进", os.path.join(ROOT, "test_cases/synthetic_urban/visualization_output/02_terrain_and_flood.png"),
     "ITZI 安装验证：合成城市地形与典型时刻淹没深度。"),
    ("fig2", "城市街区地形与建筑分布", os.path.join(ROOT, "test_cases/urban_district/visualization_output/urban_01_terrain_buildings.png"),
     "城市街区测试案例：DEM、建筑掩膜与街道布局。"),
    ("fig3", "地表与 SWMM 管网耦合对比", os.path.join(ROOT, "test_cases/urban_drainage/visualization_output/drainage_01_comparison.png"),
     "参考 urban_drainage 案例：无排水与 SWMM 完整耦合的峰值水深对比。"),
    ("fig4", "深圳子区域地形与管网节点", os.path.join(ROOT, "test_cases/shenzhen_region1/visualization_output/shenzhen_00_terrain_network.png"),
     "LarNO Region1 子区域（4 km × 5.6 km）DEM 与 OSM 主干管网检查井位置。"),
    ("fig5", "深圳 8 事件峰值水深对比（ITZI 动态 vs MIKE+）", os.path.join(ROOT, "test_cases/shenzhen_region1/visualization_output/shenzhen_01_peak_comparison.png"),
     "8 个降雨事件下 MIKE+ 参考、ITZI 地表与简化管网排水情景的峰值淹没对比。"),
    ("fig6", "深圳事件峰值汇总柱状图", os.path.join(ROOT, "test_cases/shenzhen_region1/visualization_output/shenzhen_03_summary_bar.png"),
     "峰值水深相对 MIKE+ 的比值（非误差百分比，见正文说明）。"),
    ("fig7", "ITZI-SWMM 耦合峰值对比（event1/event65）", os.path.join(ROOT, "test_cases/shenzhen_region1/visualization_output/full_domain_swmm/coupled_02_summary_bar.png"),
     "子区域 ITZI 地表与 ITZI+SWMM 完整耦合结果（部分事件）。"),
    ("fig8", "SWMM 耦合削减效果", os.path.join(ROOT, "test_cases/shenzhen_region1/visualization_output/full_domain_swmm/coupled_04_swmm_reduction.png"),
     "相对仅地表情景，SWMM 耦合对峰值水深的削减比例。"),
    ("fig9", "Wellington 基准案例", os.path.join(ROOT, "benchmark_data/wellington/visualizations/wellington_summary.png"),
     "国际基准案例 Wellington 模拟结果汇总。"),
    ("fig10", "EA Test 5 基准案例", os.path.join(ROOT, "benchmark_data/ea_test5/visualizations/ea5_summary.png"),
     "欧洲基准案例 EA Test 5 模拟结果汇总。"),
]


def img_b64(path):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def load_itzi_all():
    fp = os.path.join(ROOT, "test_cases/shenzhen_region1/output/all_summary.npz")
    if not os.path.exists(fp):
        return {}
    d = np.load(fp, allow_pickle=True)
    raw = d["all_results"].item()
    out = {}
    for evt, v in raw.items():
        out[evt] = {k: float(val) if isinstance(val, (np.floating, float, int)) else val
                    for k, val in v.items()}
    return out


def pct_ratio(sim, ref):
    if ref <= 0:
        return "—"
    return f"{sim / ref * 100:.1f}"


def rel_err(sim, ref):
    if ref <= 0:
        return "—"
    return f"{(sim - ref) / ref * 100:+.1f}"


def build_table_itzi8(data):
    rows = []
    for evt in sorted(data.keys(), key=lambda x: (len(x), x)):
        r = data[evt]
        rows.append(f"""<tr>
          <td>{evt}</td>
          <td>{r['mike_peak']:.3f}</td>
          <td>{r['itzi_surf_peak']:.3f}</td>
          <td>{pct_ratio(r['itzi_surf_peak'], r['mike_peak'])}%</td>
          <td>{rel_err(r['itzi_surf_peak'], r['mike_peak'])}%</td>
          <td>{r['itzi_pipe_peak']:.3f}</td>
          <td>{pct_ratio(r['itzi_pipe_peak'], r['mike_peak'])}%</td>
          <td>{rel_err(r['itzi_pipe_peak'], r['mike_peak'])}%</td>
          <td>{r['itzi_pipe_drained']:,.0f}</td>
        </tr>""")
    return "\n".join(rows)


def build_table_swmm_partial():
    rows = [
        ("event1", 2.076, 2.234, 2.231, 20304),
        ("event65", 1.749, 2.206, 2.175, 22696),
    ]
    html = []
    for evt, mike, surf, swmm, drain in rows:
        html.append(f"""<tr>
          <td>{evt}</td><td>{mike:.3f}</td><td>{surf:.3f}</td>
          <td>{pct_ratio(surf, mike)}%</td><td>{rel_err(surf, mike)}%</td>
          <td>{swmm:.3f}</td><td>{pct_ratio(swmm, mike)}%</td>
          <td>{rel_err(swmm, mike)}%</td><td>{drain:,.0f}</td>
        </tr>""")
    return "\n".join(html)


def svg_bar_chart(data):
    """Simple inline SVG bar chart for peak ratios."""
    events = list(sorted(data.keys(), key=lambda x: (len(x), x)))
    mike = [data[e]["mike_peak"] for e in events]
    surf = [data[e]["itzi_surf_peak"] for e in events]
    n = len(events)
    w, h = 720, 280
    bw = 18
    gap = 80
    maxv = max(max(mike), max(surf)) * 1.15
    scale = (h - 60) / maxv
    bars = []
    for i, e in enumerate(events):
        x0 = 50 + i * gap
        hm = mike[i] * scale
        hs = surf[i] * scale
        bars.append(f'<rect x="{x0}" y="{h-40-hm:.1f}" width="{bw}" height="{hm:.1f}" fill="#08519C"/>')
        bars.append(f'<rect x="{x0+bw+4}" y="{h-40-hs:.1f}" width="{bw}" height="{hs:.1f}" fill="#6BAED6"/>')
        bars.append(f'<text x="{x0+bw}" y="{h-18}" font-size="10" text-anchor="middle">{e}</text>')
    return f'''<svg viewBox="0 0 {w} {h}" width="100%" xmlns="http://www.w3.org/2000/svg" role="img">
      <text x="360" y="22" text-anchor="middle" font-size="14" font-weight="bold">峰值水深对比（深圳子区域 8 事件）</text>
      {chr(10).join(bars)}
      <rect x="520" y="30" width="14" height="14" fill="#08519C"/><text x="540" y="42" font-size="11">MIKE+</text>
      <rect x="520" y="50" width="14" height="14" fill="#6BAED6"/><text x="540" y="62" font-size="11">ITZI 地表</text>
      <line x1="40" y1="{h-40}" x2="{w-20}" y2="{h-40}" stroke="#333"/>
      <text x="18" y="{h-35}" font-size="10">0</text>
      <text x="10" y="45" font-size="10">{maxv:.1f}m</text>
    </svg>'''


def main():
    itzi8 = load_itzi_all()
    fig_html = []
    fig_num = 0
    for fid, title, path, caption in FIGURES:
        b64 = img_b64(path)
        fig_num += 1
        if b64:
            fig_html.append(f'''
            <figure id="{fid}">
              <figcaption><strong>图 {fig_num}</strong> {title}</figcaption>
              <img src="data:image/png;base64,{b64}" alt="{title}"/>
              <p class="fig-note">{caption}</p>
            </figure>''')
        else:
            fig_html.append(f'''
            <figure id="{fid}">
              <figcaption><strong>图 {fig_num}</strong> {title}（待补充：文件未找到）</figcaption>
              <p class="fig-note">{caption}</p>
            </figure>''')

    svg_chart = svg_bar_chart(itzi8) if itzi8 else "<p>待补充</p>"
    table8 = build_table_itzi8(itzi8) if itzi8 else "<tr><td colspan='9'>待补充</td></tr>"
    table_swmm = build_table_swmm_partial()

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>ITZI 城市洪涝模拟与 SWMM 耦合研究技术报告</title>
<style>
  :root {{
    --primary: #1a365d;
    --accent: #2b6cb0;
    --bg: #f7fafc;
    --text: #2d3748;
    --border: #cbd5e0;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    font-family: "Segoe UI", "Microsoft YaHei", "PingFang SC", sans-serif;
    line-height: 1.75;
    color: var(--text);
    background: var(--bg);
    margin: 0;
    padding: 0;
  }}
  .page {{
    max-width: 960px;
    margin: 0 auto;
    background: #fff;
    box-shadow: 0 0 24px rgba(0,0,0,.08);
  }}
  .cover {{
    min-height: 85vh;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    text-align: center;
    padding: 3rem 2rem;
    background: linear-gradient(160deg, #1a365d 0%, #2c5282 50%, #4299e1 100%);
    color: #fff;
  }}
  .cover h1 {{ font-size: 2rem; margin: 0 0 1rem; font-weight: 700; }}
  .cover .sub {{ font-size: 1.15rem; opacity: .92; max-width: 640px; }}
  .cover .meta {{ margin-top: 2.5rem; font-size: .95rem; opacity: .85; }}
  .content {{ padding: 2.5rem 3rem 4rem; }}
  h2 {{
    color: var(--primary);
    border-left: 5px solid var(--accent);
    padding-left: .75rem;
    margin: 2.5rem 0 1rem;
    font-size: 1.45rem;
  }}
  h3 {{ color: #2c5282; font-size: 1.15rem; margin: 1.5rem 0 .75rem; }}
  p {{ text-align: justify; margin: .6rem 0; }}
  .toc {{
    background: #edf2f7;
    padding: 1.25rem 1.5rem;
    border-radius: 8px;
    margin: 1.5rem 0;
  }}
  .toc ol {{ margin: .5rem 0 0 1.2rem; padding: 0; }}
  .toc li {{ margin: .35rem 0; }}
  .toc a {{ color: var(--accent); text-decoration: none; }}
  .abstract {{
    background: #ebf8ff;
    border: 1px solid #bee3f8;
    padding: 1.25rem 1.5rem;
    border-radius: 8px;
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    margin: 1rem 0 1.5rem;
    font-size: .88rem;
  }}
  th, td {{
    border: 1px solid var(--border);
    padding: .5rem .6rem;
    text-align: center;
  }}
  th {{ background: #edf2f7; color: var(--primary); font-weight: 600; }}
  tr:nth-child(even) {{ background: #f7fafc; }}
  .table-caption {{
    font-weight: 600;
    color: var(--primary);
    margin: 1.25rem 0 .5rem;
    text-align: center;
  }}
  figure {{
    margin: 1.5rem 0 2rem;
    text-align: center;
  }}
  figure img {{
    max-width: 100%;
    height: auto;
    border: 1px solid var(--border);
    border-radius: 4px;
  }}
  figcaption, .fig-note {{
    font-size: .9rem;
    color: #4a5568;
    margin-top: .5rem;
    text-align: left;
    padding: 0 .5rem;
  }}
  .note-box {{
    background: #fffaf0;
    border-left: 4px solid #dd6b20;
    padding: .75rem 1rem;
    margin: 1rem 0;
    font-size: .92rem;
  }}
  .formula {{
    background: #f7fafc;
    padding: .75rem 1rem;
    border-radius: 6px;
    font-family: "Cambria Math", serif;
    text-align: center;
    margin: 1rem 0;
  }}
  ul {{ margin: .5rem 0; padding-left: 1.5rem; }}
  li {{ margin: .35rem 0; }}
  footer {{
    text-align: center;
    padding: 2rem;
    color: #718096;
    font-size: .85rem;
    border-top: 1px solid var(--border);
  }}
  @media print {{
    .page {{ box-shadow: none; }}
    .cover {{ min-height: auto; page-break-after: always; }}
    figure {{ page-break-inside: avoid; }}
  }}
</style>
</head>
<body>
<div class="page">

<section class="cover">
  <h1>基于 ITZI 的二维城市地表洪涝模拟<br/>及与 SWMM 管网完整耦合研究</h1>
  <p class="sub">深圳 LarNO Region1 数据集验证、MIKE+ 参考对比与多情景模拟技术报告</p>
  <p class="meta">
    项目路径：E:\\Projects\\20260518-itzi-flood<br/>
    报告日期：{date.today().isoformat()}<br/>
    文档类型：自包含 HTML 科研报告（单文件）
  </p>
</section>

<div class="content">

<section id="toc">
  <h2>目录</h2>
  <nav class="toc">
    <ol>
      <li><a href="#abstract">摘要</a></li>
      <li><a href="#background">1 研究背景与目的</a></li>
      <li><a href="#data">2 数据与软件环境</a></li>
      <li><a href="#methods">3 研究方法</a></li>
      <li><a href="#process">4 研究过程</a></li>
      <li><a href="#results">5 结果展示</a></li>
      <li><a href="#discussion">6 分析与讨论</a></li>
      <li><a href="#conclusion">7 主要结论</a></li>
      <li><a href="#outlook">8 不足与展望</a></li>
    </ol>
  </nav>
</section>

<section id="abstract">
  <h2>摘要</h2>
  <div class="abstract">
    <p>本研究在 Windows 环境下完成了开源二维洪涝模型 <strong>ITZI</strong>（基于 GRASS GIS 栅格后端）的部署，
    建立了从合成案例、城市街区案例到 <strong>深圳市 LarNO Region1</strong> 真实栅格数据的完整模拟流程。
    参考模型对比仅采用 LarNO 数据集中附带的 <strong>MIKE+ 参考水深场</strong>（<code>h.npy</code>），
    <strong>不与 LarNO 神经网络（LarNO）预测结果</strong> 进行对比。</p>
    <p>在方法上，先后实现了：（1）ITZI 偏惯性二维地表流求解；（2）沿 OSM 主干道路的简化管网排水；
    （3）借鉴 <code>urban_drainage</code> 案例的 <strong>ITZI–SWMM 双向完整耦合</strong>（堰/孔口交换）。
    深圳子区域（4 km × 5.6 km）8 个降雨事件的峰值水深表明：ITZI 地表情景相对 MIKE+ 峰值比值约 98%–129%；
    加入排水后比值约 94%–114%。ITZI+SWMM 完整耦合试算（event1）峰值比值约 107%，累计入管约 2.0×10⁴ m³。</p>
    <p><strong>关键词：</strong>ITZI；城市洪涝；SWMM；MIKE+；LarNO；管网耦合；深圳</p>
  </div>
</section>

<section id="background">
  <h2>1 研究背景与目的</h2>
  <h3>1.1 背景</h3>
  <p>城市洪涝模拟通常需要同时考虑二维地表漫流与地下排水系统的相互作用。ITZI 是面向栅格数据的分布式
  水动力–水文耦合模型，可与 SWMM 进行节点级耦合，适用于城市雨洪管理场景。</p>
  <p>本工作源于对 <a href="https://www.itzi.org/">itzi.org</a> 开源模型的本地化部署需求，并延伸至
  论文《Large-scale urban flood modeling and zero-shot high-resolution generalization with LarNO》所公开的
  深圳市 Region1（20 m 分辨率）数据集，用于在真实城市地形与建筑掩膜条件下检验 ITZI 模拟能力。</p>
  <h3>1.2 目的</h3>
  <ul>
    <li>完成 ITZI 在指定路径下的安装、测试案例运行与可视化全流程；</li>
    <li>建立可复现的 ITZI 二维地表流模拟脚本（含建筑径流重分配）；</li>
    <li>在 LarNO 数据上对比 MIKE+ 参考结果，评估峰值水深与淹没范围；</li>
    <li>实现并验证 ITZI 与 SWMM 的完整耦合框架（非仅经验排水公式）。</li>
  </ul>
</section>

<section id="data">
  <h2>2 数据与软件环境</h2>
  <table>
    <caption class="table-caption">表 1 主要数据与来源说明</caption>
    <thead>
      <tr><th>数据项</th><th>路径/来源</th><th>说明</th></tr>
    </thead>
    <tbody>
      <tr><td>DEM 地形</td><td>LarNO geodata/region1_20m/dem.npy</td><td>400×560，20 m 分辨率，约 8 km×11.2 km</td></tr>
      <tr><td>降雨与参考水深</td><td>flood/region1_20m/event*/</td><td>rainfall.npy（72×5 min）；h.npy 为 MIKE+ 参考</td></tr>
      <tr><td>OSM 管网</td><td>osm_merged_network.npz</td><td>2847 节点、6839 管段（复制自 LarNO 拓展研究）</td></tr>
      <tr><td>合成/街区案例</td><td>test_cases/synthetic_urban 等</td><td>方法验证与 SWMM 耦合模板</td></tr>
    </tbody>
  </table>
  <table>
    <caption class="table-caption">表 2 软件环境（实测记录）</caption>
    <thead>
      <tr><th>组件</th><th>版本/说明</th></tr>
    </thead>
    <tbody>
      <tr><td>操作系统</td><td>Windows 10 Enterprise LTSC</td></tr>
      <tr><td>Python</td><td>3.13（Miniconda3）</td></tr>
      <tr><td>ITZI</td><td>已安装（site-packages/itzi）</td></tr>
      <tr><td>PySWMM</td><td>2.1.0</td></tr>
      <tr><td>GRASS GIS</td><td>待补充（部分案例使用 Python API 直接驱动栅格域）</td></tr>
    </tbody>
  </table>
</section>

<section id="methods">
  <h2>3 研究方法</h2>
  <h3>3.1 ITZI 二维地表流</h3>
  <p>采用 ITZI 偏惯性（partial inertia）二维地表流模块，栅格单元 20 m，曼宁糙率建筑区取极大值（阻流），
  活动区默认 n=0.015。建筑像元降雨按活动区面积重分配至可下渗像元。模拟时长 6 h（72 个 5 min 降雨步）。</p>
  <h3>3.2 简化管网排水（情景 B）</h3>
  <p>将 OSM 合并管网节点映射至子区域栅格，采用孔口/堰流公式在节点位置抽取地表水深，作为侧向排水汇流。
  该方案计算效率高，但未求解管段水力线。</p>
  <h3>3.3 ITZI–SWMM 完整耦合</h3>
  <p>参考 <code>test_cases/urban_drainage/run_coupled_simulation.py</code> 与 ITZI 官方
  <code>DrainageSimulation.apply_coupling_to_nodes</code>：由 PySWMM 求解一维管网，每个时间步将地表水位/水深
  传入节点，计算耦合流量并写入栅格 <code>n_drain</code>。SWMM 输入文件需包含 [COORDINATES] 段（格式参照
  urban_drainage 已验证模板）。</p>
  <h3>3.4 指标说明（重要）</h3>
  <div class="note-box">
    <strong>峰值比值 ≠ 误差百分比。</strong> 表中“相对 MIKE+ 比值”定义为：
    比值 = (ITZI 峰值 / MIKE+ 峰值) × 100%。100% 表示峰值相等；108% 表示偏高 8%。
    相对误差 = (ITZI − MIKE+) / MIKE+ × 100%，与比值不同。
  </div>
  <div class="formula">
    比值 (%) = h<sub>ITZI,max</sub> / h<sub>MIKE+,max</sub> × 100 &nbsp;&nbsp;|&nbsp;&nbsp;
    相对误差 (%) = (h<sub>ITZI,max</sub> − h<sub>MIKE+,max</sub>) / h<sub>MIKE+,max</sub> × 100
  </div>
</section>

<section id="process">
  <h2>4 研究过程</h2>
  <ol>
    <li><strong>阶段一（2026-05-18）：</strong> ITZI 官网调研、环境检查、<code>test_cases</code> 下合成城市、城市街区、
    排水耦合及 Wellington/EA 基准案例建设。</li>
    <li><strong>阶段二（2026-05-19 参考）：</strong> LarNO 论文数据与拓展研究（管网沿主干道、OSM 网络合并）；
    本报告数据从 <code>E:\\Projects\\20260519-LarNO</code> <strong>复制</strong> 至本项目，未修改原路径文件。</li>
    <li><strong>阶段三：</strong> 深圳子区域 ITZI 动态模拟 8 事件（地表 + 简化管网），输出至
    <code>shenzhen_region1/output/</code>。</li>
    <li><strong>阶段四：</strong> 构建 SWMM 输入、实现 <code>run_full_domain_coupled.py</code>；
    子区域完整 8 事件 SWMM 批处理因后台任务中断，目前仅 event1/event65 具完整耦合输出文件。</li>
  </ol>
</section>

<section id="results">
  <h2>5 结果展示</h2>

  <h3>5.1 深圳子区域 8 事件 — ITZI 动态 vs MIKE+（已完成）</h3>
  <div class="chart-wrap">{svg_chart}</div>
  <p class="table-caption"><strong>表 3</strong> 深圳子区域（4 km×5.6 km）峰值水深、比值、相对误差及简化管网排水量</p>
  <table>
    <thead>
      <tr>
        <th>事件</th><th>MIKE+峰值(m)</th><th>ITZI地表(m)</th>
        <th>比值(%)</th><th>相对误差(%)</th>
        <th>ITZI+管峰值(m)</th><th>比值(%)</th><th>相对误差(%)</th>
        <th>排水量(m³)</th>
      </tr>
    </thead>
    <tbody>
      {table8}
    </tbody>
  </table>

  <h3>5.2 ITZI–SWMM 完整耦合（部分完成）</h3>
  <p class="table-caption"><strong>表 4</strong> 子区域 ITZI–SWMM 耦合试算结果（相对 MIKE+ 参考）</p>
  <table>
    <thead>
      <tr>
        <th>事件</th><th>MIKE+峰值(m)</th><th>ITZI地表(m)</th>
        <th>比值(%)</th><th>相对误差(%)</th>
        <th>ITZI+SWMM(m)</th><th>比值(%)</th><th>相对误差(%)</th>
        <th>SWMM排水(m³)</th>
      </tr>
    </thead>
    <tbody>
      {table_swmm}
    </tbody>
  </table>
  <p>注：8 事件批处理完整耦合结果待补充；event20–event70 的 SWMM 情景尚未稳定落盘。</p>

  <h3>5.3 图表集</h3>
  {''.join(fig_html)}

</section>

<section id="discussion">
  <h2>6 分析与讨论</h2>
  <h3>6.1 与 MIKE+ 的差异</h3>
  <p>多数事件 ITZI 地表峰值高于 MIKE+（比值 104%–129%），可能原因包括：边界条件差异（本研究未完全复现 MIKE+ 边界）、
  建筑降雨重分配、曼宁参数与下渗方案不同。event20 与 event67 地表峰值接近 MIKE+（约 98%）。</p>
  <h3>6.2 排水情景</h3>
  <p>简化管网（孔口排水）可显著降低峰值（约 6%–15%）并减少淹没像元，排水量可达 1.4×10⁵–3.5×10⁵ m³/6 h。
  SWMM 完整耦合试算（event1）排水约 2.0×10⁴ m³，峰值略低于纯地表，说明耦合框架有效，但管网规模经抽稀后能力有限。</p>
  <h3>6.3 方法学意义</h3>
  <p><code>urban_drainage</code> 案例证明了 ITZI+SWMM 耦合在技术上可行；深圳案例证明该框架可扩展到真实城市 DEM
  与 MIKE+ 对标数据，为后续全域模拟与参数率定奠定基础。</p>
</section>

<section id="conclusion">
  <h2>7 主要结论</h2>
  <ol>
    <li>成功部署 ITZI 并完成多层级测试案例（合成、街区、排水、国际基准）。</li>
    <li>深圳 LarNO 子区域 8 事件 ITZI 地表模拟已完成，峰值水深相对 MIKE+ 比值约 98%–129%（非零误差）。</li>
    <li>简化管网排水可降低洪峰并减少积水体积；ITZI–SWMM 完整耦合在 event1 试算中运行稳定，峰值比值约 107%。</li>
    <li>峰值“比值%”应解释为相对 MIKE+ 峰值的比例；相对误差约为比值减 100%。</li>
  </ol>
</section>

<section id="outlook">
  <h2>8 不足与展望</h2>
  <ul>
    <li>全域（8 km×11.2 km）SWMM 耦合 8 事件批处理尚未完整结束，需前台长时运行避免中断。</li>
    <li>大型 OSM 管网（2000+ 节点）需抽稀与拓扑重建以保证 SWMM 数值稳定。</li>
    <li>未与 LarNO 神经网络结果对比；未开展系统参数敏感性分析（待补充）。</li>
    <li>建议后续：统一边界条件、完成 8 事件 SWMM 耦合、增加淹没面积与体积指标、撰写正式论文级率定方案。</li>
  </ul>
</section>

</div>
<footer>
  <p>本报告由项目脚本自动生成，数据来源于本地模拟输出；缺失项标注为“待补充”。<br/>
  文件：report.html — 完全自包含，可离线双击浏览。</p>
</footer>

</div>
</body>
</html>"""

    with open(OUT, "w", encoding="utf-8") as f:
        f.write(html)
    size_mb = os.path.getsize(OUT) / 1024 / 1024
    print(f"Written: {OUT} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
