#!/usr/bin/env python3
"""Generate a standalone HTML report for DrainLite connected residual results."""

from __future__ import annotations

import argparse
import base64
import csv
import html
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "extended_study" / "output" / "drainlite_connected_residual"
CONNECTED = ROOT / "extended_study" / "output" / "connected_itzi_swmm"
CV_OUT = ROOT / "extended_study" / "output" / "drainlite_connected_cross_validation"
V2_OUT = ROOT / "extended_study" / "output" / "larno_drainlite_v2_package"
QUALITY_CV_OUT = ROOT / "extended_study" / "output" / "drainlite_connected_cross_validation_v2_inf1mmh_quality_filtered"
QUALITY_ANALYSIS_OUT = ROOT / "extended_study" / "output" / "drainlite_quality_stratified_v2_inf1mmh"
HYDRO_OUT = ROOT / "extended_study" / "output" / "hydrograph_timing_diagnostics"
INF_PILOT_OUT = ROOT / "extended_study" / "output" / "infiltration_pilot"
INF_SENS_OUT = ROOT / "extended_study" / "output" / "infiltration_sensitivity"
INF_SELECTION_OUT = ROOT / "extended_study" / "output" / "infiltration_model_selection"
DATASET_AUDIT = ROOT / "extended_study" / "output" / "connected_swmm_dataset" / "connected_swmm_dataset_event_summary.csv"
DATASET_META = ROOT / "LarNO-main" / "benchmark" / "urbanflood" / "flood" / "region1_20m_connected_swmm_v1" / "metadata.json"
FLOOD_DIR = ROOT / "LarNO-main" / "benchmark" / "urbanflood" / "flood" / "region1_20m_connected_swmm_v1"
WORKFLOW = ROOT / "extended_study" / "ITZI_SWMM_FIXED_WORKFLOW.md"
REPORT = OUT / "drainlite_connected_residual_report.html"
REPORT_ALIAS = OUT / "report.html"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", default="region1_20m_connected_swmm_v2_inf1mmh")
    parser.add_argument("--main-output-name", default="drainlite_connected_residual_v2_inf1mmh")
    parser.add_argument("--cv-output-name", default="drainlite_connected_cross_validation_v2_inf1mmh")
    parser.add_argument("--package-output-name", default="larno_drainlite_v2_inf1mmh_package")
    parser.add_argument("--quality-cv-output-name", default="drainlite_connected_cross_validation_v2_inf1mmh_quality_filtered")
    parser.add_argument("--quality-analysis-output-name", default="drainlite_quality_stratified_v2_inf1mmh")
    parser.add_argument("--connected-output-name", default="connected_itzi_swmm")
    parser.add_argument("--dataset-audit-name", default="connected_swmm_dataset_v2_inf1mmh")
    parser.add_argument("--hydro-output-name", default="hydrograph_timing_diagnostics_v2_inf1mmh")
    return parser.parse_args()


def configure_paths(args: argparse.Namespace) -> None:
    global OUT, CONNECTED, CV_OUT, V2_OUT, QUALITY_CV_OUT, QUALITY_ANALYSIS_OUT, HYDRO_OUT, DATASET_AUDIT, DATASET_META, FLOOD_DIR, REPORT, REPORT_ALIAS
    OUT = ROOT / "extended_study" / "output" / args.main_output_name
    CONNECTED = ROOT / "extended_study" / "output" / args.connected_output_name
    CV_OUT = ROOT / "extended_study" / "output" / args.cv_output_name
    V2_OUT = ROOT / "extended_study" / "output" / args.package_output_name
    QUALITY_CV_OUT = ROOT / "extended_study" / "output" / args.quality_cv_output_name
    QUALITY_ANALYSIS_OUT = ROOT / "extended_study" / "output" / args.quality_analysis_output_name
    HYDRO_OUT = ROOT / "extended_study" / "output" / args.hydro_output_name
    DATASET_AUDIT = ROOT / "extended_study" / "output" / args.dataset_audit_name / "v2_infiltration_event_summary.csv"
    FLOOD_DIR = ROOT / "LarNO-main" / "benchmark" / "urbanflood" / "flood" / args.dataset_name
    DATASET_META = FLOOD_DIR / "metadata.json"
    REPORT = OUT / "drainlite_connected_residual_report.html"
    REPORT_ALIAS = OUT / "report.html"


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value: object, digits: int = 3) -> str:
    if value is None:
        return "待补充"
    try:
        f = float(value)
    except Exception:
        return html.escape(str(value))
    if f != f:
        return "N/A"
    if abs(f) >= 1000:
        return f"{f:,.0f}"
    return f"{f:.{digits}f}"


def table(rows: list[dict[str, str]], columns: list[tuple[str, str]], digits: int = 3) -> str:
    if not rows:
        return "<p class='missing'>待补充：表格数据尚未生成。</p>"
    head = "".join(f"<th>{html.escape(label)}</th>" for key, label in columns)
    body = []
    for row in rows:
        cells = []
        for key, _ in columns:
            value = row.get(key, "")
            if key in {
                "event",
                "model",
                "status",
                "quality_status",
                "rule",
                "feature",
                "label",
                "dataset",
                "analysis_set",
                "validation_design",
                "use_as_training_label",
                "must_disclose_stability_warning",
            }:
                cells.append(f"<td>{html.escape(str(value))}</td>")
            else:
                cells.append(f"<td class='num'>{fmt(value, digits)}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def img_data(path: Path) -> str:
    if not path.exists():
        return ""
    mime = "image/png"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def figure(path: Path, caption: str, explanation: str) -> str:
    src = img_data(path)
    if not src:
        return f"<div class='figure missing'><p>{html.escape(caption)}：待补充，图片文件不存在。</p></div>"
    return f"""
    <figure>
      <img src="{src}" alt="{html.escape(caption)}">
      <figcaption>{html.escape(caption)}</figcaption>
      <p class="figtext">{html.escape(explanation)}</p>
    </figure>
    """


def main() -> int:
    args = parse_args()
    configure_paths(args)
    OUT.mkdir(parents=True, exist_ok=True)
    metadata = read_json(DATASET_META)
    model_meta = read_json(OUT / "connected_residual_run_metadata.json")
    coupled_metrics = read_csv(FLOOD_DIR / "connected_swmm_metrics_rebuilt.csv")
    if not coupled_metrics:
        coupled_metrics = read_csv(CONNECTED / "connected_itzi_swmm_metrics.csv")
    dataset_rows = read_csv(DATASET_AUDIT)
    summary = read_csv(OUT / "metrics" / "connected_residual_ablation_summary.csv")
    event_metrics = read_csv(OUT / "metrics" / "connected_residual_event_metrics.csv")
    cv_summary = read_csv(CV_OUT / "metrics" / "connected_residual_cv_summary.csv")
    cv_event_metrics = read_csv(CV_OUT / "metrics" / "connected_residual_cv_event_metrics.csv")
    quality_cv_summary = read_csv(QUALITY_CV_OUT / "metrics" / "connected_residual_cv_summary.csv")
    quality_cv_events = read_csv(QUALITY_CV_OUT / "metrics" / "connected_residual_cv_event_metrics.csv")
    quality_stratified = read_csv(QUALITY_ANALYSIS_OUT / "metrics" / "quality_stratified_drainlite_summary.csv")
    quality_membership = read_csv(QUALITY_ANALYSIS_OUT / "metrics" / "quality_stratified_event_membership.csv")
    quality_mike_summary = read_csv(QUALITY_ANALYSIS_OUT / "metrics" / "quality_filtered_mike_summary.csv")
    quality_mike_events = read_csv(QUALITY_ANALYSIS_OUT / "metrics" / "quality_filtered_mike_event_metrics.csv")
    v2_quality = read_csv(V2_OUT / "metrics" / "v2_label_quality_grading.csv")
    v2_quality_summary = read_csv(V2_OUT / "metrics" / "v2_label_quality_summary.csv")
    v2_mike_summary = read_csv(V2_OUT / "metrics" / "v2_mike_reference_summary.csv")
    v2_mike_events = read_csv(V2_OUT / "metrics" / "v2_mike_reference_event_metrics.csv")
    hydro_rows = read_csv(HYDRO_OUT / "hydrograph_timing_summary.csv")
    rainfall_accounting = read_csv(HYDRO_OUT / "rainfall_accounting_summary.csv")
    infiltration_pilot = read_csv(INF_PILOT_OUT / "event68_infiltration_pilot_summary.csv")
    infiltration_sensitivity = read_csv(INF_SENS_OUT / "event68_infiltration_sensitivity_metrics.csv")
    infiltration_selection = read_csv(INF_SELECTION_OUT / "infiltration_final_selection.csv")
    importance = read_csv(OUT / "metrics" / "feature_importance_all_static.csv")[:12]
    connected_sensitivity = [
        row for row in infiltration_sensitivity if row.get("scenario") in {"mike_reference", "connected"}
    ]
    formal_quality_rows = [row for row in quality_stratified if row.get("analysis_set") == "quality_filtered_formal"]
    accepted_sensitivity_rows = [row for row in quality_stratified if row.get("analysis_set") == "accepted_only_sensitivity"]
    formal_deployable_rows = [row for row in formal_quality_rows if row.get("model") != "swmm_assisted"]
    formal_oracle_rows = [row for row in formal_quality_rows if row.get("model") == "swmm_assisted"]
    accepted_deployable_rows = [row for row in accepted_sensitivity_rows if row.get("model") != "swmm_assisted"]
    formal_surface = next((row for row in formal_quality_rows if row.get("model") == "surface_only"), {})
    formal_base = next((row for row in formal_quality_rows if row.get("model") == "base"), {})
    formal_all_static = next((row for row in formal_quality_rows if row.get("model") == "all_static"), {})
    formal_mike_surface = next((row for row in quality_mike_summary if row.get("model") == "ITZI surface-only"), {})
    formal_mike_connected = next((row for row in quality_mike_summary if row.get("model") == "ITZI-SWMM connected"), {})
    formal_mike_drainlite = next((row for row in quality_mike_summary if row.get("model") == "DrainLite all_static"), {})

    for row in coupled_metrics:
        row["connected_swmm_peak_m"] = row.get("connected_swmm_peak_m") or row.get("connected_peak_m", "")
        try:
            if row.get("quality_status"):
                row["status"] = row["quality_status"]
            else:
                ce = abs(float(row.get("continuity_error_pct", "nan")))
                row["status"] = "accepted" if ce <= 2.0 else "warning" if ce <= 8.0 else "excluded"
        except Exception:
            row["status"] = "待补充"

    cv_timeseries_html = "\n".join(
        figure(
            OUT / "figures" / f"{event}_cv_timeseries.png",
            f"图 CV-{idx}. {event} 留一验证时间过程",
            "该图使用留一事件交叉验证结果，也就是说图中的 DrainLite 曲线是在该事件未参与训练时得到的。左图为最大水深过程，中图为全域积水体积过程，右图为超过 0.03 m 的淹没面积过程。若绿色 DrainLite 曲线贴近蓝色 ITZI-SWMM 曲线，说明轻量残差模型成功复现了管网耦合带来的时序削减过程。",
        )
        for idx, event in enumerate(metadata.get("events", []), start=1)
    )

    html_text = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LarNO-DrainLite Connected Residual Report</title>
<style>
  body {{ margin: 0; font-family: "Times New Roman", "SimSun", "Songti SC", serif; color: #1f2933; background: #f5f7fb; }}
  .page {{ max-width: 1120px; margin: 0 auto; background: #fff; padding: 42px 54px 72px; box-shadow: 0 0 30px rgba(0,0,0,.08); }}
  h1 {{ font-size: 30px; line-height: 1.25; margin: 0 0 10px; color: #0f172a; }}
  h2 {{ font-size: 22px; margin: 34px 0 12px; padding-bottom: 8px; border-bottom: 2px solid #d9e2ec; color: #102a43; }}
  h3 {{ font-size: 17px; margin: 22px 0 8px; color: #243b53; }}
  p {{ line-height: 1.78; font-size: 15px; margin: 9px 0; }}
  .subtitle {{ color: #52606d; font-size: 16px; }}
  .meta {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin: 22px 0; }}
  .card {{ border: 1px solid #d9e2ec; border-radius: 6px; padding: 12px 14px; background: #f8fafc; overflow-wrap: normal; word-break: normal; }}
  .card b {{ display: block; color: #102a43; margin-bottom: 4px; }}
  .note {{ border-left: 4px solid #2f80ed; background: #eef6ff; padding: 12px 16px; margin: 18px 0; }}
  .warn {{ border-left: 4px solid #c2410c; background: #fff7ed; padding: 12px 16px; margin: 18px 0; }}
  table {{ width: 100%; border-collapse: collapse; margin: 12px 0 22px; font-size: 13px; }}
  th, td {{ border: 1px solid #d9e2ec; padding: 7px 8px; vertical-align: top; }}
  th {{ background: #eef2f7; color: #102a43; text-align: left; }}
  td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  figure {{ margin: 22px 0 30px; border: 1px solid #d9e2ec; border-radius: 6px; padding: 12px; background: #fbfdff; }}
  figure img {{ display: block; max-width: 100%; height: auto; margin: 0 auto; }}
  figcaption {{ font-weight: 700; margin: 10px 0 4px; color: #102a43; }}
  .figtext {{ font-size: 14px; color: #334e68; }}
  code {{ background: #eef2f7; padding: 2px 4px; border-radius: 4px; }}
  .missing {{ color: #8a4b0f; }}
  .toc a {{ color: #1d4ed8; text-decoration: none; margin-right: 14px; }}
  @page {{ size: A4; margin: 14mm 12mm 16mm; }}
  @media print {{
    body {{ background: white; }}
    .page {{ max-width: none; box-shadow: none; padding: 0; }}
    nav {{ display: none; }}
    h2, h3 {{ break-after: avoid; page-break-after: avoid; }}
    figure, table, .card, .note, .warn {{ break-inside: avoid; page-break-inside: avoid; }}
    p {{ orphans: 3; widows: 3; }}
  }}
</style>
</head>
<body>
<main class="page">
  <header>
    <h1>LarNO-DrainLite：基于固定 ITZI-SWMM 耦合标签的管网残差修正报告</h1>
    <p class="subtitle">本报告汇总当前“路网概化管网 + ITZI 原生 SWMM 双向耦合 + 轻量残差学习”的阶段性结果。</p>
    <div class="meta">
      <div class="card"><b>数据集</b>{html.escape(metadata.get("location", "待补充")).replace("_", "_<wbr>")}</div>
      <div class="card"><b>空间分辨率</b>{fmt(metadata.get("cell_size_m"))} m</div>
      <div class="card"><b>空间尺寸</b>{metadata.get("shape", ["待补充", "待补充"])} grid</div>
      <div class="card"><b>事件数量</b>{len(metadata.get("events", []))}</div>
      <div class="card"><b>训练事件</b>{", ".join(model_meta.get("train_events", []))}</div>
      <div class="card"><b>测试事件</b>{", ".join(model_meta.get("test_events", []))}</div>
    </div>
  </header>

  <nav class="toc">
    <a href="#summary">摘要</a>
    <a href="#workflow">模拟方式</a>
    <a href="#quality">标签质量</a>
    <a href="#formalvalidation">正式验证</a>
    <a href="#dataset">数据集</a>
    <a href="#model">模型</a>
    <a href="#validation">交叉验证</a>
    <a href="#mike">MIKE 参照</a>
    <a href="#hydrograph">过程线诊断</a>
    <a href="#results">结果</a>
    <a href="#discussion">讨论</a>
    <a href="#provenance">可追溯性</a>
  </nav>

  <section id="summary">
    <h2>摘要</h2>
    <p>本阶段的核心目标，是在不进行 LarNO 主干大规模重训的前提下，将城市排水管网效应注入已有的地表洪水预报结果。具体做法是先用固定后的 ITZI-SWMM 原生耦合模型生成带管网标签，再训练一个轻量的有符号残差模型。这里的残差定义为 <code>h_itzi_swmm_connected - h_itzi_surface</code>。残差为负，表示管网把地表水排走并降低水深；残差为正，则表示局部回灌、壅水或水位抬升。</p>
    <p>固定拆分结果仅用于探索性消融。正式主模型定义为不使用事后事件汇总量的 <code>all_static</code>，主要泛化结论来自质量筛选后的 5 事件留一验证。固定拆分中加入完整事件 SWMM 汇总指标的 <code>swmm_assisted</code> 可能得到较低误差，但这些指标在实时预报开始时不可获得，因此只能作为事后物理统计辅助对照。</p>
    <p>质量筛选后的严格留一事件交叉验证使用 event1、event67、event68、event69 和 event70，3 个 excluded 事件完全不进入训练或测试。主模型 <code>all_static</code> 的平均绝对误差为 {fmt(formal_all_static.get("mae_mm"))} 毫米，surface-only 为 {fmt(formal_surface.get("mae_mm"))} 毫米，相对改善 {fmt(formal_all_static.get("improvement_vs_surface_pct"))}%；相对不含静态管网特征的 base 模型改善 {fmt(formal_all_static.get("improvement_vs_base_pct"))}%。原 8 事件结果继续保留为诊断性敏感性分析。</p>
  </section>

  <section id="workflow">
    <h2>模拟方式固定</h2>
    <p>后续本文档中所说的“ITZI 和 SWMM 这样模拟”，专指当前固定工作流：二维地表采用 ITZI 的 <code>SurfaceFlowSimulation</code>，一维管网采用 SWMM 动态波并通过 PySWMM 调用，地表与管网交换采用 ITZI 原生 <code>DrainageSimulation.apply_coupling_to_nodes()</code>。这不是静态洼地填充，也不是直接 sink 模型。</p>
    <p>管网来自调整后的路网。由于没有真实市政管网资料，当前方法把路网作为地下排水主线的空间先验，删除孤立和断裂管段后，为每个连通组件配置概化出水口，并按照地形高程设置管底高程和管段坡度。</p>
    <div class="warn"><p>注意：该工作流是“道路对齐的概化管网耦合模型”，不能表述为真实测绘管网。SWMM 连续性误差超过 2% 的事件可以用于诊断和模型探索，但报告中必须标注稳定性警告。</p></div>
    <h3>表 1. ITZI-SWMM 事件质量表：固定耦合事件摘要</h3>
    {table(coupled_metrics, [
        ("event", "事件"),
        ("surface_peak_m", "无管网峰值/m"),
        ("connected_swmm_peak_m", "耦合峰值/m"),
        ("peak_reduction_mm", "峰值削减/mm"),
        ("final_volume_reduction_m3", "末时刻体积削减/m3"),
        ("external_inflow_m3", "入管体积/m3"),
        ("external_outflow_m3", "出流体积/m3"),
        ("continuity_error_pct", "连续性误差/%"),
        ("status", "质量标记"),
    ])}
  </section>

  <section id="quality">
    <h2>物理标签质量分级</h2>
    <p>DrainLite v2 的监督标签来自 ITZI-SWMM 原生耦合计算。由于 SWMM 动态波在强降雨、大范围概化管网和高频地表交换条件下可能出现连续性误差和非收敛步，标签不能只按“是否生成成功”判断质量。本报告采用三档分级：连续性误差绝对值不超过 2% 的事件标为 accepted；2% 到 8% 的事件标为 warning；超过 8% 或报告缺失、计算异常的事件标为 excluded。</p>
    <p>这套分级的作用是给后续论文表述划边界：accepted 只表示连续性误差达到阈值，不等于动态波完全收敛；当前 accepted 事件仍有较高的非收敛步比例，因此也必须披露稳定性风险。warning 事件可以参与主分析但需单独标注；excluded 事件只能作为诊断样本。为避免 excluded 标签进入其他折的训练集，本次进一步单独运行了 5 事件严格留一验证。原 8 事件结果只用于敏感性诊断，不再作为论文核心性能数字。</p>
    <h3>表 2.1. 标签质量等级汇总</h3>
    {table(v2_quality_summary, [
        ("quality_status", "质量等级"),
        ("n_events", "事件数"),
        ("rule", "判定规则"),
    ])}
    <h3>表 2.2. 逐事件 SWMM 稳定性和水量指标</h3>
    {table(v2_quality, [
        ("event", "事件"),
        ("quality_status", "质量等级"),
        ("continuity_error_pct", "连续性误差/%"),
        ("nonconverging_steps_pct", "非收敛步比例/%"),
        ("external_inflow_m3", "入管量/m3"),
        ("external_outflow_m3", "出流量/m3"),
        ("final_stored_m3", "最终存水量/m3"),
        ("peak_reduction_mm", "峰值削减/mm"),
        ("final_volume_reduction_m3", "末时刻体积削减/m3"),
        ("must_disclose_stability_warning", "需披露稳定性警告"),
    ])}
    {figure(V2_OUT / "figures" / "v2_label_quality_grading.png", "图 1A. ITZI-SWMM 物理标签质量分级", "柱状图显示每个事件的 SWMM 连续性误差，折线显示非收敛步比例。绿色阈值线对应 accepted 事件的 2% 边界，红色虚线对应 excluded 事件的 8% 边界。阅读这张图时要同时看两个信息：连续性误差说明整体水量守恒偏差，非收敛步比例说明动态波迭代过程中有多少时间步没有完全满足数值求解条件。当前结果说明管网削峰现象清楚，但强降雨事件仍有稳定性警告，论文中必须如实披露。")}
  </section>

  <section id="formalvalidation">
    <h2>质量筛选后的正式验证</h2>
    <p>本节给出论文采用的主结果。事件集合固定为 event1、event67、event68、event69 和 event70，其中 3 个为 accepted、2 个为 warning。每一折只用另外 4 个质量可用事件训练，测试事件从采样、模型拟合和早停验证中完全隔离；event20、event65 和 event66 不参与任何一折。平均绝对误差、均方根误差和临界成功指数只在 43,606 个高程低于 49.9 米墙体阈值的非墙体格点上计算。被排除的约 22.13% 格点代表建筑或人工抬高墙体，而不是缺失或损坏的地形数据；表中均为五个事件的事件级宏平均。</p>
    <p><code>all_static</code> 的 MAE 从 surface-only 的 {fmt(formal_surface.get("mae_mm"))} 毫米降至 {fmt(formal_all_static.get("mae_mm"))} 毫米，降低 {fmt(formal_all_static.get("improvement_vs_surface_pct"))}%。与只使用水深、降雨、地形、坐标和局部邻域特征的 <code>base</code> 相比，静态管网特征进一步降低 {fmt(formal_all_static.get("improvement_vs_base_pct"))}%。accepted-only 三事件敏感性分析仍得到相同方向，说明管网残差学习结论不是由 warning 事件单独驱动。</p>
    <p>同时需要看到边界：正式 5 事件中，surface-only 相对 MIKE 的非墙体格点全时序 MAE 为 {fmt(formal_mike_surface.get("mae_mm"))} 毫米，DrainLite 为 {fmt(formal_mike_drainlite.get("mae_mm"))} 毫米。DrainLite 将有符号峰值偏差从 {fmt(formal_mike_surface.get("peak_depth_error_mm"))} 毫米调整为 {fmt(formal_mike_drainlite.get("peak_depth_error_mm"))} 毫米，但末时刻体积偏差绝对值从 {fmt(formal_mike_surface.get("abs_final_volume_error_m3"))} 立方米增至 {fmt(formal_mike_drainlite.get("abs_final_volume_error_m3"))} 立方米。其原因是监督目标为 ITZI-SWMM 概化管网标签，而不是 MIKE。因此，本文将结论限定为恢复耦合标签中的管网效应，而不把结果解释为全面提高了对 MIKE 的逐像元拟合或体积一致性。</p>
    <h3>表 2.3. 严格 5 事件留一验证主结果</h3>
    {table(formal_deployable_rows, [
        ("model", "模型"),
        ("n_events", "事件数"),
        ("mae_mm", "标签 MAE/mm"),
        ("rmse_mm", "标签 RMSE/mm"),
        ("csi_0p03", "CSI 0.03m"),
        ("csi_0p15", "CSI 0.15m"),
        ("peak_error_mm", "峰值误差/mm"),
        ("abs_peak_error_mm", "绝对峰值误差/mm"),
        ("abs_peak_time_error_h", "绝对峰时误差/h"),
        ("mean_abs_peak_volume_error_m3", "峰值时刻体积绝对误差/m3"),
        ("mean_abs_final_volume_error_m3", "末时刻体积绝对误差/m3"),
        ("final_reduction_capture_pct", "末时刻削减捕捉率/%"),
        ("mae_to_mike_mm", "相对 MIKE MAE/mm"),
        ("improvement_vs_surface_pct", "相对 surface 改善/%"),
        ("improvement_vs_base_pct", "相对 base 改善/%"),
    ])}
    <h3>表 2.4. accepted-only 三事件敏感性结果</h3>
    {table(accepted_deployable_rows, [
        ("model", "模型"),
        ("n_events", "事件数"),
        ("mae_mm", "标签 MAE/mm"),
        ("rmse_mm", "标签 RMSE/mm"),
        ("csi_0p03", "CSI 0.03m"),
        ("csi_0p15", "CSI 0.15m"),
        ("improvement_vs_surface_pct", "相对 surface 改善/%"),
    ])}
    <h3>表 2.4A. 事后 SWMM 统计量辅助诊断</h3>
    <p><code>swmm_assisted</code> 使用完整事件结束后得到的 SWMM 汇总量，不能在真实预报起点获得，故不参与主模型排序，仅用于估计“若提前知道事件级管网响应强度”能增加多少解释能力。</p>
    {table(formal_oracle_rows, [
        ("model", "模型"),
        ("mae_mm", "标签 MAE/mm"),
        ("rmse_mm", "标签 RMSE/mm"),
        ("csi_0p03", "CSI 0.03m"),
        ("csi_0p15", "CSI 0.15m"),
        ("improvement_vs_surface_pct", "相对 surface 改善/%"),
    ])}
    <h3>表 2.5. 事件纳入规则</h3>
    {table(quality_membership, [
        ("event", "事件"),
        ("quality_status", "质量等级"),
        ("continuity_error_pct", "连续性误差/%"),
        ("nonconverging_steps_pct", "非收敛步比例/%"),
        ("in_all8_diagnostic", "纳入 8 事件诊断"),
        ("in_quality_filtered_formal", "纳入正式验证"),
        ("in_accepted_sensitivity", "纳入 accepted 敏感性"),
    ])}
    {figure(QUALITY_ANALYSIS_OUT / "figures" / "quality_filtered_validation_summary.png", "图 1B. 质量筛选后 DrainLite 正式验证结果", "左图比较严格 5 事件留一验证下各模型对 ITZI-SWMM 标签的平均绝对误差，柱子越低越好；all_static 明显低于 surface-only，并继续低于 base。中图是同一批预测相对 MIKE 外部参照的全时序像元误差，它提醒读者 DrainLite 并非以 MIKE 为监督目标。右图把 accepted 与 warning 事件分开，比较 surface-only 和 all_static；两类事件中 all_static 都保持较低误差，说明主要改善并非由某一个质量等级偶然造成。")}
    <h3>表 2.6. 正式 5 事件相对 MIKE 外部参照</h3>
    {table(quality_mike_summary, [
        ("model", "模型"),
        ("n_events", "事件数"),
        ("mae_mm", "非墙体格点 MAE/mm"),
        ("rmse_mm", "非墙体格点 RMSE/mm"),
        ("csi_0p03", "CSI 0.03m"),
        ("csi_0p15", "CSI 0.15m"),
        ("peak_depth_error_mm", "有符号峰值偏差/mm"),
        ("abs_peak_depth_error_mm", "峰值绝对误差/mm"),
        ("abs_peak_time_error_h", "峰时绝对误差/h"),
        ("peak_map_mae_mm", "峰值空间图 MAE/mm"),
        ("final_volume_error_m3", "有符号末体积偏差/m3"),
        ("abs_final_volume_error_m3", "末体积绝对误差/m3"),
    ])}
    {figure(QUALITY_ANALYSIS_OUT / "figures" / "quality_filtered_mike_summary.png", "图 1C. 正式 5 事件的 MIKE 外部合理性检查", "左图比较非墙体格点上的全时序像元平均绝对误差；surface-only 最低，说明 DrainLite 没有全面改善对 MIKE 的逐像元拟合。中图显示有符号峰值偏差，正值表示高估 MIKE 的全域峰值，负值表示低估；DrainLite 把 surface-only 的正偏调整为幅度较小的负偏。右图显示有符号末时刻体积偏差，DrainLite 与 ITZI-SWMM 均明显为负，说明当前概化管网和有效损失使后期地表存水量低于 MIKE，不能宣称体积一致性改善。")}
  </section>

  <section id="dataset">
    <h2>数据集构建</h2>
    <p>新数据集保存为 <code>{html.escape(metadata.get("location", "待补充"))}</code>。每个事件包含 72 个 5 分钟时间步，空间窗口为 200 × 280 个 20 米网格。为了让后续训练脚本不会混淆旧标签，数据集中 <code>h.npy</code> 已固定为当前 ITZI-SWMM 原生耦合标签，同时额外保存 <code>h_itzi_surface.npy</code>、<code>h_itzi_swmm_connected.npy</code> 和 <code>h_connected_residual.npy</code>。</p>
    <h3>表 2. 数据集事件质量摘要</h3>
    {table(dataset_rows, [
        ("event", "事件"),
        ("surface_peak_m", "无管网峰值/m"),
        ("connected_peak_m", "耦合峰值/m"),
        ("peak_reduction_mm", "峰值削减/mm"),
        ("mean_abs_residual_mm", "平均绝对残差/mm"),
        ("final_volume_reduction_m3", "末时刻体积削减/m3"),
        ("continuity_error_pct", "SWMM 连续性误差/%"),
    ])}
    {figure(OUT / "figures" / "connected_static_dem_network.png", "图 1. 固定概化 SWMM 管网与 DEM 叠加图", "这张图用于检查管网空间特征与地形窗口是否一致。横轴从左向右表示网格 x 方向距离，纵轴从上向下表示网格 y 方向距离；左下角比例尺为 1 千米。底图是数字高程模型，深色栅格表示管线位置，蓝色点表示参与地表交换的概化检查井或雨水口，橙色点表示连通组件的出水口。管线主要沿建筑之间的低平道路廊道布置，山体和抬高建筑格点未被作为道路。该图同时核验方向、尺度和相对位置，避免在模型输入中混用旧网络或发生上下翻转、整体平移等错位。")}
  </section>

  <section id="model">
    <h2>LarNO-DrainLite 模型</h2>
    <p>当前模型是一个可接入 LarNO 的轻量排水残差适配器，但本轮实验没有运行预训练 LarNO 主干。为单独检验管网残差是否可学习，基准输入采用覆盖全部事件的 ITZI surface-only 地表水深。模型还使用当前降雨、累计降雨、地形高程、地形坡度、空间坐标、局部邻域水深和 8 类管网静态特征，输出毫米单位的有符号残差。只有在后续把同网格、同事件的 LarNO 实际预测替换为基准场并重新验证后，才能将结果称为 LarNO 端到端管网修正性能。</p>
    <p>预测公式为 <code>h_pred = max(h_surface + residual_mm / 1000, 0)</code>。这个公式保留了水深非负约束，但不再强制“有管网一定不高于无管网”，因为真实双向耦合里可能出现局部壅水或回灌。</p>
    <h3>表 3. 消融实验汇总</h3>
    {table(summary, [
        ("model", "模型"),
        ("n_events", "测试事件数"),
        ("mae_mm", "MAE/mm"),
        ("rmse_mm", "RMSE/mm"),
        ("csi_0p03", "CSI 0.03m"),
        ("csi_0p15", "CSI 0.15m"),
        ("peak_error_mm", "有符号全域最大水深偏差/mm"),
        ("abs_peak_error_mm", "绝对全域最大水深偏差/mm"),
        ("signed_residual_mae_mm", "残差 MAE/mm"),
        ("mae_to_mike_mm", "相对 MIKE MAE/mm"),
        ("mae_improvement_vs_surface_pct", "相对 surface 改善/%"),
        ("mae_improvement_vs_base_pct", "相对 base 改善/%"),
    ])}
    {figure(OUT / "figures" / "feature_importance_all_static.png", "图 2. all_static 模型特征重要性", "横轴表示打乱某个特征后平均绝对误差增加多少毫米。越靠前的特征越重要。结果显示，无管网水深仍是最核心变量；地形高程、行列坐标和局部邻域水深决定水流汇聚位置；管网特征中，到出水口距离、局部管线密度和局部排水能力密度具有明显贡献。")}
    {figure(OUT / "figures" / "feature_importance_swmm_assisted.png", "图 3. swmm_assisted 事后诊断模型特征重要性", "该图在 all_static 基础上加入完整事件计算结束后得到的 SWMM 出流、存水、连续性误差和节点淹没等汇总指标。这些信息在实时预报起点不可获得，因此该模型是 oracle-assisted（已知事件事后统计量的理想化辅助）对照，不能作为可部署性能。其作用是判断事件级管网强度统计是否还含有额外信息。")}
    <h3>表 4. all_static 前 12 个重要特征</h3>
    {table(importance, [
        ("feature", "特征"),
        ("importance_mean", "重要性均值/mm"),
        ("importance_std", "标准差/mm"),
    ])}
  </section>

  <section id="validation">
    <h2>全 8 事件诊断性留一验证</h2>
    <p>本节保留原来的 8 事件 leave-one-event-out validation（留一事件交叉验证），用于完整记录全部现有计算结果。每次拿 1 个事件测试、其余 7 个事件训练，循环 8 次。由于 event20、event65 和 event66 在部分训练折中会作为监督标签出现，这组数字只能作为诊断和敏感性结果；正式性能结论应以上一节严格 5 事件验证为准。</p>
    <p>交叉验证重点比较 <code>surface_only</code>、<code>base</code>、<code>mask_only</code>、<code>hydraulic_only</code> 和 <code>all_static</code>。其中 <code>base</code> 不含管网静态特征，<code>all_static</code> 加入入口、出水口、管线、管径、坡度、能力、覆土深度和到出水口距离等特征。因此 <code>all_static</code> 相对 <code>base</code> 的改善就是“管网空间先验”的直接贡献。<code>swmm_assisted</code> 另列为使用事后全事件 SWMM 统计量的诊断对照，不进入可部署主模型排名。</p>
    <h3>表 5. 留一事件交叉验证汇总</h3>
    {table(cv_summary, [
        ("model", "模型"),
        ("n_events", "事件数"),
        ("mae_mm", "MAE/mm"),
        ("rmse_mm", "RMSE/mm"),
        ("csi_0p03", "CSI 0.03m"),
        ("csi_0p15", "CSI 0.15m"),
        ("peak_error_mm", "峰值误差/mm"),
        ("signed_residual_mae_mm", "残差 MAE/mm"),
        ("mae_to_mike_mm", "相对 MIKE MAE/mm"),
        ("mae_improvement_vs_surface_pct", "相对 surface 改善/%"),
        ("mae_improvement_vs_base_pct", "相对 base 改善/%"),
    ])}
    {figure(OUT / "figures" / "cv_summary_mae_rmse.png", "图 4. 留一事件交叉验证平均误差", "该图把 8 折交叉验证的平均 MAE 和 RMSE 画成柱状图。surface-only 是不做管网修正的基线，base 是只使用水深、降雨、地形和坐标的残差模型，all_static 是加入全部管网静态特征的主模型。图中 all_static 的 MAE 明显低于 surface-only 和 base，说明轻量残差修正有效，且管网特征提供了额外信息。")}
    {figure(OUT / "figures" / "cv_event_mae_by_model.png", "图 5. 留一事件交叉验证逐事件误差", "该图按事件展示不同模型的 MAE。阅读时应比较同一事件内不同颜色柱子的高低。多数事件中 all_static 都低于 base，说明管网特征对不同降雨过程具有相对稳定的帮助；若个别事件提升较小，通常与该事件峰值位置不受管网控制或 SWMM 标签稳定性有关。")}
    {figure(OUT / "figures" / "cv_volume_reduction_capture.png", "图 6. DrainLite 对末时刻净地表水量变化的恢复", "柱高按 surface-only 末时刻地表水量减去对应结果计算。正值表示耦合或预测后的地表存水量较少，负值表示双向交换和空间重分配后地表存水量反而增加；因此它是有符号净变化，不等同于累计入管量或单向排水量。黑色柱来自 ITZI-SWMM 耦合标签，蓝色柱来自留一验证的 DrainLite all_static 预测。二者越接近，说明模型越能恢复耦合产生的域内净水量响应。")}
    <h3>表 6. 留一事件逐事件指标</h3>
    {table([r for r in cv_event_metrics if r.get("model") in ["surface_only", "base", "all_static", "swmm_assisted"]], [
        ("event", "事件"),
        ("model", "模型"),
        ("mae_m", "MAE/m"),
        ("rmse_m", "RMSE/m"),
        ("peak_error_m", "峰值误差/m"),
        ("signed_residual_mae_m", "残差 MAE/m"),
        ("mae_to_mike_m", "相对 MIKE MAE/m"),
    ], digits=4)}
    {cv_timeseries_html}
  </section>

  <section id="mike">
    <h2>MIKE 外部参照分析</h2>
    <p>MIKE reference 在本研究中只作为外部合理性参照，不作为 DrainLite 的训练标签。这样处理是为了避免把两个目标混在一起：DrainLite 的直接任务是学习“ITZI-SWMM 概化管网标签相对 ITZI surface-only 的修正量”；MIKE 的作用是检查这种修正是否在峰值、水量和空间分布上仍处于合理范围。</p>
    <p>从全时序像元 MAE 看，ITZI surface-only 相对 MIKE 的误差为 {fmt(next((r.get("mae_mm") for r in v2_mike_summary if r.get("model") == "ITZI surface-only"), None))} 毫米，DrainLite all_static CV 为 {fmt(next((r.get("mae_mm") for r in v2_mike_summary if r.get("model") == "DrainLite all_static CV"), None))} 毫米，ITZI-SWMM connected 为 {fmt(next((r.get("mae_mm") for r in v2_mike_summary if r.get("model") == "ITZI-SWMM connected"), None))} 毫米。这意味着 DrainLite 并没有被证明在逐像元全时序 MAE 上更接近 MIKE。</p>
    <p>但从峰值偏差看，管网修正具有物理意义：surface-only 的平均峰值深度误差约为 {fmt(next((r.get("peak_depth_error_mm") for r in v2_mike_summary if r.get("model") == "ITZI surface-only"), None))} 毫米，ITZI-SWMM connected 为 {fmt(next((r.get("peak_depth_error_mm") for r in v2_mike_summary if r.get("model") == "ITZI-SWMM connected"), None))} 毫米，DrainLite all_static CV 为 {fmt(next((r.get("peak_depth_error_mm") for r in v2_mike_summary if r.get("model") == "DrainLite all_static CV"), None))} 毫米。对于最终采用的 1 mm/h 入渗版本，DrainLite 的平均峰值偏差接近 0，说明它没有像 2 mm/h 版本那样系统性压低峰值；但末时刻体积误差从 surface-only 的约 {fmt(next((r.get("final_volume_error_m3") for r in v2_mike_summary if r.get("model") == "ITZI surface-only"), None))} 立方米转为 DrainLite 的约 {fmt(next((r.get("final_volume_error_m3") for r in v2_mike_summary if r.get("model") == "DrainLite all_static CV"), None))} 立方米，表示体积偏差方向由偏高转为偏低。因此不能说它全面优于 MIKE，只能说它把管网削峰效应注入了 surface-only 结果，并在峰值量级上更合理。</p>
    <h3>表 6.1. 相对 MIKE reference 的总体指标</h3>
    {table(v2_mike_summary, [
        ("model", "模型"),
        ("n_events", "事件数"),
        ("mae_mm", "MAE/mm"),
        ("rmse_mm", "RMSE/mm"),
        ("csi_0p03", "CSI 0.03m"),
        ("csi_0p15", "CSI 0.15m"),
        ("peak_depth_error_mm", "峰值深度误差/mm"),
        ("peak_map_mae_mm", "峰值空间图 MAE/mm"),
        ("final_volume_error_m3", "末时刻体积误差/m3"),
    ])}
    {figure(V2_OUT / "figures" / "v2_mike_reference_summary.png", "图 14A. 相对 MIKE reference 的总体误差对比", "该图把全时序 MAE 和峰值空间图 MAE 分开画出。蓝色柱反映所有时间步、所有像元上的平均绝对误差，橙色柱反映每个像元峰值水深图的平均绝对误差。两类误差回答的问题不同：前者更关注整体时序拟合，后者更关注最大淹没风险图。当前结果说明，DrainLite 的目标不是直接拟合 MIKE，而是逼近 ITZI-SWMM 管网标签；因此 MIKE 像元 MAE 未必同步改善。")}
    {figure(V2_OUT / "figures" / "v2_mike_event_mae.png", "图 14B. 逐事件 MIKE reference MAE 对比", "该图逐事件比较三类结果相对 MIKE 的平均绝对误差。阅读时应在同一事件内比较不同模型柱子的高度。可以看到，有些事件中 ITZI surface-only 的逐像元 MAE 更小，而加入管网后峰值和体积偏差更合理。这一现象说明 MIKE 对照不能简单用单一 MAE 指标判定，应同时看峰值、体积、淹没范围和时间过程。")}
    <h3>表 6.2. 逐事件 MIKE reference 指标</h3>
    {table(v2_mike_events, [
        ("event", "事件"),
        ("model", "模型"),
        ("mae_m", "MAE/m"),
        ("rmse_m", "RMSE/m"),
        ("csi_0p03", "CSI 0.03m"),
        ("csi_0p15", "CSI 0.15m"),
        ("peak_depth_error_m", "峰值深度误差/m"),
        ("peak_map_mae_m", "峰值空间图 MAE/m"),
        ("final_volume_error_m3", "末时刻体积误差/m3"),
    ], digits=4)}
  </section>

  <section id="hydrograph">
    <h2>水文过程线诊断</h2>
    <p>在进一步检查中发现，MIKE reference 与最终 1 mm/h ITZI 结果在峰值时刻和退水过程上仍存在系统差异。MIKE 的地表水量通常在降雨峰值后不久达到峰值，并在后半程明显下降；最终 ITZI surface-only 已考虑 1 mm/h 有效入渗，但仍没有显式二维开放边界和管网排水，因此后半程蓄水偏强。ITZI-SWMM connected 加入管网后减少体积和峰值，但退水幅度仍弱于 MIKE。</p>
    <p>本节表格和过程线已重新基于最终 <code>region1_20m_connected_swmm_v2_inf1mmh</code> 数据集生成。更早的无入渗过程线只用于解释为什么需要引入有效损失，不作为最终标签结果。当前模拟从 <code>t=0</code> 开始施加第 0 帧降雨；第 0 帧覆盖 0--300 秒，保存的第 0 帧水深对应 300 秒状态，最后一帧对应 21,600 秒。现存差异主要应从封闭二维边界、统一常量入渗、建筑降雨重分配和概化管网能力等设定继续解释。</p>
    <p>另一个水量差异来自建筑降雨处理。当前区域建筑网格约占 22.13%，模拟程序把落在建筑格上的降雨全部重新分配到非建筑活动格。这保持了整个区域总降雨量不变，但会把道路和空地上的有效降雨提高到约 1.284 倍。如果 MIKE reference 中建筑降雨并不是以同样方式无损、瞬时汇入地表道路，那么 ITZI 会系统性偏蓄水，后半程也更难退水。</p>
    <h3>表 6.3. MIKE 与 ITZI 峰值时刻和退水比诊断</h3>
    {table(hydro_rows, [
        ("event", "事件"),
        ("model", "模型"),
        ("rain_peak_hour", "降雨峰值/h"),
        ("max_depth_peak_hour", "最大水深峰值/h"),
        ("max_depth_final_to_peak_ratio", "末时刻/峰值最大水深"),
        ("volume_peak_hour", "地表水量峰值/h"),
        ("volume_final_to_peak_ratio", "末时刻/峰值水量"),
        ("peaks_at_final_frame", "峰值在末帧"),
    ], digits=3)}
    <h3>表 6.4. 建筑降雨重分配水量核算</h3>
    {table(rainfall_accounting, [
        ("event", "事件"),
        ("building_fraction_pct", "建筑格比例/%"),
        ("redistribution_factor_on_active_cells", "活动格降雨放大倍数"),
        ("mean_total_rainfall_all_cells_mm", "全域平均累计雨量/mm"),
        ("mean_total_rainfall_active_original_mm", "活动格原始累计雨量/mm"),
        ("mean_total_rainfall_active_after_redistribution_mm", "重分配后活动格雨量/mm"),
        ("active_rainfall_increase_pct", "活动格雨量提高/%"),
    ], digits=3)}
    {figure(HYDRO_OUT / "figures" / "hydrograph_timing_summary.png", "图 14C. 最终 1 mm/h 数据源下 MIKE 与 ITZI 的峰值时刻和退水过程诊断", "上图比较最大水深达到峰值的时间。MIKE 多数事件较早出现峰值，最终 1 mm/h ITZI surface-only 的最大水深峰值仍普遍偏晚；加入概化管网后，部分事件的体积峰值明显提前。下图比较末时刻地表水量与峰值地表水量之比：MIKE 曲线有清楚退水，surface-only 多数事件仍接近 1，而 ITZI-SWMM connected 的比值有所下降。该图说明 1 mm/h 入渗和管网共同改善了退水，但还不能完全复制 MIKE 的过程线。")}

    <h3>入渗试算：event68，常量入渗 5 mm/h</h3>
    <p>为检验“缺少入渗/下渗损失”是否是主因之一，复制的 ITZI 模拟程序接入了原生 <code>InfConstantRate</code> 常量入渗模型，并首先对 event68 试算 5 mm/h 的活动格入渗率。这个试算没有覆盖全部 8 个事件，也不是最终率定结果；它的作用是判断趋势是否合理。</p>
    <p>结果表明，加入 5 mm/h 入渗后，event68 的过程线开始出现清楚退水：最大水深峰值时刻从无入渗 surface-only 的约 5.92 h 提前到 3.42 h，接近 MIKE 的 3.17 h；地表水量峰值也从 6.0 h 提前到 2.75 h。但 5 mm/h 对当前事件偏强，峰值水深和总体水量低于 MIKE，因此不能直接作为正式参数，应继续做 1-5 mm/h 的多事件率定。</p>
    <h3>表 6.5. event68 入渗试算指标</h3>
    {table(infiltration_pilot, [
        ("model", "模型"),
        ("max_depth_peak_hour", "最大水深峰值/h"),
        ("max_depth_peak_m", "最大水深峰值/m"),
        ("max_depth_final_to_peak_ratio", "末时刻/峰值最大水深"),
        ("volume_peak_hour", "水量峰值/h"),
        ("volume_final_to_peak_ratio", "末时刻/峰值水量"),
        ("mae_to_mike_mm", "相对 MIKE MAE/mm"),
        ("peak_map_mae_to_mike_mm", "峰值图 MAE/mm"),
    ], digits=3)}
    {figure(INF_PILOT_OUT / "figures" / "event68_infiltration_pilot_hydrograph.png", "图 14D. event68 入渗试算过程线对比", "这张图直接比较 MIKE、无入渗 ITZI、无入渗 ITZI-SWMM、5 mm/h 入渗 ITZI、5 mm/h 入渗 ITZI-SWMM。上图是最大水深过程，下图是地表水量过程。加入入渗后，曲线从持续蓄水变成有峰值、有下降段，说明你判断的方向是对的；但曲线整体低于 MIKE，说明入渗率不能拍脑袋取 5 mm/h，需要通过多事件率定找到既能形成退水、又不明显压低峰值的参数。")}

    <h3>入渗敏感性与最终参数选择</h3>
    <p>在 5 mm/h 试算显示“方向正确但过强”之后，进一步对 event68 运行了 1、2、3、4、5 mm/h 的常量有效入渗率敏感性分析。这里的入渗率不解释为实测土壤入渗能力，而是当前概化模型中的有效损失参数，它综合代表绿地/透水面下渗、局部入渗损失、未显式表达的小尺度排水损失以及 MIKE 参考模型中可能已经率定过的径流损失。</p>
    <p>event68 单事件上，2 mm/h 在峰值水深量级和 SWMM 连续性误差之间最均衡；但把 1 mm/h 和 2 mm/h 扩展到 8 个事件后发现，2 mm/h 会使相对 MIKE 的峰值和末时刻水量系统性偏低。因此最终固定采用更保守的 1 mm/h 版本作为当前主报告的数据源，2 mm/h 保留为敏感性结果，用来说明入渗参数继续增大时会出现过削减风险。</p>
    <h3>表 6.6. event68 连接管网入渗敏感性摘要</h3>
    {table(connected_sensitivity, [
        ("label", "方案"),
        ("rate_mmh", "入渗率/mm/h"),
        ("scenario", "情景"),
        ("max_depth_peak_hour", "最大水深峰值/h"),
        ("max_depth_peak_m", "最大水深峰值/m"),
        ("volume_peak_hour", "水量峰值/h"),
        ("volume_final_to_peak_ratio", "末时刻/峰值水量"),
        ("mae_to_mike_mm", "相对 MIKE MAE/mm"),
        ("peak_map_mae_to_mike_mm", "峰值图 MAE/mm"),
        ("continuity_error_pct", "连续性误差/%"),
    ], digits=3)}
    {figure(INF_SENS_OUT / "figures" / "event68_infiltration_sensitivity_hydrographs.png", "图 14E. event68 入渗率敏感性过程线", "上图展示不同入渗率下的最大水深过程，下图展示非墙体格点上的地表水量过程。阅读时应先看黑色 MIKE 曲线，再看不同颜色的 ITZI-SWMM 曲线。入渗率越大，后半程水量下降越明显，最大水深也越容易被压低。1 mm/h 仍偏蓄水，2 mm/h 开始接近 MIKE 的峰值水深量级，4 和 5 mm/h 虽然峰值时刻更接近，但水深被明显压低，因此不能作为主方案。")}
    {figure(INF_SENS_OUT / "figures" / "event68_infiltration_sensitivity_metrics.png", "图 14F. event68 入渗率敏感性指标", "该图把入渗率作为横轴，分别画出相对 MIKE 的全时序误差、峰值水深、末时刻/峰值水量比和 SWMM 连续性误差。它的作用是避免只凭一张过程线主观选参数。2 mm/h 在 event68 单事件上较均衡，但参数最终不能只由一个事件决定，因此还需要跨事件 MIKE 对照。")}
    <h3>表 6.7. 1 mm/h 与 2 mm/h 跨事件最终选择表</h3>
    {table(infiltration_selection, [
        ("label", "方案"),
        ("dataset", "数据集"),
        ("accepted_events", "accepted"),
        ("warning_events", "warning"),
        ("excluded_events", "excluded"),
        ("cv_surface_mae_to_label_mm", "surface 标签 MAE/mm"),
        ("cv_all_static_mae_to_label_mm", "all_static 标签 MAE/mm"),
        ("cv_all_static_improvement_vs_surface_pct", "标签改善/%"),
        ("mike_surface_peak_error_mm", "surface-MIKE 峰值偏差/mm"),
        ("mike_drainlite_peak_error_mm", "DrainLite-MIKE 峰值偏差/mm"),
        ("mike_drainlite_final_volume_error_m3", "DrainLite-MIKE 末体积偏差/m3"),
    ], digits=3)}
    {figure(INF_SELECTION_OUT / "figures" / "infiltration_final_selection.png", "图 14G. 入渗版本最终选择依据", "该图比较 1 mm/h 和 2 mm/h 两个全事件版本。左图是 DrainLite 相对 ITZI-SWMM 标签的留一误差，两者都能学习管网残差；中图是 DrainLite 相对 MIKE 的平均峰值偏差，1 mm/h 接近零，而 2 mm/h 明显低估；右图是末时刻体积偏差，2 mm/h 的负偏差更大。综合看，1 mm/h 更适合作为当前主报告和论文初稿的数据源。")}
  </section>

  <section id="results">
    <h2>结果展示</h2>
    <h3>代表事件峰值空间对比</h3>
    <p>以下三张图分别对应测试事件 event68、event69 和 event70。每张图包含六个子图：MIKE reference 峰值、ITZI 无管网峰值、ITZI-SWMM 耦合峰值、DrainLite 修正后峰值、DrainLite 与耦合标签的误差，以及在每个格点达到耦合峰值的时刻所取出的模型有符号残差。三个事件的四幅水深图共用同一色标；误差和残差色标分别使用三个事件绝对值的 99.5 百分位数作为统一显示上限，并用箭头色标提示仍有少量超出范围的极值。这个显示处理不裁剪原数组，也不影响任何统计量。阅读时应重点比较第三幅和第四幅；第五幅越接近零，说明空间误差越小；第六幅的蓝色区域表示预测水深降低，红色区域表示局部抬升或回灌式修正。</p>
    {figure(OUT / "figures" / "event68_connected_peak_maps.png", "图 15. event68 峰值空间对比", "event68 是强降雨测试事件之一。无管网 ITZI 在边界低洼区和汇水通道形成较高积水；ITZI-SWMM 耦合后，大片区域水深降低，但最大峰值点仍保留较高水深。DrainLite 修正后能恢复主要削减区域，误差图显示局部仍有过削减或欠削减，说明当前模型对最高积水点附近的管网控制还不够精细。")}
    {figure(OUT / "figures" / "event69_connected_peak_maps.png", "图 16. event69 峰值空间对比", "event69 与 event68 的降雨和积水结构相近，因此是检验模型稳定性的好样本。图中 DrainLite 对管网削减带的再现较连续，预测误差主要集中在局部深水边缘，说明模型已经学习到道路管网对面状积水体积的削减，但对锋面边界和极值点仍存在偏差。")}
    {figure(OUT / "figures" / "event70_connected_peak_maps.png", "图 17. event70 峰值空间对比", "event70 的总水量和峰值均较高。DrainLite 预测的峰值分布与 ITZI-SWMM 标签总体接近，尤其是中低水深范围的削减趋势较清楚。由于训练标签来自概化管网，局部极值误差仍不可避免，应在论文中作为方法局限讨论。")}
  </section>

  <section id="discussion">
    <h2>分析与结论</h2>
    <p>第一，固定后的 ITZI-SWMM 耦合模型确实产生了清楚的管网效应。多个事件中，末时刻地表水体积削减达到数万至近二十万立方米，说明此前“有无管网几乎无差别”的问题主要来自旧管网拓扑断裂或耦合过弱，而不是 ITZI-SWMM 方法本身无效。</p>
    <p>第二，轻量有符号残差路线在当前概化管网和事件集合内成立。在严格五事件留一验证中，all_static 将相对 ITZI-SWMM 耦合标签的事件宏平均 MAE 从 surface-only 的 {fmt(formal_surface.get("mae_mm"))} 毫米降至 {fmt(formal_all_static.get("mae_mm"))} 毫米，降低 {fmt(formal_all_static.get("improvement_vs_surface_pct"))}%；相对不含管网静态特征的 base 仍改善 {fmt(formal_all_static.get("improvement_vs_base_pct"))}%。这一结论证明的是“道路概化管网特征 + 物理标签驱动残差学习”能够恢复部分耦合响应，不等于已经完成 LarNO 主干改造。</p>
    <p>第三，管网静态特征有贡献，但不是唯一主导因素。特征重要性显示，无管网水深、地形和空间位置仍然是残差预测的主控制量，管网距离和局部管线密度提供额外解释力。这符合物理直觉：管网能否削减水深，首先取决于哪里积水，其次才取决于附近是否有入口、管线和出水路径。</p>
    <p>第四，相对 MIKE reference 的结果必须谨慎解释。DrainLite 学习的是 ITZI-SWMM 概化管网效应，而不是 MIKE 标签本身；因此它的主要验收标准是能否恢复耦合标签中的管网残差。MIKE 对照显示 DrainLite 缓解了 surface-only 的峰值偏高问题，但末时刻体积偏差只是由正偏转为负偏，绝对体积误差并未改善；全时序逐像元 MAE 也没有优于 surface-only。这一结果应作为论文讨论的一部分，而不是回避。</p>
    <p>第五，水文过程线诊断显示入渗损失是必要修正项，但单一常量入渗率还不能完全复现 MIKE 的时序 reference。1 mm/h 是当前跨事件更保守的选择：它避免 2 mm/h 的系统性峰值低估，并保留清楚的管网残差信号；但它仍不能让全时序像元误差优于 surface-only，也不能消除所有事件的 SWMM 连续性误差。</p>
    <div class="warn"><p>主要限制是 SWMM 动态波稳定性和概化管网不确定性。当前最终 1 mm/h 版本包含 3 个 accepted 事件、2 个 warning 事件和 3 个 excluded 事件。虽然削峰方向和空间结构清楚，但作为正式论文标签时必须披露这一点，并优先继续进行管网参数率定、出水口边界条件优化、短管段稳定性处理和空间变入渗/径流系数设置。</p></div>
  </section>

  <section id="provenance">
    <h2>数据来源与科学可追溯性</h2>
    <p>本研究没有把参照论文中的已发表结果数值当作本研究结果。公开 LarNO 基准中的数字高程模型、降雨和 MIKE 水深数组是外部来源输入；ITZI surface-only、原生 ITZI-SWMM 耦合水深、残差标签、DrainLite 留一验证预测及其统计量均由本地脚本计算。MIKE 只用于外部合理性检查，没有进入 DrainLite 的训练目标、早停验证或模型选择。</p>
    <p>独立的科学完整性审查文档逐项记录了主要结论对应的 CSV 或 NPY 证据、生成脚本、数组形状、单位、事件纳入规则、分析掩膜以及 SHA-256 文件摘要。该文档同时列出当前证据不能支持的表述，包括真实深圳市政管网、完整 LarNO 推理性能、跨区域泛化和无条件优于 MIKE。这样可以把论文正文中的科学叙述与工作过程审计分开，避免形成研究总结或审稿答辩式写法。</p>
  </section>

  <section>
    <h2>交付内容</h2>
    <p>成果包内包含固定模拟协议、八事件配对数据集、正式五事件留一验证结果、八事件诊断结果、MIKE 外部参照分析、SciencePlots 投稿图件、论文稿、独立 HTML 报告和科学完整性审查文档。所有报告图片均以 Base64 编码嵌入 HTML，表格由计算结果直接转换为 HTML，不依赖外部 CSV、图片路径或网络资源。</p>
  </section>
</main>
</body>
</html>
"""
    REPORT.write_text(html_text, encoding="utf-8")
    REPORT_ALIAS.write_text(html_text, encoding="utf-8")
    print(REPORT)
    print(REPORT_ALIAS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
