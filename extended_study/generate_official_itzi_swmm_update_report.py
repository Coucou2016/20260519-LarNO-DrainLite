#!/usr/bin/env python3
"""Generate a standalone update report for the formal ITZI-SWMM correction."""

from __future__ import annotations

import base64
import csv
import html
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
IMPORT_DIR = ROOT / "extended_study" / "output" / "official_itzi_swmm_import"
MODEL_DIR = ROOT / "extended_study" / "output" / "drainlite_official_residual"
OUT_HTML = IMPORT_DIR / "official_itzi_swmm_update_report.html"
OUT_MD = IMPORT_DIR / "official_itzi_swmm_update_report.md"
FIG_DIR = IMPORT_DIR / "figures"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def as_float(row: dict[str, str], key: str) -> float:
    try:
        return float(row[key])
    except Exception:
        return float("nan")


def table(rows: list[dict[str, object]], columns: list[tuple[str, str]]) -> str:
    head = "".join(f"<th>{html.escape(label)}</th>" for _, label in columns)
    body = []
    for row in rows:
        cells = []
        for key, _ in columns:
            value = row.get(key, "")
            if isinstance(value, float):
                text = f"{value:.4g}"
            else:
                text = str(value)
            cells.append(f"<td>{html.escape(text)}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def image_data(path: Path) -> str:
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{data}"


def make_figures(import_rows: list[dict[str, str]], summary_rows: list[dict[str, str]]) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    events = [r["event"] for r in import_rows]
    drained = np.array([as_float(r, "official_swmm_drained_m3") for r in import_rows]) / 1000.0
    reduction = np.array([as_float(r, "official_swmm_positive_reduction_volume_m3") for r in import_rows]) / 1000.0
    surcharge = np.array([as_float(r, "official_swmm_local_surcharge_volume_m3") for r in import_rows]) / 1000.0

    x = np.arange(len(events))
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - 0.25, drained, width=0.25, label="SWMM drained")
    ax.bar(x, reduction, width=0.25, label="surface minus coupled")
    ax.bar(x + 0.25, surcharge, width=0.25, label="local surcharge")
    ax.set_ylabel("Volume (10^3 m3)")
    ax.set_xticks(x)
    ax.set_xticklabels(events, rotation=35, ha="right")
    ax.set_title("Formal ITZI-SWMM drainage and local surcharge")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "official_swmm_volume_balance.png", dpi=180)
    plt.close(fig)

    models = [r["model"] for r in summary_rows]
    mae = np.array([as_float(r, "mae_mm") for r in summary_rows])
    rmse = np.array([as_float(r, "rmse_mm") for r in summary_rows])
    fig, ax = plt.subplots(figsize=(10, 5))
    idx = np.arange(len(models))
    ax.bar(idx - 0.18, mae, width=0.36, label="MAE")
    ax.bar(idx + 0.18, rmse, width=0.36, label="RMSE")
    ax.set_ylabel("Error (mm)")
    ax.set_xticks(idx)
    ax.set_xticklabels(models, rotation=35, ha="right")
    ax.set_title("DrainLite-OfficialResidual ablation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "official_residual_ablation.png", dpi=180)
    plt.close(fig)


def main() -> None:
    import_rows_raw = read_csv(IMPORT_DIR / "official_itzi_swmm_event_metrics.csv")
    summary_rows_raw = read_csv(MODEL_DIR / "metrics" / "official_residual_ablation_summary.csv")
    event_rows_raw = read_csv(MODEL_DIR / "metrics" / "official_residual_event_metrics.csv")
    make_figures(import_rows_raw, summary_rows_raw)

    import_rows: list[dict[str, object]] = []
    for r in import_rows_raw:
        import_rows.append(
            {
                "event": r["event"],
                "mike_peak_m": as_float(r, "mike_peak_max_m"),
                "surface_peak_m": as_float(r, "surface_peak_max_m"),
                "official_swmm_peak_m": as_float(r, "official_swmm_peak_max_m"),
                "swmm_drained_m3": as_float(r, "official_swmm_drained_m3"),
                "positive_reduction_m3": as_float(r, "official_swmm_positive_reduction_volume_m3"),
                "local_surcharge_m3": as_float(r, "official_swmm_local_surcharge_volume_m3"),
                "mae_vs_mike_m": as_float(r, "official_swmm_vs_mike_mae_m"),
            }
        )
    summary_rows: list[dict[str, object]] = []
    for r in summary_rows_raw:
        summary_rows.append(
            {
                "model": r["model"],
                "mae_mm": as_float(r, "mae_mm"),
                "rmse_mm": as_float(r, "rmse_mm"),
                "csi_0p03": as_float(r, "csi_0p03"),
                "csi_0p15": as_float(r, "csi_0p15"),
                "improve_surface_pct": as_float(r, "mae_improvement_vs_surface_pct"),
            }
        )

    best = min(summary_rows, key=lambda r: float(r["mae_mm"]))
    surface = next(r for r in summary_rows if r["model"] == "surface_only")
    all_static = next(r for r in summary_rows if r["model"] == "all_static")
    volume_fig = FIG_DIR / "official_swmm_volume_balance.png"
    ablation_fig = FIG_DIR / "official_residual_ablation.png"
    peak_figs = sorted((MODEL_DIR / "figures").glob("*_official_peak_maps.png"))

    html_text = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>正式 ITZI-SWMM 耦合结果接入修正版报告</title>
<style>
body {{ margin: 0; font-family: "Microsoft YaHei", "Noto Sans CJK SC", Arial, sans-serif; color: #1f2933; background: #f5f7fb; }}
.page {{ max-width: 1180px; margin: 0 auto; padding: 36px 28px 72px; background: white; }}
h1 {{ font-size: 30px; margin: 0 0 8px; }}
h2 {{ margin-top: 34px; border-left: 5px solid #2454a6; padding-left: 10px; font-size: 22px; }}
h3 {{ margin-top: 24px; font-size: 18px; }}
p {{ line-height: 1.75; font-size: 15px; }}
.meta {{ color: #52616b; margin-bottom: 22px; }}
.note {{ background: #eef5ff; border: 1px solid #bfd7ff; padding: 14px 16px; border-radius: 8px; }}
.warn {{ background: #fff7ed; border: 1px solid #fed7aa; padding: 14px 16px; border-radius: 8px; }}
table {{ border-collapse: collapse; width: 100%; margin: 12px 0 22px; font-size: 13px; }}
th, td {{ border: 1px solid #d9e2ec; padding: 8px 9px; text-align: right; }}
th:first-child, td:first-child {{ text-align: left; }}
th {{ background: #edf2f7; }}
figure {{ margin: 22px 0 30px; }}
img {{ max-width: 100%; height: auto; border: 1px solid #d8dee9; border-radius: 6px; background: white; }}
figcaption {{ font-size: 14px; color: #3e4c59; line-height: 1.65; margin-top: 8px; }}
code {{ background: #edf2f7; padding: 2px 5px; border-radius: 4px; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 14px; }}
.card {{ border: 1px solid #d9e2ec; border-radius: 8px; padding: 14px; background: #fbfdff; }}
.card b {{ display: block; font-size: 20px; margin-bottom: 4px; }}
</style>
</head>
<body>
<main class="page">
<h1>正式 ITZI-SWMM 耦合结果接入修正版报告</h1>
<p class="meta">数据源：当前工作区副本 <code>external_models/20260518-itzi-flood</code>；原始路径 <code>E:/Projects/20260518-itzi-flood</code> 未被修改。</p>

<section class="note">
<p><b>核心修正。</b>此前轻量研究中存在三类结果：概念性入口 <code>sink</code>、独立一维 <code>SWMM standalone</code> 指标，以及我后来做的 <code>ITZI-PySWMM</code> 原型。根据你的要求，正式数据集现在改为使用复制过来的完整 ITZI-SWMM 耦合模型结果，并新增 <code>h_itzi_swmm_official.npy</code>、<code>h_itzi_swmm_official_surface.npy</code>、<code>h_itzi_swmm_official_reduction.npy</code> 三类文件。旧的 <code>h.npy</code> 和 <code>h_itzi_sink.npy</code> 没有覆盖，仍只代表概念性道路排水入口模型。</p>
</section>

<h2>1. 数据与标签接入</h2>
<p>正式耦合结果来自深圳 <code>region1</code> 的 20 米分辨率子区域，空间大小为 <code>200×280</code>，每个事件保存 72 个五分钟时刻。为了让标签满足后续 LarNO 或轻量模型训练要求，我在副本脚本中只增加了“保存 5 分钟水深序列”的输出开关，物理过程仍使用原工程中的 ITZI 二维地表流求解和 SWMM 动态波管网交换。SWMM 子区域网络为 DYNWAVE 路由，包含 82 个 junction、1 个 outfall 和 29 条 conduit。</p>

{table(import_rows, [
    ("event", "Event"),
    ("mike_peak_m", "MIKE peak (m)"),
    ("surface_peak_m", "Surface peak (m)"),
    ("official_swmm_peak_m", "Official SWMM peak (m)"),
    ("swmm_drained_m3", "SWMM drained (m3)"),
    ("positive_reduction_m3", "Positive reduction (m3)"),
    ("local_surcharge_m3", "Local surcharge (m3)"),
    ("mae_vs_mike_m", "MAE vs MIKE (m)"),
])}

<figure>
<img src="{image_data(volume_fig)}" alt="Formal ITZI-SWMM volume balance">
<figcaption><b>图 1 正式 ITZI-SWMM 排水量、地表削减量与局部回灌量。</b>蓝色柱表示 SWMM 节点累计接收的排水量，橙色柱表示最终水深中 surface-only 高于 coupled 的正向削减体积，绿色柱表示 coupled 局部高于 surface-only 的回灌或水位抬升体积。读这张图时要注意：SWMM drained 是节点交换累计量，不会一比一等于最终图上的净削减，因为管网交换会改变时序汇流、局部水位和空间再分布。正式耦合不是简单“每个格点都扣水”的模型，因此局部 surcharge 是合理且必须统计的现象。</figcaption>
</figure>

<h2>2. 正式标签版 DrainLite</h2>
<p>由于正式 ITZI-SWMM 是双向耦合，不能再使用旧的非负削峰目标。新的轻量模型定义为 <code>DrainLite-OfficialResidual</code>，训练目标是 <code>1000 × (h_itzi_swmm_official - h_itzi_surface)</code>，单位为毫米。这个残差可以为负，也可以为正：负值表示排水降低水深，正值表示检查井壅水、回灌或耦合改变汇流后局部水深升高。</p>

<div class="grid">
<div class="card"><b>{float(surface["mae_mm"]):.3f} mm</b><span>surface-only 对正式耦合标签的平均绝对误差</span></div>
<div class="card"><b>{float(best["mae_mm"]):.3f} mm</b><span>最佳轻量残差模型平均绝对误差：{html.escape(str(best["model"]))}</span></div>
<div class="card"><b>{float(best["improve_surface_pct"]):.1f}%</b><span>最佳模型相对 surface-only 的误差下降</span></div>
<div class="card"><b>{float(all_static["mae_mm"]):.3f} mm</b><span>all_static 管网特征模型误差</span></div>
</div>

{table(summary_rows, [
    ("model", "Model"),
    ("mae_mm", "MAE (mm)"),
    ("rmse_mm", "RMSE (mm)"),
    ("csi_0p03", "CSI 0.03 m"),
    ("csi_0p15", "CSI 0.15 m"),
    ("improve_surface_pct", "Improvement vs surface (%)"),
])}

<figure>
<img src="{image_data(ablation_fig)}" alt="DrainLite official residual ablation">
<figcaption><b>图 2 正式标签版 DrainLite 消融结果。</b>这张图比较 surface-only、概念性 sink 标签以及多个轻量残差模型对正式 ITZI-SWMM 标签的拟合误差。可以看到，surface-only 本身已经非常接近正式耦合标签，说明正式管网在当前子区域的净影响较弱。base 模型只使用水深、降雨、地形、坐标和时间等基础特征，已经把误差降低到约 0.54 毫米；加入全部静态管网特征后没有继续改善，说明当前正式耦合标签中的管网信号较弱，且训练事件数量只有 5 个，管网特征贡献容易被局部状态变量掩盖。</figcaption>
</figure>

<h2>3. 空间对比</h2>
<p>下列空间图展示三个测试事件。每张图的上排依次为 MIKE reference、ITZI surface-only、正式 ITZI-SWMM；下排为轻量残差模型、预测误差和有符号残差。误差图用于判断模型是在错误位置修正，还是沿着低洼区和管网附近给出合理调整。有符号残差图中，负值代表排水削减，正值代表局部回灌或壅水效应。</p>
"""

    for i, fig in enumerate(peak_figs, start=3):
        event = fig.name.split("_")[0]
        html_text += f"""
<figure>
<img src="{image_data(fig)}" alt="{html.escape(event)} official residual maps">
<figcaption><b>图 {i} {html.escape(event)} 正式耦合标签与轻量残差模型空间对比。</b>这张图应先看三张参考图：MIKE reference 是原 LarNO benchmark 的外部参照，surface-only 是不接入 SWMM 时的地表动力学结果，Official ITZI-SWMM 是本次从正式耦合模型导入的新标签。再看下排：DrainLite official residual 表示轻量模型是否能逼近正式耦合结果，Prediction - official error 显示偏高和偏低的位置，Predicted signed residual 则解释模型给出的修正方向。若残差主要集中在已有积水区域和管网影响区，说明模型没有凭空制造修正；若误差集中在峰值洼地，则说明轻量模型仍难以完全表达二维地表流和一维管网交换的局部非线性。</figcaption>
</figure>
"""

    html_text += """
<h2>4. 结论与后续口径</h2>
<p>这次修正后，正式论文和报告应采用三层结果口径：第一，<code>ITZI + sink</code> 是概念性道路排水入口模型，可用于说明“道路对齐排水先验”的可控削峰效果；第二，<code>Official ITZI-SWMM</code> 是复制自你原完整校准工程的正式双向耦合标签，应作为后续正式耦合数据集的主参考；第三，<code>DrainLite-OfficialResidual</code> 是面向正式标签的轻量化机器学习版本，它学习的是有符号残差，而不是单调削峰。</p>
<p class="warn"><b>需要避免的表述。</b>不能再把旧的 ITZI-PySWMM 原型或 standalone SWMM 指标写成正式耦合 reference；也不能把正式 ITZI-SWMM 简化理解成所有位置水深都下降。正式耦合的创新点应写为：在低算力条件下，利用正式 ITZI-SWMM 双向耦合结果构造管网感知标签，并通过有符号残差学习把管网排水和局部回灌效应注入 LarNO/ITZI 类地表预报结果。</p>
</main>
</body>
</html>
"""
    OUT_HTML.write_text(html_text, encoding="utf-8")

    md = [
        "# 正式 ITZI-SWMM 耦合结果接入修正版报告",
        "",
        "正式耦合标签已从当前工作区副本导入，原始路径未修改。",
        "",
        "## 关键结论",
        "",
        f"- 最佳正式残差模型：{best['model']}，MAE={float(best['mae_mm']):.3f} mm。",
        f"- Surface-only 基线 MAE={float(surface['mae_mm']):.3f} mm。",
        "- 正式 ITZI-SWMM 标签允许局部回灌，因此不能使用单调削峰约束。",
        "- 旧 ITZI + sink 标签保留为概念性入口排水对照，不再作为正式耦合 reference。",
        "",
        f"HTML 报告：`{OUT_HTML}`",
    ]
    OUT_MD.write_text("\n".join(md), encoding="utf-8")
    print(f"Wrote {OUT_HTML}")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
