#!/usr/bin/env python3
"""Write acceptance criteria and current gap audit for LarNO drainage study."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "extended_study" / "output" / "drainage_acceptance"
BENCH = ROOT / "LarNO-main" / "benchmark" / "urbanflood"
CODE = ROOT / "LarNO-main" / "code" / "urbanflood_larfno"
DRAINAGE_LOC = "region1_20m_drainage_v1"

FEATURES = [
    "drain_inlet_mask",
    "drain_outfall_mask",
    "pipe_mask",
    "pipe_diameter",
    "pipe_slope",
    "pipe_capacity",
    "pipe_cover_depth",
    "distance_to_outfall",
]


def valid_source_events() -> list[str]:
    flood = BENCH / "flood" / "region1_20m"
    events = []
    for event_dir in sorted([p for p in flood.iterdir() if p.is_dir()], key=lambda p: p.name):
        h = event_dir / "h.npy"
        rain = event_dir / "rainfall.npy"
        if not h.exists() or not rain.exists():
            continue
        if h.stat().st_size < 1000 or rain.stat().st_size < 1000:
            continue
        try:
            h_arr = np.load(h, mmap_mode="r")
            rain_arr = np.load(rain, mmap_mode="r")
        except Exception:
            continue
        if tuple(h_arr.shape) == (72, 400, 560) and tuple(rain_arr.shape) == (72, 400, 560):
            events.append(event_dir.name)
    return events


def drainage_events() -> list[str]:
    flood = BENCH / "flood" / DRAINAGE_LOC
    if not flood.exists():
        return []
    return sorted([p.name for p in flood.iterdir() if p.is_dir()], key=lambda x: (len(x), x))


def has_all_features() -> bool:
    geo = BENCH / "geodata" / DRAINAGE_LOC
    return all((geo / f"{name}.npy").exists() for name in FEATURES)


def validation_ok() -> bool:
    path = ROOT / "extended_study" / "output" / "drainage_dataset_v1" / "validation_summary.txt"
    return path.exists() and "array_status: ok" in path.read_text(encoding="utf-8")


def swmm_mean_continuity() -> float | None:
    path = ROOT / "extended_study" / "output" / "swmm_network" / "swmm_summary_metrics.csv"
    if not path.exists():
        return None
    vals = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            vals.append(float(row["routing_continuity_error_pct"]))
    return sum(vals) / len(vals) if vals else None


def split_events(list_name: str) -> list[str]:
    path = CODE / "configs" / list_name
    if not path.exists():
        return []
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def swmm_parsed_events(events: list[str]) -> int:
    count = 0
    flood = BENCH / "flood" / DRAINAGE_LOC
    for event in events:
        path = flood / event / "swmm_metrics.csv"
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        if rows and rows[0].get("run_status") == "parsed":
            count += 1
    return count


def adapter_checkpoint_exists() -> bool:
    exp = ROOT / "LarNO-main" / "exp"
    return any(exp.glob("larno_drain_adapter_v1_itzi_sink_*/drainage_adapter_state_dict.pt"))


def larno_d_checkpoint_exists() -> bool:
    exp = ROOT / "LarNO-main" / "exp"
    for p in exp.glob("*"):
        if not p.is_dir():
            continue
        if p.name.startswith("larno_drain_adapter"):
            continue
        if any(p.glob("**/*state_dict.pt")) and p.name != "20260220_183648_006352":
            return True
    return False


def status(ok: bool, partial: bool = False) -> str:
    if ok:
        return "PASS"
    if partial:
        return "PARTIAL"
    return "MISSING"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    source_events = valid_source_events()
    d_events = drainage_events()
    swmm_cont = swmm_mean_continuity()
    train_events = split_events("region1_drainage_train.txt")
    test_events = split_events("region1_drainage_test.txt")
    split_ok = (
        bool(train_events)
        and bool(test_events)
        and set(train_events).isdisjoint(test_events)
        and (set(train_events) | set(test_events)).issubset(set(d_events))
    )
    dataset_ok = has_all_features() and validation_ok() and len(d_events) >= 8
    scale_ok = (
        len(source_events) > 0
        and len(d_events) >= int(0.8 * len(source_events))
        and split_ok
    )
    swmm_parsed = swmm_parsed_events(d_events)

    rows = [
        {
            "phase": "A. 数据集",
            "goal": "建立 region1_20m_drainage_v1，包含 DEM、降雨、MIKE reference、ITZI surface/sink 标签和 8 个管网特征。",
            "acceptance": "所有数组无 NaN/Inf；形状一致；h.npy 为 72x200x280；8 个管网特征与 DEM 同格网。",
            "current": f"drainage_events={len(d_events)}; features={len(FEATURES) if has_all_features() else 0}; validation_ok={validation_ok()}",
            "status": status(dataset_ok),
            "next_action": "已完成；后续只有在生成 h_coupled.npy v2 标签或调整管网特征时才需要重建。" if dataset_ok else "补齐 geodata 特征、事件标签和 validation_summary。",
        },
        {
            "phase": "B. 数据规模",
            "goal": "用本地全部有效 20 m 事件构建训练集，排除 event79。",
            "acceptance": f"本地有效事件 {len(source_events)} 个，drainage 数据集至少覆盖这些事件的 80%，并明确 train/test 拆分；正式模型选择阶段建议另设 validation split 或交叉验证。",
            "current": f"valid_source_events={len(source_events)}; drainage_events={len(d_events)}; train={len(train_events)}; test={len(test_events)}; split_ok={split_ok}",
            "status": status(scale_ok, partial=len(d_events) >= 8),
            "next_action": "已完成 12/5 train/test 拆分；正式训练前建议补 validation split 或用交叉验证，不要用最终 test 调参。" if scale_ok else "补跑缺失事件的 ITZI 72 步 surface/sink 标签，并重建数据集列表。",
        },
        {
            "phase": "C. SWMM standalone 指标",
            "goal": "保留 SWMM dynamic-wave 结果作为事件级/管网能力辅助指标。",
            "acceptance": "八事件 SWMM 均 parsed；平均 routing continuity error < 2%；报告中不称为双向耦合标签。",
            "current": f"parsed_events={swmm_parsed}/{len(d_events)}; mean_routing_continuity_pct={swmm_cont:.3f}" if swmm_cont is not None else f"parsed_events={swmm_parsed}/{len(d_events)}; summary missing",
            "status": status(swmm_cont is not None and swmm_cont < 2.0 and swmm_parsed >= 8),
            "next_action": "当前可作为八事件 standalone 指标；若要把 SWMM 指标作为模型输入，需要补齐新增事件或显式标记缺失。" if swmm_parsed < len(d_events) else "增加 SWMM 节点/管段时序输出到空间特征或事件条件；保留 standalone 限定。",
        },
        {
            "phase": "D. LarNO-D 21通道接口",
            "goal": "LarNO 原 13 通道输入扩展为 21 通道，支持管网特征并兼容预训练权重迁移。",
            "acceptance": "validation_summary 中 processed_input_shape 为 1x21x200x280x1；LarNO-D forward 成功；expanded lifting 加载无 skipped。",
            "current": (ROOT / "extended_study" / "output" / "drainage_dataset_v1" / "validation_summary.txt").read_text(encoding="utf-8").replace("\n", "; ") if validation_ok() else "validation missing",
            "status": status(validation_ok()),
            "next_action": "运行 region1_drainage_finetune.yaml 的正式训练，并输出测试指标。",
        },
        {
            "phase": "E. DrainAdapter 微调路线",
            "goal": "冻结 13 通道 LarNO 主干，训练管网 residual/sink adapter。",
            "acceptance": "至少完成一个正式训练实验，保存 drainage_adapter_state_dict.pt、train_log.csv、test_metrics.csv；adapter 相对 base MAE 至少改善 5% 或说明原因。",
            "current": f"adapter_checkpoint_exists={adapter_checkpoint_exists()}",
            "status": status(adapter_checkpoint_exists(), partial=(CODE / "train_drain_adapter.py").exists()),
            "next_action": "在可用 GPU/服务器上跑 20-50 epoch；当前 GTX 950M 仅适合 dry-run。",
        },
        {
            "phase": "F. LarNO-D 正式训练",
            "goal": "训练或微调 21 通道 LarNO-D，并与 13 通道 baseline 对比。",
            "acceptance": "保存 checkpoint、metrics、prediction maps；MAE/CSI/峰值误差至少一项优于 baseline，且 CSI 不退化。",
            "current": f"larno_d_checkpoint_exists={larno_d_checkpoint_exists()}",
            "status": status(larno_d_checkpoint_exists(), partial=(CODE / "configs" / "region1_drainage_finetune.yaml").exists()),
            "next_action": "先用 expanded lifting-only 跑 20-50 epoch，再全模型小学习率微调。",
        },
        {
            "phase": "G. 消融与创新论证",
            "goal": "证明管网特征确实贡献预测能力，而不是仅靠降雨/DEM。",
            "acceptance": "完成 no-drain、mask-only、hydraulic-only、all-features 四组对比；输出统一表格和空间图。",
            "current": "not_started",
            "status": "MISSING",
            "next_action": "在正式训练完成后批量生成 ablation configs 和评价表。",
        },
        {
            "phase": "H. ITZI-SWMM 双向耦合",
            "goal": "从 standalone SWMM 升级为地表-管网双向交换，生成 h_coupled.npy v2 标签。",
            "acceptance": "地表入流、节点壅水/溢流回灌、outfall 约束均进入质量守恒；输出 h_coupled.npy 和水量平衡审计。",
            "current": "not_started; current SWMM is standalone only",
            "status": "MISSING",
            "next_action": "设计 inlet exchange API，先做单事件短时段耦合原型。",
        },
        {
            "phase": "I. 最终报告",
            "goal": "形成可复现科研报告，说明方法、数据、模型、指标、局限和创新点。",
            "acceptance": "HTML 单文件包含数据质量、物理标签、模型结果、消融结果、结论；不夸大 standalone SWMM。",
            "current": "implementation report exists; model training/ablation results missing",
            "status": "PARTIAL",
            "next_action": "训练和消融完成后重生成最终报告。",
        },
    ]

    csv_path = OUT / "acceptance_audit.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    md = [
        "# LarNO-D 城市排水管网增强研究目标、验收条件与缺口审计",
        "",
        "## 总目标",
        "",
        "构建一个能显式考虑城市排水管网空间分布和水力参数的 LarNO 城市雨洪预报模型。模型应在相同降雨与地形条件下，利用道路/管网对齐的入口、管段、排放口和管网能力特征，学习带排水影响的时空水深响应，并相对原始 LarNO/无管网基线在带管网标签上取得可验证改进。",
        "",
        "## 分阶段验收表",
        "",
    ]
    for row in rows:
        md.extend([
            f"### {row['phase']} - {row['status']}",
            f"- 目标：{row['goal']}",
            f"- 可验收条件：{row['acceptance']}",
            f"- 当前状态：{row['current']}",
            f"- 下一步：{row['next_action']}",
            "",
        ])
    gaps = []
    if not scale_ok:
        gaps.append("数据规模或事件拆分未达标：需要补齐有效事件标签，并确保 train/test 不重叠。")
    if swmm_parsed < len(d_events):
        gaps.append(f"SWMM standalone 指标只覆盖 {swmm_parsed}/{len(d_events)} 个 drainage 事件；新增事件目前只有 missing 占位。")
    if not adapter_checkpoint_exists():
        gaps.append("DrainAdapter 还没有正式训练结果：缺少 checkpoint、train_log.csv、test_metrics.csv 和预测图。")
    if not larno_d_checkpoint_exists():
        gaps.append("LarNO-D 21 通道模型还没有正式训练结果：当前只完成接口和前向校验。")
    gaps.extend([
        "消融实验尚未完成：还不能严格证明管网 mask、水力参数和全特征组合分别带来的增益。",
        "ITZI-SWMM 双向耦合尚未实现：当前 SWMM 是 standalone 指标，不能称为完整地表-管网耦合 reference。",
        "最终科研报告仍缺少正式训练、消融和泛化测试结论。",
    ])

    md.extend(["## 当前最关键缺口", ""])
    md.extend([f"{i}. {gap}" for i, gap in enumerate(gaps, start=1)])
    md.extend([
        "",
        "## 建议推进顺序",
        "",
        "1. 固定实验协议：保留当前 12/5 train/test 拆分，正式调参时增加 validation split 或交叉验证。",
        "2. 先训练 DrainAdapter 20-50 epoch，得到第一个管网修正基线和可视化预测图。",
        "3. 再训练 LarNO-D expanded-lifting 版本，并与原 13 通道 LarNO、DrainAdapter 对比。",
        "4. 补齐四组消融实验：no-drain、mask-only、hydraulic-only、all-features。",
        "5. 若要把 SWMM 作为模型输入，进一步提取节点/管段时序或事件级指标场；当前 17 事件 standalone 汇总指标已可用于报告和辅助分析。",
        "6. 最后推进 ITZI-SWMM 双向耦合，生成 v2 的 h_coupled.npy 标签和水量平衡审计。",
    ])
    md_path = OUT / "GOALS_ACCEPTANCE_GAP_AUDIT.md"
    md_path.write_text("\n".join(md), encoding="utf-8")
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    for row in rows:
        print(f"{row['phase']}: {row['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
