#!/usr/bin/env python3
"""Generate a provenance and integrity audit for the DrainLite manuscript."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATASET_NAME = "region1_20m_connected_swmm_v2_inf1mmh"
FLOOD = ROOT / "LarNO-main" / "benchmark" / "urbanflood" / "flood" / DATASET_NAME
GEO = ROOT / "LarNO-main" / "benchmark" / "urbanflood" / "geodata" / DATASET_NAME
MAIN = ROOT / "extended_study" / "output" / "drainlite_connected_residual_v2_inf1mmh"
FILTERED_CV = ROOT / "extended_study" / "output" / "drainlite_connected_cross_validation_v2_inf1mmh_quality_filtered"
QUALITY = ROOT / "extended_study" / "output" / "drainlite_quality_stratified_v2_inf1mmh"
PACKAGE = ROOT / "extended_study" / "output" / "larno_drainlite_v2_inf1mmh_package"
OUT_MD = PACKAGE / "scientific_integrity_audit.md"
OUT_JSON = PACKAGE / "evidence_manifest.json"
EVENTS = ["event1", "event20", "event65", "event66", "event67", "event68", "event69", "event70"]
FORMAL_EVENTS = ["event1", "event67", "event68", "event69", "event70"]
ARRAYS = ["rainfall.npy", "h_itzi_surface.npy", "h_itzi_swmm_connected.npy", "h_mike_ref.npy", "h.npy"]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def find(rows: list[dict[str, str]], key: str, value: str) -> dict[str, str]:
    return next(row for row in rows if row.get(key) == value)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rel(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")


def array_audit(path: Path) -> dict[str, object]:
    array = np.load(path, mmap_mode="r")
    finite = np.isfinite(array)
    return {
        "path": rel(path),
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "finite_fraction": float(np.mean(finite)),
        "minimum": float(np.nanmin(array)),
        "maximum": float(np.nanmax(array)),
        "sha256": sha256(path),
    }


def fmt(value: object, digits: int = 3) -> str:
    number = float(value)
    return f"{number:,.{digits}f}"


def main() -> int:
    PACKAGE.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((FLOOD / "metadata.json").read_text(encoding="utf-8"))
    quality_rows = read_csv(PACKAGE / "metrics" / "v2_label_quality_grading.csv")
    formal_rows = [
        row
        for row in read_csv(QUALITY / "metrics" / "quality_stratified_drainlite_summary.csv")
        if row["analysis_set"] == "quality_filtered_formal"
    ]
    mike_rows = read_csv(QUALITY / "metrics" / "quality_filtered_mike_summary.csv")
    all_static = find(formal_rows, "model", "all_static")
    base = find(formal_rows, "model", "base")
    surface = find(formal_rows, "model", "surface_only")
    mike_surface = find(mike_rows, "model", "ITZI surface-only")
    mike_connected = find(mike_rows, "model", "ITZI-SWMM connected")
    mike_drainlite = find(mike_rows, "model", "DrainLite all_static")

    dem = np.load(GEO / "dem.npy", mmap_mode="r")
    active = np.isfinite(dem) & (dem < 49.9)
    array_records = [array_audit(FLOOD / event / name) for event in EVENTS for name in ARRAYS]
    static_names = metadata["static_features"]
    static_records = [array_audit(GEO / f"{name}.npy") for name in static_names]

    source_files = [
        ROOT / "extended_study" / "run_connected_itzi_swmm_comparison.py",
        ROOT / "extended_study" / "build_component_outfall_swmm_network.py",
        ROOT / "extended_study" / "build_connected_swmm_drainage_dataset_v2_infiltration.py",
        ROOT / "extended_study" / "train_drainlite_connected_residual.py",
        ROOT / "extended_study" / "run_drainlite_connected_cross_validation.py",
        ROOT / "extended_study" / "recompute_connected_cv_metrics.py",
        ROOT / "extended_study" / "summarize_quality_filtered_validation.py",
        ROOT / "extended_study" / "summarize_quality_filtered_mike.py",
        ROOT / "extended_study" / "publication_plot_style.py",
        ROOT / "extended_study" / "redraw_drainlite_publication_figures.py",
        ROOT / "extended_study" / "generate_drainlite_connected_diagnostics.py",
        ROOT / "extended_study" / "generate_larno_drainlite_v2_package.py",
        ROOT / "extended_study" / "generate_drainlite_connected_report.py",
        ROOT / "extended_study" / "generate_larno_drainlite_v2_inf1mmh_manuscript.py",
        ROOT / "extended_study" / "render_drainlite_documents.py",
        ROOT / "extended_study" / "generate_scientific_integrity_audit.py",
    ]
    file_hashes = [{"path": rel(path), "sha256": sha256(path)} for path in source_files]

    claims = [
        {
            "claim": "Strict five-event all-static MAE to the coupled label",
            "value": f"{fmt(all_static['mae_mm'])} mm",
            "evidence": rel(QUALITY / "metrics" / "quality_stratified_drainlite_summary.csv"),
            "generator": "extended_study/summarize_quality_filtered_validation.py",
            "status": "verified",
        },
        {
            "claim": "Surface-only MAE under the same formal protocol",
            "value": f"{fmt(surface['mae_mm'])} mm",
            "evidence": rel(QUALITY / "metrics" / "quality_stratified_drainlite_summary.csv"),
            "generator": "extended_study/summarize_quality_filtered_validation.py",
            "status": "verified",
        },
        {
            "claim": "Improvement of all-static over surface-only",
            "value": f"{fmt(all_static['improvement_vs_surface_pct'])}%",
            "evidence": rel(QUALITY / "metrics" / "quality_stratified_drainlite_summary.csv"),
            "generator": "extended_study/summarize_quality_filtered_validation.py",
            "status": "verified",
        },
        {
            "claim": "Improvement of all-static over base residual model",
            "value": f"{fmt(all_static['improvement_vs_base_pct'])}%",
            "evidence": rel(QUALITY / "metrics" / "quality_stratified_drainlite_summary.csv"),
            "generator": "extended_study/summarize_quality_filtered_validation.py",
            "status": "verified",
        },
        {
            "claim": "Five-event MIKE full-sequence MAE: surface-only / coupled / DrainLite",
            "value": (
                f"{fmt(mike_surface['mae_mm'])} / {fmt(mike_connected['mae_mm'])} / "
                f"{fmt(mike_drainlite['mae_mm'])} mm"
            ),
            "evidence": rel(QUALITY / "metrics" / "quality_filtered_mike_summary.csv"),
            "generator": "extended_study/summarize_quality_filtered_mike.py",
            "status": "verified",
        },
        {
            "claim": "Five-event MIKE signed peak bias: surface-only / DrainLite",
            "value": f"{fmt(mike_surface['peak_depth_error_mm'])} / {fmt(mike_drainlite['peak_depth_error_mm'])} mm",
            "evidence": rel(QUALITY / "metrics" / "quality_filtered_mike_summary.csv"),
            "generator": "extended_study/summarize_quality_filtered_mike.py",
            "status": "verified",
        },
    ]

    manifest = {
        "generated_from_workspace": str(ROOT),
        "dataset": DATASET_NAME,
        "formal_events": FORMAL_EVENTS,
        "excluded_events": ["event20", "event65", "event66"],
        "active_dem_cells": int(active.sum()),
        "total_grid_cells": int(active.size),
        "array_audit": array_records,
        "static_feature_audit": static_records,
        "claims": claims,
        "code_hashes": file_hashes,
    }
    OUT_JSON.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    quality_lines = "\n".join(
        f"| {row['event']} | {row['quality_status']} | {fmt(row['continuity_error_pct'])} | "
        f"{fmt(row['nonconverging_steps_pct'])} |"
        for row in quality_rows
    )
    claim_lines = "\n".join(
        f"| {item['claim']} | {item['value']} | `{item['evidence']}` | "
        f"`{item['generator']}` | {item['status']} |"
        for item in claims
    )
    hash_lines = "\n".join(f"| `{item['path']}` | `{item['sha256']}` |" for item in file_hashes)

    text = f"""# LarNO-DrainLite 科学真实性、准确性与完整性审查文档

## 1. 审查目的

本文件用于把论文正文中的科学叙事与内部证据链分开。审查对象是本工作区生成的 20 m、72 帧 DrainLite 结果，不以参照论文中的数值代替本项目计算。参照论文仅用于理解 LarNO 的研究背景、论文组织和 MIKE benchmark 的来源。

## 2. 数据归属边界

必须区分三类数据。第一类是公开基准输入，包括数字高程模型、降雨场和 `h_mike_ref.npy`；这些文件来自 LarNO benchmark，不能声称为本研究新计算的数据。第二类是本项目重新计算的物理结果，包括 `h_itzi_surface.npy` 和 `h_itzi_swmm_connected.npy`；它们由当前工作区脚本调用复制后的正式 ITZI 原生地表求解器及 ITZI-SWMM 双向交换流程生成。第三类是本项目训练和统计得到的结果，包括 DrainLite 预测数组、交叉验证指标、消融表和图件。论文中所有 DrainLite 定量结论只能来自第二、第三类数据的本地计算。

MIKE 结果只作为外部合理性参照，不是 DrainLite 的监督标签。论文不得把 MIKE 参考场表述为本研究生成，也不得把 DrainLite 对 ITZI-SWMM 标签的改善等同于对 MIKE 的全面改善。

## 3. 数据完整性核验

- 数据集：`{DATASET_NAME}`。
- 事件数：{len(EVENTS)}；每个事件均检查 `rainfall`、ITZI surface-only、ITZI-SWMM connected、MIKE reference 和训练目标五类数组。
- 网格：`{metadata['shape'][0]} x {metadata['shape'][1]}`，空间分辨率 {metadata['cell_size_m']:.0f} m，时间步数 {metadata['time_steps']}。
- 低于 49.9 m 墙体阈值的非墙体像元：{int(active.sum()):,}；总像元：{int(active.size):,}。被排除的格点表示建筑或人工抬高墙体，不是缺失或损坏的 DEM 数据。正式像元误差均使用这一固定掩膜。
- 40 个事件数组和 8 个静态管网特征的形状、数据类型、有限值比例、极值和 SHA-256 校验值均写入 `evidence_manifest.json`。
- 审查结果：所有纳入清单的数组均可读取；详细记录见 `{rel(OUT_JSON)}`。

## 4. 物理标签质量

| Event | Quality tier | SWMM continuity error (%) | Nonconverging steps (%) |
|---|---|---:|---:|
{quality_lines}

`accepted` 只表示绝对连续性误差不超过 2%，不代表动态波每个时间步都完全收敛。event1 和 event67 为 warning；event20、event65 和 event66 因绝对连续性误差超过 8%而不进入正式五事件训练和测试。所有事件的非收敛步比例仍高，因此论文需要把数值稳定性列为限制，而不能称其为完全校准的真实管网真值。

## 5. 核心论文主张的可追溯性

| Scientific claim | Verified value | Direct evidence | Generator | Status |
|---|---:|---|---|---|
{claim_lines}

表中数值由 CSV 直接读取，而不是从参照论文或人工抄录。正式主结果采用五事件严格留一事件交叉验证；被排除事件既不参与训练，也不参与测试。八事件版本只可作为诊断分析，不能与五事件正式结果混写。

## 6. 方法和代码完整性

当前证据链可从以下顺序复现：管网构建与连通性检查；ITZI surface-only 和 ITZI-SWMM 事件计算；20 m 成对标签数据集生成；DrainLite 有符号残差训练；五事件留一交叉验证；固定非墙体掩膜下重新统计；MIKE 外部对照；SciencePlots 图件和报告生成。关键脚本哈希如下：

| File | SHA-256 |
|---|---|
{hash_lines}

哈希只能证明本次审查时文件版本固定，不能代替同行复现。运行顺序及参数还记录在 `REPRODUCIBILITY.md` 和 `ITZI_SWMM_FIXED_WORKFLOW.md`。

## 7. 图件审查记录

第一轮视觉审查发现：旧管网叠图使用栅格等值线，产生大量空心轮廓并遮挡 DEM；峰值空间图的子图标号与标题拥挤，极端值使误差图近似空白；峰时诊断图图例覆盖柱状图；不同脚本使用了不一致字体和 170--220 dpi 输出。

第二轮重制统一采用 SciencePlots 的 `science` 与 `no-latex` 样式、Times New Roman、300 dpi 和色盲友好配色。管网改为栅格覆盖显示；event68--event70 的峰值误差和残差分别使用跨事件统一的 99.5 百分位对称显示范围，峰值水深子图也共用完整数据最大值。该处理仅改变色标显示，不裁剪数组或统计输入。图件重制只重新读取现有 NPY/CSV，没有重新训练模型。

第三轮读图修正了物理标签质量图的标题与图例重叠、MIKE 最终体积偏差图的数值标注遮挡，以及峰值残差子图时间含义不明确的问题。终审后仍保留的科学问题不是排版问题：MIKE 的退水过程明显强于 ITZI；静态管网特征相对 base 的增益为 {fmt(all_static['improvement_vs_base_pct'])}%；DrainLite 相对 MIKE 的全时空 MAE 没有优于 surface-only。这些结果均保留在论文讨论中。

## 8. 不得越界的结论

1. 当前结果证明的是对概化管网 ITZI-SWMM 耦合残差的轻量学习，不是实测市政管网泛化。
2. 当前实验的基础水深场是 ITZI surface-only，而不是实际 LarNO 推理输出；因此模型应表述为 LarNO-compatible adapter，而不是已经完成 LarNO 主干集成。
3. MIKE benchmark 用于外部检查，不是监督目标；MIKE 的逐像元、峰值和体积指标必须分别报告。
4. `swmm_assisted` 使用事件结束后才可知的 SWMM 汇总量，只能作为 oracle-assisted 诊断，不是可部署模型。
5. 1 mm/h 是当前概化系统的经验有效损失率，不是经过土壤观测率定的物理入渗参数。

## 9. 仍未补齐的证据

- 尚无真实 LarNO 输出作为 DrainLite 基础场的端到端实验。
- 尚无实测市政管网拓扑和水力参数验证。
- 尚无独立于 1 mm/h 选择过程的额外事件集。
- SWMM 动态节点水头、管段流量、满管率和壅水状态尚未栅格化为时变模型输入。
- 动态波非收敛步比例仍高，需要进一步改进管网参数、时间步和节点设置。

因此，审查结论是：现有论文主结果可由本地代码和输出文件追溯，适合形成一篇限定范围的轻量残差方法论文；但不能宣称已经得到真实管网条件下的 LarNO 端到端预报模型。
"""
    OUT_MD.write_text(text, encoding="utf-8")
    print(OUT_MD)
    print(OUT_JSON)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
