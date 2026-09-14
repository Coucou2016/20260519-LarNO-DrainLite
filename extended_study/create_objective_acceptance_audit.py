from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "extended_study" / "output" / "objective_acceptance_audit"
COUPLED = ROOT / "extended_study" / "output" / "itzi_swmm_coupled_prototype"
EXTVAL = ROOT / "extended_study" / "output" / "drainlite_extended_validation" / "metrics"
REPORT = ROOT / "report.html"

EXPECTED_EVENTS = [
    "event1",
    "event20",
    "event65",
    "event66",
    "event67",
    "event68",
    "event69",
    "event70",
    "event71",
    "event72",
    "event73",
    "event74",
    "event75",
    "event76",
    "event77",
    "event78",
    "event80",
]


def status(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def add(rows: list[dict[str, str]], item: str, requirement: str, evidence: str, ok: bool, caveat: str = "") -> None:
    rows.append(
        {
            "item": item,
            "requirement": requirement,
            "status": status(ok),
            "evidence": evidence,
            "caveat": caveat,
        }
    )


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def inspect_coupled_arrays() -> tuple[bool, str]:
    details = []
    ok = True
    for event in EXPECTED_EVENTS:
        path = COUPLED / event / "h_coupled_prototype.npy"
        if not path.exists():
            ok = False
            details.append(f"{event}: missing h_coupled_prototype.npy")
            continue
        arr = np.load(path, mmap_mode="r")
        event_ok = arr.shape == (72, 200, 280) and np.isfinite(arr[0]).all() and np.isfinite(arr[-1]).all()
        ok = ok and bool(event_ok)
        details.append(f"{event}: shape={arr.shape}, max={float(np.max(arr)):.3f} m")
    return ok, "; ".join(details)


def inspect_exchange_logs() -> tuple[bool, str]:
    required = {"time_s", "swmm_elapsed_s", "dt_drain_s", "surface_to_pipe_cms", "pipe_to_surface_cms", "coupled_nodes"}
    ok = True
    details = []
    for event in EXPECTED_EVENTS:
        path = COUPLED / event / "exchange_log.csv"
        if not path.exists():
            ok = False
            details.append(f"{event}: missing exchange_log.csv")
            continue
        head = pd.read_csv(path, nrows=5)
        row_count = sum(1 for _ in path.open("r", encoding="utf-8")) - 1
        event_ok = required.issubset(set(head.columns)) and row_count > 0
        ok = ok and bool(event_ok)
        details.append(f"{event}: rows={row_count}, columns_ok={event_ok}")
    return ok, "; ".join(details)


def inspect_report() -> tuple[bool, str]:
    if not REPORT.exists():
        return False, "report.html missing"
    html = REPORT.read_text(encoding="utf-8")
    img_count = len(re.findall(r"<img\b", html, re.I))
    base64_count = len(re.findall(r'src="data:image/png;base64,', html, re.I))
    non_base64 = bool(re.search(r"<img[^>]+src=[\"'](?!data:image/png;base64,)", html, re.I))
    http_urls = bool(re.search(r"https?://", html, re.I))
    ok = (
        html.lstrip().startswith("<!DOCTYPE html>")
        and "<style>" in html
        and img_count == base64_count
        and not non_base64
        and not http_urls
        and "ITZI-PySWMM 双向耦合原型 17 事件" in html
    )
    evidence = (
        f"size={REPORT.stat().st_size} bytes, images={img_count}, "
        f"base64_images={base64_count}, non_base64={non_base64}, http_urls={http_urls}"
    )
    return ok, evidence


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []

    status_df = read_csv(COUPLED / "subprocess_batch_status.csv")
    status_ok = (
        not status_df.empty
        and set(status_df["event"]) == set(EXPECTED_EVENTS)
        and len(status_df) == len(EXPECTED_EVENTS)
        and (status_df["status"] == "ok").all()
    )
    add(
        rows,
        "2.1",
        "17 个有效事件稳定完成 ITZI-PySWMM 双向耦合原型运行。",
        f"status_rows={len(status_df)}, ok_rows={int((status_df['status'] == 'ok').sum()) if not status_df.empty else 0}, events={','.join(status_df['event'].tolist()) if not status_df.empty else 'missing'}",
        status_ok,
        "event79 本地文件损坏，按研究方案排除。",
    )

    arrays_ok, arrays_evidence = inspect_coupled_arrays()
    add(rows, "2.2", "每个耦合事件输出完整 72 x 200 x 280 水深数组，且首末帧有限。", arrays_evidence, arrays_ok)

    logs_ok, logs_evidence = inspect_exchange_logs()
    add(rows, "2.3", "节点交换机制有逐步日志，包含地表入管、管网回灌、步长和耦合节点数。", logs_evidence, logs_ok)

    summary = read_csv(COUPLED / "prototype_summary_all_events.csv")
    if not summary.empty:
        exchange_ratio = (
            summary["exchange_net_vs_final_surface_delta_m3"].abs()
            / (summary["surface_to_pipe_m3"] + summary["pipe_to_surface_m3"]).clip(lower=1)
        )
        mass_ok = len(summary) == 17 and (summary["exchange_log_status"] == "ok").all() and float(exchange_ratio.max()) <= 0.02
        evidence = (
            f"rows={len(summary)}, exchange_log_ok={bool((summary['exchange_log_status'] == 'ok').all())}, "
            f"max_exchange_volume_closure_ratio={float(exchange_ratio.max()):.4f}, "
            f"mean_exchange_volume_closure_ratio={float(exchange_ratio.mean()):.4f}"
        )
    else:
        mass_ok = False
        evidence = "prototype_summary_all_events.csv missing or empty"
    add(
        rows,
        "2.4",
        "完成质量/水量交换审计：交换日志积分与最终地表体积变化量级闭合。",
        evidence,
        mass_ok,
        "SWMM .rpt 未提供原生 routing continuity error；本项采用耦合交换日志与地表体积闭合残差审计。",
    )

    if not summary.empty:
        mike_ok = {
            "coupled_vs_mike_mae_mm",
            "coupled_vs_mike_csi_0p03",
            "coupled_vs_mike_csi_0p15",
            "coupled_vs_surface_mae_mm",
            "coupled_vs_sink_mae_mm",
        }.issubset(summary.columns) and len(summary) == 17
        net_back = int((summary["exchange_net_pipe_to_surface_m3"] > 0).sum())
        net_drain = int((summary["exchange_net_pipe_to_surface_m3"] < 0).sum())
        evidence = (
            f"rows={len(summary)}, mean_coupled_vs_mike_mae_mm={summary['coupled_vs_mike_mae_mm'].mean():.3f}, "
            f"net_backflow_events={net_back}, net_drainage_events={net_drain}"
        )
    else:
        mike_ok = False
        evidence = "prototype_summary_all_events.csv missing or empty"
    add(rows, "2.5", "耦合原型与 MIKE reference、ITZI surface-only、ITZI + sink 使用统一指标比较。", evidence, mike_ok)

    spatial = read_csv(EXTVAL / "spatial_holdout_summary.csv")
    spatial_ok = not spatial.empty and set(spatial["subset"]) == {"north_west", "north_east", "south_west", "south_east"}
    add(
        rows,
        "3.1",
        "增加空间窗口 holdout 泛化验证，覆盖四个不同区域。",
        f"rows={len(spatial)}, subsets={','.join(spatial['subset'].tolist()) if not spatial.empty else 'missing'}",
        spatial_ok,
    )

    strata = read_csv(EXTVAL / "pipe_density_strata_summary.csv")
    strata_ok = not strata.empty and set(strata["subset"]) == {"low_pipe_density", "middle_pipe_density", "high_pipe_density"}
    add(
        rows,
        "3.2",
        "增加不同管网密度区域的分层泛化验证。",
        f"rows={len(strata)}, subsets={','.join(strata['subset'].tolist()) if not strata.empty else 'missing'}",
        strata_ok,
    )

    dense = read_csv(EXTVAL / "dense_peak_model_summary.csv")
    dense_ok = not dense.empty and {"hist_gbdt_peak_weighted", "hist_gbdt_all_static_refit", "surface_only"}.issubset(set(dense["model"]))
    if dense_ok:
        peak_row = dense[dense["model"] == "hist_gbdt_peak_weighted"].iloc[0]
        all_row = dense[dense["model"] == "hist_gbdt_all_static_refit"].iloc[0]
        evidence = (
            f"models={','.join(dense['model'].tolist())}, "
            f"peak_weighted_deep_0p15_mae_mm={peak_row['deep_0p15_mae_mm']:.3f}, "
            f"all_static_deep_0p15_mae_mm={all_row['deep_0p15_mae_mm']:.3f}"
        )
    else:
        evidence = f"rows={len(dense)}"
    add(rows, "4.1", "补充峰值加权/深水区增强实验，并与 all-static 和 surface-only 对比。", evidence, dense_ok)

    baselines = read_csv(EXTVAL / "paper_level_baseline_summary.csv")
    required_models = {
        "surface_only",
        "constant_rate",
        "distance_decay_empirical",
        "ridge_linear",
        "random_forest",
        "extra_trees",
        "hist_gbdt_all_static_refit",
        "hist_gbdt_peak_weighted",
    }
    baseline_ok = not baselines.empty and required_models.issubset(set(baselines["model"]))
    add(
        rows,
        "5.1",
        "补充论文级轻量基线：统一削减率、距离衰减、线性、随机森林、极端随机树、梯度提升树。",
        f"rows={len(baselines)}, models={','.join(baselines['model'].tolist()) if not baselines.empty else 'missing'}",
        baseline_ok,
    )

    report_ok, report_evidence = inspect_report()
    add(rows, "5.2", "最终报告内嵌图片和表格，并包含 17 事件耦合原型、空间泛化、峰值增强和基线对比结果。", report_evidence, report_ok)

    audit = pd.DataFrame(rows)
    audit_csv = OUT / "objective_acceptance_audit.csv"
    audit_json = OUT / "objective_acceptance_audit.json"
    audit_md = OUT / "OBJECTIVE_ACCEPTANCE_AUDIT.md"
    audit.to_csv(audit_csv, index=False, encoding="utf-8-sig")
    audit_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    all_pass = bool((audit["status"] == "PASS").all())
    lines = [
        "# Objective Acceptance Audit",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Overall status: {'PASS' if all_pass else 'FAIL'}",
        "",
        "This audit checks the active objective against current files in the workspace.",
        "",
        "| Item | Requirement | Status | Evidence | Caveat |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            "| {item} | {requirement} | {status} | {evidence} | {caveat} |".format(
                **{k: str(v).replace("|", "\\|").replace("\n", " ") for k, v in r.items()}
            )
        )
    audit_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {audit_csv}")
    print(f"Wrote {audit_json}")
    print(f"Wrote {audit_md}")
    print(f"overall_status={'PASS' if all_pass else 'FAIL'}")


if __name__ == "__main__":
    main()
