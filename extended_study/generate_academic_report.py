#!/usr/bin/env python3
"""
Generate the final standalone HTML research report.

The report is deliberately self-contained:
- CSS is embedded in <style>.
- Figures are embedded as Base64 data URIs.
- Tables are rendered directly as HTML.
- No external CDN, image path, CSV, or local resource is required at view time.
"""

from __future__ import annotations

import base64
import csv
import html
import math
from datetime import datetime
from pathlib import Path
from statistics import mean

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "extended_study" / "output"
REPORT = ROOT / "report.html"

BENCH = ROOT / "LarNO-main" / "benchmark" / "urbanflood"
DEM20 = BENCH / "geodata" / "region1_20m" / "dem.npy"
FLOOD20 = BENCH / "flood" / "region1_20m"
NET = OUT / "osm_merged_network.npz"

EVENTS = ["event1", "event20", "event65", "event66", "event67", "event68", "event69", "event70"]
SUB_20M = (slice(80, 280), slice(120, 400))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def fnum(value: object, digits: int = 3, comma: bool = False) -> str:
    if value is None:
        return "待补充"
    if isinstance(value, str):
        value = value.strip()
        if value == "":
            return "待补充"
    try:
        x = float(value)
    except (TypeError, ValueError):
        return html.escape(str(value))
    if not math.isfinite(x):
        return "待补充"
    fmt = f"{{:{',' if comma else ''}.{digits}f}}"
    return fmt.format(x)


def pct(value: object, digits: int = 1) -> str:
    return f"{fnum(value, digits)}%"


def img_data_uri(path: Path) -> str:
    raw = path.read_bytes()
    return "data:image/png;base64," + base64.b64encode(raw).decode("ascii")


def render_table(
    number: int,
    title: str,
    rows: list[dict[str, object]],
    columns: list[tuple[str, str, str]],
    note: str | None = None,
) -> str:
    thead = "".join(f"<th>{html.escape(label)}</th>" for _, label, _ in columns)
    body = []
    for row in rows:
        cells = []
        for key, _, kind in columns:
            val = row.get(key, "")
            if kind == "i":
                text = fnum(val, 0, comma=True)
            elif kind == "f1":
                text = fnum(val, 1)
            elif kind == "f2":
                text = fnum(val, 2)
            elif kind == "f3":
                text = fnum(val, 3)
            elif kind == "p1":
                text = pct(val, 1)
            elif kind == "p2":
                text = pct(val, 2)
            elif kind == "m3":
                text = fnum(val, 0, comma=True)
            else:
                text = html.escape(str(val)) if val not in (None, "") else "待补充"
            cells.append(f"<td>{text}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    note_html = f"<p class=\"table-note\">{html.escape(note)}</p>" if note else ""
    return (
        f"<figure class=\"table-figure\" id=\"table-{number}\">"
        f"<figcaption>表 {number}　{html.escape(title)}</figcaption>"
        f"<div class=\"table-wrap\"><table><thead><tr>{thead}</tr></thead>"
        f"<tbody>{''.join(body)}</tbody></table></div>{note_html}</figure>"
    )


def render_image(number: int, title: str, path: Path, note: str) -> str:
    if not path.exists():
        return (
            f"<figure class=\"figure missing\" id=\"fig-{number}\">"
            f"<figcaption>图 {number}　{html.escape(title)}</figcaption>"
            f"<p>图件缺失：{html.escape(path.name)}，待补充。</p></figure>"
        )
    return (
        f"<figure class=\"figure\" id=\"fig-{number}\">"
        f"<figcaption>图 {number}　{html.escape(title)}</figcaption>"
        f"<img alt=\"{html.escape(title)}\" src=\"{img_data_uri(path)}\">"
        f"<p class=\"fig-note\">{html.escape(note)}</p></figure>"
    )


def card(label: str, value: str, note: str = "") -> str:
    return (
        "<div class=\"card\">"
        f"<div class=\"card-label\">{html.escape(label)}</div>"
        f"<div class=\"card-value\">{html.escape(value)}</div>"
        f"<div class=\"card-note\">{html.escape(note)}</div>"
        "</div>"
    )


def make_network_alignment_figure() -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    path = OUT / "final_network_alignment.png"
    dem = np.load(DEM20).astype(float)
    net = np.load(NET, allow_pickle=True)
    nodes = [dict(x) for x in net["nodes"]]
    links = [dict(x) for x in net["links"]]
    node_by_id = {n["id"]: n for n in nodes}

    y0, y1 = SUB_20M[0].start, SUB_20M[0].stop
    x0, x1 = SUB_20M[1].start, SUB_20M[1].stop
    dem_sub = dem[y0:y1, x0:x1]

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 7.0), constrained_layout=True)
    ax = axes[0]
    dem_plot = np.where(dem >= 49.9, np.nan, dem)
    im = ax.imshow(dem_plot, cmap="terrain", origin="upper")
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="#e74c3c", lw=2.2))
    for link in links[::3]:
        a = node_by_id.get(link["from_node"])
        b = node_by_id.get(link["to_node"])
        if a and b:
            ax.plot([a["col"], b["col"]], [a["row"], b["row"]], color="#0b3d91", lw=0.35, alpha=0.18)
    ax.set_title("Full 20 m DEM domain with adjusted road/pipe network")
    ax.set_xlabel("Column index")
    ax.set_ylabel("Row index")
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Elevation / DSM value")

    ax = axes[1]
    im2 = ax.imshow(np.where(dem_sub >= 49.9, np.nan, dem_sub), cmap="terrain", origin="upper")
    shown_links = 0
    for link in links:
        a = node_by_id.get(link["from_node"])
        b = node_by_id.get(link["to_node"])
        if not a or not b:
            continue
        ar, ac = int(a["row"]), int(a["col"])
        br, bc = int(b["row"]), int(b["col"])
        if (y0 <= ar < y1 and x0 <= ac < x1) or (y0 <= br < y1 and x0 <= bc < x1):
            ax.plot([ac - x0, bc - x0], [ar - y0, br - y0], color="#0b3d91", lw=0.65, alpha=0.45)
            shown_links += 1
    sub_nodes = [n for n in nodes if y0 <= int(n["row"]) < y1 and x0 <= int(n["col"]) < x1]
    ax.scatter(
        [int(n["col"]) - x0 for n in sub_nodes],
        [int(n["row"]) - y0 for n in sub_nodes],
        s=7,
        color="#f39c12",
        edgecolor="black",
        linewidth=0.15,
        alpha=0.85,
    )
    ax.set_title(f"Formal ITZI comparison window: {len(sub_nodes)} nodes, {shown_links} visible links")
    ax.set_xlabel("Window column index")
    ax.set_ylabel("Window row index")
    cbar = fig.colorbar(im2, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Elevation / DSM value")
    fig.suptitle("Final Network-to-DEM Alignment Check", fontsize=16, fontweight="bold")
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def make_swmm_figure(swmm_rows: list[dict[str, str]]) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = OUT / "swmm_network" / "swmm_summary_bars.png"
    events = [r["event"].replace("event", "E") for r in swmm_rows]
    x = np.arange(len(events))
    runoff = np.array([float(r["surface_runoff_m3"]) for r in swmm_rows]) / 1000.0
    storage = np.array([float(r["routing_final_storage_m3"]) for r in swmm_rows]) / 1000.0
    outfall = np.array([float(r["outfall_volume_m3"]) for r in swmm_rows]) / 1000.0
    node_flood = np.array([float(r["node_flooding_volume_m3"]) for r in swmm_rows]) / 1000.0
    flooded_nodes = np.array([float(r["flooded_node_count"]) for r in swmm_rows])
    continuity = np.array([float(r["routing_continuity_error_pct"]) for r in swmm_rows])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), constrained_layout=True)
    width = 0.22
    ax = axes[0]
    ax.bar(x - 1.5 * width, runoff, width, label="Surface runoff", color="#7fb3d5", edgecolor="black")
    ax.bar(x - 0.5 * width, storage, width, label="Routing final storage", color="#73c6b6", edgecolor="black")
    ax.bar(x + 0.5 * width, node_flood, width, label="Node flooding/ponding", color="#f5b041", edgecolor="black")
    ax.bar(x + 1.5 * width, outfall, width, label="Outfall volume", color="#af7ac5", edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(events)
    ax.set_ylabel("Volume (10³ m³)")
    ax.set_title("SWMM dynamic-wave water-balance terms")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.bar(x, flooded_nodes, color="#e67e22", edgecolor="black", alpha=0.86, label="Flooded nodes")
    ax.set_xticks(x)
    ax.set_xticklabels(events)
    ax.set_ylabel("Flooded node count")
    ax.grid(True, axis="y", alpha=0.25)
    ax2 = ax.twinx()
    ax2.plot(x, continuity, color="#1f618d", marker="o", lw=2, label="Routing continuity error")
    ax2.set_ylabel("Routing continuity error (%)")
    ax.set_title("SWMM surcharge indicator and numerical continuity")
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="upper left", fontsize=8)

    fig.suptitle("Standalone SWMM Pipe-Network Branch Summary", fontsize=15, fontweight="bold")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def make_rainfall_figure() -> tuple[Path, list[dict[str, object]]]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = OUT / "rainfall_driver_summary.png"
    rows: list[dict[str, object]] = []
    ts_all = []
    for event in EVENTS:
        rain = np.load(FLOOD20 / event / "rainfall.npy")[:, SUB_20M[0], SUB_20M[1]].astype(float)
        ts = rain.mean(axis=(1, 2))
        ts_all.append(ts)
        rows.append({
            "event": event,
            "mean_total_rain_mm": float(rain.sum(axis=0).mean()),
            "max_total_rain_mm": float(rain.sum(axis=0).max()),
            "mean_peak_step_mm_per_5min": float(ts.max()),
            "time_steps": int(rain.shape[0]),
        })
    write_csv(rows, OUT / "rainfall_driver_summary.csv")

    events_short = [e.replace("event", "E") for e in EVENTS]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), constrained_layout=True)
    axes[0].bar(events_short, [r["mean_total_rain_mm"] for r in rows], color="#5dade2", edgecolor="black")
    axes[0].plot(events_short, [r["max_total_rain_mm"] for r in rows], color="#c0392b", marker="o", label="Max cell total")
    axes[0].set_ylabel("6 h cumulative rainfall (mm)")
    axes[0].set_title("Spatial mean and maximum cumulative rainfall")
    axes[0].grid(True, axis="y", alpha=0.25)
    axes[0].legend()

    t = np.arange(1, 73) * 5
    for event, ts in zip(events_short, ts_all):
        axes[1].plot(t, ts, lw=1.4, alpha=0.85, label=event)
    axes[1].set_xlabel("Time from event start (min)")
    axes[1].set_ylabel("Spatial mean rainfall (mm / 5 min)")
    axes[1].set_title("Rainfall temporal forcing used by ITZI")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(ncol=4, fontsize=8)
    fig.suptitle("Rainfall Driver Consistency Check", fontsize=15, fontweight="bold")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path, rows


def get_data_inventory(swmm_build: list[dict[str, str]], resolution_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    dem = np.load(DEM20)
    true_5m = (BENCH / "geodata" / "region1_5m" / "dem.npy").exists()
    return [
        {
            "item": "MIKE/LarNO public reference",
            "resolution": "20 m",
            "extent": f"{dem.shape[0]}x{dem.shape[1]} cells; {dem.shape[0]*20/1000:.1f}x{dem.shape[1]*20/1000:.1f} km",
            "status": f"可用；本报告分析 {len(EVENTS)} 个事件",
        },
        {
            "item": "MIKE/LarNO true high-resolution reference",
            "resolution": "5 m",
            "extent": "论文尺度约 1600x2240 cells",
            "status": "本地未发现 region1_5m 文件；真实 5 m MIKE 对比待补充" if not true_5m else "本地可用",
        },
        {
            "item": "Formal ITZI dynamic comparison window",
            "resolution": "20 m",
            "extent": "200x280 cells; 4.0x5.6 km",
            "status": "八事件地表-only 与概念性管网 sink 已完成",
        },
        {
            "item": "ITZI refined-grid sensitivity window",
            "resolution": "5 m",
            "extent": "200x280 cells; 1.0x1.4 km",
            "status": "由公开 20 m DEM/降雨细化驱动；八事件已完成",
        },
        {
            "item": "SWMM standalone dynamic-wave branch",
            "resolution": "1D network",
            "extent": f"{swmm_build[0]['nodes']} nodes; {swmm_build[0]['links']} conduits; {swmm_build[0]['subcatchments']} subcatchments",
            "status": "PySWMM 运行完成；非 ITZI-SWMM 双向耦合",
        },
        {
            "item": "5 m reference check mode",
            "resolution": "proxy_5m",
            "extent": resolution_rows[1]["dem_shape"] if len(resolution_rows) > 1 else "待补充",
            "status": "仅用于网格尺寸与绘图一致性检查，不作为真实 5 m MIKE 参考",
        },
    ]


def build_report() -> str:
    itzi20 = read_csv(OUT / "itzi_all" / "all_corrected_metrics.csv")
    swmm_build = read_csv(OUT / "swmm_network" / "swmm_build_status.csv")
    swmm = read_csv(OUT / "swmm_network" / "swmm_summary_metrics.csv")
    res_summary = read_csv(OUT / "itzi_5m_compare" / "resolution_summary.csv")
    ref5_stats = read_csv(OUT / "itzi_5m_compare" / "mike_reference_20m_vs_5m_peak_stats.csv")
    itzi5 = read_csv(OUT / "itzi_5m_refined" / "itzi_5m_refined_metrics.csv")

    network_fig = make_network_alignment_figure()
    swmm_fig = make_swmm_figure(swmm)
    rainfall_fig, rainfall_rows = make_rainfall_figure()

    inventory = get_data_inventory(swmm_build, res_summary)

    mean_surface_pct = mean(float(r["surface_vs_mike_pct"]) for r in itzi20)
    mean_pipe_pct = mean(float(r["pipe_vs_mike_pct"]) for r in itzi20)
    mean_peak_reduction = mean(float(r["pipe_peak_reduction_pct"]) for r in itzi20)
    mean_vol_reduction = mean(float(r["pipe_volume_reduction_pct"]) for r in itzi20)
    mean_flooded_reduction = mean(float(r["pipe_flooded_reduction_pct"]) for r in itzi20)
    mean_drain = mean(float(r["drained_volume_m3"]) for r in itzi20)
    mean_swmm_cont = mean(float(r["routing_continuity_error_pct"]) for r in swmm)
    mean_node_flood = mean(float(r["node_flooding_volume_m3"]) for r in swmm)
    mean_5m_peak_red = mean(float(r["pipe_peak_reduction_pct"]) for r in itzi5)
    mean_5m_vol_red = mean(float(r["pipe_volume_reduction_pct"]) for r in itzi5)

    cards = "".join([
        card("20 m ITZI surface / MIKE peak", f"{mean_surface_pct:.1f}%", "八事件平均；地表-only 略偏高"),
        card("20 m ITZI+sink / MIKE peak", f"{mean_pipe_pct:.1f}%", "概念性管网后更接近参考峰值"),
        card("20 m 管网平均削峰", f"{mean_peak_reduction:.1f}%", "全局最大水深削减"),
        card("20 m 管网体积削减", f"{mean_vol_reduction:.1f}%", "末时刻积水体积削减"),
        card("SWMM routing continuity", f"{mean_swmm_cont:.2f}%", "八事件平均，数值连续性可接受"),
        card("5 m ITZI 窗口削峰", f"{mean_5m_peak_red:.1f}%", "公开 20 m 驱动细化后的 5 m 敏感性"),
    ])

    table_no = 1
    tables: list[str] = []
    tables.append(render_table(table_no, "数据、模型与分辨率清单", inventory, [
        ("item", "对象", "s"), ("resolution", "分辨率/类型", "s"), ("extent", "范围", "s"), ("status", "状态", "s")
    ]))
    table_no += 1
    tables.append(render_table(table_no, "降雨驱动一致性检查", rainfall_rows, [
        ("event", "事件", "s"),
        ("mean_total_rain_mm", "区域平均6h雨量(mm)", "f2"),
        ("max_total_rain_mm", "单元最大6h雨量(mm)", "f2"),
        ("mean_peak_step_mm_per_5min", "区域平均峰值步雨量(mm/5min)", "f2"),
        ("time_steps", "时间步数", "i"),
    ], "rainfall.npy 按 mm/5 min 读取，在 ITZI 中转换为 m/s：rain / 1000 / 300。"))
    table_no += 1
    tables.append(render_table(table_no, "20 m ITZI 动力学模型与 MIKE reference 对比", itzi20, [
        ("event", "事件", "s"),
        ("mike_peak_m", "MIKE峰值(m)", "f3"),
        ("itzi_surface_peak_m", "ITZI地表峰值(m)", "f3"),
        ("itzi_pipe_peak_m", "ITZI+sink峰值(m)", "f3"),
        ("surface_vs_mike_pct", "地表/MIKE", "p1"),
        ("pipe_vs_mike_pct", "管网/MIKE", "p1"),
        ("pipe_peak_reduction_pct", "削峰", "p1"),
        ("pipe_volume_reduction_pct", "体积削减", "p1"),
        ("drained_volume_m3", "排水量(m³)", "m3"),
        ("pipe_flooded_reduction_pct", "淹没格削减", "p1"),
    ], "正式对比窗口为 4.0 km x 5.6 km，分辨率 20 m；MIKE reference 为公开 benchmark 中的 h.npy。"))
    table_no += 1
    tables.append(render_table(table_no, "SWMM 动态波管网分支运行状态", swmm_build, [
        ("event", "事件", "s"),
        ("status", "运行状态", "s"),
        ("rpt_exists", "RPT", "s"),
        ("out_exists", "OUT", "s"),
        ("nodes", "节点", "i"),
        ("links", "管段", "i"),
        ("subcatchments", "子汇水区", "i"),
        ("drainage_area_m2", "汇水面积(m²)", "m3"),
    ], "当前 SWMM 分支为独立 1D dynamic-wave routing；尚未与 ITZI 地表求解器做双向在线耦合。"))
    table_no += 1
    tables.append(render_table(table_no, "SWMM 水量平衡与节点溢流/蓄水指标", swmm, [
        ("event", "事件", "s"),
        ("precip_mm", "降雨(mm)", "f2"),
        ("surface_runoff_m3", "地表径流(m³)", "m3"),
        ("routing_inflow_m3", "管网入流(m³)", "m3"),
        ("routing_final_storage_m3", "末存储(m³)", "m3"),
        ("outfall_volume_m3", "出流(m³)", "m3"),
        ("flooded_node_count", "溢流/蓄水节点", "i"),
        ("node_flooding_volume_m3", "节点溢流/蓄水量(m³)", "m3"),
        ("routing_continuity_error_pct", "连续性误差", "p2"),
    ], "SWMM 的 Routing Flooding Loss 为 0 是由于 ponding/节点蓄水处理；Node Flooding Summary 中的体量用于表征管网壅水压力。"))
    table_no += 1
    tables.append(render_table(table_no, "5 m reference 可用性与 proxy 检查", res_summary, [
        ("dataset", "数据集", "s"),
        ("mode", "模式", "s"),
        ("resolution_m", "分辨率(m)", "i"),
        ("dem_shape", "栅格尺寸", "s"),
        ("domain_km", "区域尺度(km)", "s"),
        ("compare_window_cells", "对比窗口格数", "s"),
        ("compare_window_km", "对比窗口尺度(km)", "s"),
    ], "本地未发现真实 region1_5m；proxy_5m 由 20 m reference 最近邻细化，仅用于尺寸和绘图流程检查。"))
    table_no += 1
    tables.append(render_table(table_no, "20 m reference 与 proxy_5m 峰值统计", ref5_stats, [
        ("event", "事件", "s"),
        ("comparison_mode", "模式", "s"),
        ("mike20_peak_m", "20m峰值(m)", "f3"),
        ("mike5_peak_m", "5m/proxy峰值(m)", "f3"),
        ("mike5_minus_mike20_peak_m", "差值(m)", "f3"),
        ("mike20_mean_m", "20m均值(m)", "f3"),
        ("mike5_mean_m", "5m/proxy均值(m)", "f3"),
    ], "由于 proxy_5m 是最近邻细化，峰值和均值应与 20 m 保持一致；这不是真实 5 m MIKE 计算。"))
    table_no += 1
    tables.append(render_table(table_no, "5 m ITZI refined-grid 敏感性实验", itzi5, [
        ("event", "事件", "s"),
        ("mike20_peak_m", "MIKE20窗口峰值(m)", "f3"),
        ("itzi5_surface_peak_m", "ITZI5地表峰值(m)", "f3"),
        ("itzi5_pipe_peak_m", "ITZI5+sink峰值(m)", "f3"),
        ("pipe_nodes_5m_window", "管网节点", "i"),
        ("pipe_peak_reduction_pct", "削峰", "p1"),
        ("pipe_volume_reduction_pct", "体积削减", "p1"),
        ("drained_volume_m3", "排水量(m³)", "m3"),
        ("pipe_flooded_reduction_pct", "淹没格削减", "p1"),
    ], "该实验在 5 m 计算网格上真实运行 ITZI，但 DEM/降雨驱动来源于公开 20 m 数据细化，因此属于网格敏感性而非真实 5 m reference 复现。"))

    fig_no = 1
    figures: list[str] = []
    figures.append(render_image(fig_no, "全域背景与正式计算窗口", OUT / "full_domain_context.png", "用于定位公开 LarNO/MIKE 20 m 区域与本次 ITZI 正式对比窗口。"))
    fig_no += 1
    figures.append(render_image(fig_no, "最终路网/管网与 DEM 对齐复核", network_fig, "使用当前 osm_merged_network.npz 重新叠加 DEM 生成，检查路网、节点和管段相对位置。"))
    fig_no += 1
    figures.append(render_image(fig_no, "OSM 派生道路-管网分布图", OUT / "osm_pipe_network.png", "展示道路对齐管网、节点与 DEM 背景的空间关系；该图保留作为管网设计依据图件。"))
    fig_no += 1
    figures.append(render_image(fig_no, "降雨驱动时空统计", rainfall_fig, "检查八个事件的 6 h 累积雨量与 5 min 步长降雨过程，确认 ITZI 降雨输入量级。"))
    fig_no += 1
    figures.append(render_image(fig_no, "20 m 八事件峰值洪水空间对比", OUT / "itzi_all" / "all_peaks.png", "逐事件比较 MIKE reference、ITZI 地表-only、ITZI+概念性管网 sink 的峰值水深空间分布。"))
    fig_no += 1
    figures.append(render_image(fig_no, "20 m 峰值统计柱状对比", OUT / "itzi_all" / "all_bars.png", "比较八个事件的全局峰值、管网削峰比例和积水体积削减。"))
    fig_no += 1
    figures.append(render_image(fig_no, "20 m 峰值散点一致性", OUT / "itzi_all" / "all_scatter.png", "检验 ITZI 地表-only 与 ITZI+sink 相对于 MIKE reference 的峰值一致性。"))
    fig_no += 1
    figures.append(render_image(fig_no, "20 m 综合结果仪表板", OUT / "itzi_all" / "all_dashboard.png", "汇总峰值、体积、淹没格数量和管网排水效果，是本次正式 20 m 对比的主图。"))
    fig_no += 1
    figures.append(render_image(fig_no, "SWMM 动态管网分支结果汇总", swmm_fig, "展示独立 SWMM dynamic-wave 管网分支的水量平衡、出流、节点蓄水/壅水和连续性误差。"))
    fig_no += 1
    figures.append(render_image(fig_no, "20 m reference 与 5 m proxy 峰值空间检查", OUT / "itzi_5m_compare" / "mike_reference_20m_vs_5m_peaks.png", "真实 region1_5m 不在本地，因此此图仅为 proxy_5m 绘图与尺寸一致性检查。"))
    fig_no += 1
    figures.append(render_image(fig_no, "5 m ITZI refined-grid 峰值对比", OUT / "itzi_5m_refined" / "itzi_5m_refined_peak_bars.png", "公开 20 m 驱动细化后，在 5 m 网格上运行 ITZI 的窗口敏感性结果。"))
    fig_no += 1
    figures.append(render_image(fig_no, "5 m ITZI refined-grid 管网效应", OUT / "itzi_5m_refined" / "itzi_5m_refined_reduction_metrics.png", "展示 5 m 窗口内概念性管网 sink 对峰值、体积和淹没格的削减比例。"))

    generated = datetime.now().strftime("%Y-%m-%d %H:%M")
    css = """
    :root {
      --ink: #17202a;
      --muted: #5d6d7e;
      --line: #d7dbdd;
      --paper: #ffffff;
      --soft: #f5f7f9;
      --accent: #1f618d;
      --accent2: #117864;
      --warn: #a04000;
    }
    * { box-sizing: border-box; }
    html { scroll-behavior: smooth; }
    body {
      margin: 0;
      font-family: "Microsoft YaHei", "Noto Sans SC", "PingFang SC", Arial, sans-serif;
      color: var(--ink);
      background: #e9edf1;
      line-height: 1.72;
      font-size: 15.5px;
    }
    .page {
      max-width: 1180px;
      margin: 0 auto;
      background: var(--paper);
      min-height: 100vh;
      box-shadow: 0 20px 60px rgba(23, 32, 42, 0.18);
    }
    .cover {
      padding: 58px 64px 42px;
      color: white;
      background: linear-gradient(135deg, #102a43 0%, #1f618d 52%, #117864 100%);
    }
    .cover .kicker { letter-spacing: 0.12em; text-transform: uppercase; opacity: 0.82; font-size: 13px; }
    h1 { margin: 22px 0 16px; font-size: 34px; line-height: 1.22; letter-spacing: 0; }
    .subtitle { max-width: 920px; font-size: 18px; opacity: 0.94; }
    .meta-grid {
      margin-top: 34px;
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px 24px;
      font-size: 14px;
    }
    .meta-grid div { border-top: 1px solid rgba(255,255,255,0.28); padding-top: 10px; }
    nav.toc {
      padding: 28px 64px 16px;
      background: #fbfcfc;
      border-bottom: 1px solid var(--line);
    }
    nav.toc ol {
      columns: 2;
      margin: 8px 0 0;
      padding-left: 22px;
    }
    nav.toc a { color: var(--accent); text-decoration: none; }
    main { padding: 20px 64px 64px; }
    section { padding-top: 18px; margin-top: 14px; }
    h2 {
      font-size: 23px;
      margin: 30px 0 12px;
      padding-bottom: 8px;
      border-bottom: 2px solid #e5e8e8;
    }
    h3 { font-size: 18px; margin: 24px 0 8px; color: #1b4f72; }
    p { margin: 9px 0; }
    .lead {
      font-size: 16.5px;
      background: #f7fbfc;
      border-left: 4px solid var(--accent);
      padding: 14px 16px;
    }
    .cards {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 14px;
      margin: 18px 0 20px;
    }
    .card {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px 16px;
      background: #ffffff;
    }
    .card-label { color: var(--muted); font-size: 13px; }
    .card-value { font-size: 27px; font-weight: 700; color: var(--accent); margin-top: 2px; }
    .card-note { color: var(--muted); font-size: 12.5px; margin-top: 3px; }
    .figure, .table-figure {
      margin: 22px 0 28px;
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
      background: white;
    }
    figcaption {
      font-weight: 700;
      padding: 12px 14px;
      background: #eef3f6;
      border-bottom: 1px solid var(--line);
    }
    .figure img {
      display: block;
      width: 100%;
      height: auto;
      background: #fff;
    }
    .fig-note, .table-note {
      margin: 0;
      padding: 10px 14px 13px;
      color: var(--muted);
      font-size: 13px;
      background: #fbfcfc;
      border-top: 1px solid var(--line);
    }
    .table-wrap { overflow-x: auto; }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 13.2px;
    }
    th, td {
      padding: 8px 9px;
      border-bottom: 1px solid #e5e8e8;
      vertical-align: top;
      white-space: nowrap;
    }
    th {
      background: #f8f9f9;
      text-align: left;
      color: #273746;
      position: sticky;
      top: 0;
    }
    tr:nth-child(even) td { background: #fcfcfd; }
    .callout {
      border: 1px solid #f5cba7;
      background: #fff8f0;
      color: #6e2c00;
      padding: 12px 14px;
      border-radius: 8px;
      margin: 14px 0;
    }
    .ok {
      border-color: #a9dfbf;
      background: #f2fbf5;
      color: #145a32;
    }
    ul, ol { padding-left: 22px; }
    li { margin: 5px 0; }
    code {
      background: #f4f6f7;
      border: 1px solid #e5e8e8;
      border-radius: 4px;
      padding: 1px 4px;
      font-family: Consolas, "Courier New", monospace;
      font-size: 0.92em;
    }
    .two-col {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 18px;
    }
    .footer {
      margin-top: 36px;
      padding-top: 18px;
      border-top: 1px solid var(--line);
      color: var(--muted);
      font-size: 13px;
    }
    @media (max-width: 860px) {
      .cover, nav.toc, main { padding-left: 22px; padding-right: 22px; }
      nav.toc ol { columns: 1; }
      .cards, .two-col { grid-template-columns: 1fr; }
      h1 { font-size: 27px; }
      th, td { white-space: normal; }
    }
    @media print {
      body { background: white; }
      .page { box-shadow: none; max-width: none; }
      section, .figure, .table-figure { break-inside: avoid; }
      a { color: inherit; text-decoration: none; }
    }
    """

    body = f"""
    <div class="page">
      <header class="cover">
        <div class="kicker">Standalone Scientific HTML Report</div>
        <h1>LarNO 城市洪涝公开基准、ITZI 动力学复现与道路对齐管网拓展研究报告</h1>
        <p class="subtitle">基于深圳区域公开 MIKE/LarNO reference 数据，完成 ITZI 地表动力学复现、概念性道路对齐排水入口 sink、独立 SWMM 动态波管网分支，以及 5 m 网格敏感性分析。</p>
        <div class="meta-grid">
          <div><strong>研究对象</strong><br>Large-scale urban flood modeling and zero-shot high-resolution generalization with LarNO</div>
          <div><strong>本地工作目录</strong><br>E:\\Projects\\20260519-LarNO</div>
          <div><strong>报告生成时间</strong><br>{html.escape(generated)}</div>
          <div><strong>报告性质</strong><br>自包含 HTML；图片 Base64 内嵌；表格直接写入</div>
        </div>
      </header>
      <nav class="toc">
        <strong>目录</strong>
        <ol>
          <li><a href="#abstract">摘要</a></li>
          <li><a href="#background">研究背景与目的</a></li>
          <li><a href="#data">数据与可用性</a></li>
          <li><a href="#methods">方法与模型设置</a></li>
          <li><a href="#process">研究过程与质量控制</a></li>
          <li><a href="#results">结果展示</a></li>
          <li><a href="#discussion">分析与讨论</a></li>
          <li><a href="#conclusion">主要结论</a></li>
          <li><a href="#limits">不足与展望</a></li>
        </ol>
      </nav>
      <main>
        <section id="abstract">
          <h2>摘要</h2>
          <p class="lead">本报告对 LarNO 论文公开的深圳城市洪涝 benchmark 数据、ITZI 地表流动力学求解、道路对齐概念性排水入口模型，以及 SWMM 动态波管网分支进行了统一整理和复核。正式 20 m 对比中，ITZI 地表-only 峰值平均为 MIKE reference 的 {mean_surface_pct:.1f}%，加入概念性管网 sink 后平均为 {mean_pipe_pct:.1f}%，平均削峰 {mean_peak_reduction:.1f}%，末时刻积水体积平均削减 {mean_vol_reduction:.1f}%。独立 SWMM 分支已在 PySWMM 下完成八事件运行，平均 routing continuity error 为 {mean_swmm_cont:.2f}%。由于本地缺少真实 region1_5m MIKE reference，本报告不把 proxy_5m 结果表述为真实 5 m 验证；另完成了公开 20 m 驱动细化后的 5 m ITZI 窗口敏感性实验。</p>
          <div class="cards">{cards}</div>
        </section>

        <section id="background">
          <h2>研究背景与目的</h2>
          <p>LarNO 论文面向大尺度、高分辨率、长历时城市洪涝预报，使用 MIKE+ 数值模型结果作为 reference，并训练 neural operator 学习降雨、地形、排水设施与时空水深之间的映射。本研究的目标不是重新编造 LarNO 输出，而是在本地公开数据约束下，核查 reference 数据、用 ITZI 动力学模型建立可解释的物理复现分支，并进一步加入道路对齐管网因素，判断管网削峰是否合理。</p>
          <p>围绕前期发现的问题，本次最终整理重点修正三类风险：第一，正式结果不再使用静态洼地填充模型替代 ITZI，而是调用 ITZI 的 <code>SurfaceFlowSimulation</code> 与 <code>Hydrology</code>；第二，修正降雨起算与时间积分，使第一步降雨从 t=0 开始并覆盖完整 6 h；第三，保留概念性道路排水入口 sink 的同时，增加 SWMM 内置 dynamic-wave 管网分支作为独立对照。</p>
        </section>

        <section id="data">
          <h2>数据与可用性</h2>
          {tables[0]}
          <div class="callout">关键说明：本地可用的是 <code>region1_20m</code> 的 DEM、72 个 5 min 步长水深 reference 与降雨场；真实 <code>region1_5m</code> 的 MIKE reference 文件未在本地发现。因此，报告中的真实 MIKE/ITZI 对比以 20 m 为主，5 m 部分明确标注为 proxy 检查或 ITZI 5 m 网格敏感性。</div>
          {figures[0]}
          {figures[1]}
          {figures[2]}
        </section>

        <section id="methods">
          <h2>方法与模型设置</h2>
          <h3>MIKE/LarNO reference</h3>
          <p>公开 benchmark 中的 <code>h.npy</code> 被作为 MIKE+ reference 水深数据读取，维度为 72 x 400 x 560，时间分辨率为 5 min，总历时 6 h。正式对比选择事件 Event 1、20、65、66、67、68、69、70，与前期绘图和用户关注事件保持一致。</p>
          <h3>ITZI 地表动力学</h3>
          <p>ITZI 模型采用二维地表流动力学求解器，而不是静态洼地填充。正式 20 m 窗口为 rows 80:280、cols 120:400，即 200 x 280 cells、4.0 km x 5.6 km。建筑物区域按高程/高阻处理；降雨落在建筑上的部分重新分配到活动地表单元；入渗设置为 <code>InfNull</code>，以便与 reference 的强降雨地表响应做直接对比。</p>
          <h3>降雨设置</h3>
          <p>降雨场来自公开 <code>rainfall.npy</code>，按 mm/5 min 读取，进入 ITZI 前转换为 m/s，即 <code>rain / 1000 / 300</code>。本次修正后，降雨在 t=0 即开始施加，循环积分到完整 6 h，避免前期可能漏掉首个 5 min 或末段积分不完整的问题。</p>
          {tables[1]}
          {figures[3]}
          <h3>管网拓展的两条分支</h3>
          <p>概念性分支沿调整后的道路/管网节点布设排水入口，用孔口形式根据水头移除地表水，并限制每 30 s 移除量不超过局部水量的一定比例，避免数值上瞬时抽干。该分支直接进入 ITZI 地表水深演化，适合做“道路对齐排水入口”削峰敏感性分析。</p>
          <p>SWMM 分支使用同一套道路派生管网构建 SWMM dynamic-wave 1D 管网，包含节点、管段、子汇水区和 outfall，并通过 PySWMM 完成八事件独立运行。需要强调：当前 SWMM 是 standalone 1D routing 对照，不是 ITZI-SWMM 双向在线耦合；两者之间的联动边界、水位交换与检查井溢流回灌仍为后续待补充。</p>
        </section>

        <section id="process">
          <h2>研究过程与质量控制</h2>
          <ol>
            <li>核查公开数据：确认 20 m DEM 与八事件 <code>h.npy</code>/<code>rainfall.npy</code> 可用，确认真实 5 m reference 缺失。</li>
            <li>复核路网/管网位置：以当前调整后的 <code>osm_merged_network.npz</code> 为准，重新叠加 DEM 与正式计算窗口生成对齐图。</li>
            <li>重跑 20 m ITZI：分别运行地表-only 与 ITZI+概念性管网 sink，输出事件级 <code>.npz</code>、统计表和空间图。</li>
            <li>重跑 SWMM：由道路派生管网写出 SWMM <code>.inp</code>，使用 PySWMM 生成 <code>.rpt</code>/<code>.out</code>，解析水量平衡与节点壅水指标。</li>
            <li>补跑 5 m 对应实验：生成 proxy_5m reference 检查；在 1.0 km x 1.4 km 窗口内运行 5 m ITZI refined-grid 敏感性。</li>
            <li>剔除旧版错误图：前期使用静态或旧统计流程生成的局部/异常峰值图未纳入正式结果。</li>
          </ol>
          <div class="callout ok">质量控制结果：20 m ITZI、SWMM、5 m refined-grid 三组脚本均已重新执行；SWMM 旧错误文件已清理；最终报告图表均由当前输出重新读取生成。</div>
        </section>

        <section id="results">
          <h2>结果展示</h2>
          <h3>20 m ITZI 与 MIKE reference 对比</h3>
          <p>表 3 与图 5 至图 8 是本报告的正式主结果。地表-only ITZI 在多数事件中峰值略高于 MIKE reference，加入道路对齐概念性管网 sink 后，峰值整体更接近 reference，同时积水体积和淹没格数量明显降低。平均而言，地表-only 峰值为 MIKE 的 {mean_surface_pct:.1f}%，ITZI+sink 为 {mean_pipe_pct:.1f}%；管网平均削峰 {mean_peak_reduction:.1f}%，末时刻体积削减 {mean_vol_reduction:.1f}%，淹没格数量削减 {mean_flooded_reduction:.1f}%。</p>
          {tables[2]}
          {figures[4]}
          {figures[5]}
          {figures[6]}
          {figures[7]}

          <h3>SWMM 动态波管网分支</h3>
          <p>SWMM 分支在八事件中均运行成功，管网规模为 {swmm_build[0]['nodes']} 个节点、{swmm_build[0]['links']} 条管段、{swmm_build[0]['subcatchments']} 个子汇水区。结果显示，outfall 体量相对 routing inflow 较小，较多水量表现为管网末存储和节点 ponding/壅水，说明当前道路派生管网的排放能力和出口设置仍偏概念化，不能直接等同于真实规划管网能力。</p>
          {tables[3]}
          {tables[4]}
          {figures[8]}

          <h3>5 m 分辨率对应分析</h3>
          <p>本地缺少真实 5 m MIKE reference，因此本报告做了两个层级的 5 m 工作：一是 proxy_5m 检查，用于确认 20 m 到 5 m 图件和窗口映射流程；二是 5 m ITZI refined-grid 真实数值运行，但其 DEM 和降雨仍由公开 20 m 数据细化而来。5 m 窗口中，概念性管网 sink 平均削峰 {mean_5m_peak_red:.1f}%，体积削减 {mean_5m_vol_red:.1f}%。</p>
          {tables[5]}
          {tables[6]}
          {tables[7]}
          {figures[9]}
          {figures[10]}
          {figures[11]}
        </section>

        <section id="discussion">
          <h2>分析与讨论</h2>
          <h3>关于“ITZI 是否只是洼地填充”</h3>
          <p>本次正式结果不是洼地填充。ITZI 分支调用地表流动力学求解器，按 CFL 控制时间步推进水深和流量；洼地填充类快速平衡模型只适合作为早期筛查或快速近似，不能代表本报告主结果。报告中纳入的正式峰值空间图和统计表均来自重新运行后的 ITZI 动力学结果。</p>
          <h3>关于边界条件</h3>
          <p>当前公开论文文字说明 reference 采用闭合边界和干初始床面；本地数据也没有提供河道水位过程、开边界线或 outfall 边界条件。因此本次 ITZI 正式对比采用闭合边界，以避免引入无法验证的外部排水条件。如果后续要模拟“流入边界河道”的开放边界，需要补充河道位置、边界水位/水深过程、与 MIKE 相同的边界设定和外排校准资料。</p>
          <h3>关于管网削峰合理性</h3>
          <p>20 m 概念性 sink 分支表现出明显体积削减和淹没格削减，说明道路对齐入流排水入口可以产生预期的削峰效应；但 Event 67 的全局最大水深削减接近 0，同时排水量显著，这是因为全局最大值出现在不受入口控制或入口控制较弱的局部极值位置，而整体体积和淹没范围仍被削减。此类现象不应简单判为统计错误，应同时查看体积、范围和空间分布。</p>
          <h3>关于 SWMM 与概念性 sink 的关系</h3>
          <p>概念性 sink 是地表模型内部的简化排水入口，计算稳定、易于做参数敏感性；SWMM 分支则包含管段、节点、汇水区和 dynamic-wave routing，更接近真实 1D 管网机制。现阶段两者是互补对照：sink 分支用于评估排水入口对 ITZI 水深的直接影响，SWMM 分支用于评估道路派生管网本身的水力承载与壅水行为。</p>
        </section>

        <section id="conclusion">
          <h2>主要结论</h2>
          <ol>
            <li>本地公开数据足以完成 20 m MIKE reference、ITZI 地表动力学和道路对齐概念性管网 sink 的八事件对比；真实 5 m MIKE reference 暂不可用。</li>
            <li>正式 20 m 结果中，ITZI 地表-only 峰值相对 MIKE 略高，加入概念性管网后峰值更接近 reference，并显著削减末时刻积水体积与淹没格数量。</li>
            <li>SWMM dynamic-wave 管网分支已完成八事件 standalone 运行，但当前出口与管径等参数仍为概念化设置，主要用于展示完整 1D 管网机制，不宜直接解读为真实城市管网能力。</li>
            <li>5 m 部分已完成 proxy reference 检查和 ITZI 5 m refined-grid 敏感性实验；由于缺失真实 5 m MIKE reference，不能宣称已完成真实 5 m MIKE-ITZI 验证。</li>
            <li>后续将管网因素加入 LarNO 训练时，建议把道路/管网密度、节点分布、入口服务面积、SWMM 壅水指标等作为静态或事件条件特征，而不仅仅输入单一 drain inlet mask。</li>
          </ol>
        </section>

        <section id="limits">
          <h2>不足与展望</h2>
          <ul>
            <li>LarNO 神经算子模型的完整重新训练或微调尚未在本次最终流水线中执行；本报告以 MIKE reference 数据和 ITZI/SWMM 物理分支为主，LarNO 预测文件或训练权重待补充。</li>
            <li>真实 5 m MIKE reference 文件缺失，5 m 结论仅限于 proxy 检查和 ITZI refined-grid 敏感性。</li>
            <li>SWMM 分支尚未与 ITZI 做双向耦合，检查井溢流回灌、地表-管网水位交换和边界河道出流仍待实现。</li>
            <li>管径、坡度、糙率、outfall 和节点服务面积目前仍为概念性工程假设，需要用规划管网、实测管线或 MIKE/SWMM 校准资料进一步约束。</li>
            <li>开放边界设置需要河道几何和水位边界资料；在资料缺失前，保持闭合边界更利于与公开 reference 的设定一致。</li>
          </ul>
          <div class="footer">本 HTML 为完全自包含单文件报告：全部图像以内嵌 Base64 存储，全部表格由当前 CSV/NPZ 结果转换为 HTML。若后续补充真实 5 m reference 或 LarNO prediction，可重新运行报告脚本更新。</div>
        </section>
      </main>
    </div>
    """

    return (
        "<!DOCTYPE html>\n"
        "<html lang=\"zh-CN\">\n"
        "<head>\n"
        "  <meta charset=\"UTF-8\">\n"
        "  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n"
        "  <title>LarNO-ITZI-SWMM 城市洪涝拓展研究报告</title>\n"
        f"  <style>{css}</style>\n"
        "</head>\n"
        f"<body>{body}</body>\n"
        "</html>\n"
    )


def main() -> int:
    html_text = build_report()
    REPORT.write_text(html_text, encoding="utf-8")
    print(f"Wrote standalone report: {REPORT}")
    print(f"Size: {REPORT.stat().st_size / 1024 / 1024:.2f} MiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
