"""Build the reviewer-aligned manuscript, report and evidence review package."""
from __future__ import annotations
import json
import re
import shutil
import numpy as np
from generate_reviewer_v4_documents import ROOT, V4, EXP, read_csv, table, sha256
from audit_reviewer_v5_evidence import OUT

PACKAGE=OUT/"submission_package_v5"
SOURCE=V4/"submission_package_v4"
DIAG=OUT/"diagnostics"


def csv_table(name, columns=None):
    rows=read_csv(DIAG/name)
    columns=columns or list(rows[0])
    if len(columns)>7:
        keys=[k for k in ['event','scenario','model','field','subset'] if k in columns]
        values=[k for k in columns if k not in keys]
        return '\n\n'.join(table(keys+values[i:i+4],[[r.get(k,"") for k in keys+values[i:i+4]] for r in rows]) for i in range(0,len(values),4))
    return table(columns,[[r.get(k,"") for k in columns] for r in rows])


def replace_figure(text, number, replacement):
    pattern=rf"!\[Figure {number}\. .*?\]\(figures/[^\n]+\)\s*\n\s*\*\*Figure {number}\. .*?\*\*"
    text,count=re.subn(pattern,lambda _:replacement,text,flags=re.S)
    assert count==1,(number,count)
    return text


def figure(n,stem,caption):
    return f"![Figure {n}. {caption}](figures/{stem}.png)\n\n**Figure {n}. {caption}**"


def strip_table(text, number):
    return re.sub(rf"\*\*Table {number}\.[^\n]+\*\*\s*\n(?:\s*\n)?(?:\|[^\n]*\n)+","",text)


def renumber(text):
    mapping={1:1,2:2,3:3,4:4,5:5,6:6,7:7,9:8,11:9,12:10,13:11}
    text=text.replace("Figure 11d","Figure 11").replace("图 11d","图 11")
    for prefix in ["Figure ","图 "]:
        text=re.sub(re.escape(prefix)+r"(\d+)",lambda m:prefix+str(mapping.get(int(m[1]),int(m[1]))),text)
    counter=0
    def label(m):
        nonlocal counter
        counter+=1
        return f"**Table {counter}. {m[1]}**"
    return re.sub(r"\*\*Table \d+\. (.*?)\*\*",label,text)


def feature_dictionary():
    rows=[
      ["surface_depth_mm","mm","1000 B(t,x)","Contemporaneous surface-only field"],
      ["rainfall_mm","mm/5 min","Released rainfall at t,x; not redistributed forcing","Current interval"],
      ["cumulative_rainfall_mm","mm","Sum of released rainfall through t","Past and current rainfall"],
      ["dem_m","m","Released DEM at x","Static"],
      ["dem_slope","m/m","sqrt(gx^2+gy^2), numpy.gradient with 20 m spacing; wall gradients retained","Static"],
      ["time_norm","1","(t+1)/72 for zero-based index t","Simulation clock"],
      ["time_sin","1","sin(2 pi (t+1)/72)","Simulation clock"],
      ["time_cos","1","cos(2 pi (t+1)/72)","Simulation clock"],
      ["surface_depth_3x3_mean_mm","mm","1000 K3*(B M)/(K3*M); zero if denominator is zero","Current B"],
      ["surface_depth_7x7_mean_mm","mm","1000 K7*(B M)/(K7*M)","Current B"],
      ["drain_inlet_mask","0/1","One in active cells receiving a junction","Static"],
      ["pipe_mask","0/1","One in active rasterised conduit cells","Static"],
      ["drain_inlet_count","count","Number of junctions per active cell","Static"],
      ["pipe_segment_count","count","Number of rasterised conduits visiting cell","Static"],
      ["network_degree","count","Junction in-degree + out-degree; last junction wins if collocated","Static"],
      ["pipe_density_3x3","fraction","Uniform 3x3 pipe-mask mean; zero padding","Static"],
      ["pipe_density_7x7","fraction","Uniform 7x7 pipe-mask mean; zero padding","Static"],
      ["inlet_density_7x7","fraction","Uniform 7x7 inlet-mask mean; zero padding","Static"],
      ["pipe_diameter","m","Maximum conduit diameter in cell","Static"],
      ["pipe_slope","m/m","Arithmetic mean of slopes of visiting conduits, including offsets","Static"],
      ["pipe_capacity","m3/s","Sum of Manning circular full-flow capacities: A(D/4)^(2/3) sqrt(S)/n","Static descriptor, not simulated flow"],
      ["pipe_cover_depth","m","Mean endpoint cover for two-junction segments; node cover overwrites at junction cells","Static"],
      ["capacity_density_7x7","m3/s","Uniform 7x7 mean of summed capacity; zero padding","Static descriptor"],
    ]
    return table(["Feature","Unit","Calculation","Availability"],rows)+"\n\nConduits are sampled between endpoint grid indices using max(abs(delta row), abs(delta col))+1 linearly spaced points, rounded to nearest integer. Only in-bounds active cells are retained. Empty network cells are zero. Network inputs use numpy.nan_to_num with NaN mapped to zero; finite-array checks guard the published input. The surface mean uses mask normalisation, whereas static densities deliberately divide by the full kernel area. These descriptors are spatial summaries, not an alternative pipe solver. The current 2,276 coupled junctions occupy 2,276 distinct cells. The fitted prior is an additional predictor in mm, constructed from other events' C-B at the same cell and time."


def diagnostics_tables(summary):
    controls=read_csv(EXP/"metrics/final_hybrid_network_controls.csv")
    groups={}
    for row in controls:
        name=row["control"]
        key="_".join(name.split("_")[:2]) if name.startswith("shift_") else name
        groups.setdefault(key,[]).append(float(row["delta_mae_vs_aligned_mm"]))
    control_table=table(["Inference-only perturbation","Runs","Mean penalty (mm)","SD across runs (mm)"],
        [[k,len(v),f"{np.mean(v):.4f}",f"{np.std(v,ddof=1):.4f}"] for k,v in groups.items()])
    runtime=read_csv(EXP/"metrics/runtime_uncached_canonical.csv")
    clip=read_csv(EXP/"metrics/clipping_sensitivity.csv")
    timings=table(["Quantity","Value","Scope"],[
        ["Feature / assembly / estimator / postprocess (s)"," / ".join(f"{np.mean([float(r[k]) for r in runtime]):.2f}" for k in ["feature_seconds","assembly_seconds","model_seconds","postprocess_seconds"]),"Canonical seed 1907; mean of eight events"],
        ["B / correction / B+correction / C (s)"," / ".join(f"{np.mean([float(r[k]) for r in runtime]):.2f}" for k in ["surface_runtime_s","total_correction_seconds","surface_plus_correction_seconds","coupled_runtime_s"]),"Historical physical runs + in-memory correction; excludes I/O"],
        ["Mean event speed ratio; ratio of means",f"{summary['speed_mean_ratios']:.3f}; {summary['speed_ratio_means']:.3f}",f"Event-ratio SD {summary['speed_sd_ratios']:.3f}; not integrated deployment timing"],
        *[[f"Residual bound {bound}: MAE / depth-clamped cells",f"{np.mean([float(r['mae_mm']) for r in clip if r['clip_mm']==bound]):.4f} mm / {np.mean([float(r['depth_clamped_pct']) for r in clip if r['clip_mm']==bound]):.3f}%","Canonical seed only; fraction over active cell-times"] for bound in ["none","500.0","1200.0"]]
    ])
    return control_table,timings


def main():
    PACKAGE.mkdir(parents=True,exist_ok=True)
    summary=json.loads((DIAG/"summary.json").read_text())
    extremes=json.loads((DIAG/"larno_extremes.json").read_text())
    control_table,timings=diagnostics_tables(summary)
    original=(SOURCE/"manuscript.md").read_text(encoding="utf-8")
    text=original.split("## Appendix A.")[0]
    text=text.replace("Large-scale Latent Autoregressive Neural Operator (LarNO)","latent autoregressive neural operator (LarNO)")
    text=text.replace("met the predefined topology", "met the repository's topology")
    text=text.replace("reports routing continuity and non-converging steps against the stated numerical criteria", "reports routing continuity and non-converging steps; the 2% acceptance limits are specified in Section 2.4 rather than drawn on this magnified axis")
    text=text.replace("High building cells remain as hydraulic barriers but are excluded from active-cell performance statistics.",
        "Cells at DEM elevations of at least 49.9 m form the numerical barrier mask and are excluded from active-cell statistics. This threshold is a modelling proxy, not an independently verified building inventory; it may also include elevated terrain or domain-exterior cells.")
    text=text.replace("Building cells were elevated barriers. Rain falling on those cells was globally redistributed over active cells to conserve rainfall volume.",
        "Cells in the threshold-derived barrier mask were set to 50 m. Their rainfall was redistributed uniformly to active cells. This operation conserves the rectangular-domain rainfall total but changes its spatial allocation and cannot by itself identify real roof runoff.")
    text=text.replace("fields that may already contain drainage effects", "fields generated by MIKE Plus 2023 with 1D-2D drainage coupling (Cao et al., 2026, Section 4.1)")
    text=text.replace("coupling removes a more plausible aggregate water volume", "coupling brings aggregate water volume closer to this external simulation")
    text=text.replace("Better final-volume agreement under C suggests that surface-only storage is excessive", "Better final-volume agreement under C indicates lower storage discrepancy relative to MIKE")
    text=text.replace("local surcharge patches", "positive-residual patches")
    text=text.replace("native Itzï-SWMM coupling (C)", "a bidirectionally coupled 1D-2D hydrodynamic configuration using Itzï and SWMM (C)")
    text=text.replace("The authors declare no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.","The declaration of competing interests requires confirmation by all authors before submission.")
    text=text.replace("Their penalties show that correctly aligned fields are preferable to displaced or destroyed fields.","These tests expose sensitivity to inference-time distribution shifts; their penalties are not estimates of the benefit of adding network predictors during training.")
    text=text.replace("Final-model displacement and destructive controls confirm that this small increment depends on correct network alignment", "Final-model perturbations reveal sensitivity to input alignment, while retrained ablations quantify the smaller incremental predictive contribution")
    text=text.replace("### 3.6 End-to-end timing and post-processing","### 3.6 Recorded computation time and post-processing")
    oldtiming="The observed B-plus-DrainLite workflow required 297.7 s, compared with 3817.6 s for C, giving a mean speed ratio of 13.1. This is a workflow measurement on one workstation rather than a hardware-normalised benchmark."
    text=text.replace(oldtiming,f"Adding recorded B time to in-memory correction time gives 297.7 s, compared with 3817.6 s for C. The ratio of these means is {summary['speed_ratio_means']:.2f}; the mean of eight event-wise ratios is {summary['speed_mean_ratios']:.2f} (event SD {summary['speed_sd_ratios']:.2f}). Neither number includes disk I/O, prior preparation or a newly timed integrated run.")
    text=replace_figure(text,1,figure(1,"fig01_workflow","Residual learning with an offline archive. The prior requires C-B labels from other historical events at each cell and simulation time. Both the outer test event and each fitting row's own event are excluded from its prior. Current B, rainfall, terrain and fixed network fields enter the estimator; current-event C is used only for held-event scoring."))
    text=replace_figure(text,2,figure(2,"fig02_network_audit","Conceptual-network maps. (a) Terrain, active-mask edge, selected link directions and outfall-connected terminal junctions; these are not surveyed river mouths or actual outfall coordinates. (b) Conduit diameter. (c) Junction degree. OpenStreetMap contributors provided the road geometry. Numeric slope and cover diagnostics are retained in the evidence supplement."))
    text=replace_figure(text,5,figure(5,"fig05_event68_physical_maps","Cell-wise temporal maxima for event68. Panels (a-d) show max_t(MIKE), max_t(A), max_t(B) and max_t(C). Panel (e) is max_t(C)-max_t(B); panel (f) is max_t(B)-max_t(A). These differences do not measure instantaneous exchange or max_t(C-B). Supplementary Figure S1 compares the definitions at a common time. Display saturation is separate from untruncated statistics."))
    text=replace_figure(text,6,figure(6,"fig06_fixed_network_residual_maps","Held-event fields at maximum C surface volume; each row labels its exact time. All rows share fixed ranges: B depth 0-0.5 m and signed residual/error -200 to 200 mm. Saturation fractions and extrema are reported in the evidence supplement; statistics use untruncated arrays."))
    text=replace_figure(text,7,figure(7,"fig07_skill_and_sampling_seeds","Performance by predictor group. Circles and error bars show five-sampling-seed means and SD; diamonds denote a single fitted model or a deterministic baseline and have no estimated repeat uncertainty. Panels show MAE, RMSE and CSI. Single-seed subgroup estimates should not be ranked as five-seed averages."))
    text=replace_figure(text,8,"**Table 90. Inference-time perturbations of the fixed fitted hybrid; not retraining gains.**\n\n"+control_table)
    text=replace_figure(text,10,"")
    text=replace_figure(text,11,"**Table 91. Runtime scope and post-processing, canonical seed 1907.**\n\n"+timings+"\n\n"+figure(11,"fig11_event_selection","Event inclusion within the 17 locally readable rainfall/reference cases. The eight paired events tend to have higher rainfall; event77 has the largest reference maximum despite not being paired. This local subset is not the entire 80-event public benchmark."))
    extreme_caption=(f"Independent public LarNO comparison for event68 at {extremes['time_h']:.2f} h (frame {extremes['frame_1based']}, the MIKE global-depth peak). The top row shows MIKE, LarNO and signed error with ranges reaching the frame extrema, without high-end saturation. The lower maps isolate cells exceeding 0.5 m in either field; the last panel compares domain-maximum depth through time. Event-wide maxima are {extremes['all_grid_reference_global_max_m']:.3f} m and {extremes['all_grid_prediction_global_max_m']:.3f} m. At the displayed time, {extremes['raw_negative_frame_pct']:.2f}% of all-grid LarNO predictions are negative; these are zeroed only for the depth display, not for the signed-error statistics. This is an upstream checkpoint audit, not a DrainLite result.")
    text=replace_figure(text,12,figure(12,"fig12_larno_reproduction",extreme_caption))
    text=replace_figure(text,13,figure(13,"fig13_physical_sensitivity","Event68 physical settings shown as a ratio heatmap. Each cell prints the original measured quantity and its ratio to baseline. Colour compares each column with its own baseline, not different physical units. Exclude changes rainfall volume as well as allocation. No surrogate was retrained under these alternative settings."))
    # Keep the main paper focused; detailed per-event and conditional tables remain in the supplement.
    for number in [3,4,6]:text=strip_table(text,number)
    text=text.replace("as shown by the canonical-seed values below.","as shown by the event-by-seed matrix in the evidence supplement.")
    rain=read_csv(DIAG/"rainfall_event.csv")
    e65=next(r for r in rain if r["event"]=="event65")
    rainfall_para=f"""\n\nRainfall volume was recomputed from every raw five-minute field. Table 1 uses the full rectangular sum, whereas the event-inclusion plot uses the mean over active cells. These are different spatial supports. For event65, the active-cell mean is {float(e65['active_mean_total_mm']):.5f} mm and the rectangle mean is {float(e65['rectangle_mean_total_mm']):.5f} mm; the rectangular total is {float(e65['raw_m3']):.1f} m3. Multiplying the active mean by 89.6 km2 is therefore invalid. The archived 576-step audit separates active-cell rainfall, reassigned barrier-cell rainfall and reconstructed model injection. Surveyed building and exterior areas cannot be separated with this threshold mask.\n\n"""
    text=text.replace("### 2.2 Construction",rainfall_para+"### 2.2 Construction")
    net=summary['network']
    network_para=f"""\n\nRe-parsing all conduit inverts and offsets found {net['zero_or_negative_slopes']} zero or negative slopes. There are {net['near_design_floor_count']} conduits ({100*net['near_design_floor_count']/net['conduits']:.2f}%) within 0.0005 +/- 0.000002 m/m. This geometric count identifies concentration near the design floor; individual pre-adjustment slopes were not retained. None of the 221 synthetic NORMAL outfalls has its own coordinate record. Figure 2 marks their connecting terminal junctions instead. NORMAL is a normal-depth boundary, not an observed river or tidal hydrograph. The 0.8-1.2 m diameter range defines a controlled conceptual system, without a claim of survey-based calibration.\n\n"""
    text=text.replace("### 2.3 Matched",network_para+"### 2.3 Matched")
    uncertainty=f"""\n\nThe mean network increment is {summary['network_gain_mean_mm']:.6f} mm. Its SD across event-specific seed means is {summary['event_gain_sd_mm']:.6f} mm, compared with {summary['seed_macro_gain_sd_mm']:.6f} mm across seed-specific event means. An event-unit percentile bootstrap gives [{summary['event_bootstrap_95pct_mm'][0]:.6f}, {summary['event_bootstrap_95pct_mm'][1]:.6f}] mm (20,000 resamples). This interval assumes exchangeable events; observation dates and storm-family membership have not been verified. It does not establish an engineering-significant gain or generalisation beyond this rainfall distribution.\n\n"""
    text=text.replace("### 3.4 Improvements",uncertainty+"### 3.4 Improvements")
    text=text.replace("## 4. Discussion",f"""## 4. Discussion

### Physical evidence and unresolved uncertainty

The numerical label, the surrogate approximation and the incremental network predictors answer different questions. Small closure errors support internal accounting; they do not validate the conceptual network against actual drainage observations. Recorded exchanges are cumulative at 30-minute output times. The sign audit distinguishes water entering SWMM from return flow, but the available archives do not contain every routing-step storage and outfall term. A complete time-resolved joint ledger and a routed single-inlet/single-pipe benchmark remain required. The supplementary native-function check uses a prescribed-head node test double and does not substitute for that benchmark.

The selected events have mean active-cell rainfall {summary['selected_rain_mm']:.2f} mm, versus {summary['other_readable_rain_mm']:.2f} mm for the nine other readable local cases. Their missing paired labels reflect work not yet performed, rather than an established inability to simulate them. Event numbering is not evidence of independent weather. Validation is consequently conditional on these eight events.

Positive C-B is a difference between two evolving surface solutions. It can reflect redistribution as well as actual node return flow, so it must not be labelled surcharge without a local exchange record. The full model barely changes the positive-residual error relative to the dynamic hybrid. Static fields and contemporaneous B do not uniquely specify the internal network state. Non-negative depth also does not enforce water conservation; volume-error trajectories in the supplement quantify the remaining discrepancy.

The spatiotemporal prior encodes fixed location and time even though explicit coordinates are absent. Cross-configuration transfer, a coarser prior, shifted rainfall timing, multi-event roof-runoff alternatives and pipe-capacity/tailwater perturbations have not been tested here. These experiments are needed before claims of physical robustness or network intervention response.
""")
    text=text.replace("## Appendix", "## Appendix")
    text += "\n\n## Reproducibility record\n\nThe V5 figure and evidence revision uses frozen V3 physical arrays and V4 seed-1907/full-five-seed outputs from source commit `"+summary["source_commit"]+"`. The V3 tuned/clipped estimator is historical, not the current primary model. Current fitting uses `run_final_hybrid_major_revision.py` with fixed 140 iterations and no residual clipping. `audit_reviewer_v5_evidence.py` recomputes rainfall, signed residual, volume and event-seed diagnostics. V5 figure sources and input paths are recorded in the companion audit. No physical array was replaced for this revision.\n"
    text=renumber(text)
    (PACKAGE/"manuscript.md").write_text(text,encoding="utf-8")

    # A separate detailed report retains the methodological trail without turning the paper into a response letter.
    report=(SOURCE/"report.md").read_text(encoding="utf-8").split("## Supplement.")[0]
    for n in [1,2,5,6,7,12,13]:
        match=re.search(rf"!\[Figure { {1:1,2:2,5:5,6:6,7:7,12:10,13:11}[n]}\. .*?\]\(figures/[^\n]+\)\s*\n\s*\*\*Figure .*?\*\*",text,re.S)
        replacement=match[0] if match else ""
        # Convert back to original numbering before the common final renumber pass.
        if n==13: replacement=replacement.replace("Figure 11.","Figure 13.")
        if n==12: replacement=replacement.replace("Figure 10.","Figure 12.")
        report=replace_figure(report,n,replacement)
    report=replace_figure(report,8,"**Table 90. 固定模型推理期扰动，不能等同于重训增益。**\n\n"+control_table)
    report=replace_figure(report,10,"")
    report=replace_figure(report,11,"**Table 91. 历史计算记录与内存修正耗时；规范种子 1907。**\n\n"+timings+"\n\n"+figure(11,"fig11_event_selection","八场配对事件与全部本地可读事件的降雨及外部参照严重度。"))
    for n in [3,4,6]:report=strip_table(report,n)
    for n in [1,2,5,6,7,8,10,11,12,13]:
        report=re.sub(rf"\*\*图 {n} 逐项解释。\*\*[^\n]*", "",report)
    report=report.replace("建筑物面积", "阈值屏障掩膜面积").replace("端到端", "组合计算")
    report=report.replace("reports routing continuity and non-converging steps against the stated numerical criteria", "reports routing continuity and non-converging steps; the 2% acceptance limits are specified in the methods rather than drawn on this magnified axis")
    report += f"""

## 本轮数据核查与图表解释

### 研究对象和物理模型

本研究的物理标签由本项目的双向耦合一维管网—二维地表水动力配置计算。英文统一采用 “bidirectionally coupled 1D-2D hydrodynamic model”，首次出现时明确二维部分为 Itzï、管网部分为 SWMM。PySWMM 是调用接口，不是第三套物理模型。MIKE 参照来自原论文第 4.1 节说明的 MIKE Plus 2023 一维—二维耦合模拟，已经包含排水影响；其具体管网未公开，不能作为无管网场重复叠加本项目残差。

### 地形、降雨和管网的真实含义

计算矩形为 89.6 平方千米，活动单元约 42.21 平方千米。程序以高程 49.9 米阈值区分屏障，这不能独立证明其余单元全是建筑物。降雨总体积取全部矩形单元的原始降雨之和，事件清单的平均降雨却只取活动单元，两者不能直接相乘。event65 的两种平均值分别为 {float(e65['rectangle_mean_total_mm']):.5f} 和 {float(e65['active_mean_total_mm']):.5f} 毫米。这解释了审稿时由面积乘均值产生的差别。576 行逐时段审计已保留原始输入、活动区直接降雨、屏障区转移降雨和按程序重建的注入量；重建不是新增的运行时流量计。

图 2 现在用三个地图分开表达地形、管径和节点连接度，避免两种色标叠加。三角形标的是连接合成出流口的末端节点，不是已知河道排放口。全部 {net['conduits']} 根管段均为正坡，{net['near_design_floor_count']} 根接近设计下限，占 {net['near_design_floor_count']/net['conduits']*100:.2f}%。这项统计不隐藏，但移到审查材料；它只能说明几何分布，不能证明工程参数已校准。

### 峰值地图与瞬时差值

图 5 的差值必须读作两个独立时间最大值之差 max(C)-max(B)。两边峰值可能发生于不同时间，不能解释成当时从路面流入管道的水量。补图 S1 将其与同一时刻 C(t)-B(t)、max(C-B) 并列。训练使用第二种逐时残差。图 6 改用跨事件固定色标，首列 0 至 0.5 米，其余列正负 200 毫米，行标签给出时刻。超过色标的部分只在显示上饱和，没有进入统计截断；截断比例与 50、100、200 毫米阈值计数见审查材料。

### 管网增益与模型不确定性

五种采样种子的平均增益为 {summary['network_gain_mean_mm']:.6f} 毫米，但事件间标准差为 {summary['event_gain_sd_mm']:.6f} 毫米，明显大于采样种子的 {summary['seed_macro_gain_sd_mm']:.6f} 毫米。事件自助重采样区间为 {summary['event_bootstrap_95pct_mm'][0]:.6f} 至 {summary['event_bootstrap_95pct_mm'][1]:.6f} 毫米，前提是假设八个事件可交换；当前没有气象日期资料来证明独立性。推理时打乱或清零会制造训练分布以外的输入，其惩罚不能当成重新训练后管网特征的收益。原图 8 已移为表，原图 10 与已有 MIKE 表合并，避免重复展示。

图 7 改用点和误差棒。圆点代表五种种子重复，菱形代表单次训练或确定性基线，单次结果没有“零标准差”的含义。图 9 的条件误差要结合补充材料中的单元时间数量和占比阅读。正残差并不自动等于节点回流，正文不再据此声称改善真实回流预测。

### 耗时、截断和物理敏感性

原图 11 的前三个子图已合入一张紧凑表，只保留事件覆盖散点图。八个速度比先求比再平均为 {summary['speed_mean_ratios']:.3f}，两个均值相除为 {summary['speed_ratio_means']:.3f}，是不同统计量。两者都不包括磁盘读写、离线先验生成和真实串联调用。规范种子 MAE 2.179 毫米与五种种子均值 2.169 毫米分别标识，不再混读。非负处理比例也列入同一表。

最后的敏感性图改为有数值标注的热图，每列以自己的基准归一化。颜色比较相对变化，单元格给出原值。排除屏障区降雨会减少总输入，专门在行名提示，不将其效果解释为空间重分配。此处只有 event68 的物理试验，没有多事件交叉情景训练。误差下降、数值连续性和物理真实性是三件不同的事。

### 尚未完成的证据

完整内部时间步的联合水量账本、单入口—单管道—单出流口的实际路由验证、多事件建筑降雨方案、管径和尾水敏感性、粗尺度先验及雨峰平移试验仍待完成。已有 30 分钟累计记录可核查地表收支和交换积分差，不能伪装成每个 SWMM 内部时间步的联合验证。作者身份、单位、资助和第三方再分发授权必须由作者确认，不能自动生成。
"""
    report=renumber(report)
    (PACKAGE/"report.md").write_text(report,encoding="utf-8")
    supplement="# DrainLite V5 evidence supplement\n\n## Feature dictionary\n\n"+feature_dictionary()
    supplement+="\n\n## Rainfall reconciliation\n\n"+csv_table("rainfall_event.csv")
    ledger=read_csv(DIAG/"water_ledger_saved_times.csv")
    final=[r for r in ledger if float(r['time_h'])==6.0]
    supplement+="\n\n## Saved-time surface and exchange ledger\n\nThe archive contains cumulative values at 30-minute intervals, not every solver step. The table below gives six-hour totals. Positive forward exchange enters SWMM; positive return exchange leaves SWMM. Surface closure and the difference between independently accumulated exchange integrals must be read separately. No pipe-storage or terminal-discharge time series is invented. The complete saved-time ledger is water_ledger_saved_times.csv.\n\n"
    for cols in [['event','scenario','rain_m3','loss_m3','surface_m3','surface_closure_m3'],['event','scenario','surface_net_exchange_m3','swmm_forward_exchange_m3','swmm_return_exchange_m3','exchange_integral_mismatch_m3']]:
        supplement+=table(cols,[[r[k] for k in cols] for r in final])+"\n\n"
    supplement+="\n\n## Across-event and across-seed network gains\n\n"+csv_table("event_seed_gain.csv")
    supplement+="\n\n## Signed residual errors\n\nPositive residual is not an independently identified surcharge flow. All counts are cell-times, not unique land area.\n\n"+csv_table("signed_residual_diagnostics.csv")
    supplement+="\n\n## Volume trajectories and timing\n\nFinal-hour slope is a descriptive slope, not a calibrated recession constant. Rectangular five-minute integration is used for absolute volume-error integrals.\n\n"+csv_table("hydrograph_diagnostics.csv")
    supplement+="\n\n## Display bounds and tails\n\nThreshold columns apply to the named field; only rows named error describe prediction-error tails.\n\n"+csv_table("map_display_audit.csv")
    cond=read_csv(EXP/"metrics/conditional_metrics_canonical_seed.csv")
    cond=[r for r in cond if r["model"]=="hybrid_all"]
    supplement+="\n\n## Conditional subset support\n\n"+table(["Event","Subset","Cell-times","Active %","MAE mm"],[[r['event'],r['condition'],r['n_cell_times'],r['proportion_active_pct'],r['mae_mm']] for r in cond])
    supplement+="\n\n![Supplementary Figure S1. Distinct definitions of peak and instantaneous residuals.](figures/supp_peak_definitions.png)\n\n**Supplementary Figure S1. Only the middle panel is an instantaneous C-B field. All use fixed +/-0.2 m display limits; extreme values remain in statistics.**\n"
    supplement+="\n\n![Supplementary Figure S2. Common-time MIKE inundation comparison.](figures/supp_mike_extent.png)\n\n**Supplementary Figure S2. Event68 at the MIKE surface-volume peak. Green is shared inundation, orange is missed relative to MIKE, blue is false alarm and pale grey is jointly dry. These are disagreements with a model, not observed flooding errors.**\n"
    (PACKAGE/"evidence_supplement.md").write_text(supplement,encoding="utf-8")
    pending=[
      ["Rainfall mask and volume","Recomputed","576 intervals; verified numerical redistribution, not surveyed roof mask"],
      ["Peak-map meaning","Corrected","max(C)-max(B) explicitly separated from C(t)-B(t)"],
      ["Slope zeros and outfalls","Recomputed","0 nonpositive; 626 near floor; outfall coordinates absent"],
      ["Event and seed uncertainty","Recomputed","40 event-seed gains; event bootstrap conditional on exchangeability"],
      ["Inference perturbations","Reframed","OOD sensitivity separated from retrained feature benefit"],
      ["Native exchange formula/sign","Unit check only","Prescribed-head test double; no routed pipe benchmark"],
      ["Full joint internal-step water ledger","Pending","30-minute cumulative records and final report only"],
      ["Broader paired events / weather independence","Pending","Nine readable cases unpaired; dates/families unverified"],
      ["Multi-event rainfall treatment / network capacities / tailwater","Pending","Only existing single-event setting sensitivities completed"],
      ["Coarse prior / shifted rain / cross-scenario retraining","Pending","No new predictive claims"],
      ["Unclamped depth error changes and extreme LarNO tails","Partially addressed","LarNO full-range maps and extremes rebuilt; clamp counts retained; full raw-depth change diagnostics still need inference replay"],
      ["Authors and redistribution rights","Author action","Do not invent author declarations or upstream permission"],
    ]
    audit=f"""# V5 scientific integrity and reviewer implementation audit

## Evidence boundary

Frozen source commit: `{summary['source_commit']}`. Current physical arrays, held-event predictions and training labels were not altered. MIKE rainfall/terrain/reference arrays and the public LarNO checkpoint are upstream materials, clearly attributed; A/B/C coupled simulations, fitted residual estimators and their diagnostics are this project's calculations. The manuscript does not claim all data are original observations.

## Verified findings

The raw rainfall sum and active-cell mean use different masks. Geometric slope inspection finds no zero/negative conduit slope and 626 values near the design floor. No outfall has its own coordinate record. Spatial maps mark connecting terminals only. Peak-map differences are not instantaneous residuals. The held-event network increment is 0.047773 mm, event SD 0.037333 mm and sampling-seed macro SD 0.001819 mm. All are computed, not borrowed from the LarNO paper.

## Implementation status

"""+table(["Review item","Status","Evidence or limitation"],pending)+"\n\n## Exact numerical evidence\n\n"
    audit+=table(["File","SHA256"],[[p.relative_to(ROOT).as_posix(),sha256(p)] for p in sorted(DIAG.glob('*')) if p.is_file()])
    audit+="""

## Reproduction commands

```bash
python extended_study/audit_reviewer_v5_evidence.py
python extended_study/test_v5_native_exchange.py
python extended_study/generate_reviewer_v5_figures.py
python extended_study/generate_reviewer_v5_documents.py
python extended_study/render_reviewer_v5_documents.py
```

Primary training remains `extended_study/run_final_hybrid_major_revision.py` (fixed 140 iterations, no residual clipping). Historical V3 tuning is not a reproduction command for the V4/V5 primary results. Figures 1/2/6/7/9/10/11 and supplementary maps are rebuilt by the V5 figure script; unchanged physical-quality, hydrograph and peak-map illustrations originate from the frozen V4 package. Figure 5's data formula is audited in `plot_reviewer_v3_physics.py`. Full CSV evidence remains alongside concise manuscript tables; removing duplicate plots does not remove inconvenient results.

## Writing sources and source verification

The revision follows concrete, evidence-led prose and distinguishes methods, results and interpretation. It does not manufacture an author voice, field observations or stronger conclusions to evade AI detection. Sources consulted: Manchester Academic Phrasebank (https://www.phrasebank.manchester.ac.uk/), KKKKhazix/human-writing (https://github.com/KKKKhazix/human-writing), petergyang/no-ai-slop (https://github.com/petergyang/no-ai-slop). These are editorial guidance, not scientific evidence.

MIKE's coupled origin is established in Section 4.1 of the supplied Cao et al. paper, DOI 10.1016/j.jhydrol.2026.135686. The current public model card documents 13 released input channels and the 64/16 Futian split (https://github.com/holmescao/LarNO/blob/main/huggingface/model_card.md). Current Itzi stable documentation resolves to 26.6, while the frozen calculations use 25.4; the current documentation version must not be represented as the executed engine. Normal-depth outfalls are conceptual hydraulic boundary conditions, not measurements of receiving-water levels.

## Submission readiness

This revision improves presentation and evidence transparency. It does not close the pending physical validation and scenario-generalisation requests. Numerical consistency checks alone are not grounds to declare the study calibrated or ready for unconditional submission.
"""
    (PACKAGE/"scientific_integrity_audit.md").write_text(audit,encoding="utf-8")
    print(PACKAGE)


if __name__=="__main__":main()
