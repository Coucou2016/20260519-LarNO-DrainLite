"""Editorial revision using frozen arrays; no retraining or simulation changes."""
from pathlib import Path
import csv
import json
import re
import shutil
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.text import Text
import generate_reviewer_v5_figures as figures
import generate_reviewer_v3_publication_figures as physics
import plot_reviewer_v3_physics as peak
from generate_reviewer_v5_documents import replace_figure, figure, table
from publication_plot_style import add_panel_labels

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'extended_study/output/reviewer_major_revision_v5/submission_package_v5'
OUT=ROOT/'extended_study/output/reviewer_major_revision_v6'
PACKAGE=OUT/'submission_package_v6'
FIG=PACKAGE/'figures'


def read(path):
    with path.open(encoding='utf-8-sig',newline='') as stream:
        return list(csv.DictReader(stream))


def save(fig,stem):
    for ext in ['png','pdf','svg']:
        fig.savefig(FIG/f'{stem}.{ext}',dpi=400)
    plt.close(fig)


def plot_labels(value):
    return (value.replace('SWMM numerical diagnostics','Network routing diagnostics')
            .replace('C: Itzï-SWMM','C: coupled model').replace('C: Itzi-SWMM','C: coupled model')
            .replace('Public LarNO','LarNO').replace('public LarNO','LarNO'))


def make_figures():
    for path in (SOURCE/'figures').iterdir():
        if not path.name.startswith(('fig07_','fig11_event_selection')):
            shutil.copy2(path,FIG/path.name)
    # Change label text at rendering only, preserving source plotting geometry/data.
    original=Figure.savefig
    def labelled_save(fig,*args,**kwargs):
        for artist in fig.findobj(match=Text):
            artist.set_text(plot_labels(artist.get_text()))
        return original(fig,*args,**kwargs)
    Figure.savefig=labelled_save
    try:
        physics.FIG=FIG
        physics.fig03_physical_quality()
        physics.fig04_hydrographs()
        tmp=OUT/'peak_render'
        tmp.mkdir(exist_ok=True)
        peak.peak_maps(physics.REV/'acceptance_event68_6h_full/event68',figures.GEO/'conceptual_network.inp',tmp)
        for ext in ['png','pdf','svg']:
            shutil.copy2(tmp/f'fig_physics_peak_maps.{ext}',FIG/f'fig05_event68_physical_maps.{ext}')
        figures.FIG=FIG
        figures.OUT=OUT
        figures.larno_extremes()
    finally:
        Figure.savefig=original
    compact_residuals()
    sensitivity_curves()


def compact_residuals():
    active=np.load(figures.GEO/'active_mask.npy').astype(bool)
    fig,axes=plt.subplots(4,4,figsize=(7.1,5.2),layout='compressed')
    for i,event in enumerate(['event1','event20','event68','event70']):
        b=np.load(figures.FLOOD/event/'h_itzi_surface_matched.npy',mmap_mode='r')
        c=np.load(figures.FLOOD/event/'h_itzi_swmm.npy',mmap_mode='r')
        pred=np.load(figures.old.EXP/'predictions'/event/'h_hybrid_all.npy',mmap_mode='r')
        t=int(np.argmax(c[:,active].sum(axis=1,dtype=np.float64)))
        for j,field in enumerate([b[t],(c[t]-b[t])*1000,(pred[t]-b[t])*1000,(pred[t]-c[t])*1000]):
            ax=axes[i,j]
            im=ax.imshow(np.ma.masked_where(~active,field),cmap='Blues' if j==0 else 'RdBu_r',vmin=0 if j==0 else -200,vmax=.5 if j==0 else 200)
            ax.set_xticks([]);ax.set_yticks([])
            if i==0:ax.set_title(['Input B','True C-B','Predicted C-B','Depth error'][j])
            if j==0:ax.set_ylabel(f'{event}\n{(t+1)/12:.2f} h',fontsize=7)
            if j in [0,3]:fig.colorbar(im,ax=ax,fraction=.045,pad=.015,label='m' if j==0 else 'mm',extend='max' if j==0 else 'both')
    add_panel_labels(axes.flat,x=-.04,y=1.02)
    save(fig,'fig06_fixed_network_residual_maps')


def sensitivity_curves():
    active=np.load(figures.GEO/'active_mask.npy').astype(bool)
    base=figures.old.V4/'physical_sensitivity_event68'
    rows=read(base/'physical_sensitivity_metrics.csv')
    series={}; evidence=[]
    for row in rows:
        case=row['case']
        if case=='baseline':
            bpath=figures.FLOOD/'event68/h_itzi_surface_matched.npy'
            cpath=figures.FLOOD/'event68/h_itzi_swmm.npy'
        else:
            bpath=base/case/'event68/h_B_surface_inlet_n012.npy'
            cpath=base/case/'event68/h_C_itzi_swmm.npy'
        b=np.load(bpath,mmap_mode='r');c=np.load(cpath,mmap_mode='r')
        effect=[];volume=[]
        for t in range(72):
            residual=c[t,active].astype(float)-b[t,active].astype(float)
            effect.append(np.mean(np.abs(residual))*1000)
            volume.append(-residual.sum()*400/1e6)
            evidence.append([case,t+1,(t+1)/12,effect[-1],volume[-1]])
        assert abs(np.mean(effect)-float(row['drainage_effect_mae_mm']))<.001
        assert abs(volume[-1]*1e6-float(row['final_surface_reduction_m3']))<1
        series[case]=(effect,volume)
    (OUT/'diagnostics').mkdir(exist_ok=True)
    with (OUT/'diagnostics/sensitivity_time_series.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.writer(stream);writer.writerow(['case','step','time_h','mean_absolute_residual_mm','surface_volume_difference_million_m3']);writer.writerows(evidence)
    fig,axes=plt.subplots(2,2,figsize=(7.1,4.8),sharex=True,constrained_layout=True)
    groups=[ [('baseline','Global redistribution','#222222','-'),('building_nearest','Nearest-cell redistribution','#0072b2','--'),('building_exclude','Exclude (reduced rainfall)','#d55e00',':')],
             [('baseline','Loss 1 mm/h; n=0.012','#222222','-'),('loss_0mmh','Loss 0 mm/h','#009e73','--'),('loss_2mmh','Loss 2 mm/h','#0072b2',':'),('inlet_manning_0p015','Loss 1 mm/h; n=0.015','#cc79a7','-.')]]
    time=np.arange(1,73)/12
    for col,group in enumerate(groups):
        for case,label,color,style in group:
            for row in range(2):axes[row,col].plot(time,series[case][row],label=label,color=color,linestyle=style,lw=1.3)
        axes[0,col].legend(fontsize=6.6,loc='upper left')
        axes[0,col].set_title(['Rainfall allocation','Effective loss and local roughness'][col])
    for row in range(2):
        limits=(0,max(max(series[k][row]) for k in series)*1.18)
        for col in range(2):
            axes[row,col].set_ylim(limits);axes[row,col].grid(alpha=.2)
    for ax in axes[1]:ax.set_xlabel('Time (h)')
    axes[0,0].set_ylabel('Mean |C-B| (mm)')
    axes[1,0].set_ylabel('Surface volume B-C ($10^6$ m$^3$)')
    add_panel_labels(axes.flat)
    save(fig,'fig13_physical_sensitivity')


def terminology(text):
    # Leave reference titles, software provenance and file paths in the audit intact.
    body,separator,refs=text.partition('## References')
    replacements={
      'public LarNO checkpoint':'pretrained LarNO model',
      'Public LarNO':'LarNO', 'public LarNO':'LarNO',
      'Independent public LarNO comparison':'Independent LarNO benchmark comparison',
      'upstream checkpoint audit':'independent benchmark comparison',
      'Itzï-SWMM':'1D/2D coupled hydrodynamic model',
      'ITZI-SWMM':'1D/2D coupled hydrodynamic model',
      'Itzi-SWMM':'1D/2D coupled hydrodynamic model',
      'target-side SWMM head':'target-side network head',
      'Absolute SWMM flow-routing':'Absolute network flow-routing',
      'SWMM flooding loss':'Unreturned network flooding loss',
      'water entering SWMM':'water entering the drainage network',
    }
    for source,target in replacements.items():body=body.replace(source,target)
    return body+separator+refs


def main():
    FIG.mkdir(parents=True,exist_ok=True)
    shutil.copytree(SOURCE.parent/'diagnostics',OUT/'diagnostics',dirs_exist_ok=True)
    if '--documents-only' not in sys.argv:
        make_figures()
    inventory=read(figures.old.EXP/'metrics/event_selection_inventory.csv')
    valid=[r for r in inventory if r['public_arrays_valid'].lower()=='true']
    event=next(r for r in valid if r['event']=='event68')
    rank=1+sum(float(r['mean_6h_rainfall_mm_active'])>float(event['mean_6h_rainfall_mm_active']) for r in valid)
    rationale=(f"Event 68 provides a diagnostic case with an active-cell mean six-hour rainfall of {float(event['mean_6h_rainfall_mm_active']):.2f} mm "
      f"(rank {rank} among the 17 readable events), an external-reference maximum depth of {float(event['mike_global_peak_m_active']):.3f} m, "
      "and a baseline mean absolute drainage response of 11.914 mm, above the eight-event mean of 9.266 mm. "
      "Its substantial drainage response makes changes in the physical assumptions readily measurable. "
      "These characteristics justify examining it as a high-response case, not treating it as representative of all storms. "
      "The sensitivity runs were available for this event only; a prospective event-selection rule was not recorded.")
    caption=("Time-dependent sensitivity of the paired hydrodynamic simulations for event68. "
      "Panels (a,b) show the active-cell mean absolute depth difference |C-B|; panels (c,d) show the concurrent difference in surface storage B-C, not cumulative pipe discharge. "
      "Left panels compare rainfall allocation; right panels compare effective loss and inlet-neighbourhood roughness. "
      "Exclusion removes rainfall volume as well as changing its allocation. The n=0.015 and baseline curves nearly overlap. "
      "All curves use the original 72-frame simulations, with matching vertical scales across each row; the residual estimator was not retrained.")
    for stem in ['manuscript','report']:
        text=(SOURCE/f'{stem}.md').read_text(encoding='utf-8')
        text=replace_figure(text,7,'')
        text=replace_figure(text,9,'')
        text=replace_figure(text,11,figure(11,'fig13_physical_sensitivity',caption))
        text=text.replace('Figure 7','Table 2').replace('Figure 9 and Appendix B','the event inventory in the evidence supplement')
        text=text.replace('Figure 9','the event inventory in the evidence supplement')
        text=text.replace('### 3.7 Relation to the public LarNO checkpoint','### 3.7 Benchmark reconstruction and the scope of drainage correction')
        text=text.replace('The public LarNO checkpoint was executed independently for event 68.',
          'The independent LarNO benchmark comparison assesses reconstruction of the released reference field, rather than emulation of the present conceptual network. For event68, the pretrained LarNO model was evaluated against its MIKE reference.')
        text=text.replace('DrainLite was not appended to the public checkpoint. That checkpoint was trained against MIKE fields generated by MIKE Plus 2023 with 1D-2D drainage coupling (Cao et al., 2026, Section 4.1); applying C-B to it could count drainage twice. A defensible end-to-end LarNO-DrainLite calculation requires a LarNO model trained to drainage-free targets or paired neural-operator targets under matched network interventions.',
          'The two comparisons concern different prediction targets. LarNO reconstructs MIKE reference fields that already include drainage coupling (Cao et al., 2026, Section 4.1), whereas DrainLite estimates the change introduced by the present conceptual network relative to its matched surface-only control. Their error magnitudes therefore cannot be interpreted as a direct ranking. Adding the present residual to LarNO predictions would risk counting drainage twice. A consistent integration requires a drainage-free upstream prediction or matched network-conditioned targets; it is not evaluated here.')
        text=text.replace('### 3.8 Paired physical sensitivities for event 68','### 3.8 Sensitivity to rainfall allocation and hydrological assumptions\n\n'+rationale)
        text=text.replace('A road-aligned conceptual network was constructed for the 20 m Shenzhen benchmark and coupled bidirectionally to the two-dimensional Itzï solver through the Storm Water Management Model.',
          'A road-aligned conceptual network was represented within a 1D/2D coupled hydrodynamic model for the 20 m Shenzhen benchmark.')
        text=text.replace('a bidirectionally coupled 1D-2D hydrodynamic configuration using Itzï and SWMM (C)','a 1D/2D coupled hydrodynamic configuration (C)')
        text=text.replace('Surface flow was solved with Itzï 25.4 using its damped partial-inertia formulation.',
          'The 1D/2D coupled hydrodynamic model represents surface flow with a damped partial-inertia formulation and network flow with dynamic-wave routing. Such integrated descriptions are established in urban flood modelling (Fan et al., 2017).')
        text=re.sub(r'The one-dimensional network was solved by SWMM .*?before opening the native engine\.',
          'Dynamic-wave routing used a 0.5 s maximum step, variable-step factor 0.2, 50 trials, slot surcharge representation and 0.0015 m head tolerance. Surface-network exchange was bidirectional, with relaxation and damping factors of 0.8 and 0.5. Separate node ponding was disabled to avoid duplicating surface storage. Exact software versions and interfaces are recorded in the evidence supplement.',text,flags=re.S)
        text=text.replace("activated Itzï's native DrainageSimulation interface to exchange water bidirectionally with SWMM",'activated bidirectional surface-network exchange')
        text=text.replace("A connected conceptual sewer network was coupled through Itzï's native SWMM interface",'A connected conceptual sewer network was represented in the 1D/2D coupled hydrodynamic model')
        text=text.replace('Itzï; Storm Water Management Model;','1D/2D coupled hydrodynamic model;')
        text=terminology(text)
        text=re.sub(r'Figure (\d+)',lambda m:'Figure '+str({8:7,10:8,11:9}.get(int(m[1]),int(m[1]))),text)
        if stem=='manuscript':
            text=text.replace('## References','## References\n\nFan, Y., Ao, T., Yu, H., Huang, G., Li, X., 2017. A Coupled 1D-2D Hydrodynamic Model for Urban Flood Inundation. Advances in Meteorology, 2819308. https://doi.org/10.1155/2017/2819308.\n')
            text=text.replace('The V5 figure and evidence revision uses','This V6 presentation revision retains the V5 evidence audit and uses')
            text=text.replace('V5 figure sources and input paths are recorded in the companion audit.','The V6 figure sources and editorial changes are recorded in the companion audit; the earlier numerical evidence remains unchanged.')
        else:
            # Supersede V5 figure-by-figure editorial trail, which would otherwise misnumber the new paper.
            text=text.split('## 本轮数据核查与图表解释')[0]
            text=text.split('## 附录：公共事件覆盖表')[0]
            text=text.replace('C 在 B 的基础上启用 Itzï 25.4 内置的 DrainageSimulation，并由 SWMM 5.2.4 动力波求解管网。','C 在 B 的基础上采用一维/二维耦合水动力模型，开启地表与管网之间的双向水量交换。')
            text=text.replace('SWMM 流量连续性','管网流量连续性').replace('目标侧 SWMM 动态状态','目标侧管网动态状态')
            text=text.replace('Unreturned network flooding loss、warning 和 error 均为零','未返回地表的管网溢流损失、运行警告和错误记录均为零')
            text+='\n\n## 本轮论文表达与图表调整\n\n正文和图例统一采用一维/二维耦合水动力模型（1D/2D coupled hydrodynamic model），简称耦合模型。具体软件与版本保留在补充材料，主文不再按调用接口拆解模型。LarNO 是模型名称，不再加 Public；其预训练权重来自原作者，来源与本项目方法明确区分。\n\n原图 7 与已有性能表重复，删除图而保留表 2 的均值、标准差和重复次数。原图 9 的 17 个事件转入补充材料清单，保留每个事件的降雨、参考最大水深和是否参与配对计算。正文因此保留 9 幅图、5 张表。图 6 使用相同数据和色标，仅压缩行间空白。\n\n### 敏感性过程线如何阅读\n\n最后一图的上排表示同一时刻有无管网的平均绝对水深差，下排表示两种模拟的地表储水量之差，不是累计排水量。左列比较降雨分配，右列比较损失率和入口附近糙率；每行共享纵轴范围。黑色曲线为基准，其他曲线何时偏离，表示相应假设何时开始影响响应。排除屏障区降雨减少总输入，不能把其较小响应只归因于空间分配。糙率试验与基准接近重合是原计算结果，不人为拉开显示。\n\n### 为什么分析 event68\n\n'+rationale+'\n\n该事件用于放大观察有明显排水响应情况下的配置差异，不意味着它代表全部降雨。未记录事先确定的选择规则，因此不把本次解释写成预注册选择。第 3.7 节从论文的比较目标出发解释 LarNO 与 DrainLite 的区别，不再围绕检查点操作展开。此前列出的物理验证缺口没有因这次排版修改而消失。\n'
        if stem=='report':
            text+='\n\n### 保留的数据口径与证据边界\n\n矩形计算区为 89.6 平方千米，活动区约 42.21 平方千米。原始降雨体积取矩形全域之和，而事件清单的平均降雨针对活动区；两者不能直接相乘。49.9 米地形阈值是屏障掩膜代理，不是独立核实的建筑清单。576 个时段的降雨分项核查及 30 分钟累计收支记录保留在证据补充材料中，后者不等于全部内部求解时间步的联合账本。\n\n管网没有非正坡度管段，但 626 根、约 27.5% 接近设计坡度下限。221 个合成出流口没有独立坐标记录，地图表示与之相连的末端节点，不是真实受纳河道口。网络增益约 0.047773 毫米，事件间标准差约 0.037333 毫米，不能与采样种子标准差约 0.001819 毫米混淆。正残差也不自动等于节点回灌，可能包含地表水重新分布。原图 5 的独立时刻峰值之差不应解释为瞬时交换；图 6 的色标显示截断不影响原数组统计。\n'
        (PACKAGE/f'{stem}.md').write_text(text,encoding='utf-8')
    supplement=(SOURCE/'evidence_supplement.md').read_text(encoding='utf-8')
    supplement+='\n\n## Implementation provenance\n\nThe 1D/2D coupled hydrodynamic model uses Itzi 25.4, SWMM 5.2.4 and PySWMM 2.1.0. These identify the numerical implementation and calling interface, not separate experimental models. No engine or coupling equation was changed in the editorial revision.\n\n## Complete readable-event inventory\n\n'
    supplement+=table(['Event','Paired A/B/C','Mean 6-h rain (mm)','MIKE maximum (m)'],[[r['event'],r['used_in_paired_v3'],f"{float(r['mean_6h_rainfall_mm_active']):.3f}",f"{float(r['mike_global_peak_m_active']):.3f}"] for r in valid])
    (PACKAGE/'evidence_supplement.md').write_text(supplement,encoding='utf-8')
    audit=(SOURCE/'scientific_integrity_audit.md').read_text(encoding='utf-8')
    audit+='\n\n## V6 editorial change record\n\nThe physical arrays and learned predictions are unchanged. Figure 2 is copied byte-for-byte; Figures 3-5 retain data and layouts with terminology-only label changes. Figure 6 retains its fields, time selection and colour bounds. The skill plot is removed because Table 2 already reports its results. The event scatter is replaced by a full supplementary inventory. Sensitivity time series are recomputed from the six existing paired simulations, and their temporal mean absolute residual and final-volume differences are checked against the original summary CSV (tolerances 0.001 mm and 1 m3). Figure labels use LarNO; implementation provenance remains in the supplement.\n\nTerminology source: Fan et al. (2017), https://doi.org/10.1155/2017/2819308. The established term coupled 1D-2D hydrodynamic model motivates the consistent 1D/2D wording; it does not imply independent validation of this implementation. Event68 rationale describes measured characteristics retrospectively, not a fabricated prospective selection rule. Pending physical experiments remain pending.\n'
    (PACKAGE/'scientific_integrity_audit.md').write_text(audit,encoding='utf-8')
    print(PACKAGE)


if __name__=='__main__':main()
