"""Audit original routing reports and distinguish global from local diagnostics."""
from pathlib import Path
import csv
import hashlib
import json
import re
import shutil
import numpy as np
import matplotlib.pyplot as plt
from publication_plot_style import configure_publication_style,add_panel_labels
from generate_reviewer_v5_documents import replace_figure,figure,table

ROOT=Path(__file__).resolve().parents[1]
PHYS=ROOT/'extended_study/output/reviewer_major_revision_v3/formal_matched_full'
SOURCE=ROOT/'extended_study/output/reviewer_major_revision_v6/submission_package_v6'
OUT=ROOT/'extended_study/output/reviewer_major_revision_v7'
PACKAGE=OUT/'submission_package_v7'
MANUAL='https://www.epa.gov/system/files/documents/2022-04/swmm-users-manual-version-5.2.pdf'


def read(path):
    with path.open(encoding='utf-8-sig',newline='') as stream:return list(csv.DictReader(stream))


def write_csv(path,rows):
    with path.open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def audit_reports():
    rows=read(PHYS/'physics_quality.csv'); findings=[];nodes=[]
    for row in rows:
        path=PHYS/row['event']/'swmm_C.rpt'
        text=path.read_text(encoding='utf-8',errors='replace')
        global_error=float(re.search(r'Continuity Error \(%\)\s*\.+\s*([-\d.]+)',text)[1])
        nonconv=float(re.search(r'% of Steps Not Converging\s*:\s*([\d.]+)',text)[1])
        assert global_error==float(row['swmm_flow_routing_continuity_error_pct'])
        assert nonconv==float(row['swmm_steps_not_converging_pct'])
        section=text.split('Highest Continuity Errors')[1].split('Time-Step Critical Elements')[0]
        top=re.findall(r'Node\s+(\S+)\s+\(([-\d.]+)%\)',section)
        inflows=text.split('Node Inflow Summary')[1].split('Node Surcharge Summary')[0]
        lookup={}
        for line in inflows.splitlines():
            tokens=line.split()
            if len(tokens)>=9 and tokens[1] in ['JUNCTION','OUTFALL','STORAGE','DIVIDER']:
                lookup[tokens[0]]=tokens
        for node,pct in top:
            token=lookup[node]
            i_m3=float(token[7])*1000
            error=float(token[8])
            # stats.c uses inflow denominator; statsrpt.c uses outflow denominator.
            converted=100*error/(100+error)
            assert abs(converted-float(pct))<.02
            nodes.append(dict(event=row['event'],node=node,ranking_error_pct_inflow=float(pct),
              inflow_table_error_pct_outflow=error,reported_total_inflow_m3=i_m3,
              estimated_imbalance_m3=i_m3*float(pct)/100,
              estimate_note='Approximation from rounded report inflow; not an independently integrated node ledger'))
        maximum=max(top,key=lambda pair:abs(float(pair[1])))
        findings.append(dict(event=row['event'],global_continuity_pct=global_error,
          nonconverging_steps_pct=nonconv,top_node=maximum[0],top_node_error_pct_inflow=float(maximum[1]),
          combined_error_pct_rain=float(row['combined_mass_error_pct_rain']),
          assessment='Global screen passes; local hydraulic adequacy unresolved',report_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    write_csv(OUT/'diagnostics/routing_report_audit.csv',findings)
    write_csv(OUT/'diagnostics/reported_top_node_errors.csv',nodes)
    return rows,findings,nodes


def physical_results_figure(rows):
    configure_publication_style()
    x=np.arange(len(rows));fig,axes=plt.subplots(1,3,figsize=(7.1,2.65),constrained_layout=True)
    values=lambda k:np.array([float(r[k]) for r in rows])
    axes[0].bar(x-.18,values('roughness_effect_mae_mm'),.36,label='Roughness |B-A|')
    axes[0].bar(x+.18,values('drainage_effect_mae_mm'),.36,label='Drainage |C-B|')
    axes[0].set(yscale='log',ylabel='Mean absolute depth difference (mm)',title='Effect separation')
    axes[1].plot(x,values('matched_control_final_volume_m3')/1e6,'o-',label='Matched surface B')
    axes[1].plot(x,values('coupled_final_volume_m3')/1e6,'s-',label='Coupled model C')
    axes[1].set(ylabel='Final surface volume ($10^6$ m$^3$)',title='Surface storage')
    axes[2].bar(x-.18,values('B_vs_MIKE_final_volume_abs_error_m3')/1e6,.36,label='B vs MIKE')
    axes[2].bar(x+.18,values('C_vs_MIKE_final_volume_abs_error_m3')/1e6,.36,label='C vs MIKE')
    axes[2].set(ylabel='Absolute volume error ($10^6$ m$^3$)',title='External comparison')
    for ax in axes:
        ax.set_xticks(x,[r['event'].replace('event','E') for r in rows]);ax.legend(fontsize=6);ax.grid(axis='y',alpha=.2)
    add_panel_labels(axes)
    for ext in ['png','pdf','svg']:fig.savefig(PACKAGE/f'figures/fig03_physical_results.{ext}',dpi=400)
    plt.close(fig)


def table_block(text,n):
    return re.search(rf'\*\*Table {n}\.[^\n]+\*\*\s*\n(?:\s*\n)?(?:\|[^\n]*\n)+',text)[0]


def main():
    (OUT/'diagnostics').mkdir(parents=True,exist_ok=True)
    shutil.copytree(SOURCE,PACKAGE,dirs_exist_ok=True)
    rows,findings,nodes=audit_reports()
    physical_results_figure(rows)
    original=(SOURCE/'manuscript.md').read_text(encoding='utf-8')
    moved='\n\n## Numerical screening and methodological controls (V7)\n\n'+table_block(original,1)+'\n\n'+table_block(original,3)+'\n\n'+table_block(original,5)
    diagnostic_table=table(['Event','Global balance (%)','Nonconverging steps (%)','Top local node','Local balance (% inflow)','Assessment'],
        [[r['event'],r['global_continuity_pct'],r['nonconverging_steps_pct'],r['top_node'],r['top_node_error_pct_inflow'],'Local review required'] for r in findings])
    judgement=("Global routing continuity errors range from -0.178% to -0.060%, and reported nonconverging steps from 0.00% to 0.04%. "
      "Continuity error is a numerical water-budget residual, not a disconnected-pipe count. A nonzero residual is not automatically a model failure, "
      "but small global values do not establish local accuracy. In event68 the reported node N01520 has a 63.01% inflow-normalised local imbalance. "
      "Its rounded cumulative inflow is about 114 m3, corresponding to approximately 71.8 m3 of imbalance; this is not 63% of whole-network water. "
      "The same node's inflow table reports 170.331% because that table uses outflow as its denominator. These are different normalisations of the same budget discrepancy, not conflicting results. "
      "The existing aggregate screening therefore does not close the numerical validation. The reported local errors must be checked against node-level exchanges and shorter-step results before the labels can be described as locally validated.")
    findings_text='# Numerical quality assessment\n\n'+judgement+'\n\n'+diagnostic_table
    findings_text+='\n\n## Interpretation and required follow-up\n\nThe EPA manual (Section 8.5) treats continuity as a numerical diagnostic, calls for investigation of excessive local errors at relevant nodes, and recommends time series at one-minute or shorter resolution for initial instability screening. Its example of a 10% excessive global error is not a universal safe threshold. The repository limits of 2% routing/nonconvergence and 0.5% combined error are project screens, not proof of physical validity. A printed 0.00% is rounded and does not prove zero failed iterations.\n\nA nonconverging-step percentage measures steps that fail the iteration stopping criterion; it is neither a percent of wrong depths nor an estimate of mass loss. The current report prints two decimal places and cannot recover exact failed-step counts or their timing.\n\nNext acceptance: instrument inflow, outflow, storage and signed exchange at the flagged nodes; repeat event68 and the event with the largest local discrepancy with reduced maximum routing steps (0.5, 0.25, 0.125 s) and reviewed head tolerance/iteration settings; compare node hydrographs, local budgets and surface labels. Do not alter the physical network merely to improve a diagnostic. No convergence rerun was executed by this audit, and no label or learned prediction has been replaced.\n\n## Sources\n\n- [EPA manual, Sections 8.5 and 9.1]('+MANUAL+')\n- [Engine 5.2.4 local ranking formula](https://raw.githubusercontent.com/USEPA/Stormwater-Management-Model/v5.2.4/src/solver/stats.c)\n- [Engine 5.2.4 inflow-table formula](https://raw.githubusercontent.com/USEPA/Stormwater-Management-Model/v5.2.4/src/solver/statsrpt.c)\n'
    (OUT/'numerical_quality_assessment.md').write_text(findings_text,encoding='utf-8')
    effects=table(['Event','Rain volume (10^6 m3)','Roughness effect (mm)','Drainage effect (mm)','Final B-C volume (10^6 m3)'],
       [[r['event'],f"{float(r['rain_volume_m3'])/1e6:.3f}",f"{float(r['roughness_effect_mae_mm']):.3f}",f"{float(r['drainage_effect_mae_mm']):.3f}",f"{(float(r['matched_control_final_volume_m3'])-float(r['coupled_final_volume_m3']))/1e6:.3f}"] for r in rows])
    runtime=read(ROOT/'extended_study/output/reviewer_major_revision_v4/final_hybrid_controls/metrics/runtime_uncached_canonical.csv')
    speed=table(['Computation','Mean time (s)','SD across events (s)'],[[label,f"{np.mean([float(r[key]) for r in runtime]):.1f}",f"{np.std([float(r[key]) for r in runtime],ddof=1):.1f}"] for label,key in [('Surface-only B','surface_runtime_s'),('DrainLite correction','total_correction_seconds'),('B + correction','surface_plus_correction_seconds'),('Coupled model C','coupled_runtime_s')]])
    for stem in ['manuscript','report']:
        text=(SOURCE/f'{stem}.md').read_text(encoding='utf-8')
        text=text.replace(table_block(text,1),'**Table 1. Magnitude of the paired physical responses.**\n\n'+effects+'\n')
        text=text.replace(table_block(text,3),'')
        text=text.replace(table_block(text,5),'**Table 5. Computational cost from recorded physical runs and in-memory correction.**\n\n'+speed+'\n\nDisk I/O and offline preparation are excluded; this is not an integrated deployment benchmark.\n')
        text=replace_figure(text,3,figure(3,'fig03_physical_results','Paired physical responses. (a) Mean absolute drainage and roughness effects. (b) Final surface storage. (c) Final-volume discrepancy relative to MIKE. Numerical diagnostics are retained in the supplement; local node-balance validation remains unresolved.'))
        text=re.sub(r'Table (\d+)',lambda m:'Table '+str({4:3,5:4}.get(int(m[1]),int(m[1]))),text)
        text=text.replace('### 3.1 Coupling produced a resolved and numerically controlled response','### 3.1 Magnitude of the coupled drainage response')
        text=text.replace("The coupled calculations met the repository's topology, continuity and mass-balance conditions.","The coupled calculations met aggregate numerical screening limits, but local node water-balance discrepancies remain unresolved.")
        text=text.replace('All eight events passed the physical acceptance conditions.','All eight events passed the aggregate screening conditions; this does not establish local hydraulic accuracy.')
        text=text.replace('All eight events met the physical acceptance conditions.','All eight events met the aggregate screening conditions, but local node-balance errors remain unresolved.')
        text=text.replace('The accepted network contains','The configured network contains')
        text=text.replace("All events met the repository's topology, continuity, convergence and combined mass-balance criteria.",'Aggregate screening limits were met, but local node-balance diagnostics require further validation before the labels can be considered numerically resolved at the network scale.')
        if stem=='manuscript':
            text=text.replace('### 2.4 Physical-run acceptance criteria','### 2.4 Numerical quality assessment')
            start=text.index("The repository applies the following acceptance conditions;")
            end=text.index('**Table 1.',start)
            text=text[:start]+judgement+' Detailed thresholds and node diagnostics are retained in the evidence supplement (Rossman and Simon, 2022, Section 8.5).\n\n'+text[end:]
            text=text.replace('Rossman, L.A., Simon, M.A., 2022.','Rossman, L.A., Simon, M.A., 2022.')
        else:
            text=re.sub(r'\*\*图 3 逐项解释。\*\*[^\n]*','**图 3 逐项解释。** 三个子图只展示关键结果：糙率与管网作用的量级差、最终地表储水量、相对 MIKE 的最终体积差。数值诊断移至补充材料，不表示其问题已消失。',text)
            text=text.replace('9 幅图、5 张表','9 幅图、4 张表')
            text+='\n\n## 数值误差核查后的修正判断\n\n连续性误差是进水、出水和存水的数值收支残差，不是管网断开。离散求解不要求其恰好为零，但需要证明误差足够小且不影响研究结论。未收敛步比例则反映迭代判据未满足的时间步，两者不能混为一谈。\n\n八场事件的全网连续性误差为 -0.178% 至 -0.060%，未收敛步报告为 0.00% 至 0.04%。然而 event68 的 N01520 局部误差为进水量的 63.01%，约对应 114 立方米进水中的 71.8 立方米收支差，不能因全网平均较小而忽略。其入流表中的 170.331% 使用出水量作分母，不能与 63.01% 直接比较。\n\n因此，现阶段不能宣称管网数值质量已经全面验证。原始标签与训练结果未篡改，但其局部可信度仍须通过节点收支和减小时间步重算检验。正文表格保留物理效应、预测精度、外部对比和计算代价四类关键结果；过程性阈值、扰动清单和裁剪诊断移入补充材料，完整保留。\n'
        (PACKAGE/f'{stem}.md').write_text(text,encoding='utf-8')
    supplement=(SOURCE/'evidence_supplement.md').read_text(encoding='utf-8')+moved+'\n\n'+findings_text
    (PACKAGE/'evidence_supplement.md').write_text(supplement,encoding='utf-8')
    audit=(SOURCE/'scientific_integrity_audit.md').read_text(encoding='utf-8')+'\n\n## V7 correction to prior quality claims\n\n'+findings_text
    (PACKAGE/'scientific_integrity_audit.md').write_text(audit,encoding='utf-8')
    print(json.dumps(findings,indent=2))


if __name__=='__main__':main()
