"""Make reviewed figures without changing frozen simulations or predictions."""
from __future__ import annotations
import json
import shutil
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
import generate_reviewer_v4_publication_figures as old
from audit_reviewer_v5_evidence import OUT, GEO, FLOOD
from audit_reviewer_v3_network import sections
from publication_plot_style import add_panel_labels

PACKAGE = OUT / "submission_package_v5"
FIG = PACKAGE / "figures"


def save(fig, stem):
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(FIG / f"{stem}.{ext}", dpi=400)
    plt.close(fig)


def network():
    data = sections(GEO / "conceptual_network.inp")
    xy = {r[0]:np.array([float(r[1]),float(r[2])])/1000 for r in data["COORDINATES"]}
    diam = {r[0]:float(r[2]) for r in data["XSECTIONS"]}
    links = [r for r in data["CONDUITS"] if r[1] in xy and r[2] in xy]
    lines = [[xy[r[1]],xy[r[2]]] for r in links]
    ids = {r[0] for r in data["JUNCTIONS"]}
    of = {r[0] for r in data["OUTFALLS"]}
    receivers = [xy[r[1]] for r in data["CONDUITS"] if r[2] in of and r[1] in xy]
    degree = {n:sum(n in r[1:3] for r in data["CONDUITS"]) for n in ids if n in xy}
    dem = np.load(GEO / "dem.npy")
    active = np.load(GEO / "active_mask.npy").astype(bool)
    fig, axes = plt.subplots(1,3,figsize=(7.5,3.25),constrained_layout=True)
    image=axes[0].imshow(np.ma.masked_where(~active,dem),origin="upper",extent=[0,11.2,0,8],cmap="terrain",vmin=0,vmax=50)
    axes[0].contour(active,levels=[.5],extent=[0,11.2,0,8],origin="upper",colors="#555555",linewidths=.3)
    axes[0].add_collection(LineCollection(lines,colors="#333333",linewidths=.25))
    pts=np.asarray(receivers)
    axes[0].scatter(pts[:,0],pts[:,1],marker="v",s=4,color="#d55e00",label="Outfall-connected terminals")
    for row in links[::90]:
        start,end=xy[row[1]],xy[row[2]]
        axes[0].annotate("",xy=end,xytext=start,arrowprops={"arrowstyle":"->","lw":.5,"color":"black"})
    axes[0].legend(fontsize=5.5,loc="lower left")
    fig.colorbar(image,ax=axes[0],orientation="horizontal",label="Terrain (m)",shrink=.9)
    lc=LineCollection(lines,array=np.array([diam[r[0]] for r in links]),cmap="viridis",linewidths=.55,clim=(.8,1.2))
    axes[1].add_collection(lc)
    fig.colorbar(lc,ax=axes[1],orientation="horizontal",label="Diameter (m)",shrink=.9)
    nodes=np.asarray([xy[n] for n in degree])
    im=axes[2].scatter(nodes[:,0],nodes[:,1],c=list(degree.values()),s=2,cmap="magma",vmin=1,vmax=max(degree.values()))
    fig.colorbar(im,ax=axes[2],orientation="horizontal",label="Incident conduits",shrink=.9)
    for ax,title in zip(axes,["Terrain and receiving interfaces","Conduit diameter","Junction degree"]):
        ax.set(xlim=(0,11.2),ylim=(0,8),xlabel="Local x (km)",ylabel="Local y (km)",title=title)
        ax.set_aspect("equal")
    add_panel_labels(axes)
    save(fig,"fig02_network_audit")


def fixed_residuals():
    active=np.load(GEO/"active_mask.npy").astype(bool)
    fig,axes=plt.subplots(4,4,figsize=(7.1,8.3),constrained_layout=True)
    for i,event in enumerate(["event1","event20","event68","event70"]):
        b=np.load(FLOOD/event/"h_itzi_surface_matched.npy",mmap_mode="r")
        c=np.load(FLOOD/event/"h_itzi_swmm.npy",mmap_mode="r")
        p=np.load(old.EXP/"predictions"/event/"h_hybrid_all.npy",mmap_mode="r")
        t=int(np.argmax(c[:,active].sum(axis=1,dtype=np.float64)))
        for j,field in enumerate([b[t],(c[t]-b[t])*1000,(p[t]-b[t])*1000,(p[t]-c[t])*1000]):
            ax=axes[i,j]
            im=ax.imshow(np.ma.masked_where(~active,field),origin="upper",cmap="Blues" if j==0 else "RdBu_r",vmin=0 if j==0 else -200,vmax=.5 if j==0 else 200)
            ax.set_xticks([]); ax.set_yticks([])
            if i==0: ax.set_title(["Input B","True C-B","Predicted C-B","Depth error"][j])
            if j==0: ax.set_ylabel(f"{event}\n{(t+1)/12:.2f} h")
            if j in [0,3]:fig.colorbar(im,ax=ax,fraction=.046,pad=.02,label="m" if j==0 else "mm",extend="max" if j==0 else "both")
    add_panel_labels(axes.flat,x=-.04,y=1.01)
    save(fig,"fig06_fixed_network_residual_maps")


def skill():
    rows=old.read_csv(old.EXP/"metrics/model_summary_mean_sd.csv")
    order=["surface_matched","prior_only","dynamic_no_prior","hybrid_dynamic","hybrid_mask","hybrid_hydraulic","hybrid_all"]
    labels=["Surface B","ST prior","Dynamic","Prior + dynamic","+ masks","+ hydraulics","+ all network"]
    lookup={r["model"]:r for r in rows if r["reference"]=="coupled_label"}
    fig,axes=plt.subplots(1,3,figsize=(7.5,3.5),constrained_layout=True)
    for ax,metric,title in zip(axes,["mae_mm","rmse_mm","csi_0p15"],["MAE (mm)","RMSE (mm)","CSI at 0.15 m"]):
        for i,model in enumerate(order):
            r=lookup[model]; repeated=int(r["n_seeds"])>1 and model not in ["surface_matched","prior_only"]
            ax.errorbar(float(r[metric+"_mean"]),i,xerr=float(r[metric+"_sd"]) if repeated else None,
                marker="o" if repeated else "D",ms=4,color="#0072b2" if repeated else "#777777",capsize=2)
        ax.set_yticks(range(7),labels if ax is axes[0] else [])
        ax.invert_yaxis();ax.set_xlabel(title);ax.grid(axis="x",alpha=.2)
    fig.suptitle("Circles: five sampling seeds; diamonds: single fit or deterministic baseline",fontsize=8)
    add_panel_labels(axes)
    save(fig,"fig07_skill_and_sampling_seeds")


def selection():
    rows=old.read_csv(old.EXP/"metrics/event_selection_inventory.csv")
    fig,ax=plt.subplots(figsize=(5.8,3.4),constrained_layout=True)
    for used,color,label in [(False,"#999999","Readable, not paired"),(True,"#d55e00","Paired A/B/C")]:
        rr=[r for r in rows if r["public_arrays_valid"].lower()=="true" and (r["used_in_paired_v3"].lower()=="true")==used]
        x=[float(r["mean_6h_rainfall_mm_active"]) for r in rr]; y=[float(r["mike_global_peak_m_active"]) for r in rr]
        ax.scatter(x,y,c=color,s=23,label=label)
        for r,xx,yy in zip(rr,x,y):
            if r["event"] in ["event77","event68"]: ax.annotate(r["event"],(xx,yy),xytext=(5,3),textcoords="offset points",fontsize=7)
    ax.set(xlabel="Active-cell mean six-hour rainfall (mm)",ylabel="MIKE maximum depth (m)")
    ax.legend();ax.grid(alpha=.15)
    save(fig,"fig11_event_selection")


def sensitivity():
    rows=old.read_csv(old.V4/"physical_sensitivity_event68/physical_sensitivity_metrics.csv")
    fields=["drainage_effect_mae_mm","final_surface_reduction_m3","C_vs_MIKE_mae_mm"]
    raw=np.array([[float(r[f]) for f in fields] for r in rows])
    relative=raw/raw[0]
    fig,ax=plt.subplots(figsize=(6.7,3.6),constrained_layout=True)
    im=ax.imshow(relative,cmap="RdBu_r",vmin=0,vmax=2,aspect="auto")
    ax.set_xticks(range(3),["Mean |C-B| (mm)","Final B-C volume\n(1000 m3)","C vs MIKE MAE (mm)"])
    ax.set_yticks(range(6),["Baseline","Loss 0 mm/h","Loss 2 mm/h","Nearest redistribution","Exclude: less rainfall input","Inlet n = 0.015"])
    for i in range(6):
        for j in range(3):
            value=raw[i,j]/1000 if j==1 else raw[i,j]
            ax.text(j,i,f"{value:.3f}\n({relative[i,j]:.2f} x baseline)",ha="center",va="center",fontsize=8)
    ax.tick_params(which="both",length=0)
    fig.colorbar(im,ax=ax,label="Ratio to baseline (not rainfall-normalised)",shrink=.8)
    save(fig,"fig13_physical_sensitivity")


def workflow():
    fig,ax=plt.subplots(figsize=(7.1,3.8),constrained_layout=True)
    ax.set(xlim=(0,1),ylim=(0,1));ax.axis("off")
    boxes=[(.02,.69,.25,.21,"Paired B and C simulations\nfor historical training events"),
           (.37,.69,.25,.21,"Residual archive C - B\nWhole-event separation"),
           (.72,.69,.26,.21,"Offline prior at each cell/time\nOther events only"),
           (.02,.21,.25,.22,"Current event\nB depth + rainfall + terrain\nStatic network fields"),
           (.37,.21,.25,.22,"Fitted DrainLite\nPredict signed residual"),
           (.72,.21,.26,.22,"Corrected depth\nmax(B + predicted residual, 0)")]
    for x,y,w,h,label in boxes:
        ax.add_patch(old.FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.008",facecolor="#edf2f5",edgecolor="#777777",lw=.7))
        ax.text(x+w/2,y+h/2,label,ha="center",va="center",fontsize=7.5)
    for p,q in [((.27,.79),(.37,.79)),((.62,.79),(.72,.79)),((.27,.32),(.37,.32)),((.62,.32),(.72,.32)),((.85,.69),(.58,.43))]:
        ax.add_patch(old.FancyArrowPatch(p,q,arrowstyle="-|>",mutation_scale=10,lw=.9))
    ax.text(.5,.56,"Fit-row priors exclude that row's event; held-event priors exclude the held event",ha="center",fontsize=7)
    ax.text(.5,.06,"Current-event C is used for scoring only. MIKE is an external model comparison.",ha="center",fontsize=7)
    save(fig,"fig01_workflow")


def supplementary():
    active=np.load(GEO/"active_mask.npy").astype(bool)
    event="event68"
    b=np.load(FLOOD/event/"h_itzi_surface_matched.npy",mmap_mode="r")
    c=np.load(FLOOD/event/"h_itzi_swmm.npy",mmap_mode="r")
    mike=np.load(FLOOD/event/"h_mike_ref.npy",mmap_mode="r")
    t=int(np.argmax(c[:,active].sum(axis=1,dtype=np.float64)))
    fields=[c.max(axis=0)-b.max(axis=0),c[t]-b[t],(c-b).max(axis=0)]
    fig,axes=plt.subplots(1,3,figsize=(7.1,2.8),constrained_layout=True)
    for ax,field,title in zip(axes,fields,["max(C) - max(B)",f"C(t) - B(t), t={(t+1)/12:.2f} h","max(C - B)"]):
        im=ax.imshow(np.ma.masked_where(~active,field),cmap="RdBu_r",vmin=-.2,vmax=.2)
        ax.set_title(title);ax.set_xticks([]);ax.set_yticks([])
    fig.colorbar(im,ax=axes,label="Depth difference (m)",shrink=.75,extend="both")
    save(fig,"supp_peak_definitions")
    fig,axes=plt.subplots(1,3,figsize=(7.1,2.8),constrained_layout=True)
    # One common MIKE-volume-peak frame, rather than different model peak times.
    t=int(np.argmax(mike[:,active].sum(axis=1,dtype=np.float64)))
    p=np.load(old.EXP/"predictions"/event/"h_hybrid_all.npy",mmap_mode="r")
    for ax,field,title in zip(axes,[b[t],c[t],p[t]],["Surface B","Coupled C","DrainLite"]):
        truth=mike[t]>.15; wet=field>.15
        classes=np.zeros(active.shape,dtype=int)
        classes[truth & wet]=1;classes[truth & ~wet]=2;classes[~truth & wet]=3
        im=ax.imshow(np.ma.masked_where(~active,classes),cmap=ListedColormap(["#f3f3f3","#009e73","#d55e00","#0072b2"]),norm=BoundaryNorm([-.5,.5,1.5,2.5,3.5],4))
        ax.set_title(title);ax.set_xticks([]);ax.set_yticks([])
    cb=fig.colorbar(im,ax=axes,ticks=[0,1,2,3],shrink=.75)
    cb.ax.set_yticklabels(["Both dry","Overlap","Missed","False alarm"])
    fig.suptitle(f"event68: common MIKE-volume-peak time {(t+1)/12:.2f} h; threshold 0.15 m",fontsize=8)
    save(fig,"supp_mike_extent")


def larno_extremes():
    ref=np.load(old.ROOT/"LarNO-main/benchmark/urbanflood/flood/region1_20m/event68/h.npy",mmap_mode="r")
    pred=np.load(old.ROOT/"LarNO-main/exp/20260220_183648_006352/pred_results/region1_20m/epoch_992/predictions_epoch_992_sample_event68.npy",mmap_mode="r").transpose(2,0,1)
    t=int(np.argmax(ref.max(axis=(1,2))))
    vmax=float(max(ref[t].max(),pred[t].max()))
    error=(pred[t]-ref[t])*1000
    bound=float(np.abs(error).max())
    fig,axes=plt.subplots(2,3,figsize=(7.1,4.5),constrained_layout=True)
    for j,(field,title) in enumerate([(ref[t],"MIKE"),(pred[t],"Public LarNO")]):
        im=axes[0,j].imshow(field,cmap="Blues",vmin=0,vmax=vmax)
        severe=(ref[t]>.5)|(pred[t]>.5)
        axes[1,j].imshow(np.ma.masked_where(~severe,field),cmap="Blues",vmin=0,vmax=vmax)
        axes[0,j].set_title(title);axes[1,j].set_title(title+": union depth > 0.5 m",fontsize=7)
    err=axes[0,2].imshow(error,cmap="RdBu_r",vmin=-bound,vmax=bound)
    axes[0,2].set_title("Untruncated signed error")
    axes[1,2].plot(np.arange(1,73)/12,ref.max(axis=(1,2)),label="MIKE",color="#777777")
    axes[1,2].plot(np.arange(1,73)/12,pred.max(axis=(1,2)),label="LarNO",color="#0072b2",linestyle="--")
    axes[1,2].set(xlabel="Time (h)",ylabel="Maximum depth (m)");axes[1,2].legend(fontsize=6)
    for ax in list(axes[0])+list(axes[1,:2]):ax.set_xticks([]);ax.set_yticks([])
    fig.colorbar(im,ax=list(axes[:,0])+list(axes[:,1]),label="Depth (m)",shrink=.6)
    fig.colorbar(err,ax=axes[0,2],label="Error (mm)",shrink=.7)
    fig.suptitle(f"event68: MIKE global-depth-peak frame, t={(t+1)/12:.3f} h",fontsize=9)
    add_panel_labels(axes.flat,x=-.06,y=1.02)
    save(fig,"fig12_larno_reproduction")
    diagnostic={"event":"event68","time_h":(t+1)/12,"frame_1based":t+1,
        "all_grid_reference_global_max_m":float(ref.max()),"all_grid_prediction_global_max_m":float(pred.max()),
        "reference_frame_over_0p5_pct":float((ref[t]>.5).mean()*100),
        "prediction_frame_over_0p5_pct":float((pred[t]>.5).mean()*100),
        "depth_display_max_m":vmax,"error_display_abs_max_mm":bound,
        "display_high_saturation_pct":0.,"raw_negative_frame_pct":float((pred[t]<0).mean()*100)}
    (OUT/"diagnostics/larno_extremes.json").write_text(json.dumps(diagnostic,indent=2),encoding="utf-8")


def main():
    FIG.mkdir(parents=True,exist_ok=True)
    for path in (old.OUT/"figures").glob("*"):
        if not path.name.startswith(("fig08_","fig10_","fig11_")):
            shutil.copy2(path,FIG/path.name)
    old.FIG=FIG
    workflow()
    network();fixed_residuals();skill();selection();supplementary();sensitivity();larno_extremes()


if __name__=="__main__":main()
