"""Recompute review diagnostics from frozen physical arrays and held predictions."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
import numpy as np

from generate_reviewer_v4_documents import ROOT, V3, V4, EXP, EVENTS, read_csv
from run_reviewer_v3_drainlite import write_csv
from audit_reviewer_v3_network import sections
from analyze_reviewer_v3_physics import report_volumes

OUT = ROOT / "extended_study/output/reviewer_major_revision_v5"
GEO = ROOT / "LarNO-main/benchmark/urbanflood/geodata/region1_20m_drainage_v3_full"
FLOOD = ROOT / "LarNO-main/benchmark/urbanflood/flood/region1_20m_drainage_v3_full"


def main():
    out = OUT / "diagnostics"
    out.mkdir(parents=True, exist_ok=True)
    dem = np.load(GEO / "dem.npy")
    active = np.load(GEO / "active_mask.npy").astype(bool)
    assert np.array_equal(active, dem < 49.9)
    data = sections(GEO / "conceptual_network.inp")
    inv = {r[0]: float(r[1]) for r in data["JUNCTIONS"] + data["OUTFALLS"]}
    slopes = np.array([(inv[r[1]] + float(r[5]) - inv[r[2]] - float(r[6])) / float(r[3]) for r in data["CONDUITS"]])
    network = {"conduits": len(slopes), "zero_or_negative_slopes": int((slopes <= 0).sum()),
               "minimum_slope": float(slopes.min()), "maximum_slope": float(slopes.max()),
               "near_design_floor_count": int((np.abs(slopes - 0.0005) <= 2e-6).sum()),
               "near_design_floor_tolerance": 2e-6,
               "outfalls": len(data["OUTFALLS"]),
               "outfalls_with_coordinates": len({r[0] for r in data["OUTFALLS"]} & {r[0] for r in data["COORDINATES"]}),
               "note": "Near-floor count is geometric, not proof of individual algorithm adjustment history."}
    (out / "network_geometry.json").write_text(json.dumps(network, indent=2), encoding="utf-8")
    rainfall_rows, rainfall_steps, balances, hydrographs, residuals, map_rows = [], [], [], [], [], []
    for event in EVENTS:
        print(event, flush=True)
        directory = FLOOD / event
        rain = np.load(directory / "rainfall.npy", mmap_mode="r")
        b = np.load(directory / "h_itzi_surface_matched.npy", mmap_mode="r")
        c = np.load(directory / "h_itzi_swmm.npy", mmap_mode="r")
        mike = np.load(directory / "h_mike_ref.npy", mmap_mode="r")
        p = np.load(EXP / "predictions" / event / "h_hybrid_all.npy", mmap_mode="r")
        meta = json.loads((directory / "physics_metadata.json").read_text())
        injection = []
        for t, field in enumerate(rain):
            assert np.isfinite(field).all() and (field >= 0).all()
            direct = field[active].sum(dtype=np.float64) * .4
            reassigned = field[~active].sum(dtype=np.float64) * .4
            effective = np.zeros_like(field)
            effective[active] = field[active]
            effective[active] += float(field[~active].sum(dtype=np.float64)) / active.sum()
            injected = effective[active].sum(dtype=np.float64) * .4
            injection.append(injected)
            rainfall_steps.append({"event": event, "step": t + 1, "time_h": (t+1)/12,
                "raw_m3": direct + reassigned, "active_direct_m3": direct,
                "inactive_reassigned_m3": reassigned, "injected_reconstructed_m3": injected,
                "roundoff_m3": injected - direct - reassigned})
        full = rain.sum(dtype=np.float64) * .4
        rainfall_rows.append({"event": event, "rectangle_km2": dem.size * .0004,
            "active_km2": active.sum() * .0004, "inactive_proxy_km2": (~active).sum() * .0004,
            "nonfinite_dem_cells": int((~np.isfinite(dem)).sum()),
            "active_mean_total_mm": rain[:, active].sum(dtype=np.float64) / active.sum(),
            "rectangle_mean_total_mm": full / (dem.size * .4),
            "raw_m3": full, "active_direct_m3": rain[:, active].sum(dtype=np.float64)*.4,
            "inactive_reassigned_m3": rain[:, ~active].sum(dtype=np.float64)*.4,
            "injected_reconstructed_m3": sum(injection),
            "B_loss_m3": meta["records"]["B"]["infiltrated_m3"][-1],
            "C_loss_m3": meta["records"]["C"]["infiltrated_m3"][-1]})
        for scenario in ["B", "C"]:
            rec = meta["records"][scenario]
            for i, time_h in enumerate(rec["time_h"]):
                step = int(round(time_h * 12))
                net = rec["swmm_net_exchange_m3"][i]
                drainage = rec["drained_m3"][i]
                held = rec["surface_held_exchange_m3"][i]
                inp = sum(injection[:step])
                balances.append({"event": event, "scenario": scenario, "time_h": time_h,
                    "rain_m3": inp, "loss_m3": rec["infiltrated_m3"][i],
                    "surface_m3": rec["vol_m3"][i], "surface_net_exchange_m3": held,
                    "swmm_forward_exchange_m3": drainage, "swmm_return_exchange_m3": drainage-net,
                    "swmm_net_exchange_m3": net, "exchange_integral_mismatch_m3": held-net,
                    "surface_closure_m3": inp-rec["infiltrated_m3"][i]-held-rec["vol_m3"][i]})
        bv, cv, mv, pv = [x[:, active].sum(axis=1, dtype=np.float64)*400 for x in (b,c,mike,p)]
        tpeak = int(np.argmax(cv))
        for name, series in [("B",bv),("C",cv),("DrainLite",pv)]:
            hydrographs.append({"event":event,"model":name,
                "volume_peak_h": (np.argmax(series)+1)/12, "mike_volume_peak_h":(np.argmax(mv)+1)/12,
                "peak_time_difference_h": (np.argmax(series)-np.argmax(mv))/12,
                "volume_rmse_vs_mike_m3": float(np.sqrt(np.mean((series-mv)**2))),
                "last_hour_volume_slope_m3h": float(np.polyfit(np.arange(12)/12,series[-12:],1)[0]),
                "volume_bias_vs_C_mean_m3": float(np.mean(series-cv)),
                "volume_error_vs_C_maxabs_m3": float(np.max(np.abs(series-cv))),
                "volume_error_vs_C_integral_abs_m3h": float(np.abs(series-cv).sum()/12)})
        rr = (c[:, active]-b[:, active])*1000
        pr = (p[:, active]-b[:, active])*1000
        for label, mask in [("net_negative_lt_minus5mm",rr < -5),("positive_gt5mm",rr > 5)]:
            err = pr[mask]-rr[mask]
            residuals.append({"event":event,"subset":label,"n_cell_times":int(mask.sum()),
                "active_cell_time_pct":float(mask.mean()*100),"mae_mm":float(np.abs(err).mean()),
                "bias_mm":float(err.mean()),"max_abs_error_mm":float(np.abs(err).max()),
                "wrong_sign_pct":float((np.sign(pr[mask])!=np.sign(rr[mask])).mean()*100)})
        for label, field, bound in [("B",b[tpeak],.5),("C_minus_B",c[tpeak]-b[tpeak],.2),
                                    ("pred_minus_B",p[tpeak]-b[tpeak],.2),("error",p[tpeak]-c[tpeak],.2)]:
            v = field[active]
            map_rows.append({"event":event,"field":label,"time_h":(tpeak+1)/12,"display_abs_bound_m":bound,
                "clipped_pct":float((np.abs(v)>bound).mean()*100),"min_m":float(v.min()),"max_m":float(v.max()),
                **{f"abs_gt_{mm}mm_pct":float((np.abs(v)>mm/1000).mean()*100) for mm in [50,100,200]}})
    for name, rows in [("rainfall_event",rainfall_rows),("rainfall_steps",rainfall_steps),("water_ledger_saved_times",balances),
                       ("hydrograph_diagnostics",hydrographs),("signed_residual_diagnostics",residuals),("map_display_audit",map_rows)]:
        write_csv(out / f"{name}.csv",rows)
    metrics = read_csv(EXP / "metrics/event_metrics_all_seeds.csv")
    lookup = {(r["event"],r["seed"],r["model"]):float(r["mae_mm"]) for r in metrics if r["reference"]=="coupled_label"}
    seeds = sorted({r["seed"] for r in metrics},key=int)
    matrix = np.array([[lookup[e,s,"hybrid_dynamic"]-lookup[e,s,"hybrid_all"] for s in seeds] for e in EVENTS])
    write_csv(out / "event_seed_gain.csv",[{"event":e,**{f"seed_{s}_gain_mm":matrix[i,j] for j,s in enumerate(seeds)},
        "seed_mean_gain_mm":matrix[i].mean()} for i,e in enumerate(EVENTS)])
    rng = np.random.default_rng(20260926)
    means = matrix.mean(axis=1)
    boot = means[rng.integers(0,len(EVENTS),(20000,len(EVENTS)))].mean(axis=1)
    inventory = read_csv(EXP / "metrics/event_selection_inventory.csv")
    selected = [float(r["mean_6h_rainfall_mm_active"]) for r in inventory if r["used_in_paired_v3"].lower()=="true"]
    other = [float(r["mean_6h_rainfall_mm_active"]) for r in inventory if r["public_arrays_valid"].lower()=="true" and r["used_in_paired_v3"].lower()!="true"]
    runtime = read_csv(EXP / "metrics/runtime_uncached_canonical.csv")
    speed = [float(r["coupled_runtime_s"])/float(r["surface_plus_correction_seconds"]) for r in runtime]
    summary = {"source_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        "network":network,"network_gain_mean_mm":float(means.mean()),"event_gain_sd_mm":float(means.std(ddof=1)),
        "seed_macro_gain_sd_mm":float(matrix.mean(axis=0).std(ddof=1)),
        "event_bootstrap_95pct_mm":np.quantile(boot,[.025,.975]).tolist(),
        "bootstrap_scope":"Conditional on eight events being exchangeable; weather-date independence unverified",
        "selected_rain_mm":float(np.mean(selected)),"other_readable_rain_mm":float(np.mean(other)),
        "speed_mean_ratios":float(np.mean(speed)),"speed_sd_ratios":float(np.std(speed,ddof=1)),
        "speed_ratio_means":sum(float(r["coupled_runtime_s"]) for r in runtime)/sum(float(r["surface_plus_correction_seconds"]) for r in runtime),
        "raw_array_changes":False,"saved_water_ledger_interval_minutes":30,
        "full_joint_step_ledger_available":False,
        "wall_proxy_is_verified_building_mask":False}
    (out / "summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2))


if __name__ == "__main__":
    main()
