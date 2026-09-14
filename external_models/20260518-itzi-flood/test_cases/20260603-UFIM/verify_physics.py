#!/usr/bin/env python3
"""Verify UFIM coupled simulation uses full physics (ITZI SurfaceFlow + SWMM DYNWAVE)."""
from __future__ import annotations

import ast
import json
import os
import re
import sys

import numpy as np

from build_swmm_from_shapefiles import read_network, read_subcatchments
from sample_utils import CASE_DIR, dem_source_stats, load_dem, prepare_surface_fields, sample_dir

RUN_SCRIPT = os.path.join(CASE_DIR, "run_coupled_simulation.py")


def check_run_script_imports() -> dict:
    with open(RUN_SCRIPT, encoding="utf-8") as f:
        src = f.read()
    tree = ast.parse(src)
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            for alias in node.names:
                imports.append(f"{mod}.{alias.name}" if mod else alias.name)

    has_surface = any("surfaceflow" in i for i in imports) or "SurfaceFlowSimulation" in src
    has_drainage = any("drainage" in i for i in imports) or "DrainageSimulation" in src
    has_apply = "apply_coupling_to_nodes" in src
    has_sf_step = bool(re.search(r"sf_sim\.step\s*\(\s*\)", src))
    has_drainage_none = "swmm_inp=None" in src
    bad_patterns = [
        (r"depression_fill|DepressionFill", "depression filling"),
        (r"instant_routing", "instant routing"),
        (r"pipe_nodes\s*=", "simplified pipe_nodes hack"),
    ]
    issues = []
    for pat, desc in bad_patterns:
        if re.search(pat, src):
            issues.append(desc)

    return {
        "name": "run_coupled_simulation physics imports",
        "pass": (
            has_surface
            and has_drainage
            and has_apply
            and has_sf_step
            and has_drainage_none
            and not issues
        ),
        "detail": (
            f"SurfaceFlowSimulation={has_surface}, sf_sim.step={has_sf_step}, "
            f"DrainageSimulation={has_drainage}, apply_coupling_to_nodes={has_apply}, "
            f"surface-only swmm_inp=None={has_drainage_none}"
            + (f"; forbidden: {issues}" if issues else "")
        ),
    }


def _count_inp_section(path: str, section: str) -> int:
    n = 0
    sec = None
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("["):
                sec = line.strip("[]")
                continue
            if sec == section and line and not line.startswith(";"):
                n += 1
    return n


def check_swmm_dynwave(sample: str) -> dict:
    inp = os.path.join(sample_dir(sample), "network", "drainage.inp")
    if not os.path.exists(inp):
        return {"name": "SWMM DYNWAVE", "pass": False, "detail": f"missing {inp}"}
    with open(inp, encoding="utf-8") as f:
        text = f.read()
    has_dyn = bool(re.search(r"FLOW_ROUTING\s+DYNWAVE", text, re.I))
    has_coord = "[COORDINATES]" in text
    n_sub = _count_inp_section(inp, "SUBCATCHMENTS")
    nodes, _, _ = read_network(sample)
    id_map = {n["raw_id"]: n["id"] for n in nodes}
    n_shp_sub = len(read_subcatchments(sample, id_map))
    sub_ok = n_sub == n_shp_sub if n_shp_sub > 0 else True
    return {
        "name": "SWMM DYNWAVE + COORDINATES + SUBCATCHMENTS",
        "pass": has_dyn and has_coord and sub_ok,
        "detail": (
            f"DYNWAVE={has_dyn}, COORDINATES={has_coord}, "
            f"inp_subs={n_sub}, shp_subs={n_shp_sub}"
        ),
    }


def check_dem_unchanged(sample: str) -> dict:
    stats = dem_source_stats(sample)
    meta, dem_prep, _, _, _, _ = prepare_surface_fields(sample)
    meta_src, dem_src = load_dem(sample)

    valid_src = dem_src[~np.isnan(dem_src)]
    valid_prep = dem_prep[~np.isnan(dem_src)]

    max_diff = float(np.max(np.abs(valid_prep - valid_src))) if valid_src.size else 0.0
    src_min, src_max = stats["min"], stats["max"]
    prep_on_valid = dem_prep.copy()
    prep_on_valid[np.isnan(dem_src)] = np.nan
    pmin = float(np.nanmin(prep_on_valid))
    pmax = float(np.nanmax(prep_on_valid))

    return {
        "name": "DEM valid cells unchanged",
        "pass": max_diff < 1e-3 and abs(pmin - src_min) < 1e-3 and abs(pmax - src_max) < 1e-3,
        "detail": (
            f"source [{src_min:.3f}, {src_max:.3f}], prepared valid [{pmin:.3f}, {pmax:.3f}], "
            f"max_diff={max_diff:.6f}"
        ),
    }


def check_network_counts(sample: str) -> dict:
    nodes, links, _ = read_network(sample)
    junctions = [n for n in nodes if n["type"] != "O"]
    outfalls = [n for n in nodes if n["type"] == "O"]
    inp = os.path.join(sample_dir(sample), "network", "drainage.inp")
    j_inp = o_inp = c_inp = 0
    sec = None
    with open(inp, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("["):
                sec = line.strip("[]")
                continue
            if not line or line.startswith(";"):
                continue
            if sec == "JUNCTIONS":
                j_inp += 1
            elif sec == "OUTFALLS":
                o_inp += 1
            elif sec == "CONDUITS":
                c_inp += 1
    ok = j_inp == len(junctions) and o_inp == len(outfalls) and c_inp == len(links)
    return {
        "name": "Node/link counts vs shapefiles",
        "pass": ok,
        "detail": (
            f"shp J={len(junctions)} O={len(outfalls)} C={len(links)} | "
            f"inp J={j_inp} O={o_inp} C={c_inp}"
        ),
    }


def check_coupling_runtime(sample: str) -> dict:
    """Verify NPZ outputs show real SWMM coupling effect, not duplicated runs."""
    npz_path = os.path.join(sample_dir(sample), "output", "coupled_results.npz")
    if not os.path.isfile(npz_path):
        return {"name": "Runtime coupling effect", "pass": False, "detail": "missing coupled_results.npz"}
    d = np.load(npz_path, allow_pickle=True)
    h_s = d["h_final_surface"]
    h_w = d["h_final_swmm"]
    rec_w = d["rec_swmm"].item()
    summary = d["summary"].item()

    drained = float(rec_w.get("drained_m3", [0])[-1])
    vol_s = float(summary.get("surface_vol_m3", 0))
    vol_w = float(summary.get("swmm_vol_m3", 0))
    flooded_diff = int(summary.get("surface_flooded", 0)) - int(summary.get("swmm_flooded", 0))
    depth_diff_cells = int(np.sum((h_s - h_w) > 0.01))
    identical = np.allclose(h_s, h_w, atol=1e-4)

    ok = not identical and drained > 0 and (vol_w < vol_s or flooded_diff > 0 or depth_diff_cells > 0)
    return {
        "name": "Runtime coupling effect",
        "pass": ok,
        "detail": (
            f"identical_depths={identical}, drained={drained:.0f}m3, "
            f"vol_s/vol_w={vol_s:.0f}/{vol_w:.0f}, cells_swmm_lower={depth_diff_cells}"
        ),
    }


def check_integration_log(sample: str) -> dict:
    log = os.path.join(sample_dir(sample), "output", "data_integration.json")
    if not os.path.isfile(log):
        return {"name": "data_integration.json", "pass": False, "detail": "missing — run simulation"}
    with open(log, encoding="utf-8") as f:
        data = json.load(f)
    physics = data.get("physics", {})
    ok = "SurfaceFlowSimulation" in str(physics.get("surface_solver", ""))
    ok &= "DYNWAVE" in str(physics.get("drainage_solver", ""))
    folders = data.get("folders", {})
    n_full = sum(1 for f in folders.values() if str(f.get("integrated", "")).startswith("已接入"))
    n_partial = sum(1 for f in folders.values() if str(f.get("integrated", "")).startswith("部分接入"))
    n_missing = sum(1 for f in folders.values() if str(f.get("integrated", "")).startswith("未接入"))
    return {
        "name": "Simulation data integration log",
        "pass": ok and n_full >= 5,
        "detail": f"physics={physics}, full={n_full}, partial={n_partial}, missing={n_missing}/10",
    }


def verify_sample(sample: str) -> dict:
    checks = [
        check_swmm_dynwave(sample),
        check_dem_unchanged(sample),
        check_network_counts(sample),
        check_coupling_runtime(sample),
        check_integration_log(sample),
    ]
    return {
        "sample": sample,
        "pass": all(c["pass"] for c in checks),
        "checks": checks,
    }


def main() -> int:
    global_checks = [check_run_script_imports()]
    results = [{"sample": "global", "pass": global_checks[0]["pass"], "checks": global_checks}]
    for s in ["sample1", "sample2", "sample3"]:
        results.append(verify_sample(s))

    out_path = os.path.join(CASE_DIR, "output", "physics_verification.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("UFIM Physics Verification")
    print("=" * 60)
    all_pass = True
    for r in results:
        status = "PASS" if r["pass"] else "FAIL"
        print(f"\n{r['sample']}: {status}")
        if not r["pass"]:
            all_pass = False
        for c in r["checks"]:
            mark = "OK" if c["pass"] else "FAIL"
            print(f"  [{mark}] {c['name']}: {c['detail']}")
    print(f"\nSaved: {out_path}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
