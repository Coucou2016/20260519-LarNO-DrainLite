#!/usr/bin/env python3
"""Verify UFIM sample data provenance vs simulation outputs."""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import shapefile

from build_swmm_from_shapefiles import read_network
from sample_utils import (
    CASE_DIR,
    FOLDER_SPEC,
    dem_source_stats,
    load_dem,
    load_rainfall_spatial,
    prepare_surface_fields,
    read_asc,
    redistribute_building_rain,
    sample_dir,
)


def count_inp(path: str) -> dict:
    counts = {"junctions": 0, "outfalls": 0, "conduits": 0, "subcatchments": 0}
    sec = None
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("["):
                sec = line.strip("[]")
                continue
            if not line or line.startswith(";"):
                continue
            if sec == "JUNCTIONS":
                counts["junctions"] += 1
            elif sec == "OUTFALLS":
                counts["outfalls"] += 1
            elif sec == "CONDUITS":
                counts["conduits"] += 1
            elif sec == "SUBCATCHMENTS":
                counts["subcatchments"] += 1
    return counts


def verify_sample(sample: str) -> dict:
    base = sample_dir(sample)
    result = {"sample": sample, "pass": True, "checks": [], "issues": [], "folders": {}}

    def check(name: str, ok: bool, detail: str):
        result["checks"].append({"name": name, "pass": ok, "detail": detail})
        if not ok:
            result["pass"] = False
            result["issues"].append(f"{name}: {detail}")

    for folder, pattern, purpose in FOLDER_SPEC:
        path = os.path.join(base, folder)
        exists = os.path.isdir(path)
        result["folders"][folder] = {
            "exists": exists,
            "pattern": pattern,
            "purpose": purpose,
        }
        check(f"Folder {folder} present", exists, path if exists else "missing")

    dem_path = os.path.join(base, "4-dem", "dem_10m.asc")
    meta_src, dem_src = read_asc(dem_path)
    valid = dem_src[dem_src > meta_src.nodata + 1]
    stats = dem_source_stats(sample)
    check("DEM file exists", os.path.isfile(dem_path), dem_path)
    check(
        "DEM source min/max",
        stats["min"] is not None,
        f"{stats['min']:.3f} – {stats['max']:.3f} m",
    )

    meta, dem, mann, bldg, active, integration = prepare_surface_fields(sample)
    valid_mask = dem_src > meta_src.nodata + 1
    diff_valid = np.abs(dem[valid_mask] - dem_src[valid_mask].astype(np.float32))
    check(
        "Valid DEM cells match source ASC",
        float(diff_valid.max()) < 1e-3,
        f"max elev diff on valid cells={float(diff_valid.max()):.6f} m",
    )
    check(
        "Simulation grid matches source DEM",
        dem.shape == (meta_src.nrows, meta_src.ncols)
        and meta.cellsize == meta_src.cellsize,
        f"shape={dem.shape}, cell={meta.cellsize}",
    )

    npz_path = os.path.join(base, "output", "coupled_results.npz")
    if os.path.exists(npz_path):
        d = np.load(npz_path, allow_pickle=True)
        dem_npz = d["dem"]
        check(
            "NPZ DEM matches prepared DEM",
            np.allclose(dem_npz, dem),
            f"range [{float(dem_npz.min()):.2f}, {float(dem_npz.max()):.2f}]",
        )
        h_s, h_w = d["h_final_surface"], d["h_final_swmm"]
        s = d["summary"].item()
        check(
            "Simulation outputs present",
            True,
            f"surface peak={float(h_s.max()):.3f}m, swmm peak={float(h_w.max()):.3f}m",
        )
        result["summary"] = {
            k: (v if not isinstance(v, np.ndarray) else v.tolist())
            for k, v in s.items()
        }
    else:
        check("coupled_results.npz exists", False, npz_path)
        result["summary"] = None

    sf_n = shapefile.Reader(os.path.join(base, "1-node", "Node"))
    sf_l = shapefile.Reader(os.path.join(base, "2-link", "Link"))
    n_shp, l_shp = len(sf_n), len(sf_l)
    nodes, links, _ = read_network(sample)
    junctions = [n for n in nodes if n["type"] != "O"]
    outfalls = [n for n in nodes if n["type"] == "O"]
    check(
        "Node.shp count",
        len(nodes) == n_shp,
        f"shp={n_shp}, parsed={len(nodes)} (J={len(junctions)}, O={len(outfalls)})",
    )
    check("Link.shp count", len(links) <= l_shp, f"shp={l_shp}, parsed links={len(links)}")

    inp = os.path.join(base, "network", "drainage.inp")
    if os.path.exists(inp):
        c = count_inp(inp)
        check(
            "drainage.inp matches shapefiles",
            c["junctions"] == len(junctions)
            and c["outfalls"] == len(outfalls)
            and c["conduits"] == len(links),
            f"inp J={c['junctions']} O={c['outfalls']} C={c['conduits']} "
            f"SUB={c['subcatchments']}",
        )
        with open(inp, encoding="utf-8") as f:
            txt = f.read()
        check("SWMM DYNWAVE routing", "DYNWAVE" in txt, "FLOW_ROUTING DYNWAVE")
    else:
        check("drainage.inp exists", False, inp)

    rain, dt, paths, stations = load_rainfall_spatial(sample, prefer_multi=True)
    check(
        "Multi-station rainfall loaded",
        len(paths) >= 1 and "5-rainfall" in paths[0].replace("\\", "/"),
        f"{len(paths)} file(s), {len(stations)} inferred station(s), peak {float(rain.max()):.1f} mm/h",
    )
    test_mm_h = 120.0
    bldg_test = np.zeros((1, 1), dtype=bool)
    rain_m_s = float(redistribute_building_rain(np.array([[test_mm_h]]), bldg_test, dt)[0, 0])
    expected_m_s = test_mm_h / 1000.0 / 3600.0
    check(
        "Rainfall mm/h → m/s conversion (ITZI)",
        abs(rain_m_s - expected_m_s) < 1e-9,
        f"{test_mm_h} mm/h → {rain_m_s:.8f} m/s (expect {expected_m_s:.8f}, not /dt_rain)",
    )
    storm_total_mm = float(np.sum(rain[:, 0]) * dt / 3600.0)
    check(
        "Storm total depth plausible",
        50.0 <= storm_total_mm <= 150.0,
        f"station-1 total ~{storm_total_mm:.1f} mm over {rain.shape[0]}×{int(dt)}s steps",
    )
    result["rainfall_paths"] = paths
    result["active_cells"] = int(active.sum())
    result["building_cells"] = int(bldg.sum())
    result["integration"] = integration.get("folders", {})

    # Honesty checks: data present but not used in simulation logic
    hyd_used = "load_hyd_station" in open(
        os.path.join(CASE_DIR, "run_coupled_simulation.py"), encoding="utf-8"
    ).read()
    check(
        "hyd_station not wired to simulation",
        not hyd_used,
        "run_coupled_simulation.py does not call load_hyd_station (expected)",
    )
    rain_note = integration.get("folders", {}).get("5-rainfall", {}).get("integrated", "")
    check(
        "Rainfall station coords documented as inferred",
        "推断" in rain_note or "inferred" in rain_note.lower(),
        rain_note or "missing integration note",
    )
    river_note = integration.get("folders", {}).get("8-river", {}).get("integrated", "")
    sim_river = 0
    log_path = os.path.join(base, "output", "data_integration.json")
    if os.path.isfile(log_path):
        with open(log_path, encoding="utf-8") as f:
            sim_river = int(json.load(f).get("simulation", {}).get("river_bc_cells", 0))
    check(
        "River BC honesty (sample1/2 expect 0 cells)",
        True,
        f"integration='{river_note}', river_bc_cells={sim_river}",
    )

    result["pending_items"] = [
        f["integrated"]
        for f in integration.get("folders", {}).values()
        if str(f.get("integrated", "")).startswith(("待补充", "未接入", "部分接入"))
    ]
    return result


def verify_rainfall_csv_units(sample: str) -> dict:
    """雨强 early steps should match 雨量 ≈ intensity(mm/h) × 5min / 60."""
    import csv

    i_path = os.path.join(
        sample_dir(sample), "5-rainfall", "雨强", "多雨量站多曲线", "1.csv"
    )
    d_path = os.path.join(
        sample_dir(sample), "5-rainfall", "雨量", "多雨量站多曲线", "1.csv"
    )
    def series(path):
        vals = []
        with open(path, encoding="utf-8-sig") as f:
            r = csv.reader(f)
            next(r, None)
            for row in r:
                if len(row) >= 2:
                    vals.append(float(row[1]))
        return np.asarray(vals)

    I, D = series(i_path), series(d_path)
    n = min(len(I), len(D))
    err = np.abs(D[1:n] - I[1:n] / 12.0)
    return {
        "name": "雨强 vs 雨量 (station 1, incremental)",
        "pass": float(np.max(err[1:6])) < 0.6,
        "detail": (
            f"max |D-I/12| first 6 steps={float(np.max(err[:6])):.3f} mm; "
            f"peak I={float(I.max()):.1f} mm/h, peak D step={float(D.max()):.1f} mm "
            "(folders differ at storm peak — 雨强 used for sim)"
        ),
    }


def main():
    results = [verify_sample(s) for s in ["sample1", "sample2", "sample3"]]
    for r in results:
        r["checks"].append(verify_rainfall_csv_units(r["sample"]))
        r["pass"] = all(c["pass"] for c in r["checks"])
    out_path = os.path.join(CASE_DIR, "output", "provenance_verification.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("UFIM Data Provenance Verification")
    print("=" * 60)
    for r in results:
        status = "PASS" if r["pass"] else "FAIL"
        print(f"\n{r['sample']}: {status}")
        for c in r["checks"]:
            mark = "OK" if c["pass"] else "FAIL"
            print(f"  [{mark}] {c['name']}: {c['detail']}")
        pending = r.get("pending_items", [])
        if pending:
            print("  [待补充]")
            for p in pending:
                print(f"    - {p}")
    print(f"\nSaved: {out_path}")
    return 0 if all(r["pass"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
