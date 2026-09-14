# UFIM Test Case Adversarial Audit Report

**Case:** `test_cases/20260603-UFIM`  
**Audit date:** 2026-06-03  
**Overall credibility:** **PARTIAL PASS**

The core ITZI+SWMM coupled simulation is **genuine** (real `SurfaceFlowSimulation.step()`, real `apply_coupling_to_nodes`, paired surface-only vs coupled runs, real UFIM shapefile/ASC/CSV inputs). Several integration claims, report text, and verify scripts were **overstated or stale**; those are corrected below.

---

## Executive summary

| Area | Verdict | Summary |
|------|---------|---------|
| A. Physics honesty | **PASS** | Full 2D surface dynamics + SWMM DYNWAVE coupling; surface-only truly disables drainage |
| B. Data completeness | **PARTIAL** | 7/10 folders materially used; 3 partial; hyd station not used |
| C. Results credibility | **PASS** | NPZ/ASC outputs physically plausible; SWMM coupling changes results; table 5 matches NPZ |
| D. Documentation | **FAIL → fixed** | `report.html` had stale abstract/discussion numbers; PNGs missing; integration labels overstated |

---

## Findings

| # | Issue | Severity | Evidence | Status |
|---|-------|----------|----------|--------|
| 1 | `report.html` abstract cited wrong rainfall source (单站) and wrong peak depths (6.76/5.93, 8.78/8.84) | **High** | `report.html:180-182`, `503`; NPZ summary 7.66/7.57, 5.37/5.33 | **Fixed** — `generate_report.py` now pulls dynamic summary |
| 2 | Integration JSON marked river/tidal as fully “已接入” while sample1/2 have `river_bc_cells=0` | **High** | `sample_utils.py:544-546`; `data_integration.json` boundary | **Fixed** — honest 部分接入 labels |
| 3 | `build_swmm_from_shapefiles.py` read subcatchments but wrote `subcatchments=[]` | **Medium** | `build_swmm_from_shapefiles.py:264` (old) | **Fixed** — 224/204/142 subs in inp + dummy RainGage |
| 4 | `9-hyd_station` never loaded; “仅报告验证对比” was false | **Medium** | No `load_hyd_station` in `run_coupled_simulation.py` | **Fixed** — labeled 未接入 |
| 5 | `verify_physics.py` only grep/AST; could not detect fake coupling | **Medium** | Old `verify_physics.py` | **Fixed** — adds NPZ runtime coupling check |
| 6 | `visualization_output/*.png` missing (report embeds placeholders) | **Medium** | Empty `visualization_output/` | **Fixed** — `visualize_results.py --all` run; 10 PNGs embedded |
| 7 | Rain station XY invented from subcatchment centroids (not in CSV) | **Medium** | `sample_utils.py:328-337` | **Open** — documented as 部分接入 |
| 8 | `雨量/多雨量站多曲线/5.csv` exists but intensity run uses only 4 files | **Low** | `sample1/5-rainfall/` tree | **Open** — by design (雨强 folder has 4) |
| 9 | Outlook section said river/tidal “待补充” while code partially implements them | **Medium** | `report.html:524` vs `setup_boundary_arrays` | **Fixed** |
| 10 | Output named `coupled_results.npz` not `simulation_results.npz` | **Low** | README vs user checklist | **Open** — naming only |
| 11 | sample1/2 `river_bc_cells=0` — geometry not bug | **Info** | River near geo edge but grid-edge cells blocked by mask/nodata | **Open** — expected with current logic |
| 12 | `check_integration_log` counted “已接入” only; hid partial items | **Low** | Old verify script | **Fixed** |

---

## A. Physics honesty (detailed)

### What is real

1. **Surface flow:** `run_coupled_simulation.py:224-226` calls `sf_sim.step()` / `solve_dt()` each sub-step via `SurfaceFlowSimulation` — not depression fill or instant routing.
2. **SWMM coupling:** `run_coupled_simulation.py:198-218` uses `drainage.step()` + `drainage.apply_coupling_to_nodes(surface_states, cell_area)` — same pattern as gold standard `urban_drainage/run_coupled_simulation.py:417-433`.
3. **Surface-only:** `run_coupled_simulation.py:301` passes `swmm_inp=None`; no drainage object, no `n_drain` updates.
4. **DEM:** Valid cells match source ASC exactly (`max_diff=0`); only nodata → median fill (`sample_utils.py:488-491`).
5. **No fake DEM hacks:** No building +10 m, slope injection, or cut/fill beyond nodata fill.

### Caveats

- Manning lowered to 0.012 near coupled inlets (`run_coupled_simulation.py:153-157`) — physical grate shortcut, not fake physics.
- Infiltration disabled (`InfNull`) — documented simplification.
- Loop order differs from urban_drainage (hydrology → drain → surface vs drain after hydrology before `update_ext_array`) — both use ITZI API; not evidence of fake run.

### verify_physics.py (before/after)

- **Before:** AST import grep + inp file text.
- **After:** Also checks `sf_sim.step()` in source, `swmm_inp=None` for surface-only, NPZ shows `drained_m3>0`, non-identical depth fields, subcatchments in inp when shapefile has them.

---

## B. Data folder audit (1–10)

| Folder | Used in simulation? | How |
|--------|---------------------|-----|
| 1-node | **Yes** | SWMM inp + ITZI node coupling coordinates |
| 2-link | **Yes** | SWMM conduits DYNWAVE |
| 3-subcatchment | **Partial** | Now written to SWMM inp; rain still on ITZI surface only |
| 4-dem | **Yes** | ITZI terrain |
| 5-rainfall | **Partial** | 4× intensity CSV, IDW; coords inferred |
| 6-boundary | **Yes** | `mask.shp` → blocked cells (simulation domain) |
| 7-lulc | **Yes** | Manning mapping |
| 8-river | **Partial** | `river_boundary_mask` + edge mask; 0 cells sample1/2 |
| 9-hyd_station | **No** | CSV never read by run script |
| 10-tidal_station | **Partial** | `load_tidal_series` only if river BC cells exist |

### Rainfall CSV structure

- Format: `time,value` — 5-min intensity (mm/h), peak ~262.8 mm/h, start 2025-5-12.
- Multi-station: `雨强/多雨量站多曲线/1.csv`–`4.csv` (4 stations). Fifth file only under `雨量/` subtree.

### hyd_station CSV

- Format: `time,value` — stage/discharge observations (~0.74 m); **not compared** to simulation.

---

## C. Results credibility

### NPZ spot-check (sample1)

- Surface peak 7.660 m, SWMM 7.570 m; thousands of cells with SWMM lower depth.
- Cumulative drain ~2,023 m³; surface vol 1.36×10⁶ m³ vs SWMM 5.50×10⁵ m³.
- No NaN in depth fields; ASC files match NPZ maxima.

### Report vs NPZ

- Table 5 in regenerated `report.html` matches `coupled_results.npz` summary dict.
- Old abstract/discussion numbers did **not** match — **fixed**.

### Visualizations

- `visualization_output/` was **empty** at audit time; report figures were missing. Regenerate with `python visualize_results.py --all`.

---

## D. Known gaps — confirmed

| Gap | Finding |
|-----|---------|
| SUBCATCHMENTS in inp | Was empty; **fixed** in `build_swmm_from_shapefiles.py` |
| Rain station coordinates | Still inferred; needs coordinate file from data provider |
| hyd station calibration | **Not implemented** |
| sample1/2 river BC = 0 | **Correct** for current logic: river polyline does not intersect active **raster** edge cells (edge cells blocked by mask/nodata) |
| sample3 river BC = 12 | Tidal WSE applied at mid-sim time (~−0.85 m offset) |

---

## What IS genuinely correct and defensible

1. Real UFIM DEM, Node/Link shapefiles, and rainfall CSVs drive the simulation — not synthetic random fields.
2. SWMM network counts match shapefiles (e.g. sample1: J=230, O=16, C=237).
3. Paired surface-only vs coupled methodology matches `urban_drainage` reference architecture.
4. SWMM coupling measurably reduces stored volume and flooded extent in all three samples.
5. `6-boundary/mask.shp` restricts the computational domain (not just plotting).
6. README simulation results table matches latest NPZ summaries.

---

## What remains incomplete

1. **Rain gauge coordinates** — need metadata from UFIM data pack.
2. **Hyd station validation** — requires loading CSV and comparison module.
3. **River open boundary** — extend beyond grid-edge intersection or un-block edge cells where river meets boundary.
4. **Visualization PNGs** — must run `visualize_results.py --all` after audit.
5. **Re-run simulations** — not required for audit fixes except if rebuilding inp with subcatchments should be validated under pyswmm (recommended: `build_swmm_from_shapefiles.py sample1` smoke test).

---

## Files changed in this audit

- `sample_utils.py` — honest integration status strings
- `build_swmm_from_shapefiles.py` — RainGage + SUBCATCHMENTS/SUBAREAS/INFILTRATION
- `verify_physics.py` — runtime coupling + subcatchment checks
- `verify_data_provenance.py` — hyd_station + rainfall honesty checks
- `generate_report.py` — dynamic abstract, fixed discussion/outlook
- `README.md` — integration status table
- `AUDIT_REPORT.md` — this document

---

## Recommended next commands

```powershell
cd test_cases/20260603-UFIM
python build_swmm_from_shapefiles.py sample1
python build_swmm_from_shapefiles.py sample2
python build_swmm_from_shapefiles.py sample3
python verify_physics.py
python verify_data_provenance.py
python visualize_results.py --all
python generate_report.py
```

Re-run `run_coupled_simulation.py` only if subcatchment inp changes affect coupling behavior (unlikely for ITZI surface rain path).
