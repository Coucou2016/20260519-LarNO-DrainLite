# UFIM Rainfall Units Audit (2026-06-03)

## Executive summary

**Root cause: confirmed unit bug in code (not bad CSV units).**

`redistribute_building_rain()` converted rainfall intensity from **mm/h to m/s** by dividing by `(1000 × dt_rain)` with `dt_rain = 300 s`, instead of the ITZI-standard `(1000 × 3600)`. That applied **12× too much rain** over each 5-minute step.

After fixing and re-running all three samples, peak depths dropped from **~4–8 m to ~0.9–2.4 m**, consistent with a ~97 mm design storm, zero infiltration (`InfNull`), and limited SWMM drainage.

---

## 1. Rainfall CSV inspection (all samples)

### Layout

| Folder | Meaning | Used in sim? |
|--------|---------|--------------|
| `5-rainfall/雨强/多雨量站多曲线/*.csv` | Intensity time series | **Yes** (4 stations) |
| `5-rainfall/雨量/多雨量站多曲线/*.csv` | Depth time series | No (cross-check only) |
| `5-rainfall/雨强/单雨量站单曲线/1.csv` | Single-station intensity | No (multi preferred) |

All three samples (`sample1`–`sample3`) share the same numeric hyetograph structure (identical `1.csv` values).

### Format

```csv
time,value
2025/5/12 0:00,11.52
2025/5/12 0:05,11.952
...
2025/5/12 1:35,262.8
```

- **Header:** `time,value` (no unit column; README documents units).
- **Timestep:** 5 minutes (`0:00`, `0:05`, … → `dt = 300 s`).
- **雨强 values:** interpreted as **mm/h** (peak **262.8 mm/h** at 01:35).
- **雨量 values:** incremental **mm per 5-min step** (not cumulative).

### Excerpt — station 1 (`雨强` vs `雨量`)

| Time | 雨强 (mm/h) | 雨量 (mm) | 雨强÷12 (mm/5min) |
|------|-------------|-----------|-------------------|
| 0:00 | 11.52 | 0.00 | — |
| 0:05 | 11.95 | 0.95 | 0.996 |
| 0:10 | 12.46 | 1.05 | 1.038 |
| 0:15 | 13.00 | 1.17 | 1.083 |
| 1:35 | **262.8** | 1.52 | **21.9** |

Early steps: `雨量 ≈ 雨强 / 12` (because mm/h × 5 min / 60 min = mm per step).

Late storm: **folders diverge** — `雨量` peaks at 0:50 (16.7 mm/step) while `雨强` peaks at 1:35 (262.8 mm/h). Same file index is **not** the same storm phase in both folders. Simulation correctly uses **雨强 only**.

### Storm totals (station 1, 25 × 5 min)

| Metric | Value |
|--------|-------|
| Σ intensity × dt / 3600 | **~97.1 mm** |
| Σ 雨量 (if incremental) | ~76.2 mm |
| Duration | 2.08 h |

---

## 2. Code path and ITZI expectation

### Load chain

1. `load_rainfall_spatial()` → reads `雨强/多雨量站多曲线/*.csv`, assumes **mm/h**, `dt_s = 300`.
2. `interpolate_rainfall_idw()` → spatial field in **mm/h** per step.
3. `run_coupled_simulation.py` → every `dt_rain`, updates `domain` rain array via `redistribute_building_rain()`.

### ITZI (`apply_hydrology` in `itzi/flow.pyx`)

- **`rain` array must be in m/s.**
- Standard config path (`itzi.py` `update_input_arrays`): `mm/h ÷ (1000 × 3600)`.

### Bug (before fix)

```python
# sample_utils.py — WRONG
field[active] = (rain_mm[active] + extra) / 1000.0 / dt_s   # dt_s=300 → 12× too large
```

```python
# sample_utils.py — FIXED
field[active] = (rain_mm_h[active] + extra) / 1000.0 / 3600.0
```

**Mechanism:** For constant rate over 300 s, depth added = `rain_m_s × 300`. Wrong rate gives 12× depth per step → peaks ~12× too high (modulated by routing, drainage, domain).

**Not the issue:** IDW, double timestep loop, or using `雨量` folder. CSV units for `雨强` are consistent with mm/h.

---

## 3. Physical plausibility (after fix)

| Sample | Active cells | Area (km²) | Storm total | Mean depth if all ponded | Sim peak (surf / SWMM) |
|--------|--------------|------------|-------------|------------------------|-------------------------|
| sample1 | 17,173 | ~2.47 | 97 mm | ~0.10 m | **2.44 / 2.44 m** |
| sample2 | 13,342 | ~2.13 | 97 mm | ~0.10 m | **2.36 / 2.34 m** |
| sample3 | 8,362 | ~0.84 | 97 mm | ~0.10 m | **0.98 / 0.90 m** |

- **DEM:** metres (`cellsize 10`, elevations ~0–30 m, `NODATA -9999`). `h` is **water depth above DEM**, not WSE mislabelled.
- **Infiltration:** `InfNull` → **0 mm/h infiltration** (inflates depth vs real urban LULC).
- **Drainage:** SWMM coupled; sample1 drained only **~279 m³** vs **~119,000 m³** surface volume — network capacity limited vs rainfall volume.
- **Peaks > mean depth:** Topographic ponding in low cells (2–2.4 m local max) is plausible with intense burst (262.8 mm/h) and no infiltration, though still high for validation against observations.

---

## 4. DEM / NPZ units

| Field | Unit | Notes |
|-------|------|-------|
| `dem` | m | Unchanged from `dem_10m.asc` |
| `h`, `h_final_*` | m | Depth above ground |
| `rec_*['vol_m3']` | m³ | `Σ h × cell_area` |
| `rain_mm_h` in NPZ | mm/h | Input series, not m/s |

---

## 5. Before / after peak depths

| Sample | Before (bug) surf / SWMM | After (fix) surf / SWMM | Ratio (surf) |
|--------|--------------------------|-------------------------|--------------|
| sample1 | 7.66 / 7.57 m | **2.44 / 2.44 m** | ~3.1× (not full 12× due to hydraulics) |
| sample2 | 5.37 / 5.33 m | **2.36 / 2.34 m** | ~2.3× |
| sample3 | 4.02 / 3.28 m | **0.98 / 0.90 m** | ~4.1× |

---

## 6. Recommendations

1. **Keep fix** in `redistribute_building_rain()` — always divide mm/h by `3600`, never by rainfall step length.
2. **Enable LULC-based infiltration** when ready (replace `InfNull`) — will reduce peaks further.
3. **Document** that `雨量/` hyetographs are not aligned with `雨强/` at peak time; do not load `雨量` as intensity without conversion.
4. **Validation:** Compare 0.9–2.4 m peaks to gauge data; consider whether design storm 262.8 mm/h is intended for this 2 h window.
5. **SWMM rebuild:** `[SUBCATCHMENTS]` SnowPack column must not be literal `*` (SWMM ERROR 209); fixed in `build_swmm_from_shapefiles.py`.

---

## 7. Verification added

`verify_data_provenance.py` now checks:

- `redistribute_building_rain(120 mm/h) == 120/1e6/3600 m/s`
- Storm total ~97 mm for station 1
- Early-step consistency `雨量 ≈ 雨强/12` (informational; folders differ at peak)

Re-run after any rainfall pipeline change:

```bash
python run_coupled_simulation.py --all --rebuild-swmm
python verify_data_provenance.py
python visualize_results.py
python generate_report.py
```
