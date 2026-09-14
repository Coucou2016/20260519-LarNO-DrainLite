# UFIM Sample Datasets — ITZI + SWMM Coupled Flood Simulation

Three urban flood modeling samples with DEM, drainage network (shapefiles), rainfall, and boundary data. This folder contains scripts to build SWMM networks, run bi-directional ITZI–SWMM coupled simulations, and visualize results.

## Sample Data Structure

Each sample folder (`sample1`, `sample2`, `sample3`) contains:

| Folder | Content |
|--------|---------|
| `1-node/` | Manhole/outfall points (`Node.shp`) — NodeID, invert/surface elevation, Type (J=junction, O=outfall) |
| `2-link/` | Pipe links (`Link.shp`) — upstream/downstream nodes, diameter, length, roughness |
| `3-subcatchment/` | SWMM-style subcatchments (`Subcatchment.shp`) — not used directly in coupling |
| `4-dem/` | Terrain DEM (`dem_10m.asc`, 10 m resolution, projected CRS) |
| `5-rainfall/` | Rainfall time series (雨强=intensity mm/h, 雨量=depth mm; 5-min steps) |
| `6-boundary/` | Domain mask polygon (`mask.shp`) |
| `7-lulc/` | Land use (`lulc.asc`, 2 m — resampled to DEM grid) |
| `8-river/` | River centerlines (`river.shp`) — **部分接入**: bctype=3 only on grid-edge river cells |
| `9-hyd_station/` | Stage/discharge gauge (`hyd_station1.csv`) — **未接入** |
| `10-tidal_station/` | Tidal level (`tidal_station1.csv`) — **部分接入**: WSE when river BC cells exist |

### Sample extents

| Sample | DEM grid | Domain size | Nodes | Links |
|--------|----------|-------------|-------|-------|
| sample1 | 222×144 @ 10 m | 2.22 × 1.44 km | 246 | 237 |
| sample2 | 99×226 @ 10 m | 0.99 × 2.26 km | 212 | 204 |
| sample3 | 203×67 @ 10 m | 2.03 × 0.67 km | 147 | 142 |

## Data integration status (honest)

| Folder | Status | Notes |
|--------|--------|-------|
| `1-node` | **已接入** | SWMM nodes + ITZI coupling via `[COORDINATES]` |
| `2-link` | **已接入** | SWMM DYNWAVE conduits |
| `3-subcatchment` | **部分接入** | Shapefile → `[SUBCATCHMENTS]` in inp (dummy RainGage; no SWMM rain) |
| `4-dem` | **已接入** | Source ASC; nodata median fill only |
| `5-rainfall` | **部分接入** | 4 intensity CSVs + IDW; station XY inferred from subcatchment centroids |
| `6-boundary` | **已接入** | `mask.shp` blocks cells (n=100), restricts active domain |
| `7-lulc` | **已接入** | Manning n from resampled LULC |
| `8-river` | **部分接入** | bctype=3 only where river × grid edge × active; sample1/2=0, sample3=12 |
| `9-hyd_station` | **未接入** | CSV present; not loaded, not calibrated |
| `10-tidal_station` | **部分接入** | Drives WSE on river BC cells when they exist |

See `AUDIT_REPORT.md` for adversarial audit details.

## Assumptions & 待补充 Items

1. **Rainfall**: Multi-station IDW from `5-rainfall/雨强/多雨量站多曲线/*.csv` (4 stations); station XY inferred from subcatchment centroids (CSV lacks coords — 待补充 exact station locations). The `雨量/多雨量站` folder has 5 CSVs but is not used for intensity runs.
2. **LULC → Manning**: Classes 10–15 mapped to n=0.020–0.035; nodata + outside `6-boundary/mask.shp` blocked (n=100).
3. **DEM**: Source elevations unchanged on valid cells; nodata filled with valid median only (no +10 m raise, no ≥25 m building cut).
4. **River/tidal**: Domain-edge river corridor cells use tidal WSE (bctype=3) where river intersects **raster** boundary and cell is active; sample1/2 have 0 edge cells, sample3 has 12.
5. **Subcatchments**: Shapefile written to SWMM `[SUBCATCHMENTS]` with dummy RainGage; ITZI still applies all rain on surface.
6. **Hyd station**: Not loaded in simulation (未接入).
7. **Infiltration**: Disabled (`InfNull`).

## Requirements

- Python 3 with ITZI (`E:\Miniconda3\Lib\site-packages\itzi`)
- pyswmm 2.1.0
- numpy, matplotlib, pyshp (`shapefile`)

## How to Run

From `test_cases/20260603-UFIM/`:

```powershell
# 1) Build SWMM network from shapefiles (per sample)
python build_swmm_from_shapefiles.py sample1

# 2) Run paired simulations (surface-only + SWMM coupled)
python run_coupled_simulation.py --sample sample1
python run_coupled_simulation.py --all              # all three samples
python run_coupled_simulation.py --all --rebuild-swmm

# 3) Generate figures
python visualize_results.py --sample sample1
python visualize_results.py --all
```

## Outputs

Per sample (`sampleN/output/`):

- `coupled_results.npz` — DEM, final depths, time series, summary dict
- `depth_surface_final.asc` — final depth, surface-only scenario
- `depth_swmm_final.asc` — final depth, SWMM coupled scenario
- `network/drainage.inp` — SWMM input with `[COORDINATES]` for ITZI coupling

Case-level:

- `output/all_samples_summary.npz` — cross-sample summary (after `--all`)
- `visualization_output/` — PNG maps and time series

## Scripts

| Script | Purpose |
|--------|---------|
| `build_swmm_from_shapefiles.py` | Node/Link shp → SWMM `.inp` |
| `run_coupled_simulation.py` | ITZI 2D + SWMM bi-directional coupling |
| `visualize_results.py` | Depth maps with network overlay, time series, summary |
| `sample_utils.py` | Load DEM/LULC/rainfall/boundary/river, coordinate transforms |
| `verify_data_provenance.py` | Data folder + DEM/network provenance checks |
| `verify_physics.py` | ITZI SurfaceFlow + SWMM DYNWAVE physics checks |
| `generate_report.py` | Self-contained HTML report |

## Reference Patterns

- `test_cases/urban_drainage/run_coupled_simulation.py` — ITZI+SWMM coupling loop
- `test_cases/urban_drainage/create_swmm_network.py` — SWMM inp format with `[COORDINATES]`
- `test_cases/shenzhen_region1/run_full_domain_coupled.py` — full-domain coupled run

## Simulation Results (latest run)

| Sample | Rain peak (mm/h) | Surface peak (m) | SWMM peak (m) | Flooded cells (surf / SWMM) | Drained (m³) |
|--------|------------------|------------------|---------------|-----------------------------|--------------|
| sample1 | 263 | 7.66 | 7.57 | 14,259 / 10,978 | 2,023 |
| sample2 | 263 | 5.37 | 5.33 | 11,127 / 9,038 | 30,542 |
| sample3 | 263 | 4.03 | 3.28 | 6,293 / 5,310 | 3,999 |

SWMM coupling reduces peak depth and stored volume by draining surface water into the pipe network.
