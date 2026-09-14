# Fixed ITZI-SWMM Coupling Workflow

This file defines the working meaning of "ITZI 和 SWMM 这样模拟" for the
LarNO drainage extension study.

## Scope

Use this workflow when no surveyed drainage network is available and the
drainage network is conceptually generated from the adjusted road network.

This workflow is:

- ITZI 2D dynamic surface-flow simulation using `itzi.surfaceflow.SurfaceFlowSimulation`.
- SWMM 1D dynamic-wave pipe-network routing using `pyswmm`.
- Native ITZI-SWMM exchange using `itzi.drainage.DrainageSimulation` and
  `apply_coupling_to_nodes()`.
- Road-aligned conceptual drainage network generated from the current adjusted
  OSM/road-derived network.

This workflow is not:

- A depression-filling/static equilibrium model.
- A direct `pipe/sink` or orifice-sink shortcut.
- A standalone SWMM-only model.
- A claim that a real surveyed municipal drainage network has been reproduced.

## Fixed Network Construction

Run:

```powershell
python extended_study\build_component_outfall_swmm_network.py
```

The script writes:

```text
external_models\20260518-itzi-flood\test_cases\shenzhen_region1\input_data\networks\swmm_connected_sub.inp
```

Fixed assumptions:

- Source network: current road-derived `osm_merged_network.npz`.
- Subregion: same 20 m Shenzhen Region1 window used by the ITZI comparison.
- Components: retain connected road-network components with at least 3 nodes.
- Outfalls: add one synthetic outfall per retained connected component.
- Connectivity: every retained junction must have a path to an outfall.
- Isolated nodes and broken conduits are removed.
- Pipe invert: ground elevation minus 2 m conceptual cover depth.
- Pipe offset: 0 m.
- Surface coupling: all retained road-network junctions have coordinates and
  are coupled to the 2D surface as conceptual manhole/inlet exchange points.
- SWMM routing: dynamic wave, 2 s routing step, variable step enabled, maximum
  20 trials.

Current audit target:

```text
retained_junctions = 1169
retained_outfalls = 12
retained_conduits = 2887
coupled_surface_junctions = 1169
disconnected_junctions = 0
zero_degree_junctions = 0
pyswmm_validation = ok
```

## Standalone SWMM Gate

Run:

```powershell
python extended_study\run_connected_swmm_validation.py
```

Acceptance:

- `continuity_error_pct = 0.0` for the no-external-inflow standalone network
  check.
- `nonconverging_steps_pct = 0.0` for the no-external-inflow standalone network
  check.

This gate only proves the pipe topology and SWMM file are valid. It does not
prove that every coupled rainfall event has perfect dynamic-wave stability.

## Coupled Event Run

Run events one at a time:

```powershell
python extended_study\run_connected_itzi_swmm_comparison.py --events event1
python extended_study\run_connected_itzi_swmm_comparison.py --events event20
```

One-event-per-process is intentional on Windows because the SWMM native engine
can fail when multiple PySWMM simulations are created in the same Python
process.

The comparison script runs both scenarios:

- `surface-only`: ITZI dynamic surface flow, no SWMM.
- `connected ITZI-SWMM`: the same ITZI surface model plus native ITZI-SWMM
  coupling through `DrainageSimulation`.

## Rainfall Timing Convention

The rainfall array is interpreted as depth per 5-minute frame, not as mm/h.
For a 72-frame event, frame 0 must be applied from `t = 0` to `t = 300 s`,
and frame 71 must be integrated from `t = 21300 s` to `t = 21600 s`.

The copied runner was corrected so that:

```python
next_rain = 0.0
while t < duration_s:
    ...
```

This avoids delaying all rainfall by one 5-minute interval and avoids stopping
the simulation before the final rainfall frame is fully integrated. Existing
result files created before this correction should be treated as pre-fix
diagnostic outputs until the target events are rerun.

## Hydrograph Timing Diagnostic

Run:

```powershell
python extended_study\diagnose_hydrograph_timing.py
```

This diagnostic compares rainfall timing, maximum water depth, surface-water
volume, and flooded area among `MIKE reference`, `ITZI surface-only`, and
`ITZI-SWMM connected`.

Current diagnostic finding:

- MIKE reference generally has a clear falling limb: event-wise final/peak
  surface-volume ratios average about 0.52.
- ITZI surface-only has no meaningful falling limb: the surface volume peaks at
  the final frame in all eight events, with final/peak ratio approximately 1.0.
- ITZI-SWMM connected improves storage reduction but remains much weaker than
  MIKE, with final/peak surface-volume ratio about 0.96 on average.

Therefore, hydrograph timing should be treated as a calibration target before
using the current ITZI-SWMM labels as strict physical truth. The main candidate
causes are absence of calibrated infiltration/loss terms, no explicit 2D open
boundary outflow, conceptual rather than surveyed drainage capacity, and the
pre-fix rainfall timing issue.

## Boundary And Building-Rainfall Notes

ITZI supports open 2D surface boundaries through the `bctype` raster:

- `bctype = 0` or `1`: closed boundary.
- `bctype = 2`: open boundary.
- `bctype = 3`: fixed water-surface-elevation boundary using `bcval`.

The current copied runner does not set `bctype`, so the default is closed
boundary. This is consistent with a strict closed-domain test, but it cannot
represent runoff leaving the 2D raster through river/channel edges unless those
outlets are represented by SWMM outfalls or another explicit loss/outflow term.

The current runner also redistributes rainfall falling on building cells to
non-building active cells. In the present 20 m subregion, buildings occupy
about 22.13% of cells, which increases active-cell rainfall by about 28.4%.
This assumption may be reasonable as an instantaneous roof-runoff
approximation, but it is not necessarily identical to the MIKE reference setup.
It should be tested against at least one no-redistribution scenario before
claiming close hydrograph timing agreement.

## Infiltration / Loss Update

The copied runner now supports a constant ITZI infiltration model through:

```powershell
python extended_study\run_connected_itzi_swmm_comparison.py --events event68 --infiltration-mmh 5
```

Implementation details:

- `infiltration_mmh = 0` keeps the previous `InfNull` behavior.
- `infiltration_mmh > 0` uses ITZI `InfConstantRate`.
- The rate is applied only to non-building active cells.
- Cumulative infiltration is recorded as `infiltrated_m3`.
- Outputs with infiltration are written to a separate directory such as
  `extended_study\output\connected_itzi_swmm_inf_5mmh`, so previous no-loss
  results are not overwritten.

Pilot result for `event68` with `5 mm/h`:

- Hydrograph timing improves: maximum-depth peak shifts from about 5.92 h
  without infiltration to about 3.42 h with infiltration, close to MIKE's
  3.17 h.
- The falling limb appears in both maximum-depth and volume curves.
- The chosen 5 mm/h rate is too strong for final use: peak depth and surface
  water volume are underpredicted relative to MIKE.

Therefore, infiltration should be calibrated rather than fixed directly at
5 mm/h. Recommended next sensitivity values are `1, 2, 3, 4, 5 mm/h`, first on
`event68`, then on the full eight-event set once the rate range is narrowed.

Outputs:

```text
extended_study\output\connected_itzi_swmm\
```

Key files:

- `connected_itzi_swmm_metrics.csv`
- `connected_itzi_swmm_metrics.json`
- `event*/event*_connected_itzi_swmm.npz`
- `event*_swmm_connected.rpt`
- `figures/event*_connected_itzi_swmm_timeseries.png`
- `figures/event*_connected_itzi_swmm_spatial.png`

The event `.npz` files use these fixed array keys:

- `h_ref`: MIKE reference time series, shape `72 x 200 x 280`, unit m.
- `h_surf`: ITZI surface-only time series, shape `72 x 200 x 280`, unit m.
- `h_swmm_connected`: ITZI-SWMM native coupled time series, shape
  `72 x 200 x 280`, unit m.
- `h_surf_final`: final surface-only water depth, shape `200 x 280`, unit m.
- `h_swmm_connected_final`: final coupled water depth, shape `200 x 280`, unit m.

## Coupling Flow Sign

ITZI defines coupling-flow sign as follows:

- negative flow: water leaves the 2D surface and enters the drainage network.
- positive flow: water leaves the drainage network and returns to the 2D
  surface.

Therefore the cumulative drainage-volume diagnostic must use:

```python
drained_m3 += max(-q, 0) * dt
```

The copied runner has been corrected accordingly.

## Current Validation Status

Final fixed-network runs completed:

| Event | Surface peak (m) | ITZI-SWMM peak (m) | Peak reduction (mm) | Final surface-volume reduction (m3) | SWMM continuity error (%) | Status |
|---|---:|---:|---:|---:|---:|---|
| event1 | 2.234 | 2.103 | 130.929 | 128113 | -0.515 | accepted for labels |
| event20 | 1.160 | 1.103 | 56.737 | 74993 | -7.608 | usable for diagnostic figures; flag as SWMM stability warning |

Interpretation:

- The repaired network produces a clear drainage effect in the native
  ITZI-SWMM coupled model.
- `event1` passes the practical water-balance gate.
- `event20` demonstrates drainage effect but should be marked with a hydraulic
  stability warning because the SWMM report continuity error exceeds the
  preferred 2% threshold and dynamic-wave nonconvergence is high.

## Fixed Wording For Reports

Use:

> ITZI-SWMM native coupled simulation with a road-aligned conceptual drainage
> network. The 2D surface is solved by ITZI, the 1D pipe network is solved by
> SWMM dynamic wave through PySWMM, and surface-pipe exchange is handled by
> ITZI's `DrainageSimulation.apply_coupling_to_nodes()` using conceptual
> manhole/inlet nodes derived from the road network.

Do not use:

> real surveyed sewer-network reference

Do not use:

> standalone SWMM label

Do not use:

> pipe/sink coupled model
