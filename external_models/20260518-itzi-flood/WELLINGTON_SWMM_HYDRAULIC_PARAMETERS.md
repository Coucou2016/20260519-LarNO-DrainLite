# Wellington SWMM Hydraulic Parameter Audit

This gate checks whether the Wellington SWMM drainage network has enough real
hydraulic parameter evidence to support a strict reproduction.

It is separate from topology and coupling checks. A network can be connected and
still be unsuitable for official comparison if conduit size, invert level,
roughness, storage depth, outfall level, pump curves, or pump controls are
defaulted or estimated.

## Run

Windows:

```powershell
.\run_wellington_swmm_hydraulic_parameter_audit.ps1 -Strict
```

WSL/Linux:

```bash
bash run_wellington_swmm_hydraulic_parameter_audit.sh --strict
```

Default inputs:

- `wellington_real/model/wellington_drainage.inp`
- `wellington_real/reports/build_swmm_from_gis.json`
- `wellington_real/reports/swmm_pump_control_audit.json`

Output:

- `wellington_real/reports/swmm_hydraulic_parameter_audit.json`

## Verdict Meaning

- `PASS`: SWMM hydraulic parameters are internally complete and no default or
  estimated parameter evidence was found.
- `PASS_WITH_WARNINGS`: the file is usable, but there are warnings such as many
  identical roughness values or free outfalls that need source explanation.
- `PASS_EXPLORATORY`: the network can be used for exploratory modelling, but
  default or estimated parameter evidence exists.
- `FAIL`: the network is missing required hydraulic records, contains invalid
  values, or strict mode found default/estimated parameter evidence.

For the current Wellington objective, only `PASS` should be treated as eligible
for strict official reproduction. `PASS_EXPLORATORY` is not enough.

## What Is Checked

- Required SWMM sections: `JUNCTIONS` or `STORAGE`, `OUTFALLS`, `CONDUITS`,
  `XSECTIONS`.
- Every conduit has positive length and roughness.
- Every conduit has an `XSECTIONS` record with positive `geom1`.
- Junction elevations and max depths are numeric and positive where required.
- Outfall elevations are numeric.
- Pump networks are cross-checked against the pump/control audit.
- SWMM comments and GIS-to-SWMM build reports are scanned for terms such as
  `default`, `estimated`, `assumed`, `fallback`, `placeholder`, and `synthetic`.

## Why This Matters

The Wellington public GIS can provide real network geometry, but official-grade
reproduction also needs source-traceable hydraulic parameters. If the builder
had to invent a diameter, roughness, invert, storage depth, or pump curve, the
result can still be useful as a research/exploratory case, but it should not be
presented as the official Wellington model.
