# Wellington GIS to SWMM conversion

The real coupled workflow needs an explicit, traceable conversion from public
Wellington stormwater GIS assets to a SWMM dynamic-wave `.inp` file.

The converter is:

```bash
python benchmark_data/wellington_build_swmm_from_gis.py
```

It reads canonical downloaded GeoJSON files under:

```text
wellington_real/raw/
```

and writes:

```text
wellington_real/swmm/wellington_cbd_from_gis.inp
wellington_real/reports/build_swmm_from_gis.json
```

Before running it, copy and confirm the field map:

```bash
copy benchmark_data\wellington_swmm_field_map.template.json wellington_real\metadata\wellington_swmm_field_map.json
```

Then edit `wellington_real/metadata/wellington_swmm_field_map.json` to match the
actual ArcGIS metadata fields downloaded from Wellington Water.

Generate a field-discovery report first:

```bash
python benchmark_data/wellington_arcgis_field_discovery.py
```

It writes:

```text
wellington_real/reports/arcgis_field_discovery.json
```

Use that report to avoid guessing which ArcGIS attributes represent inverts,
diameters, node IDs, upstream/downstream IDs, and outfall types.

## Strict mode

Default mode is strict. It fails if the real GIS data and confirmed field map do
not provide enough evidence for:

- node identity and coordinates;
- node invert or level evidence;
- pipe identity;
- upstream/downstream node relation or geometry-to-node snapping;
- pipe diameter or cross-section dimensions;
- outfall/outlet nodes;
- source layer/ObjectID or asset provenance.

This is deliberate. A strict failure means the public data is not yet sufficient
for a defensible official reproduction.

## Exploratory mode

There is an explicit escape hatch:

```bash
python benchmark_data/wellington_build_swmm_from_gis.py --allow-estimated-hydraulics
```

That mode may write a runnable exploratory SWMM file, but the report verdict is
`PASS_EXPLORATORY`. It must not be used to claim an exact official reproduction.

## Provenance

The generated `.inp` includes comments such as:

```text
;; source_layer=Stormwater Pipe source_object=...
```

The provenance audit checks these markers:

```bash
python benchmark_data/wellington_swmm_provenance_audit.py
python benchmark_data/wellington_swmm_topology_audit.py
python benchmark_data/wellington_swmm_inflow_coupling_audit.py
python benchmark_data/wellington_swmm_pump_control_audit.py
python benchmark_data/wellington_swmm_hydraulic_parameter_audit.py
```

The inflow audit prevents an empty dynamic-wave network from being treated as a
valid coupled run. SWMM must have rainfall-runoff, direct inflows, RDII/DWF, or
verified ITZI surface-drainage exchange depending on the final coupling design.

The pump/control audit checks whether downloaded pumpstation assets require
SWMM `[PUMPS]`, pump curves, and controls/rules.

The hydraulic-parameter audit checks junction depths, conduit lengths,
roughness, cross-sections, and flags exploratory/default hydraulic markers.
