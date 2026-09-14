# Wellington LINZ terrain and buildings

The Wellington coupled model must use real terrain. The public target flood
polygons are not terrain and must never be used to build the surface model.

## DEM

The required DEM source is:

```text
https://data.linz.govt.nz/layer/105023-wellington-lidar-1m-dem-2019-2020/
```

Because this is a large raster, it may be downloaded manually from the LINZ Data
Service or by a separate LDS/API workflow. After the file exists locally, register
it with:

```bash
python benchmark_data/wellington_register_linz_dem.py --dem path/to/wellington_dem.tif --copy-to-raw
```

This writes:

```text
wellington_real/reports/linz_dem_registration.json
wellington_real/metadata/linz_wellington_dem_source.json
wellington_real/raw/wellington_linz_dem.tif
```

The registration records source URL, checksum, declared vertical datum, expected
EPSG:2193 CRS, expected 1 m resolution, and raster metadata when `rasterio` is
available.

## Buildings

Building outlines are optional for a first hydraulic run but expected for a
defensible urban CBD representation. The public source is LINZ NZ Building
Outlines:

```text
https://data.linz.govt.nz/layer/101290-nz-building-outlines/
```

Download a Wellington-domain clip through LDS WFS with:

```bash
set LINZ_API_KEY=your_linz_key
python benchmark_data/linz_lds_capabilities_probe.py --layer-id 101290
python benchmark_data/wellington_download_linz_buildings_wfs.py --bbox xmin,ymin,xmax,ymax
```

If LDS `GetCapabilities` reports a prefixed type name, pass it explicitly:

```bash
python benchmark_data/wellington_download_linz_buildings_wfs.py --bbox xmin,ymin,xmax,ymax --type-name data.linz.govt.nz:layer-101290
```

The script writes:

```text
wellington_real/raw/linz_building_outlines.geojson
wellington_real/reports/linz_buildings_download.json
wellington_real/metadata/linz_buildings_source.json
```

## Terrain gate

Run:

```bash
python benchmark_data/wellington_terrain_source_gate.py
```

It fails if the LINZ DEM is not registered. Building outlines are reported as an
advisory item until the chosen building treatment is documented.

Then run the cross-source CRS/datum audit:

```bash
python benchmark_data/wellington_crs_datum_consistency_audit.py
python benchmark_data/wellington_domain_coverage_audit.py
```

It checks that DEM, ArcGIS stormwater/target layers, building outlines, and
vertical-datum evidence are consistent with NZTM / EPSG:2193 and NZVD2016.

## Building treatment

Copy and complete the building-treatment evidence template:

```bash
copy benchmark_data\wellington_building_treatment.template.json wellington_real\metadata\wellington_building_treatment.json
```

Then run:

```bash
python benchmark_data/wellington_building_treatment_audit.py
python benchmark_data/wellington_building_integration_audit.py
```

This records whether buildings are represented as blocked cells, a raised DEM,
roughness zones, or explicitly ignored for a first exploratory run. Exact
official reproduction should not ignore buildings unless the official model did
the same.
