# Wellington public ArcGIS downloads

Use this script to download the public Wellington Water/WCC ArcGIS layers used
by the real coupled case:

```bash
python benchmark_data/wellington_download_public_arcgis_sources.py
```

For a CBD-only extraction, pass an EPSG:2193 bounding box:

```bash
python benchmark_data/wellington_download_public_arcgis_sources.py --bbox xmin,ymin,xmax,ymax
```

If the official target has already been downloaded, derive a matching model
domain bbox from it:

```bash
python benchmark_data/wellington_derive_domain_bbox.py --buffer-m 250
```

This writes:

```text
wellington_real/metadata/wellington_cbd_domain_bbox.json
wellington_real/reports/domain_bbox.json
```

Use the reported `bbox_argument` for subsequent ArcGIS/LINZ building/DEM clip
commands.

The script downloads model input layers from the Wellington Water 3 Waters
service:

- Stormwater Pumpstation, layer 22.
- Stormwater Node, layer 23.
- Stormwater Pipe, layer 25.
- Stormwater Open Channel, layer 26.
- Stormwater Connection Pipe, layer 27.

It downloads official comparison targets from the WCC 100yr climate-change
freeboard flood-depth service:

- North CBD, layer 10.
- Southern CBD, layer 14.

Important rule:

```text
stormwater layers = model inputs
flood-depth layers = comparison targets only
```

The audit that enforces this separation is:

```bash
python benchmark_data/wellington_official_target_not_input_audit.py
```

The ArcGIS source manifest builder is:

```bash
python benchmark_data/wellington_source_manifest_from_arcgis_downloads.py
```

After DEM and buildings are registered, build and audit the full source
manifest:

```bash
python benchmark_data/wellington_full_source_manifest.py
python benchmark_data/wellington_manifest_role_audit.py
```

Outputs are written under:

```text
wellington_real/raw/
wellington_real/metadata/
wellington_real/reports/public_arcgis_downloads.json
wellington_real/reports/source_manifest.arcgis.json
wellington_real/reports/source_manifest.full.json
wellington_real/reports/source_manifest.json
wellington_real/reports/manifest_role_audit.json
wellington_real/reports/official_target_not_input_audit.json
```

This does not download the LINZ DEM. The DEM is still a separate required input
because the ArcGIS flood-depth polygons are not terrain and must not be used to
construct the surface model.
