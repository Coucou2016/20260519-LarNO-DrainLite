# Current Wellington ITZI/SWMM progress

Date: 2026-05-29

## What is implemented locally

The repository now contains a guarded Wellington real-data workflow for a
coupled ITZI + SWMM case. The workflow is built around public/official source
data rather than hand-drawn pipes or synthetic terrain:

- Wellington Water/WCC stormwater network ArcGIS REST service.
- Wellington Water modelled flood-depth ArcGIS REST service for the 100 yr ARI
  climate-change/freeboard products.
- LINZ Wellington City LiDAR 1 m DEM source.
- SWMM dynamic-wave integrity checks.
- ITZI drainage coupling checks.
- Surface-to-drainage exchange audit.
- Source provenance, vertical datum, rainfall forcing, outfall/tailwater, and
  official comparison gates.

The new Level 1 wrapper is:

```bash
python benchmark_data/wellington_level1_real_pipeline.py --keep-going
```

It writes:

```text
wellington_real/reports/level1_real_pipeline.json
```

The raw downloaded-input gate is:

```bash
python benchmark_data/wellington_raw_input_gate.py
```

It writes:

```text
wellington_real/reports/raw_input_gate.json
```

## What is not yet honestly complete

The full Wellington case has not been completed end-to-end in the current
Windows sandbox session because local command execution is currently failing
before commands start:

```text
windows sandbox: setup refresh failed with status exit code: 1
```

That means I have not successfully downloaded the large LINZ Wellington DEM or
run a new ITZI simulation during this interrupted environment state. The code
and runbooks are prepared so the large Wellington download and processing can
continue as soon as local shell/WSL execution is available again.

## How ITZI is compiled/run on Windows

The current project runbooks prefer WSL Ubuntu or Docker Linux for the heavy
GRASS/GDAL/ITZI batch workflow. That is an engineering choice for reproducibility,
not a claim that ITZI cannot run on Windows.

The ITZI installation documentation now states that Windows 11 has been tested
and that the Windows installation steps are the same as GNU/Linux after
installing `uv`. ITZI still depends on GRASS GIS 8.4 or above.

The intended robust Windows setup for this project is:

1. Windows stores the project files, for example:

   ```text
   E:\Projects\20260518-itzi-flood
   ```

2. WSL Ubuntu or a Linux Docker image mounts that same folder.

3. Inside the Linux environment, GRASS GIS, ITZI, Python geospatial libraries,
   SWMM/PySWMM, GDAL, and raster tools are installed.

4. The actual ITZI model run happens inside Linux/GRASS, while outputs are
   written back to the Windows project directory.

For the WSL/Docker path, the important distinction is:

- Host operating system: Windows.
- Hydraulic/GIS runtime: Linux user space through WSL or Docker.
- Shared model files and reports: the Windows project folder.

This remains the practical path for large GIS preprocessing because GRASS/GDAL
batch behavior is usually easier to reproduce in Linux environments.

## Current data honesty status

There are two levels of Wellington work:

- Level 1: real-data coupled local run. This needs the public DEM/network/depth
  downloads and can proceed with transparent assumptions.
- Level 3: official Wellington Water/WCC reproduction. This additionally needs
  official rainfall hyetograph, sea/tailwater or outfall boundary, vertical
  datum handling, freeboard/scenario definition, and acceptance criteria.

The pipeline is intentionally designed not to silently promote Level 1 to Level
3. If official forcing and boundary evidence is missing, the final official gate
must fail.
