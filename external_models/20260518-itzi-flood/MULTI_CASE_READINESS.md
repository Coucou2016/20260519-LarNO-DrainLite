# Multi-case readiness

Additional candidates are tracked in:

```text
benchmark_data/multi_case_reproduction_matrix.json
```

Audit them with:

```bash
python benchmark_data/multi_case_readiness_audit.py
```

The audit intentionally keeps cases at low readiness until the evidence supports
the next level:

- Level 0: at least one relevant public source found.
- Level 1: verified DEM source plus verified public stormwater network source.
- Level 2: Level 1 plus official/public flood target source.
- Level 3: Level 2 plus official rainfall, outfall/tailwater, datum/freeboard,
  scenario metadata, and acceptance metrics.

Probe candidate ArcGIS services with:

```bash
python benchmark_data/generic_arcgis_service_probe.py --case-id nz_hamilton_city --url "https://services1.arcgis.com/R6s0QqCMQdwKY6yp/ArcGIS/rest/services/Stormwater%20Dataset%20-%20Hamilton%20City%20Council/FeatureServer" --sample-layers
```

This writes probe reports under:

```text
wellington_real/reports/candidate_probes/
```

Current practical priority:

1. Wellington CBD: strongest Level 2 candidate.
2. Christchurch: promising, but service/layer/export details need probing.
3. Hamilton: public stormwater FeatureServer identified; flood target endpoint
   and DEM still need confirmation.
4. Auckland: public flood target FeatureServer identified; stormwater network
   export endpoint and DEM still need confirmation.
5. UK EA: useful 2D benchmark archive, not a pipe-coupled SWMM benchmark unless
   suitable network data is found.
