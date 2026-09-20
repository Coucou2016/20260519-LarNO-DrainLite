# Audited execution environment

The V3 physical labels and V4 residual experiments were produced on 64-bit
Windows with Python 3.13.12. Package versions are fixed in
`requirements-lock.txt`; the coupling entry point additionally asserts Itzi
25.4, PySWMM 2.1.0 and swmm-toolkit 0.17.0 at runtime.

The coupled solver is sensitive to native-library versions. Reproduction of the
published physical arrays therefore requires the exact versions above. The
artifact verifier does not open the native solver and can be run on another
platform after all Git LFS objects have been materialised.

Install the review environment with:

```bash
python -m pip install -r requirements-lock.txt
```

The recorded repository does not depend on a machine-specific Miniconda path.
Legacy GRASS GIS demonstrations under `external_models` are outside the
canonical Shenzhen V3/V4 path and may require a separately configured
`GRASS_BASE` installation.
