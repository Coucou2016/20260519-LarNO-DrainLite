# Third-party notices

This research snapshot combines local additions with upstream software and
reference data. Inclusion in one Git repository does not replace or broaden
the original licenses.

## LarNO

- Repository: https://github.com/holmescao/LarNO
- Dataset page: https://holmescao.github.io/datasets/LarNO
- Article DOI: https://doi.org/10.1016/j.jhydrol.2026.135686

LarNO source, checkpoint material, rainfall arrays, and MIKE reference arrays
must be used under the terms supplied by their authors and publishers. The
upstream README states that the project is MIT licensed, but the referenced
root licence file was absent at upstream commit
`9235736cbb07ef18ea79e68ab14e1667d83260b0` when checked on 2026-09-15. The
licence status of each dataset/checkpoint artifact should therefore be
confirmed with the LarNO authors before a further redistribution.

## Itzi

The installed Itzï 25.4 runtime is used as the two-dimensional surface solver
and as the host of the native drainage exchange path. Itzï declares
GPL-2.0-or-later. The copied project workflow is not a vendored copy of the
complete Itzï package and is not relicensed by this repository.

## EPA SWMM and PySWMM

SWMM 5.2.4 supplies the one-dimensional dynamic-wave sewer solver; PySWMM
2.1.0 provides the Python interface used by the coupling workflow. PySWMM is
BSD-2-Clause. EPA SWMM remains subject to its upstream EPA distribution terms.

## MIKE reference data

MIKE depth fields are used solely as external plausibility references. MIKE
software itself is not included. The repository does not imply ownership of
the underlying commercial model.

## OpenStreetMap-derived road alignment

The conceptual pipe alignment was derived from road information and must retain
the applicable OpenStreetMap attribution and Open Database License obligations
where the underlying geometries are redistributed.

Required attribution: `© OpenStreetMap contributors`,
<https://www.openstreetmap.org/copyright>, Open Database License 1.0. The exact
download timestamp was not recoverable from the retained artifacts and is
therefore recorded as unavailable rather than inferred.

## Reference article

The user-supplied PDF and Markdown copy of the LarNO paper are intentionally
excluded. Only the DOI and bibliographic references are retained.

See `LICENSE_MAP.md` for path-level scope and unresolved permission boundaries.
