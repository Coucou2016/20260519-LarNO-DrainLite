#!/usr/bin/env python3
"""Shared utilities for UFIM sample datasets (DEM, LULC, rainfall, boundaries, network)."""
from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import shapefile

CASE_DIR = os.path.dirname(os.path.abspath(__file__))

FOLDER_SPEC = [
    ("1-node", "Node.shp", "SWMM nodes + ITZI coupling coordinates"),
    ("2-link", "Link.shp", "SWMM conduits"),
    ("3-subcatchment", "Subcatchment.shp", "SWMM subcatchments"),
    ("4-dem", "dem_10m.asc", "ITZI DEM (source elevations, nodata mask only)"),
    ("5-rainfall", "*.csv", "Rainfall time series (intensity / depth)"),
    ("6-boundary", "mask.shp", "Domain active mask"),
    ("7-lulc", "lulc.asc", "Manning n mapping"),
    ("8-river", "river.shp", "River open-boundary polyline"),
    ("9-hyd_station", "hyd_station*.csv", "Validation reference"),
    ("10-tidal_station", "tidal_station*.csv", "Tidal / open-boundary stage"),
]

LULC_MANNING = {
    10: 0.020,
    12: 0.035,
    13: 0.030,
    14: 0.025,
    15: 0.025,
}


@dataclass
class GridMeta:
    ncols: int
    nrows: int
    xll: float
    yll: float
    cellsize: float
    nodata: float

    @property
    def xur(self) -> float:
        return self.xll + self.ncols * self.cellsize

    @property
    def yur(self) -> float:
        return self.yll + self.nrows * self.cellsize

    @property
    def width_m(self) -> float:
        return self.ncols * self.cellsize

    @property
    def height_m(self) -> float:
        return self.nrows * self.cellsize


@dataclass
class DataIntegration:
    sample: str
    folders: Dict[str, dict] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"sample": self.sample, "folders": self.folders}


def sample_dir(sample: str) -> str:
    return os.path.join(CASE_DIR, sample)


def read_asc(path: str) -> Tuple[GridMeta, np.ndarray]:
    with open(path, encoding="utf-8", errors="replace") as f:
        hdr: Dict[str, float] = {}
        for _ in range(6):
            parts = f.readline().split()
            key = parts[0].lower().replace("_value", "")
            hdr[key] = float(parts[1])
        data = np.loadtxt(f, dtype=np.float32)
    meta = GridMeta(
        ncols=int(hdr["ncols"]),
        nrows=int(hdr["nrows"]),
        xll=float(hdr["xllcorner"]),
        yll=float(hdr["yllcorner"]),
        cellsize=float(hdr["cellsize"]),
        nodata=float(hdr.get("nodata", -9999)),
    )
    return meta, data


def write_asc(path: str, meta: GridMeta, data: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"ncols         {meta.ncols}\n")
        f.write(f"nrows         {meta.nrows}\n")
        f.write(f"xllcorner     {meta.xll}\n")
        f.write(f"yllcorner     {meta.yll}\n")
        f.write(f"cellsize      {meta.cellsize}\n")
        f.write(f"NODATA_value  {meta.nodata}\n")
        np.savetxt(f, data, fmt="%.4f")


def proj_to_local(x: float, y: float, meta: GridMeta) -> Tuple[float, float]:
    return x - meta.xll, y - meta.yll


def local_to_rowcol(x_local: float, y_local: float, meta: GridMeta) -> Tuple[int, int]:
    col = int(x_local / meta.cellsize)
    row = meta.nrows - 1 - int(y_local / meta.cellsize)
    col = max(0, min(meta.ncols - 1, col))
    row = max(0, min(meta.nrows - 1, row))
    return row, col


def rowcol_to_local(row: int, col: int, meta: GridMeta) -> Tuple[float, float]:
    x = col * meta.cellsize + meta.cellsize * 0.5
    y = (meta.nrows - 1 - row) * meta.cellsize + meta.cellsize * 0.5
    return x, y


def resample_lulc_to_dem(lulc: np.ndarray, lmeta: GridMeta, dmeta: GridMeta) -> np.ndarray:
    out = np.full((dmeta.nrows, dmeta.ncols), lmeta.nodata, dtype=np.float32)
    for r in range(dmeta.nrows):
        y_local = (dmeta.nrows - 1 - r) * dmeta.cellsize + dmeta.cellsize * 0.5
        y_proj = dmeta.yll + y_local
        for c in range(dmeta.ncols):
            x_local = c * dmeta.cellsize + dmeta.cellsize * 0.5
            x_proj = dmeta.xll + x_local
            lr = lmeta.nrows - 1 - int((y_proj - lmeta.yll) / lmeta.cellsize)
            lc = int((x_proj - lmeta.xll) / lmeta.cellsize)
            if 0 <= lr < lmeta.nrows and 0 <= lc < lmeta.ncols:
                out[r, c] = lulc[lr, lc]
    return out


def load_dem(sample: str) -> Tuple[GridMeta, np.ndarray]:
    """Load source DEM; nodata -> NaN. No elevation edits."""
    path = os.path.join(sample_dir(sample), "4-dem", "dem_10m.asc")
    meta, dem = read_asc(path)
    dem = dem.astype(np.float32)
    dem[dem <= meta.nodata + 1] = np.nan
    return meta, dem


def dem_source_stats(sample: str) -> dict:
    meta, dem = load_dem(sample)
    valid = dem[~np.isnan(dem)]
    return {
        "path": os.path.join(sample, "4-dem", "dem_10m.asc"),
        "nrows": meta.nrows,
        "ncols": meta.ncols,
        "cellsize": meta.cellsize,
        "nodata": meta.nodata,
        "min": float(valid.min()) if valid.size else None,
        "max": float(valid.max()) if valid.size else None,
    }


def load_lulc_on_dem(sample: str, dmeta: GridMeta) -> np.ndarray:
    lulc_path = os.path.join(sample_dir(sample), "7-lulc", "lulc.asc")
    lmeta, lulc = read_asc(lulc_path)
    return resample_lulc_to_dem(lulc, lmeta, dmeta)


def _point_in_ring(x: float, y: float, ring: List[Tuple[float, float]]) -> bool:
    inside = False
    n = len(ring)
    if n < 3:
        return False
    j = n - 1
    for i in range(n):
        xi, yi = ring[i]
        xj, yj = ring[j]
        if ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / (yj - yi + 1e-30) + xi
        ):
            inside = not inside
        j = i
    return inside


def rasterize_domain_mask(sample: str, meta: GridMeta) -> np.ndarray:
    """True = inside computational domain (mask.shp)."""
    path = os.path.join(sample_dir(sample), "6-boundary", "mask")
    if not os.path.exists(path + ".shp"):
        return np.ones((meta.nrows, meta.ncols), dtype=bool)

    try:
        from matplotlib.path import Path as MplPath
    except ImportError:
        MplPath = None

    sf = shapefile.Reader(path)
    rings: List[List[Tuple[float, float]]] = []
    for shp in sf.shapes():
        if len(shp.points) >= 3:
            rings.append(shp.points)

    cols = np.arange(meta.ncols) * meta.cellsize + meta.cellsize * 0.5 + meta.xll
    rows = meta.yll + (meta.nrows - 1 - np.arange(meta.nrows)) * meta.cellsize + meta.cellsize * 0.5
    xx, yy = np.meshgrid(cols, rows)
    points = np.column_stack([xx.ravel(), yy.ravel()])

    inside = np.zeros(meta.nrows * meta.ncols, dtype=bool)
    if MplPath is not None:
        for ring in rings:
            inside |= MplPath(ring).contains_points(points)
    else:
        for i, (x, y) in enumerate(points):
            for ring in rings:
                if _point_in_ring(x, y, ring):
                    inside[i] = True
                    break
    return inside.reshape(meta.nrows, meta.ncols)


def load_subcatchment_centroids(sample: str, meta: GridMeta) -> List[Tuple[float, float]]:
    path = os.path.join(sample_dir(sample), "3-subcatchment", "Subcatchment")
    if not os.path.exists(path + ".shp"):
        return []
    sf = shapefile.Reader(path)
    centroids: List[Tuple[float, float]] = []
    for shp in sf.shapes():
        if not shp.points:
            continue
        xs = [p[0] for p in shp.points]
        ys = [p[1] for p in shp.points]
        cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
        centroids.append(proj_to_local(cx, cy, meta))
    return centroids


def _farthest_point_sample(
    points: List[Tuple[float, float]], k: int
) -> List[Tuple[float, float]]:
    if len(points) <= k:
        return points
    pts = np.asarray(points, dtype=np.float64)
    chosen = [int(np.argmax(pts[:, 0] + pts[:, 1]))]
    for _ in range(k - 1):
        dists = np.min(
            np.linalg.norm(pts - pts[chosen][:, None], axis=2), axis=0
        )
        chosen.append(int(np.argmax(dists)))
    return [tuple(pts[i]) for i in chosen]


def _list_rainfall_csvs(sample: str, prefer_multi: bool, use_intensity: bool) -> List[str]:
    rain_root = os.path.join(sample_dir(sample), "5-rainfall")
    candidates: List[str] = []
    for root, _, files in os.walk(rain_root):
        for fn in files:
            if not fn.endswith(".csv"):
                continue
            rel = os.path.relpath(os.path.join(root, fn), rain_root)
            if use_intensity and "雨强" not in rel and "intensity" not in rel.lower():
                continue
            if not use_intensity and "雨量" not in rel and "depth" not in rel.lower():
                continue
            candidates.append(os.path.join(root, fn))

    def score(p: str) -> Tuple[int, int, str]:
        rel = os.path.relpath(p, rain_root)
        multi = 0 if ("多" in rel or "multi" in rel.lower()) else 1
        single = 0 if ("单" in rel or "single" in rel.lower()) else 1
        if prefer_multi:
            return (multi, single, rel)
        return (single, multi, rel)

    candidates.sort(key=score)
    if not candidates:
        raise FileNotFoundError(f"No rainfall CSV in {rain_root}")

    if prefer_multi:
        folder = os.path.dirname(candidates[0])
        multi = sorted(
            (f for f in os.listdir(folder) if f.endswith(".csv")),
            key=lambda x: int(os.path.splitext(x)[0]) if x[0].isdigit() else x,
        )
        return [os.path.join(folder, f) for f in multi]
    return [candidates[0]]


def _read_csv_series(path: str) -> np.ndarray:
    values: List[float] = []
    with open(path, encoding="utf-8-sig", errors="replace") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) < 2:
                continue
            try:
                values.append(float(row[1]))
            except ValueError:
                continue
    if not values:
        raise ValueError(f"No rainfall values in {path}")
    return np.asarray(values, dtype=np.float32)


def load_rainfall_series(
    sample: str, prefer_multi: bool = True
) -> Tuple[np.ndarray, float, str]:
    """Single-station intensity (mm/h), 5-min steps."""
    series, dt, paths, _ = load_rainfall_spatial(sample, prefer_multi=prefer_multi)
    return series[:, 0], dt, paths[0]


def load_rainfall_spatial(
    sample: str, prefer_multi: bool = True
) -> Tuple[np.ndarray, float, List[str], List[Tuple[float, float]]]:
    """
    Multi-station rainfall intensities (mm/h).
    Returns (T, K), dt_s, source_paths, station_xy_local.
    Station coords inferred via spatial spread of subcatchment centroids
    when CSVs lack coordinates (documented in integration report).
    """
    paths = _list_rainfall_csvs(sample, prefer_multi=prefer_multi, use_intensity=True)
    series_list = [_read_csv_series(p) for p in paths]
    n = min(len(s) for s in series_list)
    stacked = np.column_stack([s[:n] for s in series_list])

    meta, _ = load_dem(sample)
    centroids = load_subcatchment_centroids(sample, meta)
    k = stacked.shape[1]
    if len(centroids) >= k:
        stations = _farthest_point_sample(centroids, k)
    else:
        # Fallback: spread stations across domain
        stations = [
            (meta.width_m * (i + 1) / (k + 1), meta.height_m * 0.5) for i in range(k)
        ]
    return stacked, 300.0, paths, stations


def interpolate_rainfall_idw(
    intensity_mm_h: np.ndarray,
    stations_xy: List[Tuple[float, float]],
    meta: GridMeta,
    active: np.ndarray,
    power: float = 2.0,
) -> np.ndarray:
    """Spatial rainfall field (mm/h) for one timestep."""
    H, W = active.shape
    field = np.zeros((H, W), dtype=np.float32)
    if intensity_mm_h.size == 1:
        field[active] = float(intensity_mm_h[0])
        return field

    st = np.asarray(stations_xy, dtype=np.float64)
    vals = np.asarray(intensity_mm_h, dtype=np.float64)
    rows, cols = np.where(active)
    for idx in range(len(rows)):
        r, c = rows[idx], cols[idx]
        x, y = rowcol_to_local(r, c, meta)
        d = np.linalg.norm(st - np.array([x, y]), axis=1)
        d = np.maximum(d, meta.cellsize * 0.5)
        w = 1.0 / np.power(d, power)
        field[r, c] = float(np.sum(w * vals) / np.sum(w))
    return field


def load_tidal_series(sample: str) -> Tuple[np.ndarray, float, str]:
    base = os.path.join(sample_dir(sample), "10-tidal_station")
    for fn in os.listdir(base):
        if fn.startswith("tidal") and fn.endswith(".csv"):
            path = os.path.join(base, fn)
            return _read_csv_series(path), 3600.0, path
    raise FileNotFoundError(f"No tidal CSV in {base}")


def load_hyd_station_series(sample: str) -> Tuple[np.ndarray, float, str]:
    base = os.path.join(sample_dir(sample), "9-hyd_station")
    for fn in os.listdir(base):
        if fn.startswith("hyd") and fn.endswith(".csv"):
            path = os.path.join(base, fn)
            return _read_csv_series(path), 300.0, path
    raise FileNotFoundError(f"No hyd station CSV in {base}")


def river_boundary_mask(
    sample: str, meta: GridMeta, max_dist_cells: float = 1.5
) -> np.ndarray:
    """Cells within max_dist_cells * cellsize of river polyline (subsampled segments)."""
    path = os.path.join(sample_dir(sample), "8-river", "river")
    mask = np.zeros((meta.nrows, meta.ncols), dtype=bool)
    if not os.path.exists(path + ".shp"):
        return mask

    sf = shapefile.Reader(path)
    max_d = max_dist_cells * meta.cellsize
    segments: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
    for shp in sf.shapes():
        pts = shp.points
        step = max(1, len(pts) // 500)
        for i in range(0, len(pts) - 1, step):
            p0 = proj_to_local(pts[i][0], pts[i][1], meta)
            p1 = proj_to_local(pts[min(i + step, len(pts) - 1)][0],
                                pts[min(i + step, len(pts) - 1)][1], meta)
            segments.append((p0, p1))

    if not segments:
        return mask

    xs = [s[0][0] for s in segments] + [s[1][0] for s in segments]
    ys = [s[0][1] for s in segments] + [s[1][1] for s in segments]
    pad = max_d * 2
    r0 = max(0, meta.nrows - 1 - int((max(ys) + pad) / meta.cellsize))
    r1 = min(meta.nrows, meta.nrows - int((min(ys) - pad) / meta.cellsize))
    c0 = max(0, int((min(xs) - pad) / meta.cellsize))
    c1 = min(meta.ncols, int((max(xs) + pad) / meta.cellsize) + 1)

    for r in range(r0, r1):
        for c in range(c0, c1):
            x, y = rowcol_to_local(r, c, meta)
            pt = np.array([x, y])
            for p0, p1 in segments:
                v = np.array(p1) - np.array(p0)
                w = pt - np.array(p0)
                t = np.clip(np.dot(w, v) / (np.dot(v, v) + 1e-12), 0, 1)
                proj = np.array(p0) + t * v
                if np.linalg.norm(pt - proj) <= max_d:
                    mask[r, c] = True
                    break
    return mask


def tidal_stage_at_time(tidal: np.ndarray, dt_tidal: float, t_s: float) -> float:
    idx = int(t_s / dt_tidal)
    idx = min(idx, len(tidal) - 1)
    return float(tidal[idx])


def setup_boundary_arrays(
    sample: str,
    meta: GridMeta,
    dem: np.ndarray,
    active: np.ndarray,
    bldg: np.ndarray,
    sim_duration_s: float,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    ITZI bctype / bcval: 0=default, 3=fixed WSE on river corridor.
    """
    bctype = np.zeros(dem.shape, dtype=np.int32)
    bcval = dem.copy().astype(np.float32)
    info: dict = {"river_cells": 0, "tidal_source": None}

    edge = np.zeros(dem.shape, dtype=bool)
    edge[0, :] = edge[-1, :] = edge[:, 0] = edge[:, -1] = True
    river_mask = river_boundary_mask(sample, meta) & active & edge
    if not river_mask.any():
        return bctype, bcval, info

    try:
        tidal, dt_t, tidal_path = load_tidal_series(sample)
        info["tidal_source"] = tidal_path
        wse_off = tidal_stage_at_time(tidal, dt_t, min(sim_duration_s * 0.5, len(tidal) * dt_t))
        bctype[river_mask] = 3
        bcval[river_mask] = dem[river_mask] + max(wse_off, 0.0)
        bctype[bldg] = 0
        info["river_cells"] = int(river_mask.sum())
        info["tidal_wse_offset_m"] = wse_off
    except FileNotFoundError:
        pass
    return bctype, bcval, info


def prepare_surface_fields(
    sample: str,
) -> Tuple[GridMeta, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Returns meta, dem_hydro, mannings, building_mask, active_mask, integration_info.
    DEM elevations from source ASC unchanged on valid cells; nodata -> median fill only.
    """
    meta, dem_src = load_dem(sample)
    lulc_on_dem = load_lulc_on_dem(sample, meta)
    inside = rasterize_domain_mask(sample, meta)

    nodata_mask = np.isnan(dem_src)
    bldg = nodata_mask | ~inside

    dem_hydro = dem_src.copy()
    valid = dem_hydro[~nodata_mask]
    fill_val = float(np.nanmedian(valid)) if valid.size else 0.0
    dem_hydro[nodata_mask] = fill_val

    mannings = np.full(dem_hydro.shape, 0.015, dtype=np.float32)
    for cls, n in LULC_MANNING.items():
        mannings[lulc_on_dem == cls] = n
    mannings[bldg] = 100.0

    active = ~bldg
    info = audit_data_integration(sample)
    return meta, dem_hydro, mannings, bldg, active, info


def audit_data_integration(sample: str) -> dict:
    """Inventory all numbered folders and integration status."""
    base = sample_dir(sample)
    folders: Dict[str, dict] = {}
    for folder, pattern, purpose in FOLDER_SPEC:
        path = os.path.join(base, folder)
        status = "missing"
        detail = ""
        if os.path.isdir(path):
            files = os.listdir(path)
            status = "present"
            if folder == "5-rainfall":
                multi = _list_rainfall_csvs(sample, True, True)
                single = _list_rainfall_csvs(sample, False, True)
                detail = f"multi={len(multi)} files, single={os.path.basename(single[0])}"
            elif folder.startswith(("1-", "2-", "3-", "6-", "8-")):
                shp = [f for f in files if f.endswith(".shp")]
                detail = ", ".join(shp) if shp else "no shp"
            else:
                detail = ", ".join(files[:5])
        folders[folder] = {
            "pattern": pattern,
            "purpose": purpose,
            "status": status,
            "detail": detail,
            "integrated": _integration_status(folder, path, sample),
        }
    return {"sample": sample, "folders": folders}


def _integration_status(folder: str, path: str, sample: str) -> str:
    if not os.path.isdir(path):
        return "未接入（目录缺失）"
    if folder == "1-node":
        return "已接入 SWMM+ITZI 耦合"
    if folder == "2-link":
        return "已接入 SWMM DYNWAVE"
    if folder == "3-subcatchment":
        return "部分接入（shapefile→SWMM [SUBCATCHMENTS]；dummy RainGage，雨仍施于 ITZI 地表）"
    if folder == "4-dem":
        return "已接入 ITZI（源高程，仅 nodata 中值填充）"
    if folder == "5-rainfall":
        return "部分接入（4 站雨强 CSV + IDW；站点 XY 由子汇水区质心推断，非实测坐标）"
    if folder == "6-boundary":
        return "已接入域内 mask（阻塞区 n=100）"
    if folder == "7-lulc":
        return "已接入 Manning 映射"
    if folder == "8-river":
        return "部分接入（river.shp→bctype=3；仅栅格边界×河道；见 simulation.river_bc_cells）"
    if folder == "9-hyd_station":
        return "未接入（CSV 存在；未加载、未率定、未对比）"
    if folder == "10-tidal_station":
        return "部分接入（潮位 CSV→WSE；仅当 river_bc_cells>0 时生效）"
    return "待补充"


def save_integration_log(sample: str, info: dict, extra: dict | None = None) -> str:
    out = os.path.join(sample_dir(sample), "output", "data_integration.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    payload = {**info, **(extra or {})}
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return out


def load_network_for_plot(sample: str) -> Tuple[List[dict], List[dict], GridMeta]:
    """Nodes and links in row/col for visualization."""
    from build_swmm_from_shapefiles import read_network

    nodes, links, meta = read_network(sample)
    for n in nodes:
        n["row"], n["col"] = local_to_rowcol(n["x_local"], n["y_local"], meta)
    link_lines: List[dict] = []
    node_by_id = {n["id"]: n for n in nodes}
    raw_to_id = {n["raw_id"]: n["id"] for n in nodes}
    sf_l = shapefile.Reader(os.path.join(sample_dir(sample), "2-link", "Link"))
    for rec in sf_l.iterShapeRecords():
        attrs = rec.record.as_dict() if hasattr(rec.record, "as_dict") else {}
        us = str(attrs.get("Us_Node", ""))
        ds = str(attrs.get("Ds_Node", ""))
        pts = rec.shape.points
        if len(pts) >= 2:
            line_rc = []
            for x, y in pts:
                xl, yl = proj_to_local(x, y, meta)
                line_rc.append(local_to_rowcol(xl, yl, meta))
        else:
            nu = node_by_id.get(raw_to_id.get(us, ""), None)
            nd = node_by_id.get(raw_to_id.get(ds, ""), None)
            if nu and nd:
                line_rc = [(nu["row"], nu["col"]), (nd["row"], nd["col"])]
            else:
                continue
        link_lines.append({"points": line_rc})
    return nodes, link_lines, meta


def redistribute_building_rain(
    rain_mm_h: np.ndarray, bldg: np.ndarray, dt_s: float | None = None
) -> np.ndarray:
    """
    Convert spatial rainfall intensity (mm/h) to ITZI rain array (m/s).

    ITZI hydrology expects ``rain`` in m/s (see itzi.flow.apply_hydrology). The
    standard TimedArray path divides mm/h by (1000 * 3600), not by the rainfall
    step length — dt_s is only used for building redistribution bookkeeping.
    """
    del dt_s  # rainfall step length; not used in mm/h → m/s conversion
    active = ~bldg
    n_active = int(active.sum())
    field = np.zeros_like(rain_mm_h, dtype=np.float32)
    if n_active == 0:
        return field
    extra = float(np.sum(rain_mm_h[bldg])) / n_active
    field[active] = (rain_mm_h[active] + extra) / 1000.0 / 3600.0
    return field
