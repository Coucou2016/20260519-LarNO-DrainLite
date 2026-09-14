#!/usr/bin/env python3
"""
Generate synthetic test data for ITZI flood simulation.
Creates: DEM, Manning's n friction, rainfall time series
Scenario: 500m x 500m urban area with 5m resolution (100x100 grid)
"""
import numpy as np
import os

# Configuration
GRID_SIZE = 100          # 100x100 cells
CELL_SIZE = 5.0          # 5m resolution
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(OUT_DIR, "input_data")
os.makedirs(DATA_DIR, exist_ok=True)

# ============================================================
# 1. Digital Elevation Model (DEM) - synthetic urban terrain
# ============================================================
x = np.linspace(0, (GRID_SIZE-1) * CELL_SIZE, GRID_SIZE)
y = np.linspace(0, (GRID_SIZE-1) * CELL_SIZE, GRID_SIZE)
xx, yy = np.meshgrid(x, y)

# Base slope: 2% from north to south
base_elev = 50.0 - yy * 0.02

# Add a depression (pond/basin) in the center
cx, cy = GRID_SIZE//2, GRID_SIZE//2
dist = np.sqrt((xx - cx*CELL_SIZE)**2 + (yy - cy*CELL_SIZE)**2)
depression = -1.5 * np.exp(-dist**2 / (2 * 30**2))

# Add some terrain roughness (small hills and depressions)
np.random.seed(42)
roughness = np.random.randn(GRID_SIZE, GRID_SIZE) * 0.15
# Smooth the roughness
from scipy.ndimage import gaussian_filter
roughness = gaussian_filter(roughness, sigma=2)

# Add a channel/stream from north to south through the depression
channel_dist = np.abs(xx - cx*CELL_SIZE)
channel = -1.0 * np.exp(-channel_dist**2 / (2 * 8**2))
channel = gaussian_filter(channel, sigma=(1, 3))

# Combine all terrain features
dem = base_elev + depression + roughness + channel
dem = dem.astype(np.float32)

# Save DEM
dem_file = os.path.join(DATA_DIR, "dem.asc")
with open(dem_file, 'w') as f:
    f.write(f"ncols         {GRID_SIZE}\n")
    f.write(f"nrows         {GRID_SIZE}\n")
    f.write(f"xllcorner     0.0\n")
    f.write(f"yllcorner     0.0\n")
    f.write(f"cellsize      {CELL_SIZE}\n")
    f.write(f"NODATA_value  -9999\n")
    np.savetxt(f, dem, fmt='%.3f')

print(f"DEM saved to {dem_file} (min={dem.min():.2f}, max={dem.max():.2f})")

# ============================================================
# 2. Manning's n friction coefficient
# ============================================================
mannings = np.full((GRID_SIZE, GRID_SIZE), 0.03, dtype=np.float32)
# Higher friction in "built-up areas" (sides)
built_up = np.abs(xx - cx*CELL_SIZE) > 100
mannings[built_up] = 0.06
# Lower friction along the channel
mannings[channel_dist < 15] = 0.02
# Very low friction in the depression (like a pond)
mannings[dist < 20] = 0.015

mannings_file = os.path.join(DATA_DIR, "mannings.asc")
with open(mannings_file, 'w') as f:
    f.write(f"ncols         {GRID_SIZE}\n")
    f.write(f"nrows         {GRID_SIZE}\n")
    f.write(f"xllcorner     0.0\n")
    f.write(f"yllcorner     0.0\n")
    f.write(f"cellsize      {CELL_SIZE}\n")
    f.write(f"NODATA_value  -9999\n")
    np.savetxt(f, mannings, fmt='%.4f')
print(f"Manning's n saved to {mannings_file}")

# ============================================================
# 3. Rainfall time series - 2-hour hyetograph
# ============================================================
# Design storm: 2-hour event with 50 mm total rainfall
# Peak at 45 minutes (Chicago design storm pattern)
duration_min = 120
dt_rain = 5  # 5-minute intervals
n_steps = duration_min // dt_rain

t_min = np.arange(0, duration_min, dt_rain)
# Normalized hyetograph (Chicago storm from Huff 1st quartile)
t_peak = 45  # peak at 45 min
intensity = np.zeros(n_steps)
for i, t in enumerate(t_min):
    if t < t_peak:
        intensity[i] = 50 * (t / t_peak) ** 0.5
    else:
        intensity[i] = 50 * ((duration_min - t) / (duration_min - t_peak)) ** 1.5

# Scale to total = 50 mm
total_rain = np.sum(intensity) * dt_rain / 60
intensity = intensity * 50 / total_rain
rain_mmh = intensity.astype(np.float32)

# Create rainfall maps (each map is a raster with uniform rainfall)
rain_dir = os.path.join(DATA_DIR, "rainfall")
os.makedirs(rain_dir, exist_ok=True)
rain_maps = []
for i, r in enumerate(rain_mmh):
    rain_map = np.full((GRID_SIZE, GRID_SIZE), r, dtype=np.float32)
    map_file = os.path.join(rain_dir, f"rain_{i:04d}.asc")
    with open(map_file, 'w') as f:
        f.write(f"ncols         {GRID_SIZE}\n")
        f.write(f"nrows         {GRID_SIZE}\n")
        f.write(f"xllcorner     0.0\n")
        f.write(f"yllcorner     0.0\n")
        f.write(f"cellsize      {CELL_SIZE}\n")
        f.write(f"NODATA_value  -9999\n")
        np.savetxt(f, rain_map, fmt='%.2f')
    rain_maps.append((map_file, t_min[i]))

print(f"Rainfall maps saved to {rain_dir}/ ({n_steps} maps, total={total_rain:.1f} mm)")

# ============================================================
# 4. Boundary conditions - open boundary at south edge
# ============================================================
bc_type = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int32)
bc_value = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
# Open boundary at the southern edge (last row)
bc_type[-1, :] = 1   # type 1 = open (fixed water level)
bc_value[-1, :] = dem[-1, :] - 0.1  # slightly below ground

bc_type_file = os.path.join(DATA_DIR, "bctype.asc")
with open(bc_type_file, 'w') as f:
    f.write(f"ncols         {GRID_SIZE}\n")
    f.write(f"nrows         {GRID_SIZE}\n")
    f.write(f"xllcorner     0.0\n")
    f.write(f"yllcorner     0.0\n")
    f.write(f"cellsize      {CELL_SIZE}\n")
    f.write(f"NODATA_value  -9999\n")
    np.savetxt(f, bc_type, fmt='%d')

bc_value_file = os.path.join(DATA_DIR, "bcvalue.asc")
with open(bc_value_file, 'w') as f:
    f.write(f"ncols         {GRID_SIZE}\n")
    f.write(f"nrows         {GRID_SIZE}\n")
    f.write(f"xllcorner     0.0\n")
    f.write(f"yllcorner     0.0\n")
    f.write(f"cellsize      {CELL_SIZE}\n")
    f.write(f"NODATA_value  -9999\n")
    np.savetxt(f, bc_value, fmt='%.3f')

print("Boundary condition files saved.")

# ============================================================
# 5. Save rainfall metadata for GRASS time series registration
# ============================================================
# Create a simple time info file
time_file = os.path.join(DATA_DIR, "rainfall", "time_info.txt")
with open(time_file, 'w') as f:
    f.write("# Rain maps with relative times (minutes from start)\n")
    for i, (_, t) in enumerate(rain_maps):
        f.write(f"rain_{i:04d}.asc\t{t:.0f}\n")
print(f"Rainfall time info saved to {time_file}")

print("\n=== Test data generation complete ===")
print(f"Data directory: {DATA_DIR}")
print(f"Grid: {GRID_SIZE}x{GRID_SIZE} @ {CELL_SIZE}m resolution")
print(f"Domain: {GRID_SIZE*CELL_SIZE}m x {GRID_SIZE*CELL_SIZE}m")
