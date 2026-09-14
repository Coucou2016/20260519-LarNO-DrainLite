#!/usr/bin/env python3
"""Analyze the Shenzhen Futian DEM to understand terrain and identify road networks."""
import numpy as np
import os, sys

DEM_PATH = r"e:\Projects\20260519-LarNO\LarNO-main\benchmark\urbanflood\geodata\region1_20m\dem.npy"

def main():
    dem = np.load(DEM_PATH, allow_pickle=True)
    print(f"DEM shape: {dem.shape} (H x W)")
    print(f"Resolution: 20m → domain = {dem.shape[0]*20/1000:.1f}km x {dem.shape[1]*20/1000:.1f}km")
    print(f"Elevation range: {dem.min():.2f}m - {dem.max():.2f}m")
    print(f"Mean elevation: {dem.mean():.2f}m")

    # Mask: building pixels (wall_height=50m)
    building_mask = dem >= 49.9
    terrain_mask = ~building_mask
    valid_dem = dem[terrain_mask]

    print(f"\nBuilding pixels (wall_height=50): {building_mask.sum():,} / {building_mask.size:,} ({100*building_mask.sum()/building_mask.size:.1f}%)")
    print(f"Valid terrain: min={valid_dem.min():.2f}m, max={valid_dem.max():.2f}m, mean={valid_dem.mean():.2f}m")

    # Compute slope (gradient)
    dy, dx = np.gradient(dem, 20.0)  # gradient per 20m cell
    slope = np.sqrt(dx**2 + dy**2)
    valid_slope = slope[terrain_mask]
    print(f"\nTerrain slope: min={valid_slope.min():.4f}, max={valid_slope.max():.4f}, mean={valid_slope.mean():.4f}")
    print(f"Slope percentiles: 10%={np.percentile(valid_slope,10):.4f}, 50%={np.percentile(valid_slope,50):.4f}, 90%={np.percentile(valid_slope,90):.4f}")

    # Identify likely road corridors (low slope, low elevation variability)
    # Roads typically follow valleys / low-gradient paths
    aspect = np.arctan2(dy, dx)

    # Find main flow paths (N→S due to terrain slope)
    print(f"\nTerrain gradient dx (E→W): mean={dx[terrain_mask].mean():.4f}, std={dx[terrain_mask].std():.4f}")
    print(f"Terrain gradient dy (S→N): mean={dy[terrain_mask].mean():.4f}, std={dy[terrain_mask].std():.4f}")

    # Dominant slope direction
    mean_dx = dx[terrain_mask].mean()
    mean_dy = dy[terrain_mask].mean()
    print(f"\nDominant slope direction: dx={mean_dx:.4f}, dy={mean_dy:.4f}")
    print(f"Overall terrain dips toward: {'North' if mean_dy > 0 else 'South'}-{'East' if mean_dx > 0 else 'West'}")

    # Save analysis
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, "dem_analysis.npz"),
             dem=dem, building_mask=building_mask, slope=slope, dx=dx, dy=dy,
             terrain_mask=terrain_mask)

if __name__ == "__main__":
    main()
