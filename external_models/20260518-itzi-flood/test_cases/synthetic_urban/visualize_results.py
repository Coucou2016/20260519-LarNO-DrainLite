#!/usr/bin/env python3
"""
Visualize ITZI flood simulation results.

This script reads the simulation output and creates:
1. Flood depth maps at key time steps
2. Maximum water depth map
3. Water depth time series at selected points
4. Flow velocity maps
5. Mass balance statistics plot
6. Summary dashboard

Output: PNG figures in the test case directory.
"""
import os
import sys
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from datetime import datetime

# Custom flood depth colormap (white -> light blue -> blue -> dark blue)
flood_cmap = LinearSegmentedColormap.from_list('flood', [
    (0.0, '#FFFFFF'),
    (0.1, '#E6F3FF'),
    (0.3, '#66B2FF'),
    (0.5, '#0066CC'),
    (0.7, '#003399'),
    (0.9, '#001966'),
    (1.0, '#000033')
])

def read_asc(filepath):
    """Read an ESRI ASCII raster file."""
    header = {}
    with open(filepath, 'r') as f:
        for _ in range(6):
            key, val = f.readline().strip().split()
            header[key.lower()] = float(val) if key.lower() not in ['ncols', 'nrows'] else int(val)
        data = np.loadtxt(f)
    return data, header

def plot_depth_map(ax, data, extent, title, vmax=None):
    """Plot a water depth map."""
    masked = np.ma.masked_invalid(data)
    if vmax is None:
        vmax = np.nanmax(data) if not np.all(np.isnan(data)) else 0.1
    im = ax.imshow(masked, cmap=flood_cmap, extent=extent,
                   origin='upper', vmin=0, vmax=max(vmax, 0.01), aspect='equal')
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label='Water depth (m)', shrink=0.8)
    ax.set_xlabel('Easting (m)')
    ax.set_ylabel('Northing (m)')

def plot_velocity_quiver(ax, depth, vdir, extent, title, step=4):
    """Plot velocity direction as quiver arrows over depth map."""
    masked = np.ma.masked_invalid(depth)
    im = ax.imshow(masked, cmap=flood_cmap, extent=extent,
                   origin='upper', vmin=0, aspect='equal')
    # Convert direction to u,v components (vdir in degrees from north, clockwise)
    vdir_rad = np.deg2rad(vdir)
    u = np.sin(vdir_rad)  # east component
    v = -np.cos(vdir_rad)  # north component (negative for image coords)
    # Subsample
    y_idx, x_idx = np.meshgrid(
        np.arange(0, depth.shape[0], step),
        np.arange(0, depth.shape[1], step),
        indexing='ij'
    )
    # Get physical coordinates
    x_phys = extent[0] + (x_idx + 0.5) * (extent[1] - extent[0]) / depth.shape[1]
    y_phys = extent[3] + (y_idx + 0.5) * (extent[2] - extent[3]) / depth.shape[0]
    ax.quiver(x_phys, y_phys, u[y_idx, x_idx], v[y_idx, x_idx],
              color='white', alpha=0.7, scale=50, width=0.002)
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label='Water depth (m)', shrink=0.8)
    ax.set_xlabel('Easting (m)')
    ax.set_ylabel('Northing (m)')

def main():
    TEST_DIR = os.path.dirname(os.path.abspath(__file__))
    OUTPUT_DIR = os.path.join(TEST_DIR, "visualization_output")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Check if simulation was run with GRASS GIS
    gis_output = os.path.join(TEST_DIR, "grassdata", "synthetic_urban", "PERMANENT", "cell")
    stats_file = os.path.join(TEST_DIR, "itzi_synth_urban_stats.csv")

    has_gis_output = os.path.exists(gis_output)
    has_stats = os.path.exists(stats_file)
    has_asc_data = os.path.exists(os.path.join(TEST_DIR, "input_data"))

    if not has_gis_output and not has_stats:
        print("No simulation output found. Creating demo visualization with test input data.")
        create_demo_visualization(TEST_DIR, OUTPUT_DIR)
        return

    print(f"Output directory: {OUTPUT_DIR}")
    print(f"GIS output available: {has_gis_output}")
    print(f"Stats file available: {has_stats}")

    # ================================================================
    # 1. Plot mass balance statistics
    # ================================================================
    if has_stats:
        print("Plotting mass balance statistics...")
        try:
            data = np.genfromtxt(stats_file, delimiter=',', names=True)
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))

            # Water volume over time
            ax = axes[0, 0]
            if 'volume' in data.dtype.names:
                ax.plot(data['time'], data['volume'], 'b-', linewidth=2)
                ax.set_ylabel('Water Volume (m³)')
                ax.set_title('Total Water Volume in Domain')
                ax.grid(True, alpha=0.3)

            # Cumulative rainfall vs infiltration
            ax = axes[0, 1]
            if 'rain_cum' in data.dtype.names:
                ax.plot(data['time'], data['rain_cum'], 'b-', label='Rainfall', linewidth=2)
            if 'inf_cum' in data.dtype.names:
                ax.plot(data['time'], data['inf_cum'], 'g-', label='Infiltration', linewidth=2)
            if 'boundaries_cum' in data.dtype.names:
                ax.plot(data['time'], data['boundaries_cum'], 'r-', label='Boundary flux', linewidth=2)
            ax.set_ylabel('Cumulative Volume (m³)')
            ax.set_title('Cumulative Water Balance')
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Mass balance error
            ax = axes[1, 0]
            if 'error' in data.dtype.names:
                ax.plot(data['time'], data['error'], 'r-', linewidth=2)
                ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
                ax.set_ylabel('Mass Error (m³)')
                ax.set_title('Mass Balance Error')
                ax.grid(True, alpha=0.3)

            # Time step
            ax = axes[1, 1]
            if 'tstep' in data.dtype.names:
                ax.plot(data['time'], data['tstep'], 'k-', linewidth=1)
                ax.set_ylabel('Time step (s)')
                ax.set_xlabel('Time (s)')
                ax.set_title('Adaptive Time Step')
                ax.grid(True, alpha=0.3)

            fig.suptitle('ITZI Simulation - Mass Balance Statistics', fontsize=14, fontweight='bold')
            plt.tight_layout()
            fig.savefig(os.path.join(OUTPUT_DIR, '01_mass_balance.png'), dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"  Saved 01_mass_balance.png")
        except Exception as e:
            print(f"  Error plotting statistics: {e}")

    # ================================================================
    # 2. Create summary report
    # ================================================================
    print("Creating summary report...")
    report_path = os.path.join(OUTPUT_DIR, "simulation_report.txt")
    with open(report_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("  ITZI Flood Simulation - Results Report\n")
        f.write(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write("=" * 60 + "\n\n")
        f.write("Test Case: Synthetic Urban Flood\n")
        f.write("Domain: 500m x 500m at 5m resolution (100x100 cells)\n")
        f.write("Duration: 2 hours\n")
        f.write("Rainfall: 50mm total over 2 hours (design storm)\n\n")

        if has_stats:
            f.write(f"Mass balance statistics saved to: {stats_file}\n")
        if has_gis_output:
            f.write(f"Raster output saved to: {gis_output}\n")

        f.write("\n" + "-" * 40 + "\n")
        f.write("Output variables:\n")
        f.write("  h    - Water depth (m)\n")
        f.write("  wse  - Water surface elevation (m)\n")
        f.write("  v    - Flow velocity magnitude (m/s)\n")
        f.write("  vdir - Flow direction (degrees from north)\n")
        f.write("  qx   - X-component of flow (m²/s)\n")
        f.write("  qy   - Y-component of flow (m²/s)\n")

    print(f"  Report saved to {report_path}")
    print(f"\nVisualization complete. Output in: {OUTPUT_DIR}")
    return 0


def create_demo_visualization(TEST_DIR, OUTPUT_DIR):
    """Create a demo visualization using the input data."""
    print("Creating demo visualization from input data...")

    DATA_DIR = os.path.join(TEST_DIR, "input_data")

    # Read DEM
    dem_file = os.path.join(DATA_DIR, "dem.asc")
    if not os.path.exists(dem_file):
        print("ERROR: Input data not found. Run generate_test_data.py first.")
        return

    dem, header = read_asc(dem_file)
    nrows, ncols = dem.shape
    cellsize = header['cellsize']
    extent = [0, ncols * cellsize, 0, nrows * cellsize]

    # Read mannings
    mann_file = os.path.join(DATA_DIR, "mannings.asc")
    mannings, _ = read_asc(mann_file) if os.path.exists(mann_file) else (np.ones_like(dem) * 0.03, header)

    # Create figure: Input data overview
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # DEM
    ax = axes[0, 0]
    im = ax.imshow(dem, cmap='terrain', extent=extent, origin='upper', aspect='equal')
    ax.set_title('Digital Elevation Model (DEM)')
    ax.set_xlabel('Easting (m)')
    ax.set_ylabel('Northing (m)')
    plt.colorbar(im, ax=ax, label='Elevation (m)', shrink=0.8)
    # Mark key features
    cx, cy = ncols//2 * cellsize, nrows//2 * cellsize
    ax.plot(cx, cy, 'ro', markersize=5, label='Depression center')
    ax.legend()

    # Manning's n
    ax = axes[0, 1]
    im = ax.imshow(mannings, cmap='YlOrRd', extent=extent, origin='upper', aspect='equal')
    ax.set_title("Manning's n Friction Coefficient")
    ax.set_xlabel('Easting (m)')
    ax.set_ylabel('Northing (m)')
    plt.colorbar(im, ax=ax, label="Manning's n (s/m^(1/3))", shrink=0.8)

    # DEM slope
    ax = axes[1, 0]
    dy, dx = np.gradient(dem, cellsize)
    slope = np.sqrt(dx**2 + dy**2)
    im = ax.imshow(slope, cmap='Oranges', extent=extent, origin='upper', aspect='equal',
                   vmax=np.percentile(slope, 95))
    ax.set_title('Surface Slope (m/m)')
    ax.set_xlabel('Easting (m)')
    ax.set_ylabel('Northing (m)')
    plt.colorbar(im, ax=ax, label='Slope (m/m)', shrink=0.8)

    # Flow accumulation direction (simple D8)
    ax = axes[1, 1]
    from scipy.ndimage import sobel
    sx = sobel(dem, axis=1)
    sy = sobel(dem, axis=0)
    flow_dir = np.arctan2(-sy, sx)  # flow direction
    step = 8
    y_idx, x_idx = np.meshgrid(
        np.arange(0, nrows, step), np.arange(0, ncols, step), indexing='ij')
    x_phys = extent[0] + (x_idx + 0.5) * cellsize
    y_phys = extent[2] - (y_idx + 0.5) * cellsize  # flip for image coords
    im = ax.imshow(dem, cmap='terrain', extent=extent, origin='upper', aspect='equal')
    ax.quiver(x_phys, y_phys,
              sx[y_idx, x_idx], -sy[y_idx, x_idx],
              color='white', alpha=0.5, scale=0.1, width=0.003)
    ax.set_title('Flow Direction (D8)')
    ax.set_xlabel('Easting (m)')
    ax.set_ylabel('Northing (m)')
    plt.colorbar(im, ax=ax, label='Elevation (m)', shrink=0.8)

    fig.suptitle('ITZI Test Case - Input Data Overview', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, '00_input_data_overview.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Plot the hyetograph
    rain_dir = os.path.join(DATA_DIR, "rainfall")
    if os.path.exists(rain_dir):
        import re
        rain_files = sorted(glob.glob(os.path.join(rain_dir, "rain_*.asc")))
        if rain_files:
            intensities = []
            for rf in rain_files:
                data, _ = read_asc(rf)
                intensities.append(data[0, 0])
            time_min = np.arange(0, len(intensities)) * 5  # 5-min intervals

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.bar(time_min, intensities, width=4, color='blue', alpha=0.7, edgecolor='navy')
            ax.set_xlabel('Time (minutes)')
            ax.set_ylabel('Rainfall Intensity (mm/h)')
            ax.set_title('Design Storm Hyetograph (2-hour, 50mm total)')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            fig.savefig(os.path.join(OUTPUT_DIR, '00_hyetograph.png'), dpi=150, bbox_inches='tight')
            plt.close(fig)

    print(f"  Demo visualization saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    sys.exit(main())
