#!/usr/bin/env python3
"""
Create comprehensive visualizations from the ITZI core simulation results.
Generates:
  1. Flood depth maps at key time steps (multi-panel)
  2. Maximum inundation map
  3. Water volume time series
  4. Hyetograph + hydrograph combined plot
  5. 3D terrain + flood visualization
  6. Summary dashboard
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import glob

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "core_sim_output")
VIZ_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "visualization_output")
CELL_SIZE = 5.0
DOMAIN_SIZE = 500.0
GRID_SIZE = 100

os.makedirs(VIZ_DIR, exist_ok=True)

# Custom flood depth colormap
flood_cmap = LinearSegmentedColormap.from_list('flood', [
    (0.0, '#F7FBFF'),
    (0.1, '#DEEBF7'),
    (0.2, '#C6DBEF'),
    (0.3, '#9ECAE1'),
    (0.4, '#6BAED6'),
    (0.5, '#4292C6'),
    (0.6, '#2171B5'),
    (0.7, '#08519C'),
    (0.8, '#08306B'),
    (1.0, '#041838'),
])

extent = [0, DOMAIN_SIZE, 0, DOMAIN_SIZE]  # xmin, xmax, ymin, ymax


def read_asc(filepath):
    """Read ESRI ASCII raster."""
    with open(filepath) as f:
        header = {}
        for _ in range(6):
            key, val = f.readline().strip().split()
            header[key.lower()] = float(val) if key.lower() not in ['ncols', 'nrows'] else int(val)
        data = np.loadtxt(f)
    return data, header


def main():
    print("Creating visualizations...")

    # Load results
    results = np.load(os.path.join(OUTPUT_DIR, "simulation_results.npz"))
    time_h = results['time']
    volume = results['volume']
    hmax_series = results['hmax']
    dem = results['dem']
    mannings = results['mannings']

    # Load depth maps
    depth_files = sorted(glob.glob(os.path.join(OUTPUT_DIR, "depth_t*.asc")))
    depth_maps = {}
    for df in depth_files:
        t_label = os.path.basename(df).replace("depth_t", "").replace(".asc", "").replace("h", "")
        depth_maps[float(t_label)] = np.loadtxt(df)

    # ================================================================
    # Figure 1: Multi-panel Flood Depth Evolution
    # ================================================================
    print("  Figure 1: Flood depth evolution...")
    fig, axes = plt.subplots(3, 4, figsize=(18, 12))
    key_times = sorted(depth_maps.keys())
    vmax = max(np.nanmax(d) for d in depth_maps.values())

    for idx, t_h in enumerate(key_times[:12]):
        ax = axes[idx // 4, idx % 4]
        h = depth_maps[t_h]
        h_masked = np.ma.masked_where(h <= 0.001, h)
        im = ax.imshow(h_masked, cmap=flood_cmap, extent=extent,
                       origin='upper', vmin=0, vmax=vmax, aspect='equal')
        ax.set_title(f't = {t_h:.1f}h')
        ax.set_xlabel('Easting (m)', fontsize=8)
        ax.set_ylabel('Northing (m)', fontsize=8)

    fig.suptitle('ITZI Flood Simulation - Water Depth Evolution',
                 fontsize=14, fontweight='bold')
    cbar = fig.colorbar(im, ax=axes, label='Water Depth (m)',
                        shrink=0.6, pad=0.02)
    plt.tight_layout()
    fig.savefig(os.path.join(VIZ_DIR, '01_flood_depth_evolution.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ================================================================
    # Figure 2: Terrain + Max Flood + Velocity
    # ================================================================
    print("  Figure 2: Terrain and maximum flood...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # DEM
    ax = axes[0]
    im = ax.imshow(dem, cmap='terrain', extent=extent, origin='upper', aspect='equal')
    ax.set_title('Digital Elevation Model')
    ax.set_xlabel('Easting (m)')
    ax.set_ylabel('Northing (m)')
    plt.colorbar(im, ax=ax, label='Elevation (m)', shrink=0.8)

    # Max flood depth (final timestep)
    ax = axes[1]
    h_final = depth_maps[max(depth_maps.keys())]
    h_final_masked = np.ma.masked_where(h_final <= 0.001, h_final)
    im = ax.imshow(h_final_masked, cmap=flood_cmap, extent=extent,
                   origin='upper', aspect='equal')
    ax.set_title(f'Flood Depth at t={max(depth_maps.keys()):.1f}h (max={np.nanmax(h_final):.2f}m)')
    ax.set_xlabel('Easting (m)')
    ax.set_ylabel('Northing (m)')
    plt.colorbar(im, ax=ax, label='Water Depth (m)', shrink=0.8)

    # Flow direction overlay on DEM
    ax = axes[2]
    dy, dx = np.gradient(dem, CELL_SIZE)
    step = 8
    y_idx, x_idx = np.meshgrid(
        np.arange(0, GRID_SIZE, step),
        np.arange(0, GRID_SIZE, step),
        indexing='ij'
    )
    x_phys = x_idx * CELL_SIZE + CELL_SIZE / 2
    y_phys = (GRID_SIZE - y_idx - 1) * CELL_SIZE + CELL_SIZE / 2
    im = ax.imshow(dem, cmap='terrain', extent=extent, origin='upper', aspect='equal')
    ax.quiver(x_phys, y_phys,
              dx[y_idx, x_idx], -dy[y_idx, x_idx],
              color='white', alpha=0.6, scale=0.3, width=0.003)
    ax.set_title('Terrain with Flow Directions')
    ax.set_xlabel('Easting (m)')
    ax.set_ylabel('Northing (m)')
    plt.colorbar(im, ax=ax, label='Elevation (m)', shrink=0.8)

    fig.suptitle('Terrain Analysis and Maximum Inundation',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(VIZ_DIR, '02_terrain_and_flood.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ================================================================
    # Figure 3: Time Series - Volume, Depth, Hyetograph
    # ================================================================
    print("  Figure 3: Time series plots...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Water volume over time
    ax = axes[0, 0]
    ax.plot(time_h, volume, 'b-o', linewidth=2, markersize=4)
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Water Volume (m³)')
    ax.set_title('Total Water Volume in Domain')
    ax.grid(True, alpha=0.3)

    # Max depth over time
    ax = axes[0, 1]
    ax.plot(time_h, hmax_series, 'r-s', linewidth=2, markersize=4)
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Maximum Water Depth (m)')
    ax.set_title('Maximum Inundation Depth')
    ax.grid(True, alpha=0.3)

    # Hyetograph
    ax = axes[1, 0]
    rain_s = np.arange(0, 7200, 300)
    dt_rain = 300
    t_peak = 2700
    intensity_mmh = np.where(
        rain_s < t_peak,
        50 * (rain_s / t_peak) ** 0.5,
        50 * ((7200 - rain_s) / (7200 - t_peak)) ** 1.5
    )
    total = np.sum(intensity_mmh) * dt_rain / 3600
    intensity_mmh = intensity_mmh * 50 / total
    ax.bar(rain_s / 3600, intensity_mmh, width=0.07, color='blue', alpha=0.7)
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Rainfall Intensity (mm/h)')
    ax.set_title('Design Storm Hyetograph (50mm total)')
    ax.grid(True, alpha=0.3)

    # Manning's n distribution
    ax = axes[1, 1]
    im = ax.imshow(mannings, cmap='YlOrRd', extent=extent, origin='upper', aspect='equal')
    ax.set_title("Manning's n Roughness Coefficient")
    ax.set_xlabel('Easting (m)')
    ax.set_ylabel('Northing (m)')
    plt.colorbar(im, ax=ax, label="Manning's n (s/m^(1/3))", shrink=0.8)

    fig.suptitle('Hydrologic Time Series and Forcing Data',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(VIZ_DIR, '03_time_series.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ================================================================
    # Figure 4: Summary Dashboard
    # ================================================================
    print("  Figure 4: Summary dashboard...")
    fig = plt.figure(figsize=(16, 12))

    # Main title
    fig.suptitle('ITZI Flood Simulation - Summary Dashboard',
                 fontsize=16, fontweight='bold')

    # Flood extent summary
    ax1 = fig.add_subplot(2, 3, 1)
    h_final = depth_maps[max(depth_maps.keys())]
    flood_extent = (h_final > 0.005).sum() * CELL_SIZE * CELL_SIZE
    dry_area = DOMAIN_SIZE * DOMAIN_SIZE - flood_extent
    ax1.pie([flood_extent, dry_area], labels=['Flooded', 'Dry'],
            colors=['#2171B5', '#DEEBF7'], autopct='%1.1f%%',
            explode=(0.05, 0))
    ax1.set_title(f'Flood Extent at 2h\n({flood_extent:.0f} m² flooded)')

    # Depth statistics
    ax2 = fig.add_subplot(2, 3, 2)
    flooded_cells = h_final[h_final > 0.001]
    if len(flooded_cells) > 0:
        ax2.hist(flooded_cells, bins=30, color='#4292C6', edgecolor='white', alpha=0.8)
        ax2.axvline(np.mean(flooded_cells), color='red', linestyle='--',
                    label=f'Mean: {np.mean(flooded_cells):.2f}m')
        ax2.axvline(np.median(flooded_cells), color='green', linestyle='--',
                    label=f'Median: {np.median(flooded_cells):.2f}m')
        ax2.legend(fontsize=8)
    ax2.set_xlabel('Water Depth (m)')
    ax2.set_ylabel('Number of Cells')
    ax2.set_title('Flood Depth Distribution at 2h')

    # Cross-section through depression
    ax3 = fig.add_subplot(2, 3, 3)
    center_row = GRID_SIZE // 2
    xs = np.arange(GRID_SIZE) * CELL_SIZE
    ax3.fill_between(xs, dem[center_row, :], dem[center_row, :] - 2,
                     alpha=0.3, color='brown', label='Terrain')
    ax3.plot(xs, dem[center_row, :], 'brown', linewidth=1.5)
    ax3.fill_between(xs, dem[center_row, :],
                     dem[center_row, :] + h_final[center_row, :],
                     alpha=0.6, color='#2171B5', label='Water Surface')
    ax3.set_xlabel('Distance (m)')
    ax3.set_ylabel('Elevation (m)')
    ax3.set_title('E-W Cross Section Through Depression')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # Volume balance
    ax4 = fig.add_subplot(2, 3, 4)
    total_rain_vol = 0.05 * DOMAIN_SIZE * DOMAIN_SIZE  # 50mm over domain
    ax4.axhline(y=total_rain_vol, color='green', linestyle='--',
                label=f'Total Rain Input: {total_rain_vol:.0f} m³')
    ax4.plot(time_h, volume, 'b-o', linewidth=2, markersize=4,
             label='Water in Domain')
    ax4.set_xlabel('Time (hours)')
    ax4.set_ylabel('Volume (m³)')
    ax4.set_title('Mass Balance')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    # Final state map
    ax5 = fig.add_subplot(2, 3, 5)
    h_final_masked = np.ma.masked_where(h_final <= 0.001, h_final)
    im = ax5.imshow(h_final_masked, cmap=flood_cmap, extent=extent,
                    origin='upper', aspect='equal')
    ax5.set_title(f'Final Flood Depth (t=2h)\nMax: {h_final.max():.2f}m | Vol: {volume[-1]:.0f} m³')
    ax5.set_xlabel('Easting (m)')
    ax5.set_ylabel('Northing (m)')
    plt.colorbar(im, ax=ax5, label='Depth (m)', shrink=0.8)

    # Key metrics text
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')
    metrics_text = f"""
    SIMULATION SUMMARY
    ==================

    Domain: {DOMAIN_SIZE}m x {DOMAIN_SIZE}m
    Resolution: {CELL_SIZE}m ({GRID_SIZE}x{GRID_SIZE})
    Duration: 2 hours

    RAINFALL
    Total: 50 mm (design storm)
    Peak: {intensity_mmh.max():.1f} mm/h at 45 min

    RESULTS (at t=2h)
    Max flood depth: {h_final.max():.2f} m
    Mean flood depth: {np.mean(flooded_cells):.3f} m
    Flooded area: {flood_extent:.0f} m²
    Water volume: {volume[-1]:.0f} m³
    Mass balance: {volume[-1]/total_rain_vol*100:.1f}%

    ITZI Version: 25.4
    Engine: 2D damped partial inertia
    """
    ax6.text(0.1, 0.5, metrics_text, transform=ax6.transAxes,
             fontsize=10, fontfamily='monospace',
             verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='#F7FBFF', alpha=0.8))

    plt.tight_layout()
    fig.savefig(os.path.join(VIZ_DIR, '04_summary_dashboard.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ================================================================
    # Figure 5: 3D Visualization (if possible)
    # ================================================================
    print("  Figure 5: 3D terrain visualization...")
    try:
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Subsample for 3D
        step = 4
        x_3d = np.arange(0, GRID_SIZE, step) * CELL_SIZE
        y_3d = np.arange(0, GRID_SIZE, step) * CELL_SIZE
        xx_3d, yy_3d = np.meshgrid(x_3d, y_3d)

        dem_sub = dem[::step, ::step]
        dem_sub = np.flipud(dem_sub)
        h_sub = h_final[::step, ::step]
        h_sub = np.flipud(h_sub)

        # Terrain surface
        ax.plot_surface(xx_3d, yy_3d, dem_sub, cmap='terrain',
                        alpha=0.8, linewidth=0, antialiased=True)

        # Water surface (slightly above terrain)
        water_elev = dem_sub + h_sub
        has_water = h_sub > 0.005
        if has_water.any():
            ax.plot_surface(xx_3d, yy_3d, water_elev, cmap=flood_cmap,
                            alpha=0.5, linewidth=0, antialiased=True)

        ax.set_xlabel('Easting (m)')
        ax.set_ylabel('Northing (m)')
        ax.set_zlabel('Elevation (m)')
        ax.set_title('3D Terrain with Flood Inundation at t=2h')
        ax.view_init(elev=25, azim=-60)

        fig.savefig(os.path.join(VIZ_DIR, '05_3d_terrain_flood.png'),
                    dpi=150, bbox_inches='tight')
        plt.close(fig)
    except Exception as e:
        print(f"    3D visualization skipped: {e}")

    print(f"\n  All visualizations saved to: {VIZ_DIR}")

    # List generated files
    for f in sorted(os.listdir(VIZ_DIR)):
        if f.endswith('.png'):
            size_kb = os.path.getsize(os.path.join(VIZ_DIR, f)) / 1024
            print(f"    {f} ({size_kb:.0f} KB)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
