#!/usr/bin/env python3
"""
Urban District Flood Visualization
Creates detailed figures for the urban flood simulation.
"""
import os, sys, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import matplotlib.patches as mpatches

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "core_sim_output")
VIZ_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "visualization_output")
CELL_SIZE = 2.0
DOMAIN_SIZE = 400.0
GRID_SIZE = 200
os.makedirs(VIZ_DIR, exist_ok=True)

# Colormaps
flood_cmap = LinearSegmentedColormap.from_list('flood', [
    (0.0, '#F7FBFF'), (0.15, '#C6DBEF'), (0.3, '#6BAED6'),
    (0.5, '#2171B5'), (0.7, '#08306B'), (1.0, '#041838'),
])
landuse_cmap = ListedColormap(['#E8D5B7', '#C0C0C0', '#90EE90', '#8B4513'])
extent = [0, DOMAIN_SIZE, 0, DOMAIN_SIZE]


def main():
    print("Loading urban simulation results...")
    data = np.load(os.path.join(OUTPUT_DIR, "urban_simulation_results.npz"))
    time_h = data['time']
    volume = data['volume']
    hmax_series = data['hmax']
    flooded_cells = data['flooded_cells']
    dem = data['dem']
    mannings = data['mannings']
    building_mask = data['building_mask']
    park_mask = data['park_mask']
    h_final = data['h_final']
    v_final = data['v_final']
    hmax_arr = data['hmax_arr']

    # Load time snapshots
    depth_files = sorted(glob.glob(os.path.join(OUTPUT_DIR, "depth_urban_t*.asc")))
    depth_maps = {}
    for df in depth_files:
        label = os.path.basename(df).replace("depth_urban_t", "").replace(".asc", "").replace("h", "")
        depth_maps[float(label)] = np.loadtxt(df)

    # ================================================================
    # Fig 1: Urban Terrain + Building Layout
    # ================================================================
    print("  Figure 1: Urban terrain and building layout...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # DEM with buildings
    ax = axes[0, 0]
    im = ax.imshow(np.flipud(dem), cmap='terrain', extent=extent, aspect='equal')
    ax.set_title('Urban District DEM with Building Blocks')
    ax.set_xlabel('Easting (m)'); ax.set_ylabel('Northing (m)')
    plt.colorbar(im, ax=ax, label='Elevation (m)', shrink=0.8)

    # Building mask overlay on DEM
    ax = axes[0, 1]
    dem_bg = np.where(building_mask, np.nan, dem)
    im = ax.imshow(np.flipud(dem_bg), cmap='terrain', extent=extent, aspect='equal', alpha=0.8)
    # Overlay buildings in red
    bldg_display = np.where(building_mask, 1.0, np.nan)
    ax.imshow(np.flipud(bldg_display), cmap=ListedColormap(['#8B0000']),
              extent=extent, aspect='equal', alpha=0.7)
    ax.set_title('Building Footprints (24 buildings)')
    ax.set_xlabel('Easting (m)'); ax.set_ylabel('Northing (m)')
    plt.colorbar(im, ax=ax, label='Ground Elevation (m)', shrink=0.8)

    # Manning's n surface classification
    ax = axes[1, 0]
    # Classify: 1=street, 2=sidewalk, 3=park, 4=building
    surface = np.zeros_like(mannings, dtype=int)
    surface[mannings > 0.1] = 4    # buildings
    surface[(mannings > 0.03) & (mannings <= 0.1)] = 3  # park
    surface[(mannings > 0.02) & (mannings <= 0.03)] = 2  # sidewalk
    surface[mannings <= 0.02] = 1  # street

    colors_surface = ['#808080', '#D3D3D3', '#228B22', '#8B0000']
    labels_surface = ['Street (n=0.015)', 'Sidewalk (n=0.025)', 'Park/Grass (n=0.04)', 'Building (n=0.5)']
    cmap_surface = ListedColormap(colors_surface)
    im = ax.imshow(np.flipud(surface), cmap=cmap_surface, extent=extent,
                   aspect='equal', vmin=0.5, vmax=4.5)
    ax.set_title('Surface Classification & Manning\'s n')
    ax.set_xlabel('Easting (m)'); ax.set_ylabel('Northing (m)')
    patches = [mpatches.Patch(color=c, label=l) for c, l in zip(colors_surface, labels_surface)]
    ax.legend(handles=patches, loc='lower right', fontsize=7, ncol=2)

    # 3D terrain view
    from mpl_toolkits.mplot3d import Axes3D
    ax = axes[1, 1]
    ax.remove()
    ax = fig.add_subplot(2, 2, 4, projection='3d')
    step = 5
    x3d = np.arange(0, GRID_SIZE, step) * CELL_SIZE
    y3d = np.arange(0, GRID_SIZE, step) * CELL_SIZE
    xx3d, yy3d = np.meshgrid(x3d, y3d)
    dem_sub = np.flipud(dem[::step, ::step])
    buildings_sub = np.flipud(building_mask[::step, ::step].astype(float))
    dem_display = dem_sub.copy()
    dem_display[buildings_sub > 0] = np.nan
    ax.plot_surface(xx3d, yy3d, dem_display, cmap='terrain', alpha=0.7, linewidth=0)
    bldg_elev = dem_sub.copy()
    bldg_elev[buildings_sub == 0] = np.nan
    ax.plot_surface(xx3d, yy3d, bldg_elev, color='darkred', alpha=0.6, linewidth=0)
    ax.set_xlabel('E (m)'); ax.set_ylabel('N (m)'); ax.set_zlabel('Elev (m)')
    ax.set_title('3D Urban District View')
    ax.view_init(elev=35, azim=-45)

    fig.suptitle('Urban District - Terrain and Building Layout', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(VIZ_DIR, 'urban_01_terrain_buildings.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ================================================================
    # Fig 2: Flood Evolution in Urban Setting
    # ================================================================
    print("  Figure 2: Flood evolution in urban district...")
    key_times = sorted(depth_maps.keys())
    vmax_flood = max(np.nanmax(depth_maps[t]) for t in key_times)

    fig, axes = plt.subplots(3, 4, figsize=(20, 14))
    for idx, t_h in enumerate(key_times[:12]):
        ax = axes[idx // 4, idx % 4]
        h = depth_maps[t_h]
        h_masked = np.ma.masked_where(h <= 0.001, h)
        # Show buildings as grey overlay
        bg = np.where(building_mask, 0.3, 0)
        ax.imshow(np.flipud(bg), cmap=ListedColormap(['#FFFFFF', '#D3D3D3']),
                  extent=extent, aspect='equal', alpha=0.5)
        im = ax.imshow(np.flipud(h_masked), cmap=flood_cmap, extent=extent,
                       origin='upper', vmin=0, vmax=vmax_flood, aspect='equal')
        ax.set_title(f't = {t_h:.1f}h')
        ax.set_xlabel('E (m)', fontsize=7); ax.set_ylabel('N (m)', fontsize=7)

    fig.suptitle('Urban Flood Evolution - 2h Design Storm (50mm)',
                 fontsize=14, fontweight='bold')
    cbar = fig.colorbar(im, ax=axes, label='Water Depth (m)', shrink=0.5, pad=0.02)
    plt.tight_layout()
    fig.savefig(os.path.join(VIZ_DIR, 'urban_02_flood_evolution.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ================================================================
    # Fig 3: Time Series Analysis
    # ================================================================
    print("  Figure 3: Time series analysis...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Water volume
    ax = axes[0, 0]
    ax.plot(time_h, volume, 'b-o', linewidth=2, markersize=5)
    ax.fill_between(time_h, 0, volume, alpha=0.2, color='blue')
    total_rain = 0.05 * DOMAIN_SIZE * DOMAIN_SIZE
    ax.axhline(y=total_rain, color='green', linestyle='--', label=f'Total Rain Input ({total_rain:.0f} m3)')
    ax.set_xlabel('Time (hours)'); ax.set_ylabel('Water Volume (m3)')
    ax.set_title('Water Volume in Domain'); ax.legend(); ax.grid(True, alpha=0.3)

    # Max depth
    ax = axes[0, 1]
    ax.plot(time_h, hmax_series, 'r-s', linewidth=2, markersize=5)
    ax.set_xlabel('Time (hours)'); ax.set_ylabel('Max Water Depth (m)')
    ax.set_title('Maximum Inundation Depth'); ax.grid(True, alpha=0.3)

    # Flooded area vs time
    ax = axes[1, 0]
    flooded_area = flooded_cells * CELL_SIZE * CELL_SIZE
    ax.plot(time_h, flooded_area, 'b-o', linewidth=2, markersize=5)
    ax.set_xlabel('Time (hours)'); ax.set_ylabel('Flooded Area (m2)')
    ax.set_title('Inundated Area Over Time')
    ax.grid(True, alpha=0.3)

    # Hyetograph
    ax = axes[1, 1]
    rain_s = np.arange(0, 7200, 300)
    t_peak = 2700
    intensity_mmh = np.where(rain_s < t_peak,
        50 * (rain_s / t_peak) ** 0.5,
        50 * ((7200 - rain_s) / (7200 - t_peak)) ** 1.5)
    intensity_mmh = intensity_mmh * 50 / (np.sum(intensity_mmh) * 300 / 3600)
    ax.bar(rain_s / 3600, intensity_mmh, width=0.07, color='blue', alpha=0.7)
    ax.set_xlabel('Time (hours)'); ax.set_ylabel('Rainfall (mm/h)')
    ax.set_title('Design Storm Hyetograph (50mm)'); ax.grid(True, alpha=0.3)

    fig.suptitle('Urban Flood - Hydrologic Response', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(VIZ_DIR, 'urban_03_time_series.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ================================================================
    # Fig 4: Street-level Flow Analysis
    # ================================================================
    print("  Figure 4: Street-level flow analysis...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # Flow velocity at final time
    ax = axes[0, 0]
    v_display = np.where(v_final > 0.01, v_final, np.nan)
    im = ax.imshow(np.flipud(v_display), cmap='hot', extent=extent, aspect='equal', vmin=0)
    ax.set_title('Flow Velocity at t=2h')
    ax.set_xlabel('Easting (m)'); ax.set_ylabel('Northing (m)')
    plt.colorbar(im, ax=ax, label='Velocity (m/s)', shrink=0.8)

    # Depth along main avenue (N-S cross section)
    ax = axes[0, 1]
    avenue_col = GRID_SIZE // 2
    xs = np.arange(GRID_SIZE) * CELL_SIZE
    ax.plot(xs, dem[:, avenue_col], 'brown', linewidth=1.5, label='Terrain (Main Avenue)')
    ax.fill_between(xs, dem[:, avenue_col], dem[:, avenue_col] + h_final[:, avenue_col],
                    alpha=0.5, color='#2171B5', label='Water Surface')
    ax.set_xlabel('Distance along avenue (m)'); ax.set_ylabel('Elevation (m)')
    ax.set_title('N-S Cross Section Along Main Avenue')
    ax.legend(); ax.grid(True, alpha=0.3)

    # Building impact: depth around a specific building
    ax = axes[1, 0]
    center_row = 100
    ax.plot(xs, dem[center_row, :], 'brown', linewidth=1.5, label='Terrain')
    ax.fill_between(xs, dem[center_row, :], dem[center_row, :] + h_final[center_row, :],
                    alpha=0.5, color='#2171B5', label='Water')
    ax.set_xlabel('Distance (m)'); ax.set_ylabel('Elevation (m)')
    ax.set_title('E-W Cross Section Through Building Row')
    ax.legend(); ax.grid(True, alpha=0.3)

    # Flood depth distribution histogram
    ax = axes[1, 1]
    flooded_depth = h_final[h_final > 0.001]
    ax.hist(flooded_depth, bins=40, color='#4292C6', edgecolor='white', alpha=0.8)
    ax.axvline(np.mean(flooded_depth), color='red', linestyle='--',
               label=f'Mean: {np.mean(flooded_depth):.3f}m')
    ax.axvline(np.median(flooded_depth), color='green', linestyle='--',
               label=f'Median: {np.median(flooded_depth):.3f}m')
    ax.set_xlabel('Water Depth (m)'); ax.set_ylabel('Cell Count')
    ax.set_title('Flood Depth Distribution at t=2h'); ax.legend()

    fig.suptitle('Urban Flood - Street-Level Flow Dynamics', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(VIZ_DIR, 'urban_04_street_flow.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ================================================================
    # Fig 5: Urban Flood Summary Dashboard
    # ================================================================
    print("  Figure 5: Summary dashboard...")
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle('ITZI Urban District Flood - Summary Dashboard',
                 fontsize=16, fontweight='bold')

    # (1) Final flood map with buildings
    ax = fig.add_subplot(2, 3, 1)
    h_display = np.ma.masked_where(h_final <= 0.001, h_final)
    bldg_overlay = np.where(building_mask, 0.3, 0)
    ax.imshow(np.flipud(bldg_overlay), cmap=ListedColormap(['none', '#888888']),
              extent=extent, aspect='equal', alpha=0.6)
    im = ax.imshow(np.flipud(h_display), cmap=flood_cmap, extent=extent, aspect='equal')
    ax.set_title(f'Maximum Flood (t=2h)\nMax depth: {h_final.max():.2f}m')
    ax.set_xlabel('Easting (m)'); ax.set_ylabel('Northing (m)')
    plt.colorbar(im, ax=ax, label='Depth (m)', shrink=0.8)

    # (2) Flood extent pie
    ax = fig.add_subplot(2, 3, 2)
    flooded_area_m2 = flooded_cells[-1] * CELL_SIZE * CELL_SIZE
    total_area = DOMAIN_SIZE * DOMAIN_SIZE
    dry_area = total_area - flooded_area_m2
    ax.pie([flooded_area_m2, dry_area], labels=['Flooded', 'Dry'],
           colors=['#2171B5', '#D3D3D3'], autopct='%1.1f%%', explode=(0.05, 0))
    ax.set_title(f'Flood Extent at 2h\n({flooded_area_m2:.0f} m2 / {flooded_area_m2/total_area*100:.1f}%)')

    # (3) Depth by surface type
    ax = fig.add_subplot(2, 3, 3)
    surface_types = {
        'Streets': ~building_mask & ~park_mask & (mannings <= 0.02),
        'Sidewalks': (mannings > 0.02) & (mannings <= 0.03),
        'Park': park_mask,
        'Near Buildings': ~building_mask & (mannings > 0.03),
    }
    x_labels = []
    y_means = []
    for name, mask in surface_types.items():
        depths = h_final[mask]
        depths = depths[depths > 0.001]
        if len(depths) > 0:
            x_labels.append(name)
            y_means.append(np.mean(depths))
    bars = ax.bar(x_labels, y_means, color=['#808080', '#D3D3D3', '#228B22', '#FFD700'])
    ax.set_ylabel('Mean Flood Depth (m)')
    ax.set_title('Mean Flood Depth by Surface Type')
    for bar, val in zip(bars, y_means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val:.3f}m', ha='center', fontsize=9)

    # (4) Volume balance
    ax = fig.add_subplot(2, 3, 4)
    total_rain_vol = 0.05 * DOMAIN_SIZE * DOMAIN_SIZE
    ax.axhline(y=total_rain_vol, color='green', linestyle='--', label=f'Total Rain: {total_rain_vol:.0f} m3')
    ax.plot(time_h, volume, 'b-o', linewidth=2, markersize=5, label='Volume in Domain')
    ax.set_xlabel('Time (hours)'); ax.set_ylabel('Volume (m3)')
    ax.set_title('Mass Balance'); ax.legend(); ax.grid(True, alpha=0.3)

    # (5) 3D terrain + flood
    ax = fig.add_subplot(2, 3, 5, projection='3d')
    step = 6
    x3d = np.arange(0, GRID_SIZE, step) * CELL_SIZE
    y3d = np.arange(0, GRID_SIZE, step) * CELL_SIZE
    xx3d, yy3d = np.meshgrid(x3d, y3d)
    dem_sub = np.flipud(dem[::step, ::step])
    h_sub = np.flipud(h_final[::step, ::step])
    water_elev = dem_sub + h_sub
    has_water = h_sub > 0.005
    ax.plot_surface(xx3d, yy3d, dem_sub, cmap='terrain', alpha=0.6, linewidth=0)
    if has_water.any():
        we_display = np.where(has_water, water_elev, np.nan)
        ax.plot_surface(xx3d, yy3d, we_display, cmap=flood_cmap, alpha=0.5, linewidth=0)
    ax.set_xlabel('E (m)'); ax.set_ylabel('N (m)'); ax.set_zlabel('Elev (m)')
    ax.set_title('3D Urban Flood View'); ax.view_init(elev=30, azim=-55)

    # (6) Key metrics
    ax = fig.add_subplot(2, 3, 6)
    ax.axis('off')
    metrics = f"""
    URBAN FLOOD SUMMARY
    ====================
    Domain: {DOMAIN_SIZE}m x {DOMAIN_SIZE}m
    Resolution: {CELL_SIZE}m ({GRID_SIZE}x{GRID_SIZE})
    Buildings: 24 blocks (8-18m tall)
    Duration: 2 hours

    RAINFALL
    Total: 50 mm (design storm)
    Peak: {intensity_mmh.max():.1f} mm/h at 45 min

    RESULTS at t=2h
    Max flood depth: {h_final.max():.2f} m
    Mean flood depth: {np.mean(flooded_depth):.3f} m
    Flooded area: {flooded_area_m2:.0f} m2 ({flooded_area_m2/total_area*100:.1f}%)
    Water volume: {volume[-1]:.0f} m3
    Mass balance: {volume[-1]/total_rain_vol*100:.1f}%

    BUILDING EFFECTS
    Buildings block surface flow
    Streets act as flow channels
    Park serves as detention basin
    Street flooding: {np.mean(h_final[(mannings <= 0.02) & (h_final > 0.001)]):.3f}m mean depth

    ITZI Engine: Damped partial inertia (2D)
    """
    ax.text(0.05, 0.5, metrics, transform=ax.transAxes, fontsize=10,
            fontfamily='monospace', verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='#F0F8FF', alpha=0.8))

    plt.tight_layout()
    fig.savefig(os.path.join(VIZ_DIR, 'urban_05_dashboard.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ================================================================
    # Done
    # ================================================================
    print(f"\n  Visualizations saved to: {VIZ_DIR}")
    for f in sorted(os.listdir(VIZ_DIR)):
        if f.endswith('.png'):
            print(f"    {f} ({os.path.getsize(os.path.join(VIZ_DIR, f))/1024:.0f} KB)")


if __name__ == "__main__":
    main()
