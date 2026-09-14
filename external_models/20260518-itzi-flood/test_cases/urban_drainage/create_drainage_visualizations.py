#!/usr/bin/env python3
"""
Drainage Comparison Visualizations.
Compares surface-only vs surface+drainage coupled simulation.
"""
import os, sys, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import matplotlib.animation as animation
import matplotlib.patches as mpatches

SIM_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SIM_DIR, "core_sim_output")
VIZ_DIR = os.path.join(SIM_DIR, "visualization_output")
CELL_SIZE = 2.0; DOMAIN_SIZE = 400.0; GRID_SIZE = 200
os.makedirs(VIZ_DIR, exist_ok=True)

flood_cmap = LinearSegmentedColormap.from_list('flood', [
    (0.0, '#F7FBFF'), (0.15, '#C6DBEF'), (0.3, '#6BAED6'),
    (0.5, '#2171B5'), (0.7, '#08306B'), (1.0, '#041838'),
])
diff_cmap = plt.cm.RdBu_r
extent = [0, DOMAIN_SIZE, 0, DOMAIN_SIZE]


def main():
    print("Loading data...")
    data = np.load(os.path.join(OUTPUT_DIR, "coupled_results.npz"))
    time_h = data['time']
    vol_nodrain = data['volume_no_drain']
    vol_drain = data['volume_drain']
    hmax_nodrain = data['hmax_no_drain']
    hmax_drain = data['hmax_drain']
    flooded_nodrain = data['flooded_no_drain']
    flooded_drain = data['flooded_drain']
    drain_flow = data['drain_flow']
    dem = data['dem']
    mannings = data['mannings']
    building_mask = data['building_mask']
    h_nodrain = data['h_final_no_drain']
    h_drain = data['h_final_drain']

    n_records = len(time_h)
    if len(drain_flow) < n_records:
        drain_flow = np.pad(drain_flow, (0, n_records - len(drain_flow)), constant_values=drain_flow[-1] if len(drain_flow) > 0 else 0)
        vol_drain_full = np.pad(vol_drain, (0, n_records - len(vol_drain)), constant_values=vol_drain[-1] if len(vol_drain) > 0 else vol_drain[0])
        hmax_drain_full = np.pad(hmax_drain, (0, n_records - len(hmax_drain)), constant_values=hmax_drain[-1] if len(hmax_drain) > 0 else hmax_drain[0])
    else:
        vol_drain_full = vol_drain
        hmax_drain_full = hmax_drain

    # ================================================================
    # Figure 1: Side-by-side flood comparison
    # ================================================================
    print("  Figure 1: Side-by-side comparison...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    vmax = max(np.nanmax(h_nodrain), np.nanmax(h_drain))

    # No Drainage
    ax = axes[0]
    h = np.ma.masked_where(h_nodrain <= 0.001, h_nodrain)
    bldg_bg = np.where(building_mask, 0.3, 0)
    ax.imshow(np.flipud(bldg_bg), cmap=ListedColormap(['#FFFFFF', '#D3D3D3']),
              extent=extent, aspect='equal', alpha=0.5)
    im = ax.imshow(np.flipud(h), cmap=flood_cmap, extent=extent, aspect='equal', vmin=0, vmax=vmax)
    ax.set_title(f'Surface Only\nMax: {np.nanmax(h_nodrain):.2f}m | Vol: {vol_nodrain[-1]:.0f}m3')
    ax.set_xlabel('E (m)'); ax.set_ylabel('N (m)')
    plt.colorbar(im, ax=ax, label='Depth (m)', shrink=0.8)

    # With Drainage
    ax = axes[1]
    h = np.ma.masked_where(h_drain <= 0.001, h_drain)
    ax.imshow(np.flipud(bldg_bg), cmap=ListedColormap(['#FFFFFF', '#D3D3D3']),
              extent=extent, aspect='equal', alpha=0.5)
    im = ax.imshow(np.flipud(h), cmap=flood_cmap, extent=extent, aspect='equal', vmin=0, vmax=vmax)
    ax.set_title(f'With Drainage Network\nMax: {np.nanmax(h_drain):.2f}m | Vol: {vol_drain[-1]:.0f}m3')
    ax.set_xlabel('E (m)'); ax.set_ylabel('N (m)')
    plt.colorbar(im, ax=ax, label='Depth (m)', shrink=0.8)

    # Difference map
    ax = axes[2]
    diff = h_nodrain - h_drain
    diff_masked = np.ma.masked_where(np.abs(diff) < 0.01, diff)
    vlim = max(abs(diff_masked.min()), abs(diff_masked.max())) if diff_masked.count() > 0 else 0.2
    vlim = max(vlim, 0.05)
    im = ax.imshow(np.flipud(diff_masked), cmap=diff_cmap, extent=extent,
                   aspect='equal', vmin=-vlim, vmax=vlim)
    ax.set_title(f'Difference (No Drain - Drain)\nBlue=less flood with drainage')
    ax.set_xlabel('E (m)'); ax.set_ylabel('N (m)')
    plt.colorbar(im, ax=ax, label='Depth Diff (m)', shrink=0.8)

    fig.suptitle('Effect of Underground Drainage Network on Urban Flooding',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(VIZ_DIR, 'drainage_01_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ================================================================
    # Figure 2: Time series comparison
    # ================================================================
    print("  Figure 2: Time series comparison...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Volume comparison
    ax = axes[0, 0]
    ax.plot(time_h, vol_nodrain, 'r-o', linewidth=2, markersize=5, label='Surface Only')
    ax.plot(time_h[:len(vol_drain)], vol_drain, 'b-s', linewidth=2, markersize=5, label='With Drainage')
    ax.fill_between(time_h, vol_nodrain, vol_drain_full, alpha=0.2, color='green',
                    label=f'Drained: {vol_nodrain[-1]-vol_drain[-1]:.0f} m3')
    total_rain = 0.05 * DOMAIN_SIZE * DOMAIN_SIZE
    ax.axhline(y=total_rain, color='grey', linestyle='--', label=f'Total Rain ({total_rain:.0f} m3)')
    ax.set_xlabel('Time (hours)'); ax.set_ylabel('Water Volume (m3)')
    ax.set_title('Surface Water Volume'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Max depth comparison
    ax = axes[0, 1]
    ax.plot(time_h, hmax_nodrain, 'r-o', linewidth=2, markersize=5, label='Surface Only')
    ax.plot(time_h[:len(hmax_drain)], hmax_drain, 'b-s', linewidth=2, markersize=5, label='With Drainage')
    ax.set_xlabel('Time (hours)'); ax.set_ylabel('Max Water Depth (m)')
    ax.set_title('Maximum Inundation Depth'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Flooded area comparison
    ax = axes[1, 0]
    ax.plot(time_h, flooded_nodrain * CELL_SIZE * CELL_SIZE, 'r-o', linewidth=2, markersize=5, label='Surface Only')
    ax.plot(time_h[:len(flooded_drain)], flooded_drain * CELL_SIZE * CELL_SIZE, 'b-s', linewidth=2, markersize=5, label='With Drainage')
    ax.set_xlabel('Time (hours)'); ax.set_ylabel('Flooded Area (m2)')
    ax.set_title('Inundated Area'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Drainage flow rate
    ax = axes[1, 1]
    ax.bar(time_h[:len(drain_flow)], -np.array(drain_flow), width=0.07,
           color='blue', alpha=0.7, label='Into Drainage')
    ax.set_xlabel('Time (hours)'); ax.set_ylabel('Drainage Flow Rate (m3/s)')
    ax.set_title('Water Removed by Drainage Network'); ax.legend(); ax.grid(True, alpha=0.3)

    fig.suptitle('Urban Drainage Network Performance', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(VIZ_DIR, 'drainage_02_timeseries.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ================================================================
    # Figure 3: Drainage network map
    # ================================================================
    print("  Figure 3: Drainage network map...")
    fig, ax = plt.subplots(figsize=(12, 12))
    im = ax.imshow(np.flipud(dem), cmap='terrain', extent=extent, aspect='equal', alpha=0.7)

    # Plot manholes
    main_x = DOMAIN_SIZE / 2
    main_y = [350, 290, 230, 170, 110, 50]
    branch_data = [(80,320),(140,320),(80,200),(140,200),(260,320),(320,320),(260,200),(320,200)]

    ax.scatter([main_x]*6, main_y, c='red', s=80, marker='s', zorder=5, label='Main Manholes')
    bx, by = zip(*branch_data)
    ax.scatter(bx, by, c='orange', s=50, marker='o', zorder=5, label='Branch Manholes')
    ax.scatter([main_x], [10], c='green', s=120, marker='^', zorder=5, label='Outfall')

    # Draw pipes
    for i in range(5):
        ax.plot([main_x, main_x], [main_y[i], main_y[i+1]], 'r-', linewidth=2)
    ax.plot([main_x, main_x], [main_y[5], 10], 'r-', linewidth=3)
    # Branches
    branch_connections = [(0,1),(2,1),(3,1),(4,1),(5,1),(6,3),(7,3)]
    for bi, mi in branch_connections:
        ax.plot([bx[bi], main_x], [by[bi], main_y[mi]], 'orange', linewidth=1.5, alpha=0.7)

    ax.set_xlabel('Easting (m)'); ax.set_ylabel('Northing (m)')
    ax.set_title('Storm Drainage Network Layout')
    ax.legend(loc='lower right')
    plt.colorbar(im, ax=ax, label='Elevation (m)', shrink=0.8)
    plt.tight_layout()
    fig.savefig(os.path.join(VIZ_DIR, 'drainage_03_network.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ================================================================
    # Figure 4: Summary dashboard
    # ================================================================
    print("  Figure 4: Summary dashboard...")
    fig = plt.figure(figsize=(16, 14))
    fig.suptitle('Drainage Network Impact - Summary Dashboard', fontsize=16, fontweight='bold')

    # (1) Volume comparison bar chart
    ax = fig.add_subplot(2, 3, 1)
    labels = ['Surface Only', 'With Drainage']
    vols = [vol_nodrain[-1], vol_drain[-1]]
    bars = ax.bar(labels, vols, color=['red', 'blue'], alpha=0.7)
    ax.set_ylabel('Surface Water Volume (m3)')
    ax.set_title(f'Volume Reduction: {(vol_nodrain[-1]-vol_drain[-1])/vol_nodrain[-1]*100:.1f}%')
    for bar, v in zip(bars, vols):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50, f'{v:.0f}', ha='center')

    # (2) Depth reduction
    ax = fig.add_subplot(2, 3, 2)
    labels = ['Surface Only', 'With Drainage']
    depths = [hmax_nodrain[-1], hmax_drain[-1]]
    bars = ax.bar(labels, depths, color=['red', 'blue'], alpha=0.7)
    ax.set_ylabel('Max Depth (m)')
    ax.set_title(f'Depth Reduction: {(hmax_nodrain[-1]-hmax_drain[-1])/hmax_nodrain[-1]*100:.1f}%')
    for bar, v in zip(bars, depths):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{v:.3f}', ha='center')

    # (3) Drainage flow time series
    ax = fig.add_subplot(2, 3, 3)
    drain_vol_total = sum(abs(d) * 600 * CELL_SIZE * CELL_SIZE for d in drain_flow if d < 0)
    ax.bar(time_h[:len(drain_flow)], -np.array(drain_flow), width=0.07, color='blue', alpha=0.7)
    ax.set_xlabel('Time (hours)'); ax.set_ylabel('Flow (m3/s)')
    ax.set_title(f'Drainage Flow Rate\nTotal drained: {drain_vol_total:.0f} m3')
    ax.grid(True, alpha=0.3)

    # (4) Cross section with pipes
    ax = fig.add_subplot(2, 3, 4)
    xs = np.arange(GRID_SIZE) * CELL_SIZE
    ax.plot(xs, dem[100, :], 'brown', linewidth=1.5, label='Terrain')
    ax.fill_between(xs, dem[100, :], dem[100, :] + h_nodrain[100, :],
                    alpha=0.3, color='red', label='No Drain')
    ax.fill_between(xs, dem[100, :], dem[100, :] + h_drain[100, :],
                    alpha=0.3, color='blue', label='With Drain')
    ax.set_xlabel('Distance (m)'); ax.set_ylabel('Elevation (m)')
    ax.set_title('E-W Cross Section (Flood Depth)')
    ax.legend(); ax.grid(True, alpha=0.3)

    # (5) Drainage benefit metrics
    ax = fig.add_subplot(2, 3, 5)
    ax.axis('off')
    metrics = f"""
    DRAINAGE NETWORK
    ================
    Main trunk: 600mm pipes (N-S)
    Branches: 400mm pipes (8 lines)
    Manholes: 6 main + 8 branch
    Outfall: South boundary

    DRAINAGE IMPACT
    ================
    Volume reduction:  {(vol_nodrain[-1]-vol_drain[-1])/vol_nodrain[-1]*100:.1f}%
    Max depth reduction: {(hmax_nodrain[-1]-hmax_drain[-1])/hmax_nodrain[-1]*100:.1f}%
    Peak drainage:       {max(-np.array(drain_flow)):.1f} m3/s
    Total drained:       {drain_vol_total:.0f} m3

    SURFACE + 1D DRAINAGE COUPLING
    ==============================
    2D surface: ITZI partial inertia
    1D drainage: SWMM dynamic wave
    Coupling: bidirectional weir/orifice
    """
    ax.text(0.05, 0.5, metrics, transform=ax.transAxes, fontsize=10,
            fontfamily='monospace', verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='#F0F8FF', alpha=0.8))

    # (6) Side-by-side flood map
    ax = fig.add_subplot(2, 3, 6)
    # Show combined view
    h_diff = h_nodrain - h_drain
    diff_vmax = max(abs(h_diff.min()), abs(h_diff.max()), 0.1)
    im = ax.imshow(np.flipud(h_diff), cmap=diff_cmap, extent=extent, aspect='equal',
                   vmin=-diff_vmax, vmax=diff_vmax)
    ax.set_title('Flood Depth Difference\n(Red = No Drain deeper)')
    ax.set_xlabel('Easting (m)'); ax.set_ylabel('Northing (m)')
    plt.colorbar(im, ax=ax, label='Depth Diff (m)', shrink=0.8)

    plt.tight_layout()
    fig.savefig(os.path.join(VIZ_DIR, 'drainage_04_dashboard.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"\n  All outputs in: {VIZ_DIR}")
    for f in sorted(os.listdir(VIZ_DIR)):
        if f.endswith('.png'):
            print(f"    {f} ({os.path.getsize(os.path.join(VIZ_DIR, f))/1024:.0f} KB)")


if __name__ == "__main__":
    main()
