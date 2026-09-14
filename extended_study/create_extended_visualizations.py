#!/usr/bin/env python3
"""
Extended Study Visualization Script
=====================================
Creates comparison visualizations for the coupled surface-drainage study.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import matplotlib.patches as mpatches

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

flood_cmap = LinearSegmentedColormap.from_list('flood', [
    (0.0, '#F7FBFF'), (0.12, '#C6DBEF'), (0.25, '#6BAED6'),
    (0.4, '#3182BD'), (0.55, '#08519C'), (0.7, '#08306B'), (1.0, '#041838'),
])
diff_cmap = plt.cm.RdBu_r


def main():
    print("Extended Study — Visualization")
    results_file = os.path.join(OUT_DIR, "coupled_v2_results.npz")
    if not os.path.exists(results_file):
        results_file = os.path.join(OUT_DIR, "coupled_simulation_results.npz")
    if not os.path.exists(results_file):
        print(f"ERROR: No results found in {OUT_DIR}")
        print("Files:", os.listdir(OUT_DIR))
        return 1

    data = np.load(results_file, allow_pickle=True)
    print(f"Loaded: {list(data.keys())}")

    time_h = data['time_h']
    H, W = data['dem'].shape
    cell_size = 20.0
    extent = [0, W * cell_size, 0, H * cell_size]

    # ================================================================
    # Figure 1: Side-by-side flood comparison
    # ================================================================
    print("  Figure 1: Flood depth comparison...")
    h_a = data['h_final_a']
    h_b = data['h_final_b']
    bldg = data['bldg'] if 'bldg' in data else np.zeros_like(h_a, dtype=bool)
    dem = data['dem']

    vmax = max(np.max(h_a[h_a > 0.001]) if np.any(h_a > 0.001) else 0.1,
               np.max(h_b[h_b > 0.001]) if np.any(h_b > 0.001) else 0.1)
    vmax = max(vmax, 0.05)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Effect of Storm Drainage Network on Urban Flooding',
                 fontsize=14, fontweight='bold')

    # Surface only
    ax = axes[0]
    h_show = np.ma.masked_where(h_a < 0.001, h_a)
    bldg_bg = np.where(bldg, 0.3, 0)
    ax.imshow(np.flipud(bldg_bg), cmap=ListedColormap(['none', '#D3D3D3']),
              extent=extent, aspect='equal', alpha=0.5)
    im = ax.imshow(np.flipud(h_show), cmap=flood_cmap, extent=extent,
                   aspect='equal', vmin=0, vmax=vmax)
    ax.set_title(f'Surface Only\nMax={np.max(h_a):.3f}m, Vol={np.sum(h_a)*cell_size**2:.0f}m3')
    ax.set_xlabel('Easting (m)'); ax.set_ylabel('Northing (m)')
    plt.colorbar(im, ax=ax, label='Depth (m)', shrink=0.8)

    # With drainage
    ax = axes[1]
    h_show = np.ma.masked_where(h_b < 0.001, h_b)
    ax.imshow(np.flipud(bldg_bg), cmap=ListedColormap(['none', '#D3D3D3']),
              extent=extent, aspect='equal', alpha=0.5)
    im = ax.imshow(np.flipud(h_show), cmap=flood_cmap, extent=extent,
                   aspect='equal', vmin=0, vmax=vmax)
    ax.set_title(f'With Drainage\nMax={np.max(h_b):.3f}m, Vol={np.sum(h_b)*cell_size**2:.0f}m3')
    ax.set_xlabel('Easting (m)'); ax.set_ylabel('Northing (m)')
    plt.colorbar(im, ax=ax, label='Depth (m)', shrink=0.8)

    # Difference
    ax = axes[2]
    diff = h_a - h_b
    diff_m = np.ma.masked_where(np.abs(diff) < 0.001, diff)
    vlim = max(abs(np.min(diff)), abs(np.max(diff)), 0.01)
    im = ax.imshow(np.flipud(diff_m), cmap=diff_cmap, extent=extent,
                   aspect='equal', vmin=-vlim, vmax=vlim)
    ax.set_title(f'Difference (No Drain - Drain)\nBlue=Drainage reduces flooding')
    ax.set_xlabel('Easting (m)'); ax.set_ylabel('Northing (m)')
    plt.colorbar(im, ax=ax, label='Depth Diff (m)', shrink=0.8)

    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'extended_02_flood_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ================================================================
    # Figure 2: Time series comparison
    # ================================================================
    print("  Figure 2: Time series comparison...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Drainage Network Performance — Time Series', fontsize=14, fontweight='bold')

    vol_a = data['vol_a'] if 'vol_a' in data else []
    vol_b = data['vol_b'] if 'vol_b' in data else []
    hmax_a = data['hmax_a'] if 'hmax_a' in data else []
    hmax_b = data['hmax_b'] if 'hmax_b' in data else []
    flooded_a = data['flooded_a'] if 'flooded_a' in data else []
    flooded_b = data['flooded_b'] if 'flooded_b' in data else []

    if len(vol_a) > 0:
        ax = axes[0, 0]
        ax.plot(time_h, vol_a, 'r-o', lw=2, ms=4, label='Surface Only')
        ax.plot(time_h[:len(vol_b)], vol_b, 'b-s', lw=2, ms=4, label='With Drainage')
        ax.set_xlabel('Time (hours)'); ax.set_ylabel('Water Volume (m3)')
        ax.set_title('Surface Water Volume'); ax.legend(); ax.grid(True, alpha=0.3)

        ax = axes[0, 1]
        ax.plot(time_h, hmax_a, 'r-o', lw=2, ms=4, label='Surface Only')
        ax.plot(time_h[:len(hmax_b)], hmax_b, 'b-s', lw=2, ms=4, label='With Drainage')
        ax.set_xlabel('Time (hours)'); ax.set_ylabel('Max Depth (m)')
        ax.set_title('Maximum Inundation Depth'); ax.legend(); ax.grid(True, alpha=0.3)

        ax = axes[1, 0]
        ax.plot(time_h, np.array(flooded_a) * cell_size**2, 'r-o', lw=2, ms=4, label='Surface Only')
        ax.plot(time_h[:len(flooded_b)], np.array(flooded_b) * cell_size**2, 'b-s', lw=2, ms=4, label='With Drainage')
        ax.set_xlabel('Time (hours)'); ax.set_ylabel('Flooded Area (m2)')
        ax.set_title('Inundated Area'); ax.legend(); ax.grid(True, alpha=0.3)

        # Reduction bar chart
        ax = axes[1, 1]
        categories = ['Volume', 'Max Depth', 'Flooded Area']
        if vol_a[-1] > 0:
            reductions = [
                (vol_a[-1] - vol_b[-1]) / vol_a[-1] * 100,
                (hmax_a[-1] - hmax_b[-1]) / hmax_a[-1] * 100 if hmax_a[-1] > 0 else 0,
                (flooded_a[-1] - flooded_b[-1]) / flooded_a[-1] * 100 if flooded_a[-1] > 0 else 0,
            ]
            colors = ['#2171B5', '#6BAED6', '#C6DBEF']
            bars = ax.bar(categories, reductions, color=colors)
            ax.set_ylabel('Reduction (%)'); ax.set_title('Drainage Impact')
            for bar, val in zip(bars, reductions):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f'{val:.1f}%', ha='center', fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'extended_03_timeseries.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ================================================================
    # Figure 3: Summary Dashboard
    # ================================================================
    print("  Figure 3: Summary dashboard...")
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle('Extended Study: Coupled Urban Flood & Drainage — Summary Dashboard',
                 fontsize=16, fontweight='bold')

    # (1) DEM + buildings
    ax = fig.add_subplot(2, 3, 1)
    terrain = np.where(bldg, np.nan, dem)
    im = ax.imshow(np.flipud(terrain), cmap='terrain', extent=extent, aspect='equal')
    ax.set_title('Terrain Elevation'); ax.set_xlabel('E (m)'); ax.set_ylabel('N (m)')
    plt.colorbar(im, ax=ax, label='Elev (m)', shrink=0.8)

    # (2) Flood with drainage
    ax = fig.add_subplot(2, 3, 2)
    h_show = np.ma.masked_where(h_b < 0.001, h_b)
    bldg_bg = np.where(bldg, 0.3, 0)
    ax.imshow(np.flipud(bldg_bg), cmap=ListedColormap(['none', '#aaa']),
              extent=extent, aspect='equal', alpha=0.5)
    im = ax.imshow(np.flipud(h_show), cmap=flood_cmap, extent=extent, aspect='equal', vmin=0, vmax=vmax)
    ax.set_title(f'Flood with Drainage\nMax Depth: {np.max(h_b):.3f}m')
    ax.set_xlabel('E (m)'); ax.set_ylabel('N (m)')
    plt.colorbar(im, ax=ax, label='Depth (m)', shrink=0.8)

    # (3) Depth difference
    ax = fig.add_subplot(2, 3, 3)
    vlim = max(abs(diff.min()), abs(diff.max()), 0.01)
    im = ax.imshow(np.flipud(diff), cmap=diff_cmap, extent=extent,
                   aspect='equal', vmin=-vlim, vmax=vlim)
    ax.set_title('Depth Difference\n(Red = Drainage reduces flood)')
    ax.set_xlabel('E (m)'); ax.set_ylabel('N (m)')
    plt.colorbar(im, ax=ax, label='Diff (m)', shrink=0.8)

    # (4) Cross-section
    ax = fig.add_subplot(2, 3, 4)
    mid_row = H // 2
    xs = np.arange(W) * cell_size
    ax.fill_between(xs, dem[mid_row,:], dem[mid_row,:] + h_a[mid_row,:],
                    alpha=0.4, color='red', label='Surface Only')
    ax.fill_between(xs, dem[mid_row,:], dem[mid_row,:] + h_b[mid_row,:],
                    alpha=0.4, color='blue', label='With Drainage')
    ax.plot(xs, dem[mid_row,:], 'brown', lw=1.5)
    ax.set_xlabel('Distance (m)'); ax.set_ylabel('Elevation (m)')
    ax.set_title('Cross-Section (Mid-Domain, W→E)'); ax.legend(); ax.grid(True, alpha=0.3)

    # (5) Volume time series
    ax = fig.add_subplot(2, 3, 5)
    if len(vol_a) > 0:
        ax.plot(time_h, vol_a, 'r-o', lw=2, ms=4, label='Surface Only')
        ax.plot(time_h[:len(vol_b)], vol_b, 'b-s', lw=2, ms=4, label='With Drainage')
        expected_vol = RAIN_TOTAL_MM / 1000 * H * W * cell_size**2
        ax.axhline(y=expected_vol, color='green', ls='--', label=f'Total Rain ({expected_vol:.0f} m3)')
        ax.set_xlabel('Time (h)'); ax.set_ylabel('Volume (m3)')
        ax.set_title('Mass Balance'); ax.legend(); ax.grid(True, alpha=0.3)

    # (6) Key metrics
    ax = fig.add_subplot(2, 3, 6)
    ax.axis('off')
    vol_red = (vol_a[-1] - vol_b[-1]) / vol_a[-1] * 100 if len(vol_a) > 0 and vol_a[-1] > 0 else 0
    depth_red = (hmax_a[-1] - hmax_b[-1]) / hmax_a[-1] * 100 if len(hmax_a) > 0 and hmax_a[-1] > 0 else 0
    flood_red = (flooded_a[-1] - flooded_b[-1]) / flooded_a[-1] * 100 if len(flooded_a) > 0 and flooded_a[-1] > 0 else 0

    metrics = f"""
    EXTENDED STUDY — SHENZHEN FUTIAN
    ================================
    Coupled Urban Flood & Drainage
    Model

    DOMAIN
    Area: {H*cell_size/1000:.1f}km x {W*cell_size/1000:.1f}km
    Resolution: {cell_size:.0f}m ({H}x{W} cells)
    Simulation: {SIM_HOURS:.0f}h ({RAIN_TOTAL_MM:.0f}mm design storm)

    DRAINAGE NETWORK
    Main trunk: 800mm circular pipes
    Branches: 500mm circular pipes
    Node spacing: ~160m
    Pipe slopes follow terrain

    RESULTS
    Volume reduction: {vol_red:.1f}%
    Max depth reduction: {depth_red:.1f}%
    Flooded area reduction: {flood_red:.1f}%

    SURFACE+1D COUPLING
    2D surface: Diffusion wave
    1D drainage: Manning's equation
    Coupling: Weir/orifice exchange
    """.strip()
    ax.text(0.05, 0.5, metrics, transform=ax.transAxes, fontsize=10,
            fontfamily='monospace', verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='#F0F8FF', alpha=0.8))

    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'extended_04_dashboard.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"\n  Outputs in: {OUT_DIR}")
    for f in sorted(os.listdir(OUT_DIR)):
        if f.endswith('.png'):
            print(f"    {f} ({os.path.getsize(os.path.join(OUT_DIR, f))/1024:.0f} KB)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
