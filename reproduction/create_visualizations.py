#!/usr/bin/env python3
"""
LarNO Reproduction — Visualization & Statistics Pipeline
=========================================================
Generates comprehensive visualizations and statistics from the
Shenzhen Futian benchmark dataset (MIKE+ reference data).

Output:
  fig1_terrain.png       — DEM, building mask, slope analysis
  fig2_flood_events.png  — Flood depth maps for test events
  fig3_timeseries.png    — Time series of volume, depth, flooded area
  fig4_statistics.png    — Statistical summary (histograms, box plots)
  fig5_comparison.png    — Multi-event comparison dashboard
  statistics.xlsx        — Per-event metrics table
"""

import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from pathlib import Path

# ============================================================
# Paths
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROOT = PROJECT_ROOT / "LarNO-main" / "benchmark" / "urbanflood"
DEM_PATH = ROOT / "geodata" / "region1_20m" / "dem.npy"
FLOOD_DIR = ROOT / "flood" / "region1_20m"
OUT_DIR = Path(__file__).resolve().parent / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CELL_SIZE = 20.0  # meters
# Domain extent for plots
EXTENT_M = [0, 560 * CELL_SIZE, 0, 400 * CELL_SIZE]  # E, W, S, N in meters

flood_cmap = LinearSegmentedColormap.from_list('flood', [
    (0.0, '#F7FBFF'), (0.12, '#C6DBEF'), (0.25, '#6BAED6'),
    (0.4, '#3182BD'), (0.55, '#08519C'), (0.7, '#08306B'), (1.0, '#041838'),
])

elevation_cmap = plt.cm.terrain
diff_cmap = plt.cm.RdBu_r


def main():
    print("=" * 60)
    print("  LarNO Reproduction — Visualizations & Statistics")
    print("=" * 60)

    # Load DEM
    print("\n[1/5] Loading terrain data...")
    dem = np.load(DEM_PATH, allow_pickle=True)
    building_mask = dem >= 49.9
    dy, dx = np.gradient(dem, CELL_SIZE)
    slope = np.sqrt(dx ** 2 + dy ** 2)

    # Load test events
    print("[2/5] Loading flood event data...")
    test_events = sorted([
        d for d in os.listdir(FLOOD_DIR)
        if d.startswith("event") and os.path.isdir(os.path.join(FLOOD_DIR, d))
    ])
    print(f"  Found {len(test_events)} test events: {test_events[0]} - {test_events[-1]}")

    event_data = {}
    for evt in test_events[:8]:  # Load 8 events for detailed analysis
        evt_dir = os.path.join(FLOOD_DIR, evt)
        h_file = os.path.join(evt_dir, "h.npy")
        r_file = os.path.join(evt_dir, "rainfall.npy")
        if os.path.exists(h_file) and os.path.exists(r_file):
            h = np.load(h_file, allow_pickle=True)  # (T, H, W) in meters
            r = np.load(r_file, allow_pickle=True)  # (T, H, W) in mm/h
            event_data[evt] = {'h': h, 'rain': r}
            print(f"    {evt}: h={h.shape}, range=[{h.min():.3f}, {h.max():.3f}]m")

    # ================================================================
    # Figure 1: Terrain Analysis
    # ================================================================
    print("\n[3/5] Creating terrain analysis figure...")
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle('Shenzhen Futian District — Terrain Analysis (20m Resolution)',
                 fontsize=15, fontweight='bold')

    # (a) DEM with buildings
    ax = fig.add_subplot(2, 3, 1)
    terrain_dem = np.where(building_mask, np.nan, dem)
    im = ax.imshow(terrain_dem, cmap=elevation_cmap, extent=EXTENT_M,
                   aspect='equal', origin='lower')
    ax.set_title(f'Ground Elevation\n(min={np.nanmin(terrain_dem):.1f}m, '
                 f'max={np.nanmax(terrain_dem):.1f}m)')
    ax.set_xlabel('Easting (m)'); ax.set_ylabel('Northing (m)')
    plt.colorbar(im, ax=ax, label='Elevation (m)', shrink=0.8)

    # (b) Building footprints
    ax = fig.add_subplot(2, 3, 2)
    bldg_display = np.where(building_mask, 1.0, 0.0)
    ax.imshow(bldg_display, cmap=ListedColormap(['#F0F0F0', '#8B0000']),
              extent=EXTENT_M, aspect='equal', origin='lower')
    ax.set_title(f'Building Footprints\n({building_mask.sum():,} cells = '
                 f'{100*building_mask.sum()/building_mask.size:.1f}% of area)')
    ax.set_xlabel('Easting (m)'); ax.set_ylabel('Northing (m)')

    # (c) Terrain slope
    ax = fig.add_subplot(2, 3, 3)
    valid_slope = np.where(building_mask, np.nan, slope)
    im = ax.imshow(valid_slope, cmap='YlOrRd', extent=EXTENT_M,
                   aspect='equal', origin='lower', vmin=0, vmax=np.nanpercentile(valid_slope, 95))
    ax.set_title(f'Terrain Slope\n(mean={np.nanmean(valid_slope):.3f}, '
                 f'median={np.nanmedian(valid_slope):.3f})')
    ax.set_xlabel('Easting (m)'); ax.set_ylabel('Northing (m)')
    plt.colorbar(im, ax=ax, label='Slope (m/m)', shrink=0.8)

    # (d) Slope histogram
    ax = fig.add_subplot(2, 3, 4)
    slope_flat = valid_slope[~np.isnan(valid_slope)]
    ax.hist(np.clip(slope_flat, 0, 2), bins=50, color='#D73027', edgecolor='white', alpha=0.8)
    ax.axvline(np.mean(slope_flat), color='blue', linestyle='--', label=f'Mean: {np.mean(slope_flat):.3f}')
    ax.axvline(np.median(slope_flat), color='green', linestyle='--', label=f'Median: {np.median(slope_flat):.3f}')
    ax.set_xlabel('Slope (m/m)'); ax.set_ylabel('Cell Count')
    ax.set_title('Slope Distribution'); ax.legend()

    # (e) Elevation histogram
    ax = fig.add_subplot(2, 3, 5)
    valid_dem = dem[~building_mask]
    ax.hist(valid_dem, bins=50, color='#4575B4', edgecolor='white', alpha=0.8)
    ax.set_xlabel('Elevation (m)'); ax.set_ylabel('Cell Count')
    ax.set_title(f'Ground Elevation Distribution\n(N={len(valid_dem):,} cells)')

    # (f) Key metrics
    ax = fig.add_subplot(2, 3, 6)
    ax.axis('off')
    metrics = f"""
    SHENZHEN FUTIAN — TERRAIN SUMMARY
    ==================================
    Domain: 8.0km x 11.2km (89.6 km2)
    Resolution: 20m (400 x 560 cells)
    Total cells: 224,000

    GROUND ELEVATION
    Min: {np.nanmin(valid_dem):.2f} m
    Max: {np.nanmax(valid_dem):.2f} m
    Mean: {np.nanmean(valid_dem):.2f} m
    Std: {np.nanstd(valid_dem):.2f} m

    BUILDINGS
    Cells: {building_mask.sum():,} ({100*building_mask.sum()/building_mask.size:.1f}%)
    Represented as 50m wall (blocking flow)

    TERRAIN SLOPE
    Min: {np.nanmin(valid_slope):.4f}
    Max: {np.nanmax(valid_slope):.4f}
    Mean: {np.nanmean(valid_slope):.4f}
    Median: {np.nanmedian(valid_slope):.4f}

    BENCHMARK DATA
    Reference solver: MIKE+ (hydraulic)
    Events: {len(test_events)} test events
    Duration: 6 hours (72 x 5-min steps)
    """
    ax.text(0.05, 0.5, metrics, transform=ax.transAxes, fontsize=10,
            fontfamily='monospace', verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='#F0F8FF', alpha=0.8))

    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'fig1_terrain.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved: fig1_terrain.png")

    # ================================================================
    # Figure 2: Flood Depth Maps (Multi-Event)
    # ================================================================
    print("[4/5] Creating flood depth maps...")
    n_events = min(8, len(event_data))
    evt_names = sorted(event_data.keys())[:n_events]

    fig, axes = plt.subplots(2, 4, figsize=(24, 11))
    fig.suptitle('Flood Depth at Peak — Test Events (MIKE+ Reference)',
                 fontsize=14, fontweight='bold')

    global_vmax = 0
    for evt in evt_names:
        hmax = np.max(event_data[evt]['h'], axis=0)
        global_vmax = max(global_vmax, np.nanmax(hmax))

    for idx, evt in enumerate(evt_names):
        ax = axes[idx // 4, idx % 4]
        h = event_data[evt]['h']
        h_peak = np.max(h, axis=0)  # Max depth over time at each cell
        h_masked = np.ma.masked_where(h_peak < 0.01, h_peak)

        # Building overlay
        bldg_bg = np.where(building_mask, 0.3, 0)
        ax.imshow(bldg_bg, cmap=ListedColormap(['none', '#888888']),
                  extent=EXTENT_M, aspect='equal', origin='lower', alpha=0.5)
        im = ax.imshow(h_masked, cmap=flood_cmap, extent=EXTENT_M,
                       aspect='equal', origin='lower', vmin=0, vmax=global_vmax)

        hmax_val = np.nanmax(h_peak)
        flooded = np.sum(h_peak > 0.03) * CELL_SIZE * CELL_SIZE / 1e6
        ax.set_title(f'{evt}\nMax depth={hmax_val:.2f}m, Flooded={flooded:.1f}km²',
                    fontsize=9)
        ax.set_xlabel('E (m)', fontsize=7); ax.set_ylabel('N (m)', fontsize=7)

    cbar_ax = fig.add_axes([0.92, 0.08, 0.01, 0.84])
    fig.colorbar(im, cax=cbar_ax, label='Max Water Depth (m)')
    plt.tight_layout(rect=[0, 0, 0.91, 0.96])
    fig.savefig(os.path.join(OUT_DIR, 'fig2_flood_events.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved: fig2_flood_events.png")

    # ================================================================
    # Figure 3: Time Series Analysis
    # ================================================================
    print("[5/5] Creating time series and statistics...")
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Flood Dynamics — Time Series Analysis', fontsize=14, fontweight='bold')

    # Select representative events
    rep_events = sorted(event_data.keys())[:6]
    colors = plt.cm.Set2(np.linspace(0, 1, len(rep_events)))

    # (a) Water volume over time
    ax = axes[0, 0]
    for evt, c in zip(rep_events, colors):
        h = event_data[evt]['h']
        t = np.arange(0, h.shape[0]) * 5 / 60  # hours
        vol = [np.sum(h[i] * ~building_mask) * CELL_SIZE * CELL_SIZE for i in range(h.shape[0])]
        ax.plot(t, vol, color=c, linewidth=1.5, label=evt, alpha=0.8)
    ax.set_xlabel('Time (hours)'); ax.set_ylabel('Water Volume (m³)')
    ax.set_title('Surface Water Volume'); ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.3)

    # (b) Max depth over time
    ax = axes[0, 1]
    for evt, c in zip(rep_events, colors):
        h = event_data[evt]['h']
        t = np.arange(0, h.shape[0]) * 5 / 60
        hmax_t = [np.nanmax(h[i]) for i in range(h.shape[0])]
        ax.plot(t, hmax_t, color=c, linewidth=1.5, alpha=0.8)
    ax.set_xlabel('Time (hours)'); ax.set_ylabel('Max Water Depth (m)')
    ax.set_title('Maximum Inundation Depth'); ax.grid(True, alpha=0.3)

    # (c) Flooded area over time
    ax = axes[0, 2]
    for evt, c in zip(rep_events, colors):
        h = event_data[evt]['h']
        t = np.arange(0, h.shape[0]) * 5 / 60
        flooded = [np.sum((h[i] > 0.03) & ~building_mask) * CELL_SIZE * CELL_SIZE / 1e6
                   for i in range(h.shape[0])]
        ax.plot(t, flooded, color=c, linewidth=1.5, alpha=0.8)
    ax.set_xlabel('Time (hours)'); ax.set_ylabel('Flooded Area (km²)')
    ax.set_title('Inundated Area (>3cm depth)'); ax.grid(True, alpha=0.3)

    # (d) Peak depth histogram (all events)
    ax = axes[1, 0]
    all_peak_depths = []
    for evt in rep_events:
        h = event_data[evt]['h']
        peak = np.max(h, axis=0)
        all_peak_depths.append(peak[peak > 0.01])
    ax.hist(all_peak_depths, bins=30, histtype='stepfilled', alpha=0.6,
            color=colors, label=[f'{e}' for e in rep_events])
    ax.set_xlabel('Peak Water Depth (m)'); ax.set_ylabel('Cell Count')
    ax.set_title('Peak Depth Distribution'); ax.legend(fontsize=7)

    # (e) Event comparison: max depth vs flooded area
    ax = axes[1, 1]
    all_events = sorted(event_data.keys())
    x_vals, y_vals, labels = [], [], []
    for evt in all_events:
        h = event_data[evt]['h']
        peak = np.max(h, axis=0)
        x_vals.append(np.nanmax(peak))
        y_vals.append(np.sum((peak > 0.03) & ~building_mask) * CELL_SIZE * CELL_SIZE / 1e6)
        labels.append(evt)
    ax.scatter(x_vals, y_vals, c='#2171B5', s=60, alpha=0.7)
    for i, lbl in enumerate(labels):
        ax.annotate(lbl.replace('event', ''), (x_vals[i], y_vals[i]),
                    fontsize=7, ha='center', va='bottom')
    ax.set_xlabel('Peak Max Depth (m)'); ax.set_ylabel('Flooded Area (km²)')
    ax.set_title('Event Comparison: Depth vs Area'); ax.grid(True, alpha=0.3)

    # (f) Statistical summary
    ax = axes[1, 2]
    ax.axis('off')
    peaks = [np.nanmax(event_data[e]['h']) for e in all_events]
    volumes = [np.sum(event_data[e]['h'][-1] * ~building_mask) * CELL_SIZE * CELL_SIZE
               for e in all_events]
    flooded_areas = [np.sum((np.max(event_data[e]['h'], axis=0) > 0.03) & ~building_mask)
                      * CELL_SIZE * CELL_SIZE / 1e6 for e in all_events]

    stats = f"""
    BENCHMARK STATISTICS ({len(all_events)} events)
    ======================================
    PEAK WATER DEPTH (m)
      Mean ± Std: {np.mean(peaks):.3f} ± {np.std(peaks):.3f}
      Range: [{np.min(peaks):.3f}, {np.max(peaks):.3f}]

    FINAL VOLUME (m³)
      Mean ± Std: {np.mean(volumes)/1e6:.1f}M±{np.std(volumes)/1e6:.1f}M

    FLOODED AREA (km², >3cm)
      Mean ± Std: {np.mean(flooded_areas):.2f}±{np.std(flooded_areas):.2f}
      Range: [{np.min(flooded_areas):.2f}, {np.max(flooded_areas):.2f}]

    DOMAIN
      Area: 89.6 km²
      Grid: 400x560 @ 20m
      Valid cells: {(~building_mask).sum():,}
    """
    ax.text(0.05, 0.5, stats, transform=ax.transAxes, fontsize=10,
            fontfamily='monospace', verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='#F0F8FF', alpha=0.8))

    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'fig3_timeseries_stats.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved: fig3_timeseries_stats.png")

    # ================================================================
    # Save statistics to Excel
    # ================================================================
    try:
        import pandas as pd
        rows = []
        for evt in sorted(event_data.keys()):
            h = event_data[evt]['h']
            peak = np.max(h, axis=0)
            hmax = np.nanmax(peak)
            hmean = np.nanmean(peak[peak > 0.01]) if np.any(peak > 0.01) else 0
            flooded = np.sum((peak > 0.03) & ~building_mask) * CELL_SIZE * CELL_SIZE / 1e6
            vol_final = np.sum(h[-1] * ~building_mask) * CELL_SIZE * CELL_SIZE
            rows.append({
                'Event': evt,
                'PeakMaxDepth_m': round(hmax, 4),
                'PeakMeanDepth_m': round(hmean, 4),
                'FloodedArea_km2': round(flooded, 3),
                'FinalVolume_m3': round(vol_final, 1),
                'PeakCells_n': int(np.sum(peak > 0.03)),
            })

        df = pd.DataFrame(rows)
        xlsx_path = os.path.join(OUT_DIR, 'statistics.xlsx')
        with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='EventMetrics')
        print(f"    Saved: statistics.xlsx ({len(df)} events)")
    except Exception as e:
        print(f"    Warning: Could not save Excel: {e}")

    print(f"\n{'='*60}")
    print(f"  All outputs saved to: {OUT_DIR}")
    print(f"{'='*60}")
    for f in sorted(os.listdir(OUT_DIR)):
        if f.endswith('.png') or f.endswith('.xlsx') or f.endswith('.npz'):
            size_kb = os.path.getsize(os.path.join(OUT_DIR, f)) / 1024
            print(f"  {f} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
