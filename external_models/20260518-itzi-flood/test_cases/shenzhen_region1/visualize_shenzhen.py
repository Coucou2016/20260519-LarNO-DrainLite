#!/usr/bin/env python3
"""Visualize ITZI Shenzhen Region1 results vs MIKE+ reference."""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

CASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CASE_DIR, 'output')
VIZ_DIR = os.path.join(CASE_DIR, 'visualization_output')
DEM_PATH = os.path.join(CASE_DIR, 'input_data', 'geodata', 'region1_20m', 'dem.npy')
os.makedirs(VIZ_DIR, exist_ok=True)

CELL = 20.0
Y0, Y1, X0, X1 = 80, 280, 120, 400

flood_cmap = LinearSegmentedColormap.from_list('flood', [
    (0.0, '#F7FBFF'), (0.12, '#C6DBEF'), (0.25, '#6BAED6'),
    (0.4, '#3182BD'), (0.55, '#08519C'), (0.7, '#08306B'), (1.0, '#041838'),
])
diff_cmap = plt.cm.RdBu_r

events = ['event1', 'event20', 'event65', 'event66',
          'event67', 'event68', 'event69', 'event70']


def load_results():
    all_data = {}
    for evt in events:
        fp = os.path.join(DATA_DIR, f'{evt}_itzi.npz')
        if not os.path.exists(fp):
            print(f"Missing: {fp}")
            continue
        d = np.load(fp, allow_pickle=True)
        all_data[evt] = {
            'h_ref': d['h_ref'], 'peak_ref': d['peak_ref'],
            'h_s': d['h_itzi_surf'], 'h_p': d['h_itzi_pipe'],
            'rec_s': d['rec_s'].item(),
            'rec_p': d['rec_p'].item(),
        }
    return all_data


def fig_peak_comparison(all_data, dem, bldg):
    """8 events x 4 rows: MIKE+ / ITZI-S / ITZI-P / Diff."""
    H, W = dem.shape
    extent = [0, W * CELL, 0, H * CELL]
    bldg_bg = np.where(bldg, 0.3, 0)
    n_evts = len(all_data)
    evts = list(all_data.keys())

    global_vmax = max(
        max(np.max(all_data[e]['peak_ref']) for e in evts),
        max(np.max(all_data[e]['h_s']) for e in evts),
        max(np.max(all_data[e]['h_p']) for e in evts),
    ) * 1.1

    fig, axes = plt.subplots(4, n_evts, figsize=(2.2 * n_evts, 14))
    fig.suptitle(
        'ITZI vs MIKE+ — Shenzhen Region1 Peak Flood Depth (4km x 5.6km)',
        fontsize=14, fontweight='bold',
    )
    row_labels = ['MIKE+', 'ITZI Surface', 'ITZI + Pipes', 'Diff (ITZI+P - MIKE+)']

    for col, evt in enumerate(evts):
        d = all_data[evt]
        peak_m = np.max(d['peak_ref'])
        peak_s = np.max(d['h_s'])
        peak_p = np.max(d['h_p'])

        for row_idx, (h_data, title) in enumerate([
            (d['peak_ref'], f'MIKE+\n{peak_m:.3f}m'),
            (d['h_s'], f'ITZI-S\n{peak_s:.3f}m ({(peak_s-peak_m)/peak_m*100:+.0f}%)'),
            (d['h_p'], f'ITZI+P\n{peak_p:.3f}m ({(peak_p-peak_m)/peak_m*100:+.0f}%)'),
        ]):
            ax = axes[row_idx, col]
            h_show = np.ma.masked_where(h_data < 0.01, h_data)
            ax.imshow(np.flipud(bldg_bg), cmap=ListedColormap(['none', '#CCC']),
                      extent=extent, aspect='equal', alpha=0.5)
            ax.imshow(np.flipud(h_show), cmap=flood_cmap, extent=extent,
                      aspect='equal', vmin=0, vmax=global_vmax)
            ax.set_title(title, fontsize=8)
            if col == 0:
                ax.set_ylabel(row_labels[row_idx], fontsize=9, fontweight='bold')

        ax = axes[3, col]
        diff = d['h_p'] - d['peak_ref']
        vl = max(abs(np.min(diff)), abs(np.max(diff)), 0.1)
        ax.imshow(np.flipud(diff), cmap=diff_cmap, extent=extent,
                  aspect='equal', vmin=-vl, vmax=vl)
        ax.set_title('Diff', fontsize=8)
        if col == 0:
            ax.set_ylabel(row_labels[3], fontsize=9, fontweight='bold')

    plt.tight_layout()
    out = os.path.join(VIZ_DIR, 'shenzhen_01_peak_comparison.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {out}")


def fig_timeseries(all_data):
    """Volume and peak depth time series for all events."""
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    fig.suptitle('ITZI Time Series — Volume & Peak Depth', fontsize=14, fontweight='bold')

    for idx, evt in enumerate(all_data.keys()):
        ax = axes[idx // 4, idx % 4]
        d = all_data[evt]
        rec_s, rec_p = d['rec_s'], d['rec_p']
        ax2 = ax.twinx()
        ax.plot(rec_s['time_h'], rec_s['vol_m3'], 'b-', label='ITZI-S vol', lw=1.5)
        ax.plot(rec_p['time_h'], rec_p['vol_m3'], 'g--', label='ITZI-P vol', lw=1.5)
        ax2.plot(rec_s['time_h'], rec_s['hmax_m'], 'r-', label='ITZI-S hmax', lw=1)
        ax2.plot(rec_p['time_h'], rec_p['hmax_m'], 'm--', label='ITZI-P hmax', lw=1)
        ax.set_title(evt, fontsize=10)
        ax.set_xlabel('Time (h)')
        ax.set_ylabel('Volume (m3)')
        ax2.set_ylabel('hmax (m)')
        if idx == 0:
            ax.legend(fontsize=7, loc='upper left')
            ax2.legend(fontsize=7, loc='upper right')

    plt.tight_layout()
    out = os.path.join(VIZ_DIR, 'shenzhen_02_timeseries.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {out}")


def fig_summary_bar(all_data):
    """Bar chart: peak depth comparison across events."""
    evts = list(all_data.keys())
    mike = [all_data[e]['peak_ref'].max() for e in evts]
    surf = [all_data[e]['h_s'].max() for e in evts]
    pipe = [all_data[e]['h_p'].max() for e in evts]

    x = np.arange(len(evts))
    w = 0.25
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - w, mike, w, label='MIKE+', color='#08519C')
    ax.bar(x, surf, w, label='ITZI Surface', color='#6BAED6')
    ax.bar(x + w, pipe, w, label='ITZI + Pipes', color='#31A354')
    ax.set_xticks(x)
    ax.set_xticklabels(evts, rotation=30)
    ax.set_ylabel('Peak Depth (m)')
    ax.set_title('Peak Flood Depth — ITZI vs MIKE+ (Shenzhen Region1)')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    out = os.path.join(VIZ_DIR, 'shenzhen_03_summary_bar.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {out}")


def fig_terrain_network(dem, bldg):
    """Terrain + building + pipe network overview."""
    H, W = dem.shape
    extent = [0, W * CELL, 0, H * CELL]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    im = ax.imshow(np.flipud(dem), cmap='terrain', extent=extent, aspect='equal')
    plt.colorbar(im, ax=ax, label='Elevation (m)', shrink=0.8)
    ax.set_title('DEM — Shenzhen Sub-region')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')

    ax = axes[1]
    bldg_map = np.where(bldg, 1, 0).astype(float)
    ax.imshow(np.flipud(bldg_map), cmap=ListedColormap(['#E8F4E8', '#555']),
              extent=extent, aspect='equal', alpha=0.8)
    ax.imshow(np.flipud(dem), cmap='terrain', extent=extent, aspect='equal', alpha=0.4)

    net_path = os.path.join(CASE_DIR, 'input_data', 'networks', 'osm_merged_network.npz')
    if os.path.exists(net_path):
        net = np.load(net_path, allow_pickle=True)
        nodes = net['nodes'].tolist()
        xs, ys = [], []
        for n in nodes:
            nr, nc = n['row'] - Y0, n['col'] - X0
            if 0 <= nr < H and 0 <= nc < W:
                xs.append(nc * CELL)
                ys.append(nr * CELL)
        ax.scatter(xs, ys, c='red', s=2, alpha=0.6, label='Pipe nodes')
        ax.legend(fontsize=8)

    ax.set_title('Buildings + OSM Pipe Network')
    ax.set_xlabel('X (m)')

    plt.tight_layout()
    out = os.path.join(VIZ_DIR, 'shenzhen_00_terrain_network.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {out}")


def main():
    print("=" * 60)
    print("  Visualizing Shenzhen Region1 ITZI Results")
    print("=" * 60)

    all_data = load_results()
    if not all_data:
        print("No simulation results found. Run run_itzi_shenzhen.py first.")
        return

    dem_full = np.load(DEM_PATH)
    dem = dem_full[Y0:Y1, X0:X1]
    bldg = dem >= 49.9

    fig_terrain_network(dem, bldg)
    fig_peak_comparison(all_data, dem, bldg)
    fig_timeseries(all_data)
    fig_summary_bar(all_data)

    print(f"\nAll visualizations saved to {VIZ_DIR}")


if __name__ == "__main__":
    main()
