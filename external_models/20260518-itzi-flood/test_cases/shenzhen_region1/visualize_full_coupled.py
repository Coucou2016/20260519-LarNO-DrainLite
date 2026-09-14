#!/usr/bin/env python3
"""Visualize full-domain ITZI-SWMM coupled results vs MIKE+ reference."""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

CASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CASE_DIR, 'output', 'full_domain_swmm')
VIZ_DIR = os.path.join(CASE_DIR, 'visualization_output', 'full_domain_swmm')
DEM_PATH = os.path.join(CASE_DIR, 'input_data', 'geodata', 'region1_20m', 'dem.npy')
os.makedirs(VIZ_DIR, exist_ok=True)

CELL = 20.0
SUB_SLICE = (slice(80, 280), slice(120, 400))

flood_cmap = LinearSegmentedColormap.from_list('flood', [
    (0.0, '#F7FBFF'), (0.12, '#C6DBEF'), (0.25, '#6BAED6'),
    (0.4, '#3182BD'), (0.55, '#08519C'), (0.7, '#08306B'), (1.0, '#041838'),
])


def load_all_results():
    results = {}
    for fn in sorted(os.listdir(DATA_DIR)):
        if fn.endswith('_coupled.npz'):
            evt = fn.replace('_coupled.npz', '')
            d = np.load(os.path.join(DATA_DIR, fn), allow_pickle=True)
            results[evt] = {
                'peak_ref': d['peak_ref'],
                'h_surf': d['h_surf'],
                'h_swmm': d['h_swmm'],
                'rec_surf': d['rec_surf'].item(),
                'rec_swmm': d['rec_swmm'].item(),
                'domain': str(d.get('domain', 'sub')),
            }
    return results


def fig_peak_maps(results, dem, bldg, domain_mode):
    evts = list(results.keys())
    n = len(evts)
    fig, axes = plt.subplots(3, n, figsize=(2.5 * n, 10))
    if n == 1:
        axes = axes.reshape(3, 1)
    fig.suptitle(
        f'ITZI-SWMM Full Coupling vs MIKE+ — Peak Depth ({domain_mode} domain)',
        fontsize=13, fontweight='bold',
    )
    H, W = dem.shape
    extent = [0, W * CELL, 0, H * CELL]
    bldg_bg = np.where(bldg, 0.3, 0)
    vmax = max(
        max(np.max(r['peak_ref']) for r in results.values()),
        max(np.max(r['h_swmm']) for r in results.values()),
    ) * 1.05

    row_titles = ['MIKE+ Reference', 'ITZI Surface', 'ITZI + SWMM']
    for col, evt in enumerate(evts):
        d = results[evt]
        for row, (h, title) in enumerate([
            (d['peak_ref'], f'{evt}\nMIKE+ {np.max(d["peak_ref"]):.2f}m'),
            (d['h_surf'], f'ITZI-S {np.max(d["h_surf"]):.2f}m'),
            (d['h_swmm'], f'ITZI-SWMM {np.max(d["h_swmm"]):.2f}m'),
        ]):
            ax = axes[row, col]
            h_show = np.ma.masked_where(h < 0.01, h)
            ax.imshow(np.flipud(bldg_bg), cmap=ListedColormap(['none', '#CCC']),
                      extent=extent, aspect='equal', alpha=0.4)
            ax.imshow(np.flipud(h_show), cmap=flood_cmap, extent=extent,
                      aspect='equal', vmin=0, vmax=vmax)
            ax.set_title(title, fontsize=8)
            if col == 0:
                ax.set_ylabel(row_titles[row], fontsize=9)

    plt.tight_layout()
    out = os.path.join(VIZ_DIR, 'coupled_01_peak_maps.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {out}")


def fig_comparison_bars(results):
    evts = list(results.keys())
    mike = [np.max(results[e]['peak_ref']) for e in evts]
    surf = [np.max(results[e]['h_surf']) for e in evts]
    swmm = [np.max(results[e]['h_swmm']) for e in evts]

    x = np.arange(len(evts))
    w = 0.25
    fig, ax = plt.subplots(figsize=(max(10, len(evts) * 1.2), 5))
    ax.bar(x - w, mike, w, label='MIKE+ Reference', color='#08519C')
    ax.bar(x, surf, w, label='ITZI Surface', color='#6BAED6')
    ax.bar(x + w, swmm, w, label='ITZI + SWMM', color='#31A354')
    ax.set_xticks(x)
    ax.set_xticklabels(evts, rotation=30)
    ax.set_ylabel('Peak Depth (m)')
    ax.set_title('Peak Flood Depth — ITZI-SWMM vs MIKE+ (no LarNO NN)')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    out = os.path.join(VIZ_DIR, 'coupled_02_summary_bar.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {out}")


def fig_timeseries(results):
    evts = list(results.keys())
    ncol = min(4, len(evts))
    nrow = (len(evts) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3 * nrow))
    axes = np.atleast_2d(axes)
    fig.suptitle('Volume Time Series — Surface vs SWMM Coupled', fontsize=12, fontweight='bold')

    for idx, evt in enumerate(evts):
        ax = axes[idx // ncol, idx % ncol]
        rs = results[evt]['rec_surf']
        rw = results[evt]['rec_swmm']
        ax.plot(rs['time_h'], rs['vol_m3'], 'b-', label='ITZI-S', lw=1.5)
        ax.plot(rw['time_h'], rw['vol_m3'], 'g--', label='ITZI-SWMM', lw=1.5)
        ax.set_title(evt)
        ax.set_xlabel('Time (h)')
        ax.set_ylabel('Volume (m³)')
        if idx == 0:
            ax.legend(fontsize=8)

    for idx in range(len(evts), nrow * ncol):
        axes[idx // ncol, idx % ncol].set_visible(False)

    plt.tight_layout()
    out = os.path.join(VIZ_DIR, 'coupled_03_timeseries.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {out}")


def fig_drainage_effect(results):
    """Show peak reduction from SWMM coupling vs surface-only."""
    evts = list(results.keys())
    reduction = []
    for e in evts:
        s = np.max(results[e]['h_surf'])
        w = np.max(results[e]['h_swmm'])
        reduction.append((s - w) / s * 100 if s > 0 else 0)

    fig, ax = plt.subplots(figsize=(max(8, len(evts)), 4))
    ax.bar(evts, reduction, color='#31A354', edgecolor='black')
    ax.axhline(0, color='k', lw=0.5)
    ax.set_ylabel('Peak Depth Reduction (%)')
    ax.set_title('SWMM Coupling Effect — Peak Reduction vs Surface-Only')
    ax.set_xticklabels(evts, rotation=30)
    plt.tight_layout()
    out = os.path.join(VIZ_DIR, 'coupled_04_swmm_reduction.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {out}")


def main():
    results = load_all_results()
    if not results:
        print(f"No results in {DATA_DIR}. Run run_full_domain_coupled.py first.")
        return

    domain_mode = list(results.values())[0]['domain']
    dem_full = np.load(DEM_PATH)
    bldg_full = dem_full >= 49.9
    if domain_mode == 'full':
        dem, bldg = dem_full, bldg_full
    else:
        dem = dem_full[SUB_SLICE]
        bldg = bldg_full[SUB_SLICE]

    print(f"Visualizing {len(results)} events ({domain_mode} domain)")
    fig_peak_maps(results, dem, bldg, domain_mode)
    fig_comparison_bars(results)
    fig_timeseries(results)
    fig_drainage_effect(results)
    print(f"\nOutput: {VIZ_DIR}")


if __name__ == "__main__":
    main()
