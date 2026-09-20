#!/usr/bin/env python3
"""
Compare ITZI Model Results with MIKE+ Reference Data
======================================================
Runs ITZI simulations using actual MIKE+ rainfall forcing,
then compares surface-only and pipe-network results against
MIKE+ reference water depths.

Metrics: Peak depth, flooded area, volume, spatial correlation
"""

import os, sys, time
import numpy as np
from scipy import ndimage
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEM_PATH = os.path.join(PROJECT_ROOT, "LarNO-main", "benchmark", "urbanflood", "geodata", "region1_20m", "dem.npy")
FLOOD_DIR = os.path.join(PROJECT_ROOT, "LarNO-main", "benchmark", "urbanflood", "flood", "region1_20m")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "mike_comparison")
os.makedirs(OUT_DIR, exist_ok=True)

CELL = 20.0; CELL_AREA = CELL**2
flood_cmap = LinearSegmentedColormap.from_list('flood', [
    (0.0, '#F7FBFF'), (0.12, '#C6DBEF'), (0.25, '#6BAED6'),
    (0.4, '#3182BD'), (0.55, '#08519C'), (0.7, '#08306B'), (1.0, '#041838'),
])


def run_itzi_for_event(dem, bldg, rainfall_3d, nodes, links, subcatch_map, areas, with_pipes):
    """Run ITZI model using depression-filling with 2D rainfall."""
    from run_depression_filling import depression_flood_model_2d
    pipe_net = {'nodes': nodes, 'links': links} if with_pipes else None
    rec, h_final = depression_flood_model_2d(dem, bldg, rainfall_3d, pipe_net)
    return rec, h_final, h_final  # hmax = h_final for depression model


def compute_metrics(h_pred, h_ref, bldg, threshold=0.03):
    """Compute comparison metrics between predicted and reference depths."""
    active = ~bldg

    # Flatten valid cells
    h_pred_f = h_pred[active]
    h_ref_f = h_ref[active]

    # Mask cells where both are dry (small values)
    wet = (h_pred_f > 0.001) | (h_ref_f > 0.001)
    if wet.sum() == 0:
        return {'MAE': 0, 'RMSE': 0, 'R2': 0, 'PeakR2': 0, 'CSI': 0}

    p = h_pred_f[wet]
    r = h_ref_f[wet]

    # MAE
    mae = np.mean(np.abs(p - r))
    # RMSE
    rmse = np.sqrt(np.mean((p - r)**2))
    # R²
    ss_res = np.sum((r - p)**2)
    ss_tot = np.sum((r - np.mean(r))**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    # Peak R²: compare only cells where reference depth > 0.1m
    peak_mask = h_ref_f > 0.1
    if peak_mask.sum() > 10:
        p_peak = h_pred_f[peak_mask]
        r_peak = h_ref_f[peak_mask]
        ss_res_p = np.sum((r_peak - p_peak)**2)
        ss_tot_p = np.sum((r_peak - np.mean(r_peak))**2)
        peak_r2 = 1 - ss_res_p / ss_tot_p if ss_tot_p > 0 else 0
    else:
        peak_r2 = 0

    # CSI (Critical Success Index): flood extent overlap
    pred_flooded = h_pred_f > threshold
    ref_flooded = h_ref_f > threshold
    hits = np.sum(pred_flooded & ref_flooded)
    misses = np.sum(~pred_flooded & ref_flooded)
    false_alarms = np.sum(pred_flooded & ~ref_flooded)
    csi = hits / (hits + misses + false_alarms) if (hits + misses + false_alarms) > 0 else 0

    return {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'PeakR2': peak_r2, 'CSI': csi}


def main():
    print("=" * 60)
    print("  ITZI vs MIKE+ Comparison Study")
    print("=" * 60)

    # Load DEM and network
    print("\n[1] Loading DEM and pipe network...")
    dem = np.load(DEM_PATH, allow_pickle=True).astype(np.float64)
    bldg = dem >= 49.9; H, W = dem.shape; active = ~bldg

    net = np.load(os.path.join(os.path.dirname(OUT_DIR), 'osm_merged_network.npz'), allow_pickle=True)
    nodes = net['nodes'].tolist(); links = net['links'].tolist()

    # Build subcatchment map (reuse from previous run)
    from run_sewer_drainage_model import assign_subcatchments
    subcatch_map, areas = assign_subcatchments(dem, bldg, nodes)

    events = ['event1', 'event20', 'event65', 'event66', 'event67', 'event68', 'event69', 'event70']

    # ================================================================
    # Run ITZI for each event (surface-only and pipe-network)
    # ================================================================
    all_metrics = {}

    for evt in events:
        print(f"\n{'='*60}")
        print(f"  Processing {evt}")
        print(f"{'='*60}")

        # Load MIKE+ reference data
        h_ref = np.load(os.path.join(FLOOD_DIR, evt, 'h.npy'))  # (T, H, W)
        rainfall = np.load(os.path.join(FLOOD_DIR, evt, 'rainfall.npy'))  # (T, H, W)
        peak_ref = np.max(h_ref, axis=0)  # Peak depth at each cell
        final_ref = h_ref[-1]  # Final time step

        print(f"  MIKE+ ref: peak={np.max(peak_ref):.3f}m, flooded={np.sum(peak_ref>0.03):,} cells")

        # Run ITZI surface-only
        print("  ITZI surface-only...")
        t0 = time.time()
        rec_a, h_a, hmax_a = run_itzi_for_event(dem, bldg, rainfall, nodes, links,
                                                  subcatch_map, areas, with_pipes=False)
        print(f"    Time: {time.time()-t0:.1f}s, vol={rec_a['vol_m3'][-1]:.0f}m3")

        # Run ITZI with pipes
        print("  ITZI with pipes...")
        t0 = time.time()
        rec_b, h_b, hmax_b = run_itzi_for_event(dem, bldg, rainfall, nodes, links,
                                                  subcatch_map, areas, with_pipes=True)
        print(f"    Time: {time.time()-t0:.1f}s, vol={rec_b['vol_m3'][-1]:.0f}m3, drained={rec_b['drained_m3'][-1]:.0f}m3")

        # Compute metrics
        metrics_a = compute_metrics(h_a, final_ref, bldg)
        metrics_b = compute_metrics(h_b, final_ref, bldg)

        # Also compare peak depths
        peak_a = np.max([h_a, hmax_a], axis=0) if hasattr(hmax_a, 'shape') else hmax_a
        peak_b = h_b  # simplified

        all_metrics[evt] = {
            'mike_peak': np.max(peak_ref),
            'mike_flooded': int(np.sum(peak_ref > 0.03)),
            'mike_vol': float(np.sum(h_ref[-1][active]) * CELL_AREA),
            'itzi_surface_peak': float(np.max(h_a)),
            'itzi_surface_flooded': int(np.sum(h_a > 0.03)),
            'itzi_surface_vol': float(rec_a['vol_m3'][-1]),
            'itzi_pipe_peak': float(np.max(h_b)),
            'itzi_pipe_flooded': int(np.sum(h_b > 0.03)),
            'itzi_pipe_vol': float(rec_b['vol_m3'][-1]),
            'itzi_pipe_drained': float(rec_b['drained_m3'][-1]),
            'metrics_surface': metrics_a,
            'metrics_pipe': metrics_b,
        }

        # Save per-event data
        np.savez(os.path.join(OUT_DIR, f'{evt}_comparison.npz'),
                 h_ref=final_ref, peak_ref=peak_ref,
                 h_itzi_surf=h_a, h_itzi_pipe=h_b,
                 metrics_a=metrics_a, metrics_b=metrics_b)

    # ================================================================
    # Summary table
    # ================================================================
    print(f"\n{'='*80}")
    print("  COMPARISON SUMMARY: ITZI vs MIKE+ Reference")
    print(f"{'='*80}")
    print(f"{'Event':<10s} {'MIKE+':>10s} {'ITZI-Surf':>10s} {'ITZI-Pipe':>10s} {'SurfRed%':>8s} {'PipeRed%':>8s} {'Drained':>10s}")
    print(f"{'':10s} {'Peak(m)':>10s} {'Peak(m)':>10s} {'Peak(m)':>10s} {'':8s} {'':8s} {'(m3)':>10s}")
    print(f"{'-'*70}")

    for evt in events:
        m = all_metrics[evt]
        surf_red = (m['mike_peak'] - m['itzi_surface_peak']) / m['mike_peak'] * 100
        pipe_red = (m['mike_peak'] - m['itzi_pipe_peak']) / m['mike_peak'] * 100
        print(f"{evt:<10s} {m['mike_peak']:>10.3f} {m['itzi_surface_peak']:>10.3f} "
              f"{m['itzi_pipe_peak']:>10.3f} {surf_red:>7.1f}% {pipe_red:>7.1f}% "
              f"{m['itzi_pipe_drained']:>10.0f}")

    # Save summary
    np.savez(os.path.join(OUT_DIR, 'comparison_summary.npz'), all_metrics=all_metrics, events=events)

    # ================================================================
    # Comparison visualizations
    # ================================================================
    print(f"\n[Creating comparison visualizations...]")
    create_comparison_figures(all_metrics, events, dem, bldg)

    print(f"\n  Outputs: {OUT_DIR}")


def create_comparison_figures(all_metrics, events, dem, bldg):
    H, W = dem.shape; extent = [0, W*CELL, 0, H*CELL]
    active = ~bldg; bldg_bg = np.where(bldg, 0.3, 0)

    # === FIG 1: Peak depth comparison (multi-event) ===
    n_evts = len(events)
    fig, axes = plt.subplots(3, n_evts, figsize=(4*n_evts, 12))
    fig.suptitle('ITZI vs MIKE+: Peak Flood Depth Comparison', fontsize=14, fontweight='bold')

    # Determine global vmax
    global_vmax = 0
    for evt in events:
        data = np.load(os.path.join(OUT_DIR, f'{evt}_comparison.npz'))
        global_vmax = max(global_vmax, np.max(data['peak_ref']), np.max(data['h_itzi_surf']))

    for col, evt in enumerate(events):
        data = np.load(os.path.join(OUT_DIR, f'{evt}_comparison.npz'))
        m = all_metrics[evt]

        # Row 0: MIKE+ reference
        ax = axes[0, col]
        h_show = np.ma.masked_where(data['peak_ref'] < 0.01, data['peak_ref'])
        ax.imshow(np.flipud(bldg_bg), cmap=ListedColormap(['none', '#CCC']),
                  extent=extent, aspect='equal', alpha=0.5)
        im = ax.imshow(np.flipud(h_show), cmap=flood_cmap, extent=extent, aspect='equal', vmin=0, vmax=global_vmax)
        ax.set_title(f'MIKE+ ({evt})\nPeak={m["mike_peak"]:.2f}m', fontsize=9)
        if col == 0: ax.set_ylabel('MIKE+\nReference')

        # Row 1: ITZI surface-only
        ax = axes[1, col]
        h_show = np.ma.masked_where(data['h_itzi_surf'] < 0.01, data['h_itzi_surf'])
        ax.imshow(np.flipud(bldg_bg), cmap=ListedColormap(['none', '#CCC']),
                  extent=extent, aspect='equal', alpha=0.5)
        im = ax.imshow(np.flipud(h_show), cmap=flood_cmap, extent=extent, aspect='equal', vmin=0, vmax=global_vmax)
        mae_s = m['metrics_surface']['MAE']
        ax.set_title(f'ITZI Surf\nPeak={m["itzi_surface_peak"]:.2f}m MAE={mae_s:.3f}', fontsize=9)
        if col == 0: ax.set_ylabel('ITZI\nSurface-Only')

        # Row 2: ITZI with pipes
        ax = axes[2, col]
        h_show = np.ma.masked_where(data['h_itzi_pipe'] < 0.01, data['h_itzi_pipe'])
        ax.imshow(np.flipud(bldg_bg), cmap=ListedColormap(['none', '#CCC']),
                  extent=extent, aspect='equal', alpha=0.5)
        im = ax.imshow(np.flipud(h_show), cmap=flood_cmap, extent=extent, aspect='equal', vmin=0, vmax=global_vmax)
        mae_p = m['metrics_pipe']['MAE']
        drained = m['itzi_pipe_drained']
        ax.set_title(f'ITZI Pipe\nPeak={m["itzi_pipe_peak"]:.2f}m Drained={drained/1e3:.0f}km3', fontsize=9)
        if col == 0: ax.set_ylabel('ITZI\nWith Pipes')

    for ax_row in axes:
        for ax in ax_row:
            ax.set_xticks([]); ax.set_yticks([])

    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'mike_comparison_peaks.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: mike_comparison_peaks.png")

    # === FIG 2: Peak depth bar chart ===
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(events)); w = 0.25
    mike_peaks = [all_metrics[e]['mike_peak'] for e in events]
    surf_peaks = [all_metrics[e]['itzi_surface_peak'] for e in events]
    pipe_peaks = [all_metrics[e]['itzi_pipe_peak'] for e in events]

    ax.bar(x - w, mike_peaks, w, label='MIKE+ Reference', color='#2ECC71', alpha=0.85)
    ax.bar(x, surf_peaks, w, label='ITZI Surface-Only', color='#E74C3C', alpha=0.85)
    ax.bar(x + w, pipe_peaks, w, label='ITZI With Pipes', color='#2980B9', alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels([e.replace('event','E') for e in events])
    ax.set_ylabel('Peak Water Depth (m)'); ax.set_title('Peak Flood Depth: MIKE+ vs ITZI')
    ax.legend(); ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'mike_comparison_bars.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: mike_comparison_bars.png")

    # === FIG 3: Scatter plot (ITZI vs MIKE+) ===
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors = plt.cm.Set1(np.linspace(0, 1, len(events)))

    ax = axes[0]
    for i, evt in enumerate(events):
        m = all_metrics[evt]
        ax.scatter(m['mike_peak'], m['itzi_surface_peak'], c=[colors[i]], s=80,
                   edgecolors='black', label=evt)
    ax.plot([0, max(mike_peaks)*1.1], [0, max(mike_peaks)*1.1], 'k--', alpha=0.5)
    ax.set_xlabel('MIKE+ Peak Depth (m)'); ax.set_ylabel('ITZI Surface-Only Peak (m)')
    ax.set_title('Surface-Only vs MIKE+'); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    ax = axes[1]
    for i, evt in enumerate(events):
        m = all_metrics[evt]
        ax.scatter(m['mike_peak'], m['itzi_pipe_peak'], c=[colors[i]], s=80,
                   edgecolors='black', label=evt)
    ax.plot([0, max(mike_peaks)*1.1], [0, max(mike_peaks)*1.1], 'k--', alpha=0.5)
    ax.set_xlabel('MIKE+ Peak Depth (m)'); ax.set_ylabel('ITZI With Pipes Peak (m)')
    ax.set_title('With Pipes vs MIKE+'); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'mike_comparison_scatter.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: mike_comparison_scatter.png")

    # === FIG 4: Metrics table ===
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.axis('off')

    header = f"{'Event':<10s} {'MIKE+':>8s} {'ITZI-S':>8s} {'ITZI-P':>8s} {'MAE-S':>8s} {'MAE-P':>8s} {'CSI-S':>8s} {'CSI-P':>8s} {'R2-S':>8s} {'R2-P':>8s}"
    rows = [header, '-'*90]
    for evt in events:
        m = all_metrics[evt]
        ms = m['metrics_surface']; mp = m['metrics_pipe']
        rows.append(f"{evt:<10s} {m['mike_peak']:>8.3f} {m['itzi_surface_peak']:>8.3f} "
                    f"{m['itzi_pipe_peak']:>8.3f} {ms['MAE']:>8.4f} {mp['MAE']:>8.4f} "
                    f"{ms['CSI']:>8.3f} {mp['CSI']:>8.3f} {ms['R2']:>8.3f} {mp['R2']:>8.3f}")

    # Compute means
    avg_ms = {k: np.mean([all_metrics[e]['metrics_surface'][k] for e in events]) for k in ['MAE','RMSE','R2','PeakR2','CSI']}
    avg_mp = {k: np.mean([all_metrics[e]['metrics_pipe'][k] for e in events]) for k in ['MAE','RMSE','R2','PeakR2','CSI']}
    rows.append('-'*90)
    rows.append(f"{'MEAN':<10s} {'':>8s} {'':>8s} {'':>8s} "
                f"{avg_ms['MAE']:>8.4f} {avg_mp['MAE']:>8.4f} "
                f"{avg_ms['CSI']:>8.3f} {avg_mp['CSI']:>8.3f} "
                f"{avg_ms['R2']:>8.3f} {avg_mp['R2']:>8.3f}")

    ax.text(0.02, 0.5, '\n'.join(rows), transform=ax.transAxes, fontsize=9,
            fontfamily='monospace', verticalalignment='center')

    ax.set_title('ITZI vs MIKE+ Comparison Metrics', fontweight='bold')
    fig.savefig(os.path.join(OUT_DIR, 'mike_comparison_metrics.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: mike_comparison_metrics.png")

    # Print summary
    print(f"\n  Average Metrics (8 events):")
    print(f"  Surface-Only: MAE={avg_ms['MAE']:.4f}m, R2={avg_ms['R2']:.3f}, CSI={avg_ms['CSI']:.3f}")
    print(f"  With Pipes:   MAE={avg_mp['MAE']:.4f}m, R2={avg_mp['R2']:.3f}, CSI={avg_mp['CSI']:.3f}")


if __name__ == "__main__":
    main()
