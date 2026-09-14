#!/usr/bin/env python3
"""Create flood evolution animation for the synthetic urban test case."""
import os, sys, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.animation as animation

SIM_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SIM_DIR, "core_sim_output")
VIZ_DIR = os.path.join(SIM_DIR, "visualization_output")
CELL_SIZE = 5.0
DOMAIN_SIZE = 500.0
GRID_SIZE = 100
os.makedirs(VIZ_DIR, exist_ok=True)

flood_cmap = LinearSegmentedColormap.from_list('flood', [
    (0.0, '#F7FBFF'), (0.15, '#C6DBEF'), (0.3, '#6BAED6'),
    (0.5, '#2171B5'), (0.7, '#08306B'), (1.0, '#041838'),
])
extent = [0, DOMAIN_SIZE, 0, DOMAIN_SIZE]


def main():
    print("Loading synthetic urban results...")
    data = np.load(os.path.join(OUTPUT_DIR, "simulation_results.npz"))
    dem = data['dem']

    # Load depth snapshots
    depth_files = sorted(glob.glob(os.path.join(OUTPUT_DIR, "depth_t*.asc")))
    depth_maps = {}
    for df in depth_files:
        label = os.path.basename(df).replace("depth_t", "").replace(".asc", "").replace("h", "")
        depth_maps[float(label)] = np.loadtxt(df)

    # Interpolate
    times_orig = sorted(depth_maps.keys())
    total_frames = 120
    all_times = np.linspace(times_orig[0], times_orig[-1], total_frames)
    frames = []
    for t in all_times:
        if t in depth_maps:
            frames.append(depth_maps[t])
        else:
            for i in range(len(times_orig) - 1):
                if times_orig[i] <= t <= times_orig[i + 1]:
                    frac = (t - times_orig[i]) / (times_orig[i + 1] - times_orig[i])
                    frame = (1 - frac) * depth_maps[times_orig[i]] + frac * depth_maps[times_orig[i + 1]]
                    frames.append(frame)
                    break
    print(f"  {len(frames)} frames interpolated from {len(depth_maps)} records")

    vmax = max(np.nanmax(f) for f in frames)

    # === Animation 1: Clean flood evolution ===
    print("  Rendering GIF animation...")
    fig, ax = plt.subplots(figsize=(10, 10))
    h0 = frames[0]
    h0_masked = np.ma.masked_where(h0 <= 0.001, h0)
    im = ax.imshow(np.flipud(h0_masked), cmap=flood_cmap, extent=extent,
                   origin='upper', vmin=0, vmax=vmax, aspect='equal')
    time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                        fontsize=14, fontweight='bold', color='white',
                        verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
    ax.set_xlabel('Easting (m)', fontsize=12)
    ax.set_ylabel('Northing (m)', fontsize=12)
    ax.set_title('Flood Evolution - Synthetic Terrain (50mm/2h)', fontsize=14, fontweight='bold')
    cbar = fig.colorbar(im, ax=ax, label='Water Depth (m)', shrink=0.8)

    def update(i):
        h = frames[i]
        t = all_times[i]
        h_masked = np.ma.masked_where(h <= 0.001, h)
        im.set_array(np.flipud(h_masked))
        time_text.set_text(f'  t = {t:.1f}h  |  Max depth = {np.nanmax(h):.2f}m  |  Vol = {np.nansum(h)*25:.0f}m3  ')
        return [im, time_text]

    ani = animation.FuncAnimation(fig, update, frames=len(frames),
                                  interval=150, blit=True)
    gif_path = os.path.join(VIZ_DIR, 'flood_evolution.gif')
    ani.save(gif_path, writer='pillow', fps=8, dpi=100)
    print(f"  GIF saved: {gif_path} ({os.path.getsize(gif_path)/1024:.0f} KB)")
    plt.close(fig)

    # === Key moments snapshot ===
    print("  Saving key moments...")
    key_times = [0.17, 0.50, 0.83, 1.00, 1.33, 1.67, 2.00]
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    for idx, t_target in enumerate(key_times):
        ax = axes[idx // 4, idx % 4]
        closest_idx = np.argmin(np.abs(np.array(all_times) - t_target))
        h = frames[closest_idx]
        t_actual = all_times[closest_idx]
        h_masked = np.ma.masked_where(h <= 0.001, h)
        im = ax.imshow(np.flipud(h_masked), cmap=flood_cmap, extent=extent,
                       origin='upper', vmin=0, vmax=vmax, aspect='equal')
        ax.set_title(f't = {t_actual:.1f}h  max={np.nanmax(h):.2f}m', fontsize=9)
        ax.set_xlabel('E (m)', fontsize=7); ax.set_ylabel('N (m)', fontsize=7)
    axes[1, 3].remove()
    fig.suptitle('Flood Evolution - Key Moments (Synthetic Terrain)',
                 fontsize=14, fontweight='bold')
    fig.colorbar(im, ax=axes[1, 2], label='Water Depth (m)', shrink=0.8)
    plt.tight_layout()
    fig.savefig(os.path.join(VIZ_DIR, 'key_moments.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"\n  Outputs in: {VIZ_DIR}")


if __name__ == "__main__":
    main()
