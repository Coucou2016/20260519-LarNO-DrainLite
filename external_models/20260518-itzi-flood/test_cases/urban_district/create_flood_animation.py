#!/usr/bin/env python3
"""
Create flood evolution animations (GIF + MP4) from simulation results.
Generates smooth interpolated frames between recorded timesteps.
"""
import os, sys, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import matplotlib.animation as animation

SIM_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SIM_DIR, "core_sim_output")
VIZ_DIR = os.path.join(SIM_DIR, "visualization_output")
CELL_SIZE = 2.0
DOMAIN_SIZE = 400.0
GRID_SIZE = 200
os.makedirs(VIZ_DIR, exist_ok=True)

# Colormaps
flood_cmap = LinearSegmentedColormap.from_list('flood', [
    (0.0, '#F7FBFF'), (0.15, '#C6DBEF'), (0.3, '#6BAED6'),
    (0.5, '#2171B5'), (0.7, '#08306B'), (1.0, '#041838'),
])
extent = [0, DOMAIN_SIZE, 0, DOMAIN_SIZE]


def load_data():
    data = np.load(os.path.join(OUTPUT_DIR, "urban_simulation_results.npz"))
    depth_files = sorted(glob.glob(os.path.join(OUTPUT_DIR, "depth_urban_t*.asc")))
    depth_maps = {}
    for df in depth_files:
        label = os.path.basename(df).replace("depth_urban_t", "").replace(".asc", "").replace("h", "")
        depth_maps[float(label)] = np.loadtxt(df)
    return data, depth_maps


def interpolate_frames(depth_maps, total_frames=120):
    """Interpolate between recorded timesteps to create smooth animation."""
    times = sorted(depth_maps.keys())
    all_times = np.linspace(times[0], times[-1], total_frames)
    all_frames = []
    for t in all_times:
        if t in depth_maps:
            all_frames.append(depth_maps[t])
        else:
            # Find bracketing times
            for i in range(len(times) - 1):
                if times[i] <= t <= times[i + 1]:
                    frac = (t - times[i]) / (times[i + 1] - times[i])
                    frame = (1 - frac) * depth_maps[times[i]] + frac * depth_maps[times[i + 1]]
                    all_frames.append(frame)
                    break
    return all_times, all_frames


def create_animation(data, depth_maps):
    print("Creating flood evolution animation...")

    building_mask = data['building_mask']
    park_mask = data['park_mask']

    # Interpolate to get 120 smooth frames
    times, frames = interpolate_frames(depth_maps, total_frames=120)
    print(f"  Interpolated {len(frames)} frames from {len(depth_maps)} records")

    vmax = max(np.nanmax(f) for f in frames)
    print(f"  Max flood depth: {vmax:.2f}m")

    # ================================================================
    # Animation 1: Clean Flood Evolution
    # ================================================================
    print("  Rendering Animation 1: Flood depth evolution...")
    fig, ax = plt.subplots(figsize=(10, 10))

    # Initial frame
    h0 = frames[0]
    h0_masked = np.ma.masked_where(h0 <= 0.001, h0)
    bldg_bg = np.where(building_mask, 0.3, 0)
    ax.imshow(np.flipud(bldg_bg), cmap=ListedColormap(['#FFFFFF', '#D3D3D3']),
              extent=extent, aspect='equal', alpha=0.5, zorder=1)
    im = ax.imshow(np.flipud(h0_masked), cmap=flood_cmap, extent=extent,
                   origin='upper', vmin=0, vmax=vmax, aspect='equal', zorder=2)
    time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                        fontsize=14, fontweight='bold', color='white',
                        verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
    ax.set_xlabel('Easting (m)', fontsize=12)
    ax.set_ylabel('Northing (m)', fontsize=12)
    ax.set_title('Urban District Flood Evolution', fontsize=14, fontweight='bold')
    cbar = fig.colorbar(im, ax=ax, label='Water Depth (m)', shrink=0.8, pad=0.02)

    def update(frame_idx):
        h = frames[frame_idx]
        t = times[frame_idx]
        h_masked = np.ma.masked_where(h <= 0.001, h)
        im.set_array(np.flipud(h_masked))
        time_text.set_text(f'  t = {t:.1f} h  |  Max depth = {np.nanmax(h):.2f} m  ')
        return [im, time_text]

    ani = animation.FuncAnimation(fig, update, frames=len(frames),
                                  interval=150, blit=True)

    # Save as GIF
    gif_path = os.path.join(VIZ_DIR, 'urban_flood_evolution.gif')
    print(f"  Saving GIF (this may take a minute)...")
    ani.save(gif_path, writer='pillow', fps=8, dpi=100)
    print(f"  GIF saved: {gif_path} ({os.path.getsize(gif_path)/1024:.0f} KB)")
    plt.close(fig)

    # ================================================================
    # Animation 2: Split view - Depth + Velocity
    # ================================================================
    print("  Rendering Animation 2: Depth + Velocity comparison...")

    # Load velocity data (use depth snapshots to approximate since we saved depth only)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    h0 = frames[0]
    h0_masked = np.ma.masked_where(h0 <= 0.001, h0)
    bldg_bg = np.where(building_mask, 0.3, 0)

    # Left: Depth
    ax1.imshow(np.flipud(bldg_bg), cmap=ListedColormap(['#FFFFFF', '#D3D3D3']),
               extent=extent, aspect='equal', alpha=0.5)
    im1 = ax1.imshow(np.flipud(h0_masked), cmap=flood_cmap, extent=extent,
                     origin='upper', vmin=0, vmax=vmax, aspect='equal')
    time_text1 = ax1.text(0.02, 0.98, '', transform=ax1.transAxes,
                          fontsize=12, fontweight='bold', color='white',
                          verticalalignment='top',
                          bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
    ax1.set_title('Water Depth', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Easting (m)')
    ax1.set_ylabel('Northing (m)')
    plt.colorbar(im1, ax=ax1, label='Depth (m)', shrink=0.8)

    # Right: buildings + streets (static land use)
    surface_cmap = ListedColormap(['#808080', '#D3D3D3', '#228B22', '#8B0000'])
    surface = np.zeros_like(data['mannings'], dtype=int)
    m = data['mannings']
    surface[m > 0.1] = 4
    surface[(m > 0.03) & (m <= 0.1)] = 3
    surface[(m > 0.02) & (m <= 0.03)] = 2
    surface[m <= 0.02] = 1
    ax2.imshow(np.flipud(surface), cmap=surface_cmap, extent=extent,
               aspect='equal', alpha=0.5, vmin=0.5, vmax=4.5)
    im2 = ax2.imshow(np.flipud(h0_masked), cmap=flood_cmap, extent=extent,
                     origin='upper', vmin=0, vmax=vmax, aspect='equal')
    time_text2 = ax2.text(0.02, 0.98, '', transform=ax2.transAxes,
                          fontsize=12, fontweight='bold', color='white',
                          verticalalignment='top',
                          bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
    ax2.set_title('Flood Over Urban Surfaces', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Easting (m)')
    ax2.set_ylabel('Northing (m)')
    plt.colorbar(im2, ax=ax2, label='Depth (m)', shrink=0.8)

    fig.suptitle('Urban District Flood - Dual View Animation', fontsize=15, fontweight='bold')

    def update2(frame_idx):
        h = frames[frame_idx]
        t = times[frame_idx]
        h_masked = np.ma.masked_where(h <= 0.001, h)
        im1.set_array(np.flipud(h_masked))
        im2.set_array(np.flipud(h_masked))
        txt = f'  t = {t:.1f}h  |  Max = {np.nanmax(h):.2f}m'
        time_text1.set_text(txt)
        time_text2.set_text(txt)
        return [im1, im2, time_text1, time_text2]

    ani2 = animation.FuncAnimation(fig, update2, frames=len(frames),
                                   interval=150, blit=True)

    gif_path2 = os.path.join(VIZ_DIR, 'urban_flood_dual_view.gif')
    print(f"  Saving dual-view GIF...")
    ani2.save(gif_path2, writer='pillow', fps=8, dpi=100)
    print(f"  GIF saved: {gif_path2} ({os.path.getsize(gif_path2)/1024:.0f} KB)")
    plt.close(fig)

    # ================================================================
    # Also save key snapshot images
    # ================================================================
    print("  Saving key snapshot images...")
    key_times = [0.17, 0.50, 0.83, 1.00, 1.33, 1.67, 2.00]
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    for idx, t_target in enumerate(key_times):
        ax = axes[idx // 4, idx % 4]
        # Find closest frame
        closest_idx = np.argmin(np.abs(np.array(times) - t_target))
        h = frames[closest_idx]
        t_actual = times[closest_idx]
        h_masked = np.ma.masked_where(h <= 0.001, h)
        ax.imshow(np.flipud(bldg_bg), cmap=ListedColormap(['#FFFFFF', '#D3D3D3']),
                  extent=extent, aspect='equal', alpha=0.5)
        im = ax.imshow(np.flipud(h_masked), cmap=flood_cmap, extent=extent,
                       origin='upper', vmin=0, vmax=vmax, aspect='equal')
        ax.set_title(f't = {t_actual:.1f}h  max={np.nanmax(h):.2f}m', fontsize=9)
        ax.set_xlabel('E (m)', fontsize=7); ax.set_ylabel('N (m)', fontsize=7)
    # Remove extra subplot
    axes[1, 3].remove()

    fig.suptitle('Urban Flood - Key Moments (50mm Design Storm, 2h)',
                 fontsize=14, fontweight='bold')
    fig.colorbar(im, ax=axes[1, 2], label='Water Depth (m)', shrink=0.8)
    plt.tight_layout()
    fig.savefig(os.path.join(VIZ_DIR, 'urban_key_moments.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Combined summary snapshot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    moments = [(0.5, 'Early Stage (30min)'), (1.0, 'Mid Storm (1h)'), (2.0, 'End Storm (2h)')]
    for idx, (t_target, title) in enumerate(moments):
        ax = axes[idx]
        closest_idx = np.argmin(np.abs(np.array(times) - t_target))
        h = frames[closest_idx]
        h_masked = np.ma.masked_where(h <= 0.001, h)
        ax.imshow(np.flipud(bldg_bg), cmap=ListedColormap(['#FFFFFF', '#D3D3D3']),
                  extent=extent, aspect='equal', alpha=0.5)
        im = ax.imshow(np.flipud(h_masked), cmap=flood_cmap, extent=extent,
                       origin='upper', vmin=0, vmax=vmax, aspect='equal')
        ax.set_title(f'{title}\nMax depth: {np.nanmax(h):.2f}m', fontsize=11)
        ax.set_xlabel('Easting (m)'); ax.set_ylabel('Northing (m)')
    fig.suptitle('Urban District Flood - Before / During / After',
                 fontsize=14, fontweight='bold')
    cbar = fig.colorbar(im, ax=axes, label='Water Depth (m)', shrink=0.6, pad=0.02)
    plt.tight_layout()
    fig.savefig(os.path.join(VIZ_DIR, 'urban_before_during_after.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"\n  All outputs saved to: {VIZ_DIR}")
    for f in sorted(os.listdir(VIZ_DIR)):
        path = os.path.join(VIZ_DIR, f)
        print(f"    {f} ({os.path.getsize(path)/1024:.0f} KB)")


def main():
    print("=" * 60)
    print("  Urban Flood Animation Generator")
    print("=" * 60)
    data, depth_maps = load_data()
    create_animation(data, depth_maps)
    print("\n  Done!")


if __name__ == "__main__":
    main()
