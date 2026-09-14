#!/usr/bin/env python3
"""Visualize road-based pipe network simulation results."""
import numpy as np, os
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

flood_cmap = LinearSegmentedColormap.from_list('flood', [
    (0.0, '#F7FBFF'), (0.12, '#C6DBEF'), (0.25, '#6BAED6'),
    (0.4, '#3182BD'), (0.55, '#08519C'), (0.7, '#08306B'), (1.0, '#041838'),
])
diff_cmap = plt.cm.RdBu_r
OUT_DIR = 'output'
CELL = 20.0

data = np.load(os.path.join(OUT_DIR, 'road_based_simulation_results.npz'), allow_pickle=True)
time_h = data['time_h']; vol_a = data['vol_a']; vol_b = data['vol_b']
hmax_a = data['hmax_a']; hmax_b = data['hmax_b']
flooded_a = data['flooded_a']; flooded_b = data['flooded_b']
drained = data['drained_m3']; h_a = data['h_final_a']; h_b = data['h_final_b']
dem = data['dem']; bldg = data['bldg']
H, W = dem.shape
extent = [0, W*CELL, 0, H*CELL]

net = np.load(os.path.join(OUT_DIR, 'road_based_network_v2.npz'), allow_pickle=True)
nodes = net['nodes'].tolist(); links = net['links'].tolist()
node_lookup = {n['id']: n for n in nodes}

vmax = max(np.max(h_a[h_a>0.001]) if np.any(h_a>0.001) else 0.05,
           np.max(h_b[h_b>0.001]) if np.any(h_b>0.001) else 0.05, 0.05)
bldg_bg = np.where(bldg, 0.3, 0)
diff = h_a - h_b
vlim = max(abs(np.min(diff)), abs(np.max(diff)), 0.01)

# === FIGURE 1: Side-by-side flood comparison ===
fig, axes = plt.subplots(1, 3, figsize=(20, 7))
fig.suptitle('Road-Based Drainage Network: Flood Reduction Effect (50mm/6h Storm)',
             fontsize=14, fontweight='bold')

for idx, (ax, h, title) in enumerate([
    (axes[0], h_a, f'Surface Only\nMax={np.max(h_a):.3f}m, Vol={vol_a[-1]:.0f}m3'),
    (axes[1], h_b, f'With Road-Based Drainage\nMax={np.max(h_b):.3f}m, Vol={vol_b[-1]:.0f}m3'),
]):
    h_show = np.ma.masked_where(h < 0.001, h)
    ax.imshow(np.flipud(bldg_bg), cmap=ListedColormap(['none', '#D3D3D3']),
              extent=extent, aspect='equal', alpha=0.5)
    im = ax.imshow(np.flipud(h_show), cmap=flood_cmap, extent=extent, aspect='equal', vmin=0, vmax=vmax)
    ax.set_title(title); ax.set_xlabel('Easting (m)'); ax.set_ylabel('Northing (m)')
    plt.colorbar(im, ax=ax, label='Depth (m)', shrink=0.8)

ax = axes[2]
ax.imshow(np.flipud(np.where(bldg, 0.3, 0)), cmap=ListedColormap(['none', '#CCC']),
          extent=extent, aspect='equal', alpha=0.4)
im = ax.imshow(np.flipud(diff), cmap=diff_cmap, extent=extent, aspect='equal', vmin=-vlim, vmax=vlim)
for link in links:
    fn = node_lookup[link['from_node']]; tn = node_lookup[link['to_node']]
    ax.plot([fn['x_m'], tn['x_m']], [fn['y_m'], tn['y_m']], 'black', lw=0.4, alpha=0.3)
ax.set_title(f'Depth Reduction\n(Blue=Drainage Benefit, Drained={drained[-1]:.0f}m3)')
ax.set_xlabel('Easting (m)'); ax.set_ylabel('Northing (m)')
plt.colorbar(im, ax=ax, label='Diff (m)', shrink=0.8)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'road_based_flood_comparison.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print('Saved: road_based_flood_comparison.png')

# === FIGURE 2: Time series ===
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Road-Based Drainage Network Performance', fontsize=14, fontweight='bold')
expected = 50/1000 * np.sum(~bldg) * CELL**2

ax = axes[0, 0]
ax.plot(time_h, vol_a, 'r-o', lw=2, ms=3, label='Surface Only')
ax.plot(time_h, vol_b, 'b-s', lw=2, ms=3, label='With Road-Based Drainage')
ax.axhline(y=expected, color='green', ls='--', label=f'Total Rain ({expected:.0f}m3)')
ax.set_xlabel('Time (h)'); ax.set_ylabel('Volume (m3)')
ax.set_title(f'Surface Water Volume ({(vol_a[-1]-vol_b[-1])/vol_a[-1]*100:.1f}% reduction)')
ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[0, 1]
ax.plot(time_h, hmax_a, 'r-o', lw=2, ms=3, label='Surface Only')
ax.plot(time_h, hmax_b, 'b-s', lw=2, ms=3, label='With Drainage')
ax.set_xlabel('Time (h)'); ax.set_ylabel('Max Depth (m)')
ax.set_title('Maximum Inundation Depth'); ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[1, 0]
ax.plot(time_h, flooded_a * CELL**2, 'r-o', lw=2, ms=3, label='Surface Only')
ax.plot(time_h, flooded_b * CELL**2, 'b-s', lw=2, ms=3, label='With Drainage')
ax.set_xlabel('Time (h)'); ax.set_ylabel('Flooded Area (m2)')
ax.set_title(f'Inundated Area ({(flooded_a[-1]-flooded_b[-1])/flooded_a[-1]*100:.1f}% reduction)')
ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[1, 1]
ax.fill_between(time_h, 0, drained, color='blue', alpha=0.3)
ax.plot(time_h, drained, 'b-', lw=2, label='Cumulative Drained')
ax.set_xlabel('Time (h)'); ax.set_ylabel('Volume (m3)')
ax.set_title(f'Water Removed by Drainage (Total: {drained[-1]:.0f}m3, {drained[-1]/expected*100:.1f}% of rain)')
ax.legend(); ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'road_based_timeseries.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print('Saved: road_based_timeseries.png')

# === FIGURE 3: Dashboard ===
fig = plt.figure(figsize=(20, 15))
fig.suptitle('Extended Study: Road-Based Urban Drainage — Shenzhen Futian', fontsize=16, fontweight='bold')

# (1) Terrain
ax = fig.add_subplot(2, 3, 1)
im = ax.imshow(np.flipud(np.where(bldg, np.nan, dem)), cmap='terrain', extent=extent, aspect='equal')
ax.set_title('Terrain + Buildings'); plt.colorbar(im, ax=ax, label='Elev (m)')

# (2) Flood with drainage
ax = fig.add_subplot(2, 3, 2)
ax.imshow(np.flipud(bldg_bg), cmap=ListedColormap(['none','#aaa']), extent=extent, aspect='equal', alpha=0.5)
h_show = np.ma.masked_where(h_b < 0.001, h_b)
im = ax.imshow(np.flipud(h_show), cmap=flood_cmap, extent=extent, aspect='equal', vmin=0, vmax=vmax)
ax.set_title('Flood Depth with Drainage'); plt.colorbar(im, ax=ax, label='Depth (m)')

# (3) Difference + pipe network
ax = fig.add_subplot(2, 3, 3)
im = ax.imshow(np.flipud(diff), cmap=diff_cmap, extent=extent, aspect='equal', vmin=-vlim, vmax=vlim)
for link in links:
    fn = node_lookup[link['from_node']]; tn = node_lookup[link['to_node']]
    ax.plot([fn['x_m'], tn['x_m']], [fn['y_m'], tn['y_m']], 'black', lw=0.3, alpha=0.35)
ax.set_title('Depth Reduction + Pipe Network'); plt.colorbar(im, ax=ax, label='Diff (m)')

# (4) Cross-section
ax = fig.add_subplot(2, 3, 4)
xs = np.arange(W) * CELL
ax.fill_between(xs, dem[H//2], dem[H//2]+h_a[H//2], alpha=0.4, color='red', label='Surface Only')
ax.fill_between(xs, dem[H//2], dem[H//2]+h_b[H//2], alpha=0.4, color='blue', label='With Drainage')
ax.plot(xs, dem[H//2], 'brown', lw=1.5)
ax.set_xlabel('E (m)'); ax.set_ylabel('Elev (m)')
ax.set_title('E-W Cross Section'); ax.legend(); ax.grid(True, alpha=0.3)

# (5) Volume time series
ax = fig.add_subplot(2, 3, 5)
ax.plot(time_h, vol_a, 'r-', lw=2, label='Surface Only')
ax.plot(time_h, vol_b, 'b-', lw=2, label='With Drainage')
ax.axhline(y=expected, color='green', ls='--', label=f'Rain Input ({expected:.0f}m3)')
ax.set_xlabel('Time (h)'); ax.set_ylabel('Volume (m3)')
ax.set_title('Volume Balance'); ax.legend(); ax.grid(True, alpha=0.3)

# (6) Summary metrics
ax = fig.add_subplot(2, 3, 6); ax.axis('off')
total_len = sum(l['length'] for l in links)
metrics = (
    f"ROAD-BASED DRAINAGE NETWORK\n"
    f"============================\n\n"
    f"STUDY AREA: Shenzhen Futian\n"
    f"{H*CELL/1000:.1f}km x {W*CELL/1000:.1f}km @ 20m res\n"
    f"Buildings: {bldg.sum():,} cells ({100*bldg.sum()/bldg.size:.0f}%)\n\n"
    f"ROAD EXTRACTION\n"
    f"Method: Building-gap analysis\n"
    f"Road space = flat open ground\n"
    f"Manhole grid: 160m spacing\n\n"
    f"PIPE NETWORK\n"
    f"Manholes: {len(nodes)}, Pipes: {len(links)}\n"
    f"Total length: {total_len:.0f}m\n"
    f"Main: D=800mm, Branch: D=600mm\n"
    f"Capacity: 468 m3/s\n"
    f"Slopes follow terrain\n\n"
    f"RESULTS (50mm/6h storm)\n"
    f"Volume: {vol_a[-1]:.0f} -> {vol_b[-1]:.0f} m3\n"
    f"  ({(vol_a[-1]-vol_b[-1])/vol_a[-1]*100:.1f}% reduction)\n"
    f"MaxDepth: {hmax_a[-1]:.3f} -> {hmax_b[-1]:.3f} m\n"
    f"  ({(hmax_a[-1]-hmax_b[-1])/hmax_a[-1]*100:.1f}% reduction)\n"
    f"Flooded: {flooded_a[-1]} -> {flooded_b[-1]} cells\n"
    f"  ({(flooded_a[-1]-flooded_b[-1])/flooded_a[-1]*100:.1f}% reduction)\n"
    f"Drained: {drained[-1]:.0f} m3\n"
    f"  ({drained[-1]/expected*100:.1f}% of rainfall)\n\n"
    f"Model: Depression-filling +\n"
    f"Manning pipe flow coupling\n"
    f"Adapted from ITZI-flood"
)
ax.text(0.05, 0.5, metrics, transform=ax.transAxes, fontsize=9,
        fontfamily='monospace', verticalalignment='center',
        bbox=dict(boxstyle='round', facecolor='#F0F8FF', alpha=0.9))

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'road_based_dashboard.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print('Saved: road_based_dashboard.png')

print('Done! All visualizations saved to', OUT_DIR)
