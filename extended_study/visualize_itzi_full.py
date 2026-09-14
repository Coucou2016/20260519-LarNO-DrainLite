#!/usr/bin/env python3
"""
Full-domain ITZI vs MIKE+ visualization — matching previous global comparison style.
"""
import numpy as np, os
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output', 'itzi_full')
OUT_DIR = DATA_DIR
CELL = 20.0; CELL_AREA = CELL**2

flood_cmap = LinearSegmentedColormap.from_list('flood', [
    (0.0, '#F7FBFF'), (0.12, '#C6DBEF'), (0.25, '#6BAED6'),
    (0.4, '#3182BD'), (0.55, '#08519C'), (0.7, '#08306B'), (1.0, '#041838'),
])
diff_cmap = plt.cm.RdBu_r

events = ['event1', 'event65', 'event67']
if not all(os.path.exists(os.path.join(DATA_DIR, f'{e}_full.npz')) for e in events):
    print("Waiting for simulation data...")
    import sys; sys.exit(0)

all_data = {}
for evt in events:
    d = np.load(os.path.join(DATA_DIR, f'{evt}_full.npz'), allow_pickle=True)
    all_data[evt] = {
        'h_ref': d['h_ref'], 'peak_ref': d['peak_ref'],
        'h_itzi': d['h_itzi'],
        'rec': d['rec'].item() if hasattr(d['rec'], 'item') else d['rec'],
    }

H, W = all_data['event65']['h_ref'].shape
extent = [0, W*CELL, 0, H*CELL]
DEM_PATH = r"e:\Projects\20260519-LarNO\LarNO-main\benchmark\urbanflood\geodata\region1_20m\dem.npy"
dem = np.load(DEM_PATH); bldg = dem >= 49.9
bldg_bg = np.where(bldg, 0.3, 0)

global_vmax = max(np.max(all_data[e]['peak_ref']) for e in events)
global_vmax = max(global_vmax, max(np.max(all_data[e]['h_itzi']) for e in events)) * 1.1

# ================================================================
# FIG 1: Full-domain peak depth comparison (3 events x 3 rows)
# ================================================================
n_evts = len(events)
fig, axes = plt.subplots(3, n_evts, figsize=(5.5*n_evts, 14))
fig.suptitle('ITZI Dynamic Model vs MIKE+ — Full-Domain Peak Flood Depth (8.0km x 11.2km)',
             fontsize=14, fontweight='bold')

for col, evt in enumerate(events):
    d = all_data[evt]
    peak_m = np.max(d['peak_ref']); peak_i = np.max(d['h_itzi'])
    diff_pct = (peak_i - peak_m) / peak_m * 100

    ax = axes[0, col]
    h_show = np.ma.masked_where(d['peak_ref'] < 0.01, d['peak_ref'])
    ax.imshow(np.flipud(bldg_bg), cmap=ListedColormap(['none','#CCC']), extent=extent, aspect='equal', alpha=0.5)
    im = ax.imshow(np.flipud(h_show), cmap=flood_cmap, extent=extent, aspect='equal', vmin=0, vmax=global_vmax)
    ax.set_title(f'MIKE+ Reference\nPeak={peak_m:.3f}m', fontsize=10)
    if col == 0: ax.set_ylabel('MIKE+', fontweight='bold')

    ax = axes[1, col]
    h_show = np.ma.masked_where(d['h_itzi'] < 0.01, d['h_itzi'])
    ax.imshow(np.flipud(bldg_bg), cmap=ListedColormap(['none','#CCC']), extent=extent, aspect='equal', alpha=0.5)
    im = ax.imshow(np.flipud(h_show), cmap=flood_cmap, extent=extent, aspect='equal', vmin=0, vmax=global_vmax)
    ax.set_title(f'ITZI Dynamic (Surface)\nPeak={peak_i:.3f}m ({diff_pct:+.1f}%)', fontsize=10)
    if col == 0: ax.set_ylabel('ITZI', fontweight='bold')

    ax = axes[2, col]
    diff = d['h_itzi'] - d['peak_ref']
    vlim = max(abs(np.min(diff)), abs(np.max(diff)), 0.1)
    im = ax.imshow(np.flipud(diff), cmap=diff_cmap, extent=extent, aspect='equal', vmin=-vlim, vmax=vlim)
    ax.set_title(f'Difference (ITZI - MIKE+)\nBlue=ITZI lower, Red=ITZI higher', fontsize=10)
    if col == 0: ax.set_ylabel('Difference', fontweight='bold')

for ax_row in axes:
    for ax in ax_row:
        ax.set_xticks([]); ax.set_yticks([])

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'itzi_full_peaks.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print('1/5 itzi_full_peaks.png')

# ================================================================
# FIG 2: Time series (full domain volume/max depth)
# ================================================================
fig, axes = plt.subplots(3, 1, figsize=(14, 16))
fig.suptitle('ITZI Dynamic Model — Full-Domain Time Series vs MIKE+', fontsize=14, fontweight='bold')
colors = {'event1': '#E74C3C', 'event65': '#2980B9', 'event67': '#27AE60'}

for idx, evt in enumerate(events):
    ax = axes[idx]
    d = all_data[evt]; rec = d['rec']
    time_h = rec['time_h']
    vol_itzi = rec['vol_m3']
    mike_vol = np.sum(d['h_ref']) * CELL_AREA
    peak_i = np.max(d['h_itzi']); peak_m = np.max(d['peak_ref'])
    pct = (peak_i - peak_m) / peak_m * 100

    ax.plot(time_h, vol_itzi, 'b-o', lw=2, ms=4, label=f'ITZI Surface (final={vol_itzi[-1]:.0f}m3)')
    ax.axhline(y=mike_vol, color='red', ls='--', lw=2, label=f'MIKE+ Final ({mike_vol:.0f}m3)')
    ax.set_title(f'{evt}: ITZI Peak={peak_i:.3f}m vs MIKE+={peak_m:.3f}m ({pct:+.1f}%)', fontsize=12)
    ax.set_xlabel('Time (hours)'); ax.set_ylabel('Volume (m3)')
    ax.legend(); ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'itzi_full_timeseries.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print('2/5 itzi_full_timeseries.png')

# ================================================================
# FIG 3: Bar chart
# ================================================================
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(events)); w = 0.35
mike_peaks = [np.max(all_data[e]['peak_ref']) for e in events]
itzi_peaks = [np.max(all_data[e]['h_itzi']) for e in events]

b1 = ax.bar(x - w/2, mike_peaks, w, label='MIKE+ Reference', color='#2ECC71', alpha=0.9, edgecolor='black')
b2 = ax.bar(x + w/2, itzi_peaks, w, label='ITZI Dynamic', color='#2980B9', alpha=0.9, edgecolor='black')
ax.set_xticks(x); ax.set_xticklabels([e.replace('event','E') for e in events], fontsize=12)
ax.set_ylabel('Peak Water Depth (m)', fontsize=12)
ax.set_title('ITZI Dynamic vs MIKE+ — Full-Domain Peak Depth', fontsize=14, fontweight='bold')
ax.legend(fontsize=11); ax.grid(True, alpha=0.3, axis='y')
for b, v in zip(b1, mike_peaks): ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.03, f'{v:.3f}', ha='center', fontweight='bold')
for i, (b, v) in enumerate(zip(b2, itzi_peaks)):
    pct = (v - mike_peaks[i]) / mike_peaks[i] * 100
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.03, f'{v:.3f}\n({pct:+.1f}%)', ha='center', fontweight='bold', fontsize=9)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'itzi_full_bars.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print('3/5 itzi_full_bars.png')

# ================================================================
# FIG 4: Scatter
# ================================================================
fig, ax = plt.subplots(figsize=(8, 8))
for evt, c in zip(events, ['#E74C3C','#2980B9','#27AE60']):
    mp = np.max(all_data[evt]['peak_ref']); ip = np.max(all_data[evt]['h_itzi'])
    ax.scatter(mp, ip, c=c, s=200, edgecolors='black', lw=2, label=f'{evt} ({ip/mp*100:.0f}%)', zorder=5)
mx = max(max(mike_peaks), max(itzi_peaks)) * 1.15
ax.plot([0,mx],[0,mx],'k--',lw=1.5,alpha=0.5,label='Perfect match')
ax.fill_between([0,mx],[0,mx*0.85],[0,mx*1.15],alpha=0.1,color='green',label='+-15%')
ax.set_xlabel('MIKE+ Peak (m)'); ax.set_ylabel('ITZI Dynamic Peak (m)')
ax.set_title('ITZI vs MIKE+ — Full-Domain Peak Correlation', fontsize=14, fontweight='bold')
ax.legend(); ax.grid(True, alpha=0.3); ax.set_xlim(0,mx); ax.set_ylim(0,mx)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'itzi_full_scatter.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print('4/5 itzi_full_scatter.png')

# ================================================================
# FIG 5: Dashboard with full-domain context
# ================================================================
fig = plt.figure(figsize=(22, 16))
fig.suptitle('ITZI Dynamic Model vs MIKE+ — Full-Domain Dashboard (Shenzhen Futian, 89.6 km2)',
             fontsize=16, fontweight='bold')

# (a) MIKE+ event65
ax = fig.add_subplot(2, 3, 1)
d65 = all_data['event65']
h_s = np.ma.masked_where(d65['peak_ref']<0.01, d65['peak_ref'])
ax.imshow(np.flipud(bldg_bg), cmap=ListedColormap(['none','#CCC']), extent=extent, aspect='equal', alpha=0.5)
im = ax.imshow(np.flipud(h_s), cmap=flood_cmap, extent=extent, aspect='equal', vmin=0, vmax=global_vmax)
ax.set_title(f'MIKE+ event65 Peak: {np.max(d65["peak_ref"]):.3f}m'); plt.colorbar(im, ax=ax, label='Depth(m)', shrink=0.8)

# (b) ITZI event65
ax = fig.add_subplot(2, 3, 2)
h_s = np.ma.masked_where(d65['h_itzi']<0.01, d65['h_itzi'])
ax.imshow(np.flipud(bldg_bg), cmap=ListedColormap(['none','#CCC']), extent=extent, aspect='equal', alpha=0.5)
im = ax.imshow(np.flipud(h_s), cmap=flood_cmap, extent=extent, aspect='equal', vmin=0, vmax=global_vmax)
ax.set_title(f'ITZI Dynamic event65 Peak: {np.max(d65["h_itzi"]):.3f}m'); plt.colorbar(im, ax=ax, label='Depth(m)', shrink=0.8)

# (c) event67 diff
ax = fig.add_subplot(2, 3, 3)
d67 = all_data['event67']
diff67 = d67['h_itzi'] - d67['peak_ref']; vl67 = max(abs(np.min(diff67)), abs(np.max(diff67)), 0.1)
im = ax.imshow(np.flipud(diff67), cmap=diff_cmap, extent=extent, aspect='equal', vmin=-vl67, vmax=vl67)
ax.set_title(f'event67 Difference (ITZI-MIKE+)'); plt.colorbar(im, ax=ax, label='Diff(m)', shrink=0.8)

# (d) Bar chart
ax = fig.add_subplot(2, 3, 4)
x=np.arange(len(events)); w=0.35
b1=ax.bar(x-w/2,mike_peaks,w,label='MIKE+',color='#2ECC71',alpha=0.9)
b2=ax.bar(x+w/2,itzi_peaks,w,label='ITZI Dynamic',color='#2980B9',alpha=0.9)
ax.set_xticks(x); ax.set_xticklabels([e.replace('event','E') for e in events])
ax.set_ylabel('Peak Depth (m)'); ax.set_title('Full-Domain Peak Comparison')
ax.legend(); ax.grid(True,alpha=0.3,axis='y')
for i,(b,v) in enumerate(zip(b2,itzi_peaks)):
    pct=(v-mike_peaks[i])/mike_peaks[i]*100
    ax.text(b.get_x()+b.get_width()/2,b.get_height()+0.02,f'{pct:+.1f}%',ha='center',fontsize=9)

# (e) Volume time series
ax = fig.add_subplot(2, 3, 5)
for evt,c in zip(events,['#E74C3C','#2980B9','#27AE60']):
    rec=all_data[evt]['rec']
    ax.plot(rec['time_h'],rec['vol_m3'],lw=2,label=f'{evt}',color=c)
    mv=np.sum(all_data[evt]['h_ref'])*CELL_AREA
    ax.axhline(y=mv,ls='--',lw=1.5,color=c,alpha=0.6)
ax.set_xlabel('Time (hours)'); ax.set_ylabel('Volume (m3)')
ax.set_title('Surface Water Volume (dashed=MIKE+)'); ax.legend(); ax.grid(True,alpha=0.3)

# (f) Summary
ax = fig.add_subplot(2, 3, 6); ax.axis('off')
lines=["ITZI DYNAMIC MODEL vs MIKE+","="*28,"",
       f"DOMAIN: {H*CELL/1000:.1f} x {W*CELL/1000:.1f} km (89.6 km2)",
       f"Grid: {H}x{W} cells @ 20m resolution",
       f"Buildings: {bldg.sum():,} cells ({100*bldg.sum()/bldg.size:.0f}%)",
       "",
       "MODEL: ITZI Partial-Inertia 2D Solver",
       "  Manning friction (n=0.015 streets)",
       "  Buildings as 50m walls",
       "  2D spatially-heterogeneous rainfall",
       "  Building runoff routed to streets",
       "",
       "FULL-DOMAIN RESULTS:"]
for evt in events:
    mp=np.max(all_data[evt]['peak_ref']); ip=np.max(all_data[evt]['h_itzi'])
    lines.append(f"  {evt}: ITZI={ip:.3f}m vs MIKE+={mp:.3f}m ({(ip-mp)/mp*100:+.1f}%)")
lines+=["","Compared to depression-filling model:",
        "  - ITZI dynamic: flow resistance limits",
        "    water concentration at low points",
        "  - Better spatial distribution of flooding",
        "  - Physically-based Manning routing"]
ax.text(0.04,0.5,'\n'.join(lines),transform=ax.transAxes,fontsize=9.5,
        fontfamily='monospace',verticalalignment='center',
        bbox=dict(boxstyle='round',facecolor='#F0F8FF',alpha=0.9))
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR,'itzi_full_dashboard.png'),dpi=150,bbox_inches='tight')
plt.close(fig)
print('5/5 itzi_full_dashboard.png')

print(f"\nDone. Outputs in {OUT_DIR}")
for f in sorted(os.listdir(OUT_DIR)):
    if f.endswith('.png'): print(f"  {f} ({os.path.getsize(os.path.join(OUT_DIR,f))/1024:.0f} KB)")
