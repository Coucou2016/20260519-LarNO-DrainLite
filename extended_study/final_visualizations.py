#!/usr/bin/env python3
"""Final statistics and visualizations for OSM road-based drainage study."""
import numpy as np, os
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

OUT_DIR = 'output'
CELL = 20.0; CELL_AREA = CELL**2

flood_cmap = LinearSegmentedColormap.from_list('flood', [
    (0.0, '#F7FBFF'), (0.12, '#C6DBEF'), (0.25, '#6BAED6'),
    (0.4, '#3182BD'), (0.55, '#08519C'), (0.7, '#08306B'), (1.0, '#041838'),
])

data = np.load(os.path.join(OUT_DIR, 'osm_sewer_results.npz'), allow_pickle=True)
time_h = data['time_h']; vol_a = data['vol_a']; vol_b = data['vol_b']
hmax_a = data['hmax_a']; hmax_b = data['hmax_b']
flooded_a = data['flooded_a']; flooded_b = data['flooded_b']
drained = data['drained_m3']; h_a = data['h_final_a']; h_b = data['h_final_b']
dem = data['dem']; bldg = data['bldg']
H, W = dem.shape; extent = [0, W*CELL, 0, H*CELL]
active = ~bldg; bldg_bg = np.where(bldg, 0.3, 0)
diff = h_a - h_b; vlim = max(abs(np.min(diff)), abs(np.max(diff)), 0.01)
vmax = max(np.max(h_a[h_a>0.001]) if np.any(h_a>0.001) else 0.05,
           np.max(h_b[h_b>0.001]) if np.any(h_b>0.001) else 0.05, 0.05)
expected_rain = 0.05 * active.sum() * CELL_AREA

net = np.load(os.path.join(OUT_DIR, 'osm_merged_network.npz'), allow_pickle=True)
nodes = net['nodes'].tolist(); links = net['links'].tolist()
node_lookup = {n['id']: n for n in nodes}

n_main = sum(1 for n in nodes if n['type']=='main_trunk')
n_junc = sum(1 for n in nodes if n['type']=='junction')
n_out = sum(1 for n in nodes if n['type']=='outfall')
n_ml = sum(1 for l in links if l['type']=='main')
n_cl = sum(1 for l in links if l['type']=='collector')
total_len = sum(l['length'] for l in links)
pipe_cap = sum(np.pi*l['diameter']**2/4 * (l['diameter']/4)**(2/3) * np.sqrt(max(l['slope'],0.001))/0.013 for l in links)

vr = (vol_a[-1]-vol_b[-1])/vol_a[-1]*100
dr = (hmax_a[-1]-hmax_b[-1])/hmax_a[-1]*100 if hmax_a[-1]>0 else 0
fr = (flooded_a[-1]-flooded_b[-1])/flooded_a[-1]*100 if flooded_a[-1]>0 else 0

# === FIG 1: Flood comparison ===
fig, axes = plt.subplots(1, 3, figsize=(20, 7))
fig.suptitle('OSM Road-Based Drainage: Flood Reduction (50mm/6h Storm)', fontsize=14, fontweight='bold')
for idx, (ax, h, title) in enumerate([
    (axes[0], h_a, f'Surface Only\nMax={np.max(h_a):.3f}m | Vol={vol_a[-1]:.0f}m3'),
    (axes[1], h_b, f'With OSM Drainage\nMax={np.max(h_b):.3f}m | Vol={vol_b[-1]:.0f}m3'),
]):
    h_show = np.ma.masked_where(h < 0.001, h)
    ax.imshow(np.flipud(bldg_bg), cmap=ListedColormap(['none', '#D3D3D3']),
              extent=extent, aspect='equal', alpha=0.5)
    im = ax.imshow(np.flipud(h_show), cmap=flood_cmap, extent=extent, aspect='equal', vmin=0, vmax=vmax)
    ax.set_title(title); ax.set_xlabel('Easting (m)'); ax.set_ylabel('Northing (m)')
    plt.colorbar(im, ax=ax, label='Depth (m)', shrink=0.8)
ax = axes[2]
ax.imshow(np.flipud(np.where(bldg,0.3,0)), cmap=ListedColormap(['none','#CCC']),
          extent=extent, aspect='equal', alpha=0.4)
im = ax.imshow(np.flipud(diff), cmap=plt.cm.RdBu_r, extent=extent, aspect='equal', vmin=-vlim, vmax=vlim)
for lk in links[::10]:
    fn=node_lookup.get(lk['from_node']); tn=node_lookup.get(lk['to_node'])
    if fn and tn: ax.plot([fn['x_m'],tn['x_m']],[fn['y_m'],tn['y_m']],'black',lw=0.2,alpha=0.25)
ax.set_title(f'Depth Reduction (Blue=Improved)\nDrained={drained[-1]:.0f}m3 ({drained[-1]/expected_rain*100:.1f}% of rain)')
ax.set_xlabel('Easting (m)'); ax.set_ylabel('Northing (m)')
plt.colorbar(im, ax=ax, label='Diff (m)', shrink=0.8)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'osm_flood_comparison.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print('1/4 osm_flood_comparison.png')

# === FIG 2: Time series ===
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('OSM Drainage Network Performance', fontsize=14, fontweight='bold')
ax=axes[0,0]
ax.plot(time_h,vol_a,'r-o',lw=2,ms=3,label='Surface Only')
ax.plot(time_h,vol_b,'b-s',lw=2,ms=3,label='With OSM Drainage')
ax.axhline(y=expected_rain,color='green',ls='--',label=f'Total Rain ({expected_rain:.0f}m3)')
ax.set_xlabel('Time (h)'); ax.set_ylabel('Volume (m3)')
ax.set_title(f'Surface Water Volume ({vr:.1f}% reduction)'); ax.legend(); ax.grid(True,alpha=0.3)

ax=axes[0,1]
ax.plot(time_h,hmax_a,'r-o',lw=2,ms=3,label='Surface Only')
ax.plot(time_h,hmax_b,'b-s',lw=2,ms=3,label='With OSM Drainage')
ax.set_xlabel('Time (h)'); ax.set_ylabel('Max Depth (m)')
ax.set_title(f'Maximum Inundation Depth ({dr:.1f}% reduction)'); ax.legend(); ax.grid(True,alpha=0.3)

ax=axes[1,0]
ax.plot(time_h,np.array(flooded_a)*CELL_AREA,'r-o',lw=2,ms=3,label='Surface Only')
ax.plot(time_h,np.array(flooded_b)*CELL_AREA,'b-s',lw=2,ms=3,label='With OSM Drainage')
ax.set_xlabel('Time (h)'); ax.set_ylabel('Flooded Area (m2)')
ax.set_title(f'Inundated Area ({fr:.1f}% reduction)'); ax.legend(); ax.grid(True,alpha=0.3)

ax=axes[1,1]
ax.fill_between(time_h,0,drained,color='blue',alpha=0.3)
ax.plot(time_h,drained,'b-',lw=2,label='Cumulative Drained')
ax.set_xlabel('Time (h)'); ax.set_ylabel('Volume (m3)')
ax.set_title(f'Water Removed by Drainage (Total: {drained[-1]:.0f}m3)'); ax.legend(); ax.grid(True,alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'osm_timeseries.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print('2/4 osm_timeseries.png')

# === FIG 3: Dashboard ===
fig=plt.figure(figsize=(20,15))
fig.suptitle('Extended Study: OSM Road-Based Urban Drainage - Shenzhen Futian', fontsize=16, fontweight='bold')

ax=fig.add_subplot(2,3,1)
im=ax.imshow(np.flipud(np.where(bldg,np.nan,dem)),cmap='terrain',extent=extent,aspect='equal')
ax.set_title('DEM + Buildings'); plt.colorbar(im,ax=ax,label='Elev (m)')

ax=fig.add_subplot(2,3,2)
ax.imshow(np.flipud(bldg_bg),cmap=ListedColormap(['none','#aaa']),extent=extent,aspect='equal',alpha=0.5)
h_show=np.ma.masked_where(h_b<0.001,h_b)
im=ax.imshow(np.flipud(h_show),cmap=flood_cmap,extent=extent,aspect='equal',vmin=0,vmax=vmax)
ax.set_title(f'Flood with Drainage (Max={np.max(h_b):.2f}m)'); plt.colorbar(im,ax=ax,label='Depth (m)')

ax=fig.add_subplot(2,3,3)
im=ax.imshow(np.flipud(diff),cmap=plt.cm.RdBu_r,extent=extent,aspect='equal',vmin=-vlim,vmax=vlim)
for lk in links[::15]:
    fn=node_lookup.get(lk['from_node']); tn=node_lookup.get(lk['to_node'])
    if fn and tn: ax.plot([fn['x_m'],tn['x_m']],[fn['y_m'],tn['y_m']],'black',lw=0.25,alpha=0.3)
ax.set_title('Depth Reduction + Pipe Network'); plt.colorbar(im,ax=ax,label='Diff (m)')

ax=fig.add_subplot(2,3,4)
mid=H//2; xs=np.arange(W)*CELL
ax.fill_between(xs,dem[mid],dem[mid]+h_a[mid],alpha=0.4,color='red',label='Surface Only')
ax.fill_between(xs,dem[mid],dem[mid]+h_b[mid],alpha=0.4,color='blue',label='With Drainage')
ax.plot(xs,dem[mid],'brown',lw=1.5)
ax.set_xlabel('E (m)'); ax.set_ylabel('Elev (m)')
ax.set_title('E-W Cross Section'); ax.legend(); ax.grid(True,alpha=0.3)

ax=fig.add_subplot(2,3,5)
ax.plot(time_h,vol_a,'r-',lw=2,label='Surface Only')
ax.plot(time_h,vol_b,'b-',lw=2,label='With OSM Drainage')
ax.axhline(y=expected_rain,color='green',ls='--',label=f'Rain ({expected_rain:.0f}m3)')
ax.set_xlabel('Time (h)'); ax.set_ylabel('Volume (m3)')
ax.set_title('Volume Balance'); ax.legend(); ax.grid(True,alpha=0.3)

ax=fig.add_subplot(2,3,6); ax.axis('off')
efficiency = np.array(drained)/(np.array(vol_a)+np.array(drained)+1e-6)*100
ax.text(0.04,0.5,
f"""ROAD-BASED DRAINAGE NETWORK
============================
STUDY AREA: Shenzhen Futian
{H*CELL/1000:.1f}km x {W*CELL/1000:.1f}km @20m resolution
Buildings: {bldg.sum():,} cells ({100*bldg.sum()/bldg.size:.0f}%)
Active cells: {active.sum():,}

ROAD DATA: OpenStreetMap
Motorway/Trunk/Primary/Secondary/Tertiary
Rotated 3deg CW + aligned to DEM

PIPE NETWORK
Manholes: {len(nodes)} total
  Main trunk: {n_main}  Junctions: {n_junc}
  Outfalls: {n_out}
Pipes: {len(links)} total
  Main (D=800mm): {n_ml}
  Collector (D=600mm): {n_cl}
Total length: {total_len:.0f}m
Pipe capacity: {pipe_cap:.0f} m3/s

SIMULATION (50mm/6h Chicago storm)
  Volume:  {vol_a[-1]:.0f} -> {vol_b[-1]:.0f} m3 ({vr:.1f}% red.)
  MaxDepth: {hmax_a[-1]:.3f} -> {hmax_b[-1]:.3f} m ({dr:.1f}% red.)
  Flooded: {flooded_a[-1]} -> {flooded_b[-1]} cells ({fr:.1f}% red.)
  Drained: {drained[-1]:.0f} m3 ({drained[-1]/expected_rain*100:.1f}% of rain)

Method: Voronoi subcatchment routing
+ Manning pipe flow coupling
Adapted from ITZI-flood framework""",
transform=ax.transAxes,fontsize=9,fontfamily='monospace',verticalalignment='center',
bbox=dict(boxstyle='round',facecolor='#F0F8FF',alpha=0.9))
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR,'osm_dashboard.png'),dpi=150,bbox_inches='tight')
plt.close(fig)
print('3/4 osm_dashboard.png')

# === FIG 4: Statistics ===
fig,axes=plt.subplots(2,3,figsize=(18,11))
fig.suptitle('Extended Study - Statistical Summary',fontsize=14,fontweight='bold')

ax=axes[0,0]; bars=ax.bar(['Surface Only','With Drainage'],[vol_a[-1],vol_b[-1]],color=['#E74C3C','#2980B9'],alpha=0.85)
ax.set_ylabel('Volume (m3)'); ax.set_title(f'Final Volume ({vr:.1f}% reduction)')
for b,v in zip(bars,[vol_a[-1],vol_b[-1]]): ax.text(b.get_x()+b.get_width()/2,b.get_height()+10000,f'{v:.0f}',ha='center',fontweight='bold')
ax.grid(True,alpha=0.3,axis='y')

ax=axes[0,1]; bars=ax.bar(['Surface Only','With Drainage'],[flooded_a[-1]*CELL_AREA/1e6,flooded_b[-1]*CELL_AREA/1e6],color=['#E74C3C','#2980B9'],alpha=0.85)
ax.set_ylabel('Flooded Area (km2)'); ax.set_title(f'Inundated Area ({fr:.1f}% reduction)')
for b,v in zip(bars,[flooded_a[-1]*CELL_AREA/1e6,flooded_b[-1]*CELL_AREA/1e6]): ax.text(b.get_x()+b.get_width()/2,b.get_height()+0.1,f'{v:.2f}',ha='center',fontweight='bold')
ax.grid(True,alpha=0.3,axis='y')

ax=axes[0,2]; bars=ax.bar(['Surface Only','With Drainage'],[hmax_a[-1],hmax_b[-1]],color=['#E74C3C','#2980B9'],alpha=0.85)
ax.set_ylabel('Max Depth (m)'); ax.set_title(f'Peak Depth ({dr:.1f}% reduction)')
for b,v in zip(bars,[hmax_a[-1],hmax_b[-1]]): ax.text(b.get_x()+b.get_width()/2,b.get_height()+0.02,f'{v:.3f}',ha='center',fontweight='bold')
ax.grid(True,alpha=0.3,axis='y')

ax=axes[1,0]
h_a_f=h_a[active&(h_a>0.001)]; h_b_f=h_b[active&(h_b>0.001)]
ax.hist(np.clip(h_a_f,0,np.percentile(h_a_f,99)),bins=40,alpha=0.5,color='red',label='Surface Only')
ax.hist(np.clip(h_b_f,0,np.percentile(h_b_f,99)),bins=40,alpha=0.5,color='blue',label='With Drainage')
ax.set_xlabel('Depth (m)'); ax.set_ylabel('Cell Count'); ax.set_title('Flood Depth Distribution'); ax.legend()

ax=axes[1,1]
eff=np.array(drained)/(np.array(vol_a)+np.array(drained)+1e-6)*100
ax.plot(time_h,eff,'b-o',lw=2,ms=3)
ax.set_xlabel('Time (h)'); ax.set_ylabel('Drainage Efficiency (%)')
ax.set_title('Drainage Efficiency Over Time'); ax.grid(True,alpha=0.3)

ax=axes[1,2]; ax.axis('off')
ax.text(0.05,0.5,
f"""SIMULATION SUMMARY
==================
Domain: {H*CELL/1000:.1f} x {W*CELL/1000:.1f} km
Grid: {H}x{W} cells @ {CELL:.0f}m
Active area: {active.sum()*CELL_AREA/1e6:.2f} km2
Storm: 50mm / 6h Chicago
Total rainfall: {expected_rain:.0f} m3

PIPE NETWORK
Nodes: {len(nodes)}  Links: {len(links)}
Total length: {total_len:.0f} m
Capacity: {pipe_cap:.0f} m3/s

RESULTS
Volume reduction:    {vr:.1f}%
Depth reduction:     {dr:.1f}%
Area reduction:      {fr:.1f}%
Total drained:       {drained[-1]:.0f} m3
Peak efficiency:     {np.max(eff):.1f}%""",
transform=ax.transAxes,fontsize=10,fontfamily='monospace',verticalalignment='center',
bbox=dict(boxstyle='round',facecolor='#F0F8FF',alpha=0.9))
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR,'osm_statistics.png'),dpi=150,bbox_inches='tight')
plt.close(fig)
print('4/4 osm_statistics.png')

# === Summary ===
print(f"""
{'='*60}
  FINAL RESULTS
{'='*60}
  Volume:       {vol_a[-1]:.0f} -> {vol_b[-1]:.0f} m3  ({vr:.1f}% reduction)
  Max Depth:    {hmax_a[-1]:.3f} -> {hmax_b[-1]:.3f} m  ({dr:.1f}% reduction)
  Flooded Area: {flooded_a[-1]*CELL_AREA/1e6:.2f} -> {flooded_b[-1]*CELL_AREA/1e6:.2f} km2 ({fr:.1f}% reduction)
  Drained:      {drained[-1]:.0f} m3 ({drained[-1]/expected_rain*100:.1f}% of rainfall)
  Network:      {len(nodes)} manholes, {len(links)} pipes, {total_len:.0f}m
  Pipe capacity: {pipe_cap:.0f} m3/s
  Peak efficiency: {np.max(eff):.1f}%
{'='*60}
""")
