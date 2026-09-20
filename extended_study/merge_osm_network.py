#!/usr/bin/env python3
"""Merge OSM pipe network: add cross-connections and identify main trunk."""
import numpy as np, os

CELL = 20.0
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
data = np.load(os.path.join(SCRIPT_DIR, 'output', 'osm_pipe_network.npz'), allow_pickle=True)
nodes = data['nodes'].tolist()
links = data['links'].tolist()
dem = np.load(os.path.join(PROJECT_ROOT, 'LarNO-main', 'benchmark', 'urbanflood', 'geodata', 'region1_20m', 'dem.npy')).astype(np.float64)
H, W = dem.shape

print(f'Input: {len(nodes)} nodes, {len(links)} links')

# === Spatial index ===
bin_size = 8  # cells
grid_H = (H + bin_size - 1) // bin_size
grid_W = (W + bin_size - 1) // bin_size
bins = [[[] for _ in range(grid_W)] for _ in range(grid_H)]
for i, n in enumerate(nodes):
    gr, gc = n['row'] // bin_size, n['col'] // bin_size
    if 0 <= gr < grid_H and 0 <= gc < grid_W:
        bins[gr][gc].append(i)

# === Cross-connect nodes within 100m (5 cells) ===
max_dist = 5
added = 0
lid = len(links)
for i, ni in enumerate(nodes):
    gr, gc = ni['row'] // bin_size, ni['col'] // bin_size
    for dgr in [-1, 0, 1]:
        for dgc in [-1, 0, 1]:
            ngr, ngc = gr + dgr, gc + dgc
            if not (0 <= ngr < grid_H and 0 <= ngc < grid_W):
                continue
            for j in bins[ngr][ngc]:
                if j <= i:
                    continue
                nj = nodes[j]
                dr, dc = ni['row'] - nj['row'], ni['col'] - nj['col']
                dist = np.sqrt(dr**2 + dc**2)
                if 1 < dist <= max_dist:
                    dist_m = dist * CELL
                    slope = max(abs(ni['invert'] - nj['invert']) / max(dist_m, 0.1), 0.001)
                    lid += 1
                    links.append({
                        'id': f'C{lid:05d}', 'from_node': ni['id'], 'to_node': nj['id'],
                        'length': dist_m, 'slope': slope, 'diameter': 0.6,
                        'mannings_n': 0.013, 'type': 'collector',
                    })
                    added += 1

print(f'Cross-connections added: {added}')
print(f'Total links: {len(links)}')

# === Build adjacency ===
idx = {n['id']: i for i, n in enumerate(nodes)}
adj = {i: [] for i in range(len(nodes))}
for lk in links:
    fi, ti = idx.get(lk['from_node']), idx.get(lk['to_node'])
    if fi is not None and ti is not None:
        adj[fi].append(ti)
        adj[ti].append(fi)

# === Connected components ===
visited = set()
components = []
for start in range(len(nodes)):
    if start in visited:
        continue
    comp = []
    queue = [start]
    visited.add(start)
    while queue:
        v = queue.pop(0)
        comp.append(v)
        for nb in adj[v]:
            if nb not in visited:
                visited.add(nb)
                queue.append(nb)
    components.append(comp)

sizes = sorted([len(c) for c in components], reverse=True)
print(f'Components: {len(components)}, top sizes: {sizes[:10]}')

# === Mark main trunk ===
largest = set(components[0]) if components else set()
for ni in largest:
    if len(adj[ni]) >= 2:
        nodes[ni]['type'] = 'main_trunk'

for lk in links:
    fi, ti = idx.get(lk['from_node']), idx.get(lk['to_node'])
    if fi in largest and ti in largest:
        if nodes[fi]['type'] == 'main_trunk' or nodes[ti]['type'] == 'main_trunk':
            lk['type'] = 'main'
            lk['diameter'] = 0.8

n_main = sum(1 for n in nodes if n['type'] == 'main_trunk')
print(f'Main trunk nodes: {n_main}')
print(f'Main links: {sum(1 for l in links if l["type"]=="main")}')

# Save
np.savez(os.path.join('output', 'osm_merged_network.npz'), nodes=nodes, links=links)
print('Saved osm_merged_network.npz')
