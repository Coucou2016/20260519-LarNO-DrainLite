#!/usr/bin/env python3
"""
Extended Study: Coupled Surface-Drainage Urban Flood Simulation (V2)
=====================================================================
Improved numerical scheme with:
  - Adaptive time stepping (CFL condition)
  - Open boundary conditions
  - Stable diffusion wave solver
  - Bi-directional pipe coupling
"""

import os, sys, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from datetime import timedelta

# Paths
ROOT = r"e:\Projects\20260519-LarNO\LarNO-main\benchmark\urbanflood"
DEM_PATH = os.path.join(ROOT, "geodata", "region1_20m", "dem.npy")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUT_DIR, exist_ok=True)

# Physics
G = 9.81
MANNING_N = 0.015  # street surface
CELL_SIZE = 20.0  # m
SIM_HOURS = 3.0
RAIN_TOTAL_MM = 50.0
DT_BASE = 1.0  # base time step
RECORD_STEP = 300  # record every 5 min
ORIFICE_COEFF = 0.65
MANHOLE_AREA = 2.0  # m2


def create_terrain_and_network():
    """Load DEM and create a simplified pipe network."""
    dem_full = np.load(DEM_PATH, allow_pickle=True)
    building_mask_full = dem_full >= 49.9

    # Extract a 1km x 1km sub-region with good road features
    y0, y1 = 100, 150  # 50 cells = 1km
    x0, x1 = 200, 250  # 50 cells = 1km
    dem = dem_full[y0:y1, x0:x1].copy()
    bldg = building_mask_full[y0:y1, x0:x1].copy()
    H, W = dem.shape

    # Add gentle slope for drainage
    yy, xx = np.mgrid[0:H, 0:W]
    slope_field = (H - 1 - yy) * CELL_SIZE * 0.002  # 0.2% S→N slope
    dem = dem.astype(np.float64)
    dem[~bldg] -= slope_field[~bldg]

    # Design simple pipe network
    # Main N-S trunk along the center
    nodes, links = [], []
    nid, lid = 0, 0

    main_x = W // 2  # center column
    # Branch columns
    branch_cols = [W // 4, W // 2, 3 * W // 4]

    for col in branch_cols:
        row_list = list(range(2, H - 2, 8))  # every 160m
        prev_nid = None
        for row in row_list:
            if bldg[row, col] or dem[row, col] >= 49:
                continue
            nid += 1
            elev = float(dem[row, col])
            invert = max(elev - 2.0, 0.3)
            ntype = 'main_trunk' if col == main_x else 'branch'
            nodes.append({
                'id': f'N{nid:04d}', 'row': row, 'col': col,
                'elevation': elev, 'invert': invert, 'type': ntype,
            })
            if prev_nid is not None:
                pn = nodes[prev_nid - 1]
                dist = abs(row - pn['row']) * CELL_SIZE
                slope = max(abs(invert - pn['invert']) / max(dist, 0.1), 0.001)
                lid += 1
                diameter = 0.8 if col == main_x else 0.5
                links.append({
                    'id': f'C{lid:04d}',
                    'from_node': pn['id'], 'to_node': nodes[-1]['id'],
                    'length': dist, 'slope': slope, 'diameter': diameter,
                    'mannings_n': 0.013, 'type': 'main' if col == main_x else 'branch',
                })
            prev_nid = nid

    # Add outfall at lowest node
    if nodes:
        lowest = min(nodes, key=lambda n: n['invert'])
        nid += 1
        outfall = {
            'id': f'N{nid:04d}', 'row': lowest['row'], 'col': lowest['col'] + 1,
            'elevation': lowest['elevation'] - 2.0,
            'invert': lowest['invert'] - 2.0, 'type': 'outfall',
        }
        nodes.append(outfall)
        lid += 1
        links.append({
            'id': f'C{lid:04d}',
            'from_node': lowest['id'], 'to_node': outfall['id'],
            'length': CELL_SIZE, 'slope': 0.005, 'diameter': 1.0,
            'mannings_n': 0.013, 'type': 'main',
        })

    return dem, bldg, {'nodes': nodes, 'links': links}, (y0, x0)


class StableFloodSolver:
    """Stable 2D flood solver with diffusion wave + 1D pipe coupling."""

    def __init__(self, dem, building_mask):
        self.H, self.W = dem.shape
        self.dem = dem.astype(np.float64)
        self.bldg = building_mask
        self.active = ~building_mask

        # State
        self.h = np.zeros((self.H, self.W), dtype=np.float64)
        self.hmax = np.zeros((self.H, self.W), dtype=np.float64)
        self.vol_in = 0.0

        # Manning's n field (higher for buildings)
        self.n = np.full((self.H, self.W), MANNING_N, dtype=np.float64)
        self.n[building_mask] = 100.0

        # Pipe network
        self.pipes = None
        self.node_cells = []

    def setup_pipes(self, network):
        """Initialize pipe network from design."""
        nodes = network['nodes']
        links = network['links']
        self.n_nodes = len(nodes)
        self.n_links = len(links)

        self.node_z = np.array([n['invert'] for n in nodes])
        self.node_h = np.array([n['invert'] for n in nodes])  # water level in manhole
        self.node_Qin = np.zeros(self.n_nodes)

        self.link_from = np.zeros(self.n_links, dtype=int)
        self.link_to = np.zeros(self.n_links, dtype=int)
        self.link_len = np.zeros(self.n_links)
        self.link_slope = np.zeros(self.n_links)
        self.link_D = np.zeros(self.n_links)
        self.link_Q = np.zeros(self.n_links)

        node_id_to_idx = {n['id']: i for i, n in enumerate(nodes)}
        for i, link in enumerate(links):
            self.link_from[i] = node_id_to_idx[link['from_node']]
            self.link_to[i] = node_id_to_idx[link['to_node']]
            self.link_len[i] = link['length']
            self.link_slope[i] = link['slope']
            self.link_D[i] = link['diameter']

        # Map nodes to grid cells
        for node in nodes:
            r, c = node['row'], node['col']
            if 0 <= r < self.H and 0 <= c < self.W:
                self.node_cells.append((r, c))
            else:
                self.node_cells.append(None)

        self.pipes = True

    def step_surface(self, dt, rain_ms):
        """One surface flow step using diffusion wave."""
        h = self.h
        dem = self.dem
        wse = dem + h
        n = self.n

        # Water surface gradients
        dzdx = np.zeros_like(wse)
        dzdy = np.zeros_like(wse)
        dzdx[:, 1:] = (wse[:, 1:] - wse[:, :-1]) / CELL_SIZE
        dzdy[1:, :] = (wse[1:, :] - wse[:-1, :]) / CELL_SIZE

        Sx = np.abs(dzdx) + 1e-10
        Sy = np.abs(dzdy) + 1e-10

        # Manning's equation: Q = (1/n) * A * R^(2/3) * S^(1/2)
        # Overland flow: Qx = -(1/n) * h^(5/3) * sign(dzdx) * sqrt(|dzdx|)
        h_pow = np.maximum(h, 1e-6) ** (5.0 / 3.0)
        qx = -np.sign(dzdx) * (1.0 / n) * h_pow * np.sqrt(Sx)
        qy = -np.sign(dzdy) * (1.0 / n) * h_pow * np.sqrt(Sy)

        # CFL check
        vel_x = np.where(h > 1e-4, np.abs(qx) / h, 0)
        vel_y = np.where(h > 1e-4, np.abs(qy) / h, 0)
        max_vel = max(vel_x.max(), vel_y.max(), 0.01)

        # Adaptive sub-stepping
        cfl_dt = 0.5 * CELL_SIZE / max_vel
        n_sub = max(1, int(np.ceil(dt / cfl_dt)))
        sub_dt = dt / n_sub

        for _ in range(n_sub):
            h_safe = np.maximum(self.h, 1e-6)
            h_pow = h_safe ** (5.0 / 3.0)
            qx = -np.sign(dzdx) * (1.0 / n) * h_pow * np.sqrt(Sx)
            qy = -np.sign(dzdy) * (1.0 / n) * h_pow * np.sqrt(Sy)

            # Flux divergence
            dqdx = np.zeros_like(wse)
            dqdy = np.zeros_like(wse)
            dqdx[:, :-1] = (qx[:, 1:] - qx[:, :-1]) / CELL_SIZE
            dqdy[:-1, :] = (qy[1:, :] - qy[:-1, :]) / CELL_SIZE

            self.h = np.maximum(self.h - (dqdx + dqdy) * sub_dt + rain_ms * sub_dt, 0)
            self.h[self.bldg] = 0
            wse = self.dem + self.h

            # Update gradients each sub-step for stability
            dzdx[:, 1:] = (wse[:, 1:] - wse[:, :-1]) / CELL_SIZE
            dzdy[1:, :] = (wse[1:, :] - wse[:-1, :]) / CELL_SIZE
            Sx = np.abs(dzdx) + 1e-10
            Sy = np.abs(dzdy) + 1e-10

        self.hmax = np.maximum(self.hmax, self.h)

    def step_pipes(self, dt):
        """One pipe flow step."""
        for i in range(self.n_links):
            fi = self.link_from[i]
            ti = self.link_to[i]
            dh = self.node_h[fi] - self.node_h[ti]

            if dh <= 0 and self.link_Q[i] <= 0:
                self.link_Q[i] = 0
                continue

            slope = max(self.link_slope[i], abs(dh) / self.link_len[i])
            D = self.link_D[i]
            area = np.pi * D**2 / 4
            R = D / 4
            Q_cap = area * (1.0 / 0.013) * R**(2/3) * np.sqrt(slope)
            Q_target = np.sign(dh) * min(Q_cap, 20.0)
            alpha = min(1.0, dt / 20.0)
            self.link_Q[i] += alpha * (Q_target - self.link_Q[i])

        # Update node levels
        for i in range(self.n_nodes):
            net_Q = -self.link_Q[self.link_from == i].sum() + self.link_Q[self.link_to == i].sum()
            net_Q += self.node_Qin[i]
            storage_area = 8.0  # m2 manhole storage
            self.node_h[i] += net_Q * dt / storage_area
            self.node_h[i] = max(self.node_h[i], self.node_z[i])

        self.node_Qin.fill(0)

    def couple(self, dt):
        """Exchange flow between surface and pipes."""
        if not self.pipes:
            return

        for i, cell in enumerate(self.node_cells):
            if cell is None:
                continue
            r, c = cell
            if self.bldg[r, c]:
                continue

            surf_wl = self.dem[r, c] + self.h[r, c]
            pipe_wl = self.node_h[i]

            if surf_wl > pipe_wl and self.h[r, c] > 0.001:
                # Surface -> pipe (inlet)
                head = surf_wl - pipe_wl
                Q = ORIFICE_COEFF * MANHOLE_AREA * np.sqrt(2 * G * head)
                Q = min(Q, self.h[r, c] * CELL_SIZE**2 / dt)
                self.node_Qin[i] += Q
                self.h[r, c] -= Q * dt / CELL_SIZE**2
            elif pipe_wl > surf_wl and pipe_wl > self.dem[r, c]:
                # Pipe -> surface (surcharge)
                head = pipe_wl - max(surf_wl, self.dem[r, c])
                Q = 0.4 * MANHOLE_AREA * np.sqrt(2 * G * head)
                self.node_Qin[i] -= Q
                self.h[r, c] += Q * dt / CELL_SIZE**2

    def simulate(self, duration_s, rain_series, rain_dt, with_pipes=True):
        """Run full simulation."""
        n_steps = int(duration_s / DT_BASE)
        record_steps = int(RECORD_STEP / DT_BASE)

        records = {'time_h': [], 'volume_m3': [], 'hmax_m': [],
                   'flooded_cells': [], 'drain_removed': []}

        rain_idx = 0
        next_rain = rain_dt
        rain_rate = 0.0
        start_t = time.time()

        for step in range(n_steps):
            t = step * DT_BASE

            if t >= next_rain and rain_idx < len(rain_series):
                rain_rate = float(rain_series[rain_idx]) / 1000.0 / 3600.0  # mm/h -> m/s
                rain_idx += 1
                next_rain += rain_dt

            # Surface step
            self.step_surface(DT_BASE, rain_rate)

            # Pipe coupling (every 10s)
            if self.pipes and step % 10 == 0:
                self.couple(10.0)
                self.step_pipes(10.0)

            # Record
            if step % record_steps == 0:
                vol = float(np.sum(self.h[self.active]) * CELL_SIZE**2)
                hmax = float(np.max(self.h))
                flooded = int(np.sum(self.h > 0.03))
                records['time_h'].append(t / 3600)
                records['volume_m3'].append(vol)
                records['hmax_m'].append(hmax)
                records['flooded_cells'].append(flooded)

                if step % (record_steps * 3) == 0:
                    elapsed = time.time() - start_t
                    print(f"  t={t/3600:.1f}h  vol={vol:.0f}m3  hmax={hmax:.3f}m  "
                          f"flooded={flooded}/{self.active.sum()}  dt_sub={DT_BASE:.1f}s")

        elapsed = timedelta(seconds=int(time.time() - start_t))
        print(f"  Completed in {elapsed}, {len(records['time_h'])} records")
        return records


def main():
    print("=" * 60)
    print("  Extended Study V2: Coupled Surface-Drainage Simulation")
    print("=" * 60)

    # Load data
    print("\n[1] Loading terrain and designing pipe network...")
    dem, bldg, network, (y0, x0) = create_terrain_and_network()
    H, W = dem.shape
    print(f"  Domain: {H}x{W} cells = {H*CELL_SIZE/1000:.1f}km x {W*CELL_SIZE/1000:.1f}km")
    print(f"  Buildings: {bldg.sum()}/{bldg.size} cells ({100*bldg.sum()/bldg.size:.0f}%)")
    print(f"  Network: {len(network['nodes'])} nodes, {len(network['links'])} links")
    for ntype in ['main_trunk', 'branch', 'outfall']:
        cnt = sum(1 for n in network['nodes'] if n['type'] == ntype)
        if cnt:
            print(f"    {ntype}: {cnt}")

    # Design storm
    print("\n[2] Creating design storm...")
    duration_s = SIM_HOURS * 3600
    dt_rain = 300  # 5 min
    n_rain = int(duration_s / dt_rain)
    t_rain = np.arange(0, duration_s, dt_rain)
    t_peak = duration_s * 0.35  # peak at 35% of duration

    intensity = np.where(
        t_rain < t_peak,
        30 * (t_rain / t_peak)**0.4,
        30 * ((duration_s - t_rain) / (duration_s - t_peak))**1.2
    )
    total = np.sum(intensity) * dt_rain / 3600
    intensity = intensity * RAIN_TOTAL_MM / total
    print(f"  {n_rain} steps, {RAIN_TOTAL_MM}mm total, peak={np.max(intensity):.1f}mm/h")

    # Run paired simulations
    print("\n[3] Running paired simulations...")

    # A) Surface only
    print("\n  --- A) Surface Only ---")
    solver_a = StableFloodSolver(dem, bldg)
    results_a = solver_a.simulate(duration_s, intensity, dt_rain, with_pipes=False)

    # B) Surface + Drainage
    print("\n  --- B) Surface + Drainage ---")
    solver_b = StableFloodSolver(dem, bldg)
    solver_b.setup_pipes(network)
    results_b = solver_b.simulate(duration_s, intensity, dt_rain, with_pipes=True)

    # Summary
    print("\n[4] Results Summary:")
    print(f"  {'Metric':<25s} {'Surface Only':>15s} {'With Drainage':>15s} {'Change':>10s}")
    print(f"  {'-'*65}")
    for key, label in [('volume_m3', 'Final Volume (m3)'), ('hmax_m', 'Max Depth (m)'),
                        ('flooded_cells', 'Flooded Cells')]:
        va = results_a[key][-1]
        vb = results_b[key][-1]
        if va > 0:
            change = (vb - va) / va * 100
            print(f"  {label:<25s} {va:>15.1f} {vb:>15.1f} {change:>+9.1f}%")

    # Save
    print("\n[5] Saving results...")
    np.savez(os.path.join(OUT_DIR, 'coupled_v2_results.npz'),
             time_h=results_a['time_h'],
             vol_a=results_a['volume_m3'], vol_b=results_b['volume_m3'],
             hmax_a=results_a['hmax_m'], hmax_b=results_b['hmax_m'],
             flooded_a=results_a['flooded_cells'], flooded_b=results_b['flooded_cells'],
             h_final_a=solver_a.h, h_final_b=solver_b.h,
             hmax_arr_a=solver_a.hmax, hmax_arr_b=solver_b.hmax,
             dem=dem, bldg=bldg)

    print(f"  Saved to: {OUT_DIR}")
    print("=" * 60)
    print("  Extended Study Complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
