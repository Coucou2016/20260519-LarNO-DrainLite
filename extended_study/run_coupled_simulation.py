#!/usr/bin/env python3
"""
Extended Study: Coupled Surface-Drainage Urban Flood Simulation
================================================================
Implements a simplified but physically-grounded coupled model:
  1. 2D surface flow (diffusion wave approximation)
  2. 1D pipe flow (Manning's equation)
  3. Bi-directional coupling (weir/orifice exchange)

Compares two scenarios:
  A) Surface-only (no drainage) — baseline
  B) Surface + Drainage (coupled) — with underground pipes

Based on the ITZI-flood coupling approach adapted for the
Shenzhen Futian benchmark dataset.
"""

import os, sys, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy import ndimage
from datetime import timedelta

# ============================================================
# Paths
# ============================================================
ROOT = r"e:\Projects\20260519-LarNO\LarNO-main\benchmark\urbanflood"
DEM_PATH = os.path.join(ROOT, "geodata", "region1_20m", "dem.npy")
FLOOD_DIR = os.path.join(ROOT, "flood", "region1_20m")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# Physical constants & parameters
# ============================================================
G = 9.81  # gravity (m/s²)
MANNING_N_SURFACE = 0.015  # surface Manning's n (street/pavement)
MANNING_N_PIPE = 0.013  # pipe Manning's n (concrete)
DT = 10.0  # time step (seconds)
CELL_SIZE = 20.0  # meters
DURATION = 10800  # 3 hours (seconds)
RECORD_INTERVAL = 300  # 5 minutes

# Pipe coupling parameters
ORIFICE_COEFF = 0.65  # orifice discharge coefficient
WEIR_COEFF = 0.4  # weir coefficient (free weir)
SUBMERGED_WEIR_COEFF = 0.8  # submerged weir coefficient
MANHOLE_AREA = 2.0  # m² (typical manhole inlet area)


def load_cached_or_compute_network():
    """Load pre-designed network or create from DEM."""
    cache_path = os.path.join(OUT_DIR, 'pipe_network.npz')
    if os.path.exists(cache_path):
        data = np.load(cache_path, allow_pickle=True)
        return (data['nodes'].tolist(), data['links'].tolist(),
                data['sub_region'].tolist(), data['ns_positions'].tolist(),
                data['ew_positions'].tolist())
    return None


class PipeFlowSolver:
    """Simplified 1D pipe flow solver using Manning's equation."""

    def __init__(self, nodes, links):
        self.nodes = nodes
        self.links = links
        self.n_pipes = len(links)
        self.n_nodes = len(nodes)

        # Build node index lookup
        self.node_idx = {n['id']: i for i, n in enumerate(nodes)}

        # Initialize pipe states
        self.pipe_Q = np.zeros(self.n_pipes)  # discharge (m3/s)
        self.pipe_V = np.zeros(self.n_pipes)  # velocity (m/s)
        self.pipe_depth = np.zeros(self.n_pipes)  # flow depth in pipe (m)

        # Initialize node states
        self.node_H = np.array([n['invert'] for n in nodes])  # water level
        self.node_inflow = np.zeros(self.n_nodes)  # inflow from surface

        # Pre-compute pipe properties
        self.pipe_length = np.array([l['length'] for l in links])
        self.pipe_slope = np.array([l['slope'] for l in links])
        self.pipe_diameter = np.array([l['diameter'] for l in links])
        self.pipe_area = np.pi * (self.pipe_diameter / 2) ** 2  # full area
        self.pipe_wetted_perimeter = np.pi * self.pipe_diameter  # full perimeter

        # Link node connections
        self.link_from = np.array([self.node_idx[l['from_node']] for l in links])
        self.link_to = np.array([self.node_idx[l['to_node']] for l in links])

    def step(self, dt):
        """Advance pipe flow by one time step."""
        n_p = self.n_pipes

        for i in range(n_p):
            from_idx = self.link_from[i]
            to_idx = self.link_to[i]

            # Water level difference drives flow
            dh = self.node_H[from_idx] - self.node_H[to_idx]

            if dh <= 0 and self.pipe_Q[i] <= 0:
                # No driving head and no inertia — flow stops
                self.pipe_Q[i] = 0
                self.pipe_V[i] = 0
                self.pipe_depth[i] = 0
                continue

            # Slope = max(pipe slope, water surface slope)
            effective_slope = max(self.pipe_slope[i], abs(dh) / self.pipe_length[i])

            # Full pipe capacity using Manning's equation
            D = self.pipe_diameter[i]
            area = self.pipe_area[i]
            R = D / 4  # hydraulic radius for full circular pipe
            Q_capacity = area * (1 / self.pipe_length[i]) * (R ** (2 / 3)) * np.sqrt(effective_slope)

            # Target flow rate
            sign = 1 if dh >= 0 else -1
            Q_target = sign * min(abs(Q_capacity), 10.0)  # Limit max flow

            # Smooth approach to target
            alpha = min(1.0, dt / 30.0)  # time constant ~30s
            self.pipe_Q[i] = self.pipe_Q[i] + alpha * (Q_target - self.pipe_Q[i])

            # Update velocity and depth
            Q_abs = abs(self.pipe_Q[i])
            if Q_abs > 1e-6:
                # Compute partial depth from discharge (Manning inverse)
                theta = Q_abs / (np.sqrt(effective_slope) * (1 / 0.013))
                ratio = min(1.0, (theta / (D ** (8 / 3)))**0.5)
                self.pipe_depth[i] = ratio * D
                effective_area = self._partial_area(D, ratio)
                self.pipe_V[i] = Q_abs / max(effective_area, 1e-6)
            else:
                self.pipe_depth[i] = 0
                self.pipe_V[i] = 0

        # Update node water levels based on mass balance
        for i in range(self.n_nodes):
            inflow_sum = 0
            outflow_sum = 0
            for j in range(n_p):
                if self.link_to[j] == i:
                    inflow_sum += max(0, self.pipe_Q[j])
                if self.link_from[j] == i:
                    outflow_sum += max(0, self.pipe_Q[j])

            net_flow = inflow_sum - outflow_sum + self.node_inflow[i]
            # Simple storage routing (manhole storage)
            self.node_H[i] += net_flow * dt / 10.0  # 10 m² effective storage area
            self.node_H[i] = max(self.node_H[i], self.nodes[i]['invert'])

    @staticmethod
    def _partial_area(D, ratio):
        """Compute partial flow area for a circular pipe."""
        if ratio <= 0 or ratio >= 1:
            return np.pi * D**2 / 4 * min(ratio, 1)
        theta = 2 * np.arccos(1 - 2 * ratio)
        return (theta - np.sin(theta)) * D**2 / 8


class SurfaceFlowSolver:
    """Simplified 2D surface flow using diffusion wave approximation."""

    def __init__(self, dem, building_mask):
        self.H, self.W = dem.shape
        self.dem = dem.copy()
        self.building_mask = building_mask

        # State variables
        self.h = np.zeros((self.H, self.W), dtype=np.float32)  # water depth (m)
        self.qx = np.zeros((self.H, self.W), dtype=np.float32)  # x-discharge
        self.qy = np.zeros((self.H, self.W), dtype=np.float32)  # y-discharge
        self.hmax = np.zeros((self.H, self.W), dtype=np.float32)  # max depth

        # Building mask: no flow through buildings
        self.active = ~building_mask

        # Pre-compute Manning's n field
        self.mannings = np.full((self.H, self.W), MANNING_N_SURFACE, dtype=np.float32)
        self.mannings[building_mask] = 10.0  # effectively no flow

    def step(self, dt, rain_rate_ms, drain_rate=None):
        """
        Advance surface flow by one time step.

        Parameters
        ----------
        dt : float
            Time step (seconds)
        rain_rate_ms : float or ndarray
            Rainfall rate in m/s
        drain_rate : ndarray or None
            Drainage removal rate in m/s (negative = removal)
        """
        g = G
        n = self.mannings
        h = self.h
        dem = self.dem

        # Water surface elevation
        wse = dem + h

        # Compute gradients
        dzdx = np.zeros_like(wse)
        dzdy = np.zeros_like(wse)
        dzdx[:, 1:] = (wse[:, 1:] - wse[:, :-1]) / CELL_SIZE
        dzdy[1:, :] = (wse[1:, :] - wse[:-1, :]) / CELL_SIZE

        # Slope magnitude
        slope_x = np.abs(dzdx)
        slope_y = np.abs(dzdy)

        # Diffusion wave: q = -sign(grad) * (1/n) * h^(5/3) * sqrt(|grad|)
        # Use Manning's equation for overland flow
        h_safe = np.maximum(h, 0.0001)
        conveyance = (1.0 / n) * (h_safe ** (5 / 3))

        qx = -np.sign(dzdx) * conveyance * np.sqrt(slope_x)
        qy = -np.sign(dzdy) * conveyance * np.sqrt(slope_y)

        # Limit velocity to prevent instability (CFL condition)
        max_vel = CELL_SIZE / dt * 0.5
        vel_x = np.where(h_safe > 0.001, np.abs(qx) / h_safe, 0)
        vel_y = np.where(h_safe > 0.001, np.abs(qy) / h_safe, 0)
        qx = np.where(vel_x > max_vel, qx * max_vel / np.maximum(vel_x, 1e-6), qx)
        qy = np.where(vel_y > max_vel, qy * max_vel / np.maximum(vel_y, 1e-6), qy)

        # Flux divergence
        dqdx = np.zeros_like(wse)
        dqdy = np.zeros_like(wse)
        dqdx[:, :-1] = (qx[:, 1:] - qx[:, :-1]) / CELL_SIZE
        dqdy[:-1, :] = (qy[1:, :] - qy[:-1, :]) / CELL_SIZE

        # Update water depth
        dhdt = -(dqdx + dqdy)

        # Add rainfall (uniform over domain)
        if isinstance(rain_rate_ms, np.ndarray):
            dhdt += rain_rate_ms
        else:
            dhdt += rain_rate_ms

        # Add drainage (negative = removal)
        if drain_rate is not None:
            dhdt += drain_rate

        h_new = h + dhdt * dt
        h_new = np.maximum(h_new, 0)  # No negative depths
        h_new[self.building_mask] = 0  # No water in buildings

        self.h = h_new.astype(np.float32)
        self.qx = qx.astype(np.float32)
        self.qy = qy.astype(np.float32)
        self.hmax = np.maximum(self.hmax, self.h)


def run_simulation(dem, building_mask, rainfall_series, dt_rain,
                   network, with_drainage=True, label="Simulation"):
    """Run a single flood simulation."""
    H, W = dem.shape
    n_steps = int(DURATION // DT)
    record_steps = int(RECORD_INTERVAL // DT)

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  Drainage: {'ON' if with_drainage else 'OFF'}")
    print(f"  Duration: {DURATION/3600:.0f}h, DT: {DT}s, Steps: {n_steps}")
    print(f"{'='*60}")

    # Initialize surface solver
    surface = SurfaceFlowSolver(dem, building_mask)

    # Initialize pipe solver (if applicable)
    pipes = None
    if with_drainage and network:
        pipes = PipeFlowSolver(network['nodes'], network['links'])
        # Map pipe nodes to grid cells for coupling
        node_cells = []
        for node in network['nodes']:
            row = node['row']
            col = node['col']
            if 0 <= row < H and 0 <= col < W:
                node_cells.append((row, col))
            else:
                node_cells.append(None)
        pipe_node_cells = node_cells

    # Results storage
    records = {'time_h': [], 'volume_m3': [], 'hmax_m': [],
               'flooded_cells': [], 'drain_removed_m3': []}

    start_time = time.time()
    rain_idx = 0
    next_rain = dt_rain
    rain_rate = 0.0

    for step in range(n_steps):
        t = step * DT

        # Update rainfall
        if t >= next_rain and rain_idx < len(rainfall_series):
            rain_rate = float(rainfall_series[rain_idx]) / 1000.0 / 3600.0  # mm/h -> m/s
            rain_idx += 1
            next_rain += dt_rain

        # Surface flow step
        drain_rate = None
        if pipes and (step % 6 == 0):  # Couple every 30s for efficiency
            # Compute coupling flows
            drain_rate = np.zeros((H, W), dtype=np.float32)

            for i, node in enumerate(network['nodes']):
                cell = pipe_node_cells[i]
                if cell is None:
                    continue
                row, col = cell
                if building_mask[row, col]:
                    continue

                surf_wl = dem[row, col] + surface.h[row, col]
                pipe_wl = pipes.node_H[i]

                # Coupling: exchange flow between surface and pipe
                if surf_wl > pipe_wl and surface.h[row, col] > 0.001:
                    # Surface to pipe (inlet capture)
                    head_diff = surf_wl - pipe_wl
                    # Orifice flow
                    Q_inlet = ORIFICE_COEFF * MANHOLE_AREA * np.sqrt(2 * G * head_diff)
                    Q_inlet = min(Q_inlet, surface.h[row, col] * CELL_SIZE * CELL_SIZE / DT)
                    pipes.node_inflow[i] += Q_inlet
                    drain_rate[row, col] -= Q_inlet / (CELL_SIZE * CELL_SIZE)
                elif pipe_wl > surf_wl and pipe_wl > dem[row, col]:
                    # Pipe to surface (surcharge/flooding)
                    head_diff = pipe_wl - max(surf_wl, dem[row, col])
                    Q_surcharge = WEIR_COEFF * MANHOLE_AREA * np.sqrt(2 * G * head_diff)
                    pipes.node_inflow[i] -= Q_surcharge
                    drain_rate[row, col] += Q_surcharge / (CELL_SIZE * CELL_SIZE)

            # Pipe flow step (every 30s)
            pipes.step(30.0)
            pipes.node_inflow.fill(0)  # Reset inflows after processing

        # Surface step
        surface.step(DT, rain_rate, drain_rate)

        # Record
        if step % record_steps == 0:
            vol = np.sum(surface.h[surface.active]) * CELL_SIZE * CELL_SIZE
            hmax = float(np.max(surface.h))
            flooded = int(np.sum(surface.h > 0.03))
            drain_vol = np.sum(surface.hmax - surface.h) * CELL_SIZE * CELL_SIZE if not with_drainage else 0

            records['time_h'].append(t / 3600)
            records['volume_m3'].append(vol)
            records['hmax_m'].append(hmax)
            records['flooded_cells'].append(flooded)
            records['drain_removed_m3'].append(0.0)

            if step % (record_steps * 6) == 0:  # Print every 30 min
                elapsed = time.time() - start_time
                print(f"  t={t/3600:>4.1f}h  vol={vol:.1f}m3  hmax={hmax:.3f}m  "
                      f"flooded={flooded}  elapsed={elapsed:.0f}s")

    elapsed = timedelta(seconds=int(time.time() - start_time))
    print(f"  Completed in {elapsed}")

    return records, surface.h, surface.hmax


def main():
    print("=" * 60)
    print("  Extended Study: Coupled Surface-Drainage Simulation")
    print("  Shenzhen Futian District")
    print("=" * 60)

    # Load data
    print("\n[1] Loading terrain and network...")
    dem = np.load(DEM_PATH, allow_pickle=True)
    building_mask = dem >= 49.9

    # Use a smaller sub-region for the simulation (2km x 2km)
    y0, y1 = 90, 190  # 100 rows = 2km
    x0, x1 = 160, 260  # 100 cols = 2km
    sub_dem = dem[y0:y1, x0:x1].copy()
    sub_bldg = building_mask[y0:y1, x0:x1].copy()

    H_sub, W_sub = sub_dem.shape
    print(f"  Simulation domain: {H_sub}x{W_sub} cells = "
          f"{H_sub*CELL_SIZE/1000:.1f}km x {W_sub*CELL_SIZE/1000:.1f}km")

    # Design storm: 50mm over 6h, peak at 2h (Chicago design storm)
    dt_rain_step = 300  # 5 min
    n_rain = int(DURATION // dt_rain_step)
    t_rain = np.arange(0, DURATION, dt_rain_step)
    t_peak = 7200  # peak at 2 hours

    # Chicago hydrograph: rising limb r^0.5, falling limb r^1.5
    rain_intensity = np.where(
        t_rain < t_peak,
        30 * (t_rain / t_peak) ** 0.4,
        30 * ((DURATION - t_rain) / (DURATION - t_peak)) ** 1.2
    )
    # Scale to 50mm total
    total_raw = np.sum(rain_intensity) * dt_rain_step / 3600
    rain_intensity = rain_intensity * 50.0 / total_raw
    print(f"  Design storm: {n_rain} steps @ 5min, "
          f"total={np.sum(rain_intensity)*dt_rain_step/3600:.1f}mm, "
          f"peak={np.max(rain_intensity):.1f}mm/h")

    # Design network for sub-region
    print("\n[2] Designing pipe network for sub-region...")
    from design_pipe_network import design_pipe_network
    network = design_pipe_network(sub_dem, sub_bldg, sub_region=(0, H_sub, 0, W_sub))

    if len(network['nodes']) == 0:
        print("  WARNING: No nodes created. Adjusting design parameters...")
        # Fallback: create a simple grid network
        network = _create_simple_network(sub_dem, sub_bldg)

    print(f"  Network: {len(network['nodes'])} nodes, {len(network['links'])} links")

    # Run paired simulations
    print("\n[3] Running paired simulations...")

    # A) Surface only
    print("\n  --- A) Surface Only (No Drainage) ---")
    results_a, h_final_a, hmax_a = run_simulation(
        sub_dem, sub_bldg, rain_intensity, 300,
        network, with_drainage=False, label="A) Surface Only"
    )

    # B) Surface + Drainage
    print("\n  --- B) Surface + Drainage (Coupled) ---")
    results_b, h_final_b, hmax_b = run_simulation(
        sub_dem, sub_bldg, rain_intensity, 300,
        network, with_drainage=True, label="B) Surface + Drainage"
    )

    # Save results
    print("\n[4] Saving results...")
    np.savez(os.path.join(OUT_DIR, 'coupled_simulation_results.npz'),
             time_h=results_a['time_h'],
             volume_a=results_a['volume_m3'],
             volume_b=results_b['volume_m3'],
             hmax_a=results_a['hmax_m'],
             hmax_b=results_b['hmax_m'],
             flooded_a=results_a['flooded_cells'],
             flooded_b=results_b['flooded_cells'],
             h_final_a=h_final_a,
             h_final_b=h_final_b,
             hmax_arr_a=hmax_a,
             hmax_arr_b=hmax_b,
             dem=sub_dem,
             building_mask=sub_bldg)

    print(f"\n  Results saved to: {OUT_DIR}")
    return 0


def _create_simple_network(dem, building_mask):
    """Create a simple grid network as fallback."""
    H, W = dem.shape
    nodes, links = [], []
    nid, lid = 0, 0

    # Main lines every 500m (25 cells)
    for col in range(W // 8, W, W // 4):
        for row in range(0, H, 10):
            if not building_mask[row, col] and dem[row, col] < 49:
                nid += 1
                nodes.append({
                    'id': f'N_{nid:04d}', 'row': row, 'col': col,
                    'x_m': col * CELL_SIZE + CELL_SIZE / 2,
                    'y_m': (H - row) * CELL_SIZE - CELL_SIZE / 2,
                    'elevation': float(dem[row, col]),
                    'invert': float(max(dem[row, col] - 2.0, 0.5)),
                    'type': 'main_trunk',
                })

    # Connect consecutive nodes per column
    for col_offset in range(3):  # 3 main lines
        col_nodes = sorted(
            [n for n in nodes if n['col'] == W // 8 + col_offset * W // 4],
            key=lambda n: n['row']
        )
        for i in range(len(col_nodes) - 1):
            fn, tn = col_nodes[i], col_nodes[i + 1]
            dist = np.sqrt((fn['x_m'] - tn['x_m'])**2 + (fn['y_m'] - tn['y_m'])**2)
            if dist < 500:
                lid += 1
                links.append({
                    'id': f'C_{lid:04d}',
                    'from_node': fn['id'], 'to_node': tn['id'],
                    'length': dist,
                    'slope': max(abs(fn['invert'] - tn['invert']) / max(dist, 0.1), 0.001),
                    'diameter': 0.8, 'mannings_n': 0.013, 'type': 'main',
                })

    return {'nodes': nodes, 'links': links, 'ns_positions': [],
            'ew_positions': [], 'sub_region': [0, H, 0, W]}


if __name__ == "__main__":
    sys.exit(main())
