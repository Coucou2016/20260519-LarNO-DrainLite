#!/usr/bin/env python3
"""
ITZI Dynamic Surface Flow Model
=================================
Proper 2D overland flow model using Manning's equation with:
1. Finite-velocity flow routing (not instant depression filling)
2. Manning's friction to limit flow speed
3. CFL-limited adaptive time stepping for stability
4. Open boundary option at south edge (Shenzhen River)
5. Buildings as impervious obstacles

Physics:
  Overland flow ( Manning ):
    q = -(1/n) * h^(5/3) * sign(dz) * sqrt(|dz|)
  where q is discharge per unit width [m2/s]

  Continuity:
    dh/dt = -div(q) + rain - infiltration - drainage

Reference: ITZI partial-inertia model (itzi.surfaceflow)
Paper: Bates et al. (2010) J. Hydrology
"""

import os, sys, time
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEM_PATH = os.path.join(PROJECT_ROOT, "LarNO-main", "benchmark", "urbanflood", "geodata", "region1_20m", "dem.npy")
FLOOD_DIR = os.path.join(PROJECT_ROOT, "LarNO-main", "benchmark", "urbanflood", "flood", "region1_20m")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

CELL = 20.0; CELL_AREA = CELL**2
G = 9.81
MANNING_N = 0.015  # Street/pavement
DT_MAX = 10.0       # Max time step (s)
CFL = 0.7
HMIN = 0.0001       # Minimum water depth


class DynamicSurfaceFlow:
    """
    2D dynamic overland flow solver.

    Uses Manning's equation for flow routing with CFL-limited
    adaptive time stepping. Supports open boundaries.
    """

    def __init__(self, dem, bldg, manning_n=0.015, open_south=True):
        self.H, self.W = dem.shape
        self.dem = dem.astype(np.float64)
        self.bldg = bldg
        self.active = ~bldg
        self.n = np.full((self.H, self.W), manning_n, dtype=np.float64)
        self.n[bldg] = 1e6  # Buildings: extremely high friction

        # Open boundary: south edge (Shenzhen River)
        self.open_south = open_south
        self.bc_open = np.zeros((self.H, self.W), dtype=bool)
        if open_south:
            # Southernmost active cells: water can exit
            self.bc_open[-1, :] = self.active[-1, :]
            # Also near-south cells
            self.bc_open[-2, :] = self.active[-2, :]

        # State variables
        self.h = np.zeros((self.H, self.W), dtype=np.float64)
        self.hmax = np.zeros((self.H, self.W), dtype=np.float64)
        self.qx = np.zeros((self.H, self.W), dtype=np.float64)
        self.qy = np.zeros((self.H, self.W), dtype=np.float64)
        self.vol_out = 0.0

    def compute_fluxes(self):
        """Compute Manning's overland flow fluxes."""
        h = np.maximum(self.h, HMIN)
        wse = self.dem + h
        dzdx = np.zeros_like(wse); dzdy = np.zeros_like(wse)
        dzdx[:, 1:] = (wse[:, 1:] - wse[:, :-1]) / CELL
        dzdy[1:, :] = (wse[1:, :] - wse[:-1, :]) / CELL

        Sx = np.abs(dzdx); Sy = np.abs(dzdy)
        h53 = h ** (5.0 / 3.0)

        qx = -np.sign(dzdx) * (1.0 / self.n) * h53 * np.sqrt(np.maximum(Sx, 1e-10))
        qy = -np.sign(dzdy) * (1.0 / self.n) * h53 * np.sqrt(np.maximum(Sy, 1e-10))

        # Limit velocity for CFL
        vel_x = np.where(h > 0.001, np.abs(qx) / h, 0)
        vel_y = np.where(h > 0.001, np.abs(qy) / h, 0)
        max_vel = max(vel_x.max(), vel_y.max(), 0.001)

        # CFL time step
        dt_cfl = CFL * CELL / max_vel
        dt = min(DT_MAX, max(0.1, dt_cfl))

        self.qx = qx; self.qy = qy
        return dt

    def step(self, rain_rate_ms, dt_in=None):
        """Advance one adaptive time step."""
        # Compute fluxes and get stable dt
        if dt_in is None:
            dt = self.compute_fluxes()
        else:
            dt = min(dt_in, self.compute_fluxes())
            dt = min(dt, dt_in)

        # Flux divergence
        dqdx = np.zeros((self.H, self.W), dtype=np.float64)
        dqdy = np.zeros((self.H, self.W), dtype=np.float64)
        dqdx[:, :-1] = (self.qx[:, 1:] - self.qx[:, :-1]) / CELL
        dqdy[:-1, :] = (self.qy[1:, :] - self.qy[:-1, :]) / CELL

        # Update water depth (continuity)
        dh = -(dqdx + dqdy) * dt
        if isinstance(rain_rate_ms, np.ndarray):
            dh += rain_rate_ms * dt
        else:
            dh += rain_rate_ms * dt

        # Open boundary: allow outflow (no inflow)
        if self.open_south:
            dh[self.bc_open] = np.minimum(dh[self.bc_open], 0)

        self.h = np.maximum(self.h + dh, 0)
        self.h[self.bldg] = 0
        self.hmax = np.maximum(self.hmax, self.h)

        # Track outflow volume
        if self.open_south:
            outflow = -np.sum(dh[self.bc_open & (dh < 0)]) * CELL_AREA
            self.vol_out += outflow

        return dt

    def simulate(self, duration_s, rainfall_3d, dt_rain=300, pipe_net=None, verbose=True):
        """
        Run simulation with 2D rainfall.
        rainfall_3d: (T, H, W) in mm/5min
        """
        n_rain_steps = rainfall_3d.shape[0]
        t = 0.0; dt = 0.1; next_record = 1800  # 30 min
        rain_idx = 0; next_rain = dt_rain
        runoff_coeff = 0.90
        n_active = self.active.sum()
        cumulative_drained = 0.0

        records = {'time_h': [], 'vol_m3': [], 'hmax_m': [],
                   'flooded_cells': [], 'drained_m3': [], 'vol_out_m3': []}

        # Pipe network setup
        if pipe_net:
            nodes = pipe_net['nodes']; links = pipe_net['links']
            node_cells = [(n['row'], n['col']) for n in nodes
                         if 0 <= n['row'] < self.H and 0 <= n['col'] < self.W and not self.bldg[n['row'], n['col']]]
            pipe_cap = sum(np.pi*l['diameter']**2/4 * (max(l['slope'],0.001))**0.5 * (l['diameter']/4)**(2/3)/0.013
                          for l in links)
        else:
            node_cells = []; pipe_cap = 0

        rain_rate_ms = np.zeros((self.H, self.W), dtype=np.float64)
        step = 0
        while t < duration_s:
            step += 1

            # Update rainfall
            if t >= next_rain and rain_idx < n_rain_steps:
                rain_2d = rainfall_3d[rain_idx]  # mm/5min
                # Building runoff routing
                bldg_rain = np.sum(rain_2d[self.bldg])
                bldg_per_active = bldg_rain / n_active if n_active > 0 else 0

                rain_m = np.zeros((self.H, self.W), dtype=np.float64)
                rain_m[self.active] = (rain_2d[self.active] / 1000.0 + bldg_per_active / 1000.0) * runoff_coeff
                # Convert mm/5min → m/s
                rain_rate_ms = rain_m / dt_rain

                rain_idx += 1; next_rain += dt_rain

            # Surface flow step
            if step > 1:
                dt = self.step(rain_rate_ms)
            else:
                dt = self.step(rain_rate_ms, dt_in=0.5)

            # Pipe drainage (every 30s)
            if pipe_net and step % 3 == 0:
                for r, c in node_cells:
                    if self.h[r, c] > 0.01:
                        surf_wl = self.dem[r, c] + self.h[r, c]
                        head = max(surf_wl - (self.dem[r, c] - 2.0), 0.01)
                        Q_inlet = 0.65 * 2.0 * np.sqrt(2 * G * head)
                        Q_inlet = min(Q_inlet, self.h[r, c] * CELL_AREA / 30.0 * 0.3)
                        removal = Q_inlet * 30.0 / CELL_AREA
                        self.h[r, c] = max(0, self.h[r, c] - removal)
                        # Also drain nearby cells
                        for dr in [-1, 1]:
                            nr = r + dr
                            if 0 <= nr < self.H and not self.bldg[nr, c]:
                                self.h[nr, c] = max(0, self.h[nr, c] - removal * 0.1)
                        cumulative_drained += Q_inlet * 30.0

            t += dt

            # Record
            if t >= next_record:
                vol = float(np.sum(self.h[self.active]) * CELL_AREA)
                hmax = float(np.max(self.h))
                flooded = int(np.sum(self.h > 0.03))
                records['time_h'].append(t / 3600)
                records['vol_m3'].append(vol)
                records['hmax_m'].append(hmax)
                records['flooded_cells'].append(flooded)
                records['drained_m3'].append(float(cumulative_drained))
                records['vol_out_m3'].append(float(self.vol_out))

                if verbose:
                    print(f"  t={t/3600:.1f}h dt={dt:.2f}s vol={vol:.0f}m3 hmax={hmax:.3f}m "
                          f"flooded={flooded}/{self.active.sum()} drained={cumulative_drained:.0f}m3 out={self.vol_out:.0f}m3")
                next_record += 1800

        return records, self.h, self.hmax


def main():
    print("=" * 60)
    print("  ITZI Dynamic Surface Flow Model")
    print("=" * 60)

    dem = np.load(DEM_PATH, allow_pickle=True).astype(np.float64)
    bldg = dem >= 49.9; H, W = dem.shape
    active = ~bldg

    net = np.load(os.path.join(OUT_DIR, 'osm_merged_network.npz'), allow_pickle=True)
    nodes = net['nodes'].tolist(); links = net['links'].tolist()
    pipe_net = {'nodes': nodes, 'links': links}

    events = ['event65', 'event67', 'event1', 'event70']
    duration_s = 6 * 3600

    for evt in events:
        print(f"\n{'='*60}")
        print(f"  {evt}")
        print(f"{'='*60}")

        rainfall = np.load(os.path.join(FLOOD_DIR, evt, 'rainfall.npy'))
        h_ref = np.load(os.path.join(FLOOD_DIR, evt, 'h.npy'))
        peak_ref = np.max(h_ref, axis=0)
        print(f"  MIKE+ ref: peak={np.max(peak_ref):.3f}m")

        # Surface only (closed boundary)
        print("\n  --- Surface Only (closed boundary) ---")
        model_a = DynamicSurfaceFlow(dem.copy(), bldg, open_south=False)
        rec_a, h_a, hmax_a = model_a.simulate(duration_s, rainfall, pipe_net=None)

        # Surface only (open south boundary)
        print("\n  --- Surface Only (open south boundary) ---")
        model_b = DynamicSurfaceFlow(dem.copy(), bldg, open_south=True)
        rec_b, h_b, hmax_b = model_b.simulate(duration_s, rainfall, pipe_net=None)

        # With pipes (open south boundary)
        print("\n  --- With Pipes (open south boundary) ---")
        model_c = DynamicSurfaceFlow(dem.copy(), bldg, open_south=True)
        rec_c, h_c, hmax_c = model_c.simulate(duration_s, rainfall, pipe_net=pipe_net)

        print(f"\n  Peak comparison:")
        print(f"    MIKE+:           {np.max(peak_ref):.3f}m")
        print(f"    ITZI closed:     {np.max(h_a):.3f}m (volume={rec_a['vol_m3'][-1]:.0f}m3)")
        print(f"    ITZI open south: {np.max(h_b):.3f}m (volume={rec_b['vol_m3'][-1]:.0f}m3, outflow={rec_b['vol_out_m3'][-1]:.0f}m3)")
        print(f"    ITZI open+pipe:  {np.max(h_c):.3f}m (volume={rec_c['vol_m3'][-1]:.0f}m3, outflow={rec_c['vol_out_m3'][-1]:.0f}m3, drained={rec_c['drained_m3'][-1]:.0f}m3)")


if __name__ == "__main__":
    main()
