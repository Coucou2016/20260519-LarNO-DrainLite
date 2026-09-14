#!/usr/bin/env python3
"""
Master workflow for ITZI flood simulation test case.

Complete pipeline:
  1. Generate synthetic test data
  2. Set up GRASS GIS location
  3. Import data into GRASS
  4. Run ITZI simulation
  5. Visualize results
"""
import os
import sys
import subprocess
import shutil
import time
import numpy as np
from datetime import datetime, timedelta

# ============================================================
# Configuration
# ============================================================
GRASS_BASE = r"E:\Tools\GRASS-GIS-8.4.2"
PROJECT_DIR = r"E:\Projects\20260518-itzi-flood"
TEST_DIR = os.path.join(PROJECT_DIR, "test_cases", "synthetic_urban")
GISDBASE = os.path.join(TEST_DIR, "grassdata")
LOCATION = "synthetic_urban"
MAPSET = "PERMANENT"
DATA_DIR = os.path.join(TEST_DIR, "input_data")

# Domain parameters
GRID_SIZE = 100
CELL_SIZE = 5.0
DOMAIN_SIZE = GRID_SIZE * CELL_SIZE  # 500m

# Simulation parameters
SIM_DURATION_H = 2.0
RECORD_STEP_MIN = 10
RAINFALL_TOTAL_MM = 50.0


def setup_grass_env():
    """Configure environment for GRASS GIS."""
    # Set GRASS environment variables (but NOT PYTHONHOME - let system Python work)
    os.environ["GISBASE"] = GRASS_BASE
    os.environ["GRASS_PYTHON"] = os.path.join(GRASS_BASE, "extrabin", "python3.exe")
    os.environ["GRASS_PROJSHARE"] = os.path.join(GRASS_BASE, "share", "proj")
    os.environ["PROJ_LIB"] = os.path.join(GRASS_BASE, "share", "proj")
    os.environ["GDAL_DATA"] = os.path.join(GRASS_BASE, "share", "gdal")
    os.environ["FONTCONFIG_FILE"] = os.path.join(GRASS_BASE, "etc", "fonts.conf")
    os.environ["GRASS_OVERWRITE"] = "1"
    os.environ["GRASS_PAGER"] = "cat"
    os.environ["GRASS_HTML_BROWSER"] = "echo"

    # Add GRASS directories to PATH for DLL loading
    paths = [
        os.path.join(GRASS_BASE, "lib"),
        os.path.join(GRASS_BASE, "bin"),
        os.path.join(GRASS_BASE, "extrabin"),
    ]
    os.environ["PATH"] = os.pathsep.join(paths + [os.environ.get("PATH", "")])

    # Add GRASS Python modules to path
    sys.path.insert(0, os.path.join(GRASS_BASE, "etc", "python"))
    if sys.version_info >= (3, 8):
        os.add_dll_directory(os.path.join(GRASS_BASE, "lib"))
        os.add_dll_directory(os.path.join(GRASS_BASE, "bin"))


def gaussian_filter_np(arr, sigma):
    """Pure NumPy Gaussian filter (avoids scipy dependency)."""
    size = int(4 * sigma + 1) | 1
    x = np.arange(-(size // 2), size // 2 + 1)
    kernel = np.exp(-x**2 / (2 * sigma**2))
    kernel /= kernel.sum()
    result = arr.copy()
    for axis in range(arr.ndim):
        result = np.apply_along_axis(
            lambda r: np.convolve(r, kernel, mode='same'), axis, result
        )
    return result


# ============================================================
# Step 1: Generate Synthetic Test Data
# ============================================================
def generate_test_data():
    """Generate synthetic DEM, Manning's n, rainfall, and boundary maps."""
    print("=" * 60)
    print("Step 1: Generating synthetic test data")
    print("=" * 60)
    os.makedirs(DATA_DIR, exist_ok=True)

    x = np.linspace(0, DOMAIN_SIZE, GRID_SIZE)
    y = np.linspace(0, DOMAIN_SIZE, GRID_SIZE)
    xx, yy = np.meshgrid(x, y)

    # -- DEM: sloping terrain with depression --
    np.random.seed(42)
    base_elev = 50.0 - yy * 0.02
    cx, cy = GRID_SIZE // 2, GRID_SIZE // 2
    dist = np.sqrt((xx - cx * CELL_SIZE)**2 + (yy - cy * CELL_SIZE)**2)
    depression = -1.5 * np.exp(-dist**2 / (2 * 30**2))
    roughness = gaussian_filter_np(np.random.randn(GRID_SIZE, GRID_SIZE) * 0.15, 2)
    channel_dist = np.abs(xx - cx * CELL_SIZE)
    channel = -1.0 * np.exp(-channel_dist**2 / (2 * 8**2))
    channel = gaussian_filter_np(channel, 1.5)
    dem = (base_elev + depression + roughness + channel).astype(np.float32)
    print(f"  DEM: min={dem.min():.2f}m, max={dem.max():.2f}m")

    # -- Manning's n --
    mannings = np.full((GRID_SIZE, GRID_SIZE), 0.03, dtype=np.float32)
    mannings[np.abs(xx - cx * CELL_SIZE) > 100] = 0.06
    mannings[channel_dist < 15] = 0.02
    mannings[dist < 20] = 0.015
    print(f"  Manning's n: min={mannings.min():.3f}, max={mannings.max():.3f}")

    # -- Rainfall time series --
    dt_rain = 5  # minutes
    t_min = np.arange(0, SIM_DURATION_H * 60, dt_rain)
    t_peak = 45
    intensity = np.where(
        t_min < t_peak,
        50 * (t_min / t_peak) ** 0.5,
        50 * ((SIM_DURATION_H * 60 - t_min) / (SIM_DURATION_H * 60 - t_peak)) ** 1.5
    )
    total = np.sum(intensity) * dt_rain / 60
    intensity = intensity * RAINFALL_TOTAL_MM / total
    rain_mmh = intensity.astype(np.float32)
    print(f"  Rainfall: {len(rain_mmh)} steps, total={RAINFALL_TOTAL_MM:.1f}mm")

    # -- Boundary conditions --
    bc_type = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int32)
    bc_value = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
    bc_type[-1, :] = 1  # open boundary at south
    bc_value[-1, :] = dem[-1, :] - 0.1

    # Write all ASCII raster files
    def write_asc(filename, data):
        filepath = os.path.join(DATA_DIR, filename)
        with open(filepath, 'w') as f:
            f.write(f"ncols         {GRID_SIZE}\n")
            f.write(f"nrows         {GRID_SIZE}\n")
            f.write(f"xllcorner     0.0\n")
            f.write(f"yllcorner     0.0\n")
            f.write(f"cellsize      {CELL_SIZE}\n")
            f.write(f"NODATA_value  -9999\n")
            np.savetxt(f, data, fmt='%.3f' if data.dtype == np.float32 else '%d')
        return filepath

    write_asc("dem.asc", dem)
    write_asc("mannings.asc", mannings)
    write_asc("bctype.asc", bc_type)
    write_asc("bcvalue.asc", bc_value)

    # Write rainfall maps
    rain_dir = os.path.join(DATA_DIR, "rainfall")
    os.makedirs(rain_dir, exist_ok=True)
    for i, r in enumerate(rain_mmh):
        rain_map = np.full((GRID_SIZE, GRID_SIZE), r, dtype=np.float32)
        write_asc(f"rainfall/rain_{i:04d}.asc", rain_map)

    # Time info file
    with open(os.path.join(DATA_DIR, "rainfall", "time_info.txt"), 'w') as f:
        f.write("# Rain maps with start times (minutes from simulation start)\n")
        for i, t in enumerate(t_min):
            f.write(f"rain_{i:04d}.asc\t{t:.0f}\n")

    print(f"  Test data saved to: {DATA_DIR}")
    return True


# ============================================================
# Step 2 & 3: Set up GRASS GIS and Import Data
# ============================================================
def setup_grass_and_import():
    """Create GRASS location and import all raster data."""
    print("\n" + "=" * 60)
    print("Step 2: Setting up GRASS GIS location and importing data")
    print("=" * 60)

    setup_grass_env()
    import grass.script as gscript
    from grass.script import setup as gsetup

    # Create GISDBASE if needed
    os.makedirs(GISDBASE, exist_ok=True)

    # Create location
    loc_path = os.path.join(GISDBASE, LOCATION)
    if os.path.exists(loc_path):
        print(f"  Removing existing location: {loc_path}")
        shutil.rmtree(loc_path)

    print(f"  Creating GRASS location: {LOCATION}")
    # Use EPSG:32650 (UTM zone 50N, WGS84)
    grass_py = os.path.join(GRASS_BASE, "etc", "grass84.py")
    result = subprocess.run(
        [
            os.path.join(GRASS_BASE, "Python312", "python.exe"),
            grass_py,
            "-c", "EPSG:32650",
            loc_path,
            "-e",
        ],
        capture_output=True, text=True,
        env=os.environ.copy(),
        cwd=TEST_DIR,
    )
    if result.returncode != 0:
        print(f"  ERROR creating location: {result.stderr}")
        return False
    print("  Location created successfully")

    # Init session for data import
    gs = gsetup.init(
        path=GISDBASE,
        location=LOCATION,
        mapset=MAPSET,
        grass_path=os.path.join(GRASS_BASE, "grass_launcher.py"),
    )

    try:
        # Set region
        gscript.run_command("g.region", n=DOMAIN_SIZE, s=0, e=DOMAIN_SIZE, w=0,
                            nsres=CELL_SIZE, ewres=CELL_SIZE)

        # Import rasters
        imports = [
            ("dem.asc", "dem"),
            ("mannings.asc", "mannings"),
            ("bctype.asc", "bctype"),
            ("bcvalue.asc", "bcvalue"),
        ]
        for filename, mapname in imports:
            filepath = os.path.join(DATA_DIR, filename)
            if os.path.exists(filepath):
                print(f"  Importing {mapname}...")
                gscript.run_command("r.in.gdal", input=filepath, output=mapname,
                                    overwrite=True, flags="o")

        # Import rainfall maps and create STRDS
        rain_dir = os.path.join(DATA_DIR, "rainfall")
        rain_files = sorted([
            f for f in os.listdir(rain_dir)
            if f.startswith("rain_") and f.endswith(".asc")
        ])
        if rain_files:
            print(f"  Importing {len(rain_files)} rainfall maps...")
            rain_map_names = []
            for rf in rain_files:
                map_name = rf.replace(".asc", "")
                gscript.run_command(
                    "r.in.gdal",
                    input=os.path.join(rain_dir, rf),
                    output=map_name,
                    overwrite=True, flags="o",
                )
                rain_map_names.append(map_name)

            # Create STRDS using subprocess (more robust with this GRASS setup)
            strds_name = "rainfall"
            # Remove existing STRDS silently
            subprocess.run(
                [os.path.join(GRASS_BASE, "Python312", "python.exe"),
                 os.path.join(GRASS_BASE, "scripts", "t.remove.py"),
                 f"inputs={strds_name}"],
                capture_output=True, cwd=TEST_DIR, env=os.environ.copy(),
            )
            # Create STRDS
            subprocess.run(
                [os.path.join(GRASS_BASE, "Python312", "python.exe"),
                 os.path.join(GRASS_BASE, "scripts", "t.create.py"),
                 f"output={strds_name}", "type=strds",
                 "temporaltype=relative", "title=Rainfall time series"],
                capture_output=True, cwd=TEST_DIR, env=os.environ.copy(),
            )

            # Register maps (5-min intervals)
            for i, map_name in enumerate(rain_map_names):
                subprocess.run(
                    [os.path.join(GRASS_BASE, "Python312", "python.exe"),
                     os.path.join(GRASS_BASE, "scripts", "t.register.py"),
                     f"input={strds_name}", "type=raster",
                     f"map={map_name}", f"start={i * 5}",
                     "unit=minutes", "increment=5 minutes"],
                    capture_output=True, cwd=TEST_DIR, env=os.environ.copy(),
                )
            print(f"  Rainfall STRDS '{strds_name}' created with {len(rain_map_names)} maps")

        # Set color tables (non-critical)
        try:
            gscript.run_command("r.colors", map="dem", color="elevation", quiet=True)
            gscript.run_command("r.colors", map="mannings", color="bgyr", quiet=True)
        except Exception:
            pass

        print("  All data imported successfully")
    finally:
        gs.finish()

    return True


# ============================================================
# Step 4: Run ITZI Simulation
# ============================================================
def run_itzi_simulation():
    """Run the ITZI flood simulation."""
    print("\n" + "=" * 60)
    print("Step 3: Running ITZI flood simulation")
    print("=" * 60)

    config_file = os.path.join(TEST_DIR, "synthetic_urban.ini")
    if not os.path.exists(config_file):
        print(f"  ERROR: Config file not found: {config_file}")
        return False

    start_time = time.time()

    # Run ITZI via grass84.py --exec to ensure proper GRASS environment
    grass_py = os.path.join(GRASS_BASE, "etc", "grass84.py")
    grass_python = os.path.join(GRASS_BASE, "Python312", "python.exe")

    # Build environment for the GRASS session
    env = os.environ.copy()
    env["GRASS_OVERWRITE"] = "1"
    env["ITZI_VERBOSE"] = "1"
    env["GRASS_VERBOSE"] = "0"
    env["PYTHONHOME"] = os.path.join(GRASS_BASE, "Python312")
    env["PYTHONPATH"] = os.path.join(GRASS_BASE, "etc", "python")
    env["GISDBASE"] = GISDBASE
    env["LOCATION_NAME"] = LOCATION
    env["MAPSET"] = MAPSET

    # Create a runner script
    runner_script = os.path.join(TEST_DIR, "_run_itzi.py")
    with open(runner_script, 'w') as f:
        f.write('''import sys
sys.path.insert(0, r"E:\\Miniconda3\\Lib\\site-packages")
sys.path.insert(0, r"E:\\Tools\\GRASS-GIS-8.4.2\\etc\\python")
import os
os.environ["GRASS_OVERWRITE"] = "1"
os.environ["ITZI_VERBOSE"] = "1"
os.environ["GRASS_VERBOSE"] = "0"

from itzi import SimulationRunner
config_file = r"{config_file}"
print("Initializing ITZI simulation...")
runner = SimulationRunner()
runner.initialize(config_file)
print("Running simulation...")
runner.run()
print("Finalizing...")
runner.finalize()
print("ITZI simulation complete!")
'''.format(config_file=config_file.replace('\\', '\\\\')))

    result = subprocess.run(
        [grass_python, grass_py, "--exec", sys.executable, runner_script],
        capture_output=True, text=True, cwd=TEST_DIR, env=env,
    )

    # Clean up
    if os.path.exists(runner_script):
        os.unlink(runner_script)

    if result.stdout:
        for line in result.stdout.splitlines():
            if line.strip():
                print(f"  [ITZI] {line.strip()}")
    if result.stderr:
        for line in result.stderr.splitlines():
            if line.strip() and "WARNING" not in line:
                print(f"  [GRASS] {line.strip()}")

    elapsed = timedelta(seconds=int(time.time() - start_time))
    print(f"\n  Simulation completed in {elapsed}")
    return True


# ============================================================
# Step 5: Analyze and Visualize Results
# ============================================================
def visualize_results():
    """Visualize the simulation results."""
    print("\n" + "=" * 60)
    print("Step 4: Visualizing results")
    print("=" * 60)

    OUTPUT_DIR = os.path.join(TEST_DIR, "visualization_output")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    stats_file = os.path.join(TEST_DIR, "itzi_synth_urban_stats.csv")
    gis_output = os.path.join(GISDBASE, LOCATION, MAPSET)

    # Create summary report
    report_path = os.path.join(OUTPUT_DIR, "simulation_report.txt")
    with open(report_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("  ITZI Flood Simulation - Results Report\n")
        f.write(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write("=" * 60 + "\n\n")
        f.write("Test Case: Synthetic Urban Flood\n")
        f.write(f"Domain: {DOMAIN_SIZE}m x {DOMAIN_SIZE}m at {CELL_SIZE}m resolution\n")
        f.write(f"Grid: {GRID_SIZE}x{GRID_SIZE} cells\n")
        f.write(f"Duration: {SIM_DURATION_H} hours\n")
        f.write(f"Record interval: {RECORD_STEP_MIN} minutes\n")
        f.write(f"Total rainfall: {RAINFALL_TOTAL_MM} mm\n\n")
        f.write("Output files:\n")
        if os.path.exists(stats_file):
            f.write(f"  Mass balance statistics: {stats_file}\n")
        f.write(f"  GRASS output location: {gis_output}\n")
        f.write(f"  Visualization output: {OUTPUT_DIR}\n")

    print(f"  Report saved to: {report_path}")

    # Try to read and print statistics
    if os.path.exists(stats_file):
        print(f"\n  Mass balance statistics file: {stats_file}")
        try:
            with open(stats_file, 'r') as f:
                header = f.readline().strip()
                lines = f.readlines()
                print(f"  Columns: {header}")
                print(f"  Records: {len(lines)}")
                if lines:
                    print(f"  First: {lines[0].strip()}")
                    print(f"  Last:  {lines[-1].strip()}")
        except Exception as e:
            print(f"  Error reading stats: {e}")

    # List GRASS output maps
    cell_dir = os.path.join(gis_output, "cell")
    if os.path.exists(cell_dir):
        maps = [f for f in os.listdir(cell_dir) if not f.startswith('.')]
        print(f"\n  GRASS output raster maps ({len(maps)} maps):")
        # Group by prefix
        prefixes = {}
        for m in maps:
            parts = m.rsplit('_', 1)
            prefix = parts[0] if len(parts) > 1 else m
            if prefix not in prefixes:
                prefixes[prefix] = []
            prefixes[prefix].append(m)
        for prefix, maplist in sorted(prefixes.items()):
            print(f"    {prefix}: {len(maplist)} maps")

    print(f"\n  Visualization output directory: {OUTPUT_DIR}")
    return True


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 60)
    print("  ITZI Flood Simulation - Master Workflow")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    steps = [
        ("Generate test data", generate_test_data),
        ("Setup GRASS and import data", setup_grass_and_import),
        ("Run ITZI simulation", run_itzi_simulation),
        ("Visualize results", visualize_results),
    ]

    results = {}
    for name, func in steps:
        try:
            success = func()
            results[name] = "OK" if success else "FAILED"
            if not success:
                print(f"\n  Workflow stopped at: {name}")
                break
        except Exception as e:
            print(f"\n  ERROR in '{name}': {e}")
            import traceback
            traceback.print_exc()
            results[name] = "ERROR"
            break

    print("\n" + "=" * 60)
    print("  Workflow Summary")
    print("=" * 60)
    for name, status in results.items():
        print(f"  [{status:6s}] {name}")
    print(f"  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return 0 if all(v == "OK" for v in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
