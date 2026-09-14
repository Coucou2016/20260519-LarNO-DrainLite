#!/usr/bin/env python3
"""
Standalone ITZI simulation runner.
This script sets up the GRASS environment, creates a session,
and runs the ITZI flood simulation.
"""
import os
import sys
import time
from datetime import timedelta

# ============================================================
# Environment Setup
# ============================================================
GRASS_BASE = r"E:\Tools\GRASS-GIS-8.4.2"
GISDBASE = r"E:\Projects\20260518-itzi-flood\test_cases\synthetic_urban\grassdata"
LOCATION = "synthetic_urban"
MAPSET = "PERMANENT"
CONFIG_FILE = r"E:\Projects\20260518-itzi-flood\test_cases\synthetic_urban\synthetic_urban.ini"

# Set up environment (must be done before ANY grass import)
os.environ["GISBASE"] = GRASS_BASE
os.environ["GRASS_OVERWRITE"] = "1"
os.environ["ITZI_VERBOSE"] = "1"
os.environ["GRASS_VERBOSE"] = "0"

# Set PATH with Windows separators (critical for DLL loading)
os.environ["PATH"] = ";".join([
    os.path.join(GRASS_BASE, "lib"),
    os.path.join(GRASS_BASE, "bin"),
    os.path.join(GRASS_BASE, "extrabin"),
    os.path.join(GRASS_BASE, "Python312"),
    os.path.join(GRASS_BASE, "Python312", "Scripts"),
])

# Add DLL search directories
if sys.version_info >= (3, 8):
    for d in ["lib", "bin", "extrabin"]:
        os.add_dll_directory(os.path.join(GRASS_BASE, d))

# Add GRASS Python modules to sys.path
sys.path.insert(0, os.path.join(GRASS_BASE, "etc", "python"))

# Preload GRASS DLLs in dependency order (required for ctypes to find them)
import ctypes
_lib_dir = os.path.join(GRASS_BASE, "lib")
for _dll in ["libgrass_gis.8.4", "libgrass_datetime.8.4", "libgrass_gproj.8.4",
             "libgrass_raster.8.4", "libgrass_vector.8.4", "libgrass_dgl.8.4",
             "libgrass_g3d.8.4", "libgrass_dbmidriver.8.4", "libgrass_dbmibase.8.4"]:
    _dll_path = os.path.join(_lib_dir, _dll + ".dll")
    if os.path.exists(_dll_path):
        try:
            ctypes.CDLL(_dll_path)
        except Exception:
            pass

# ============================================================
# Main
# ============================================================
def main():
    print("=" * 60, flush=True)
    print("  ITZI Flood Simulation - Standalone Runner", flush=True)
    print("=" * 60, flush=True)

    # Initialize GRASS session
    from grass.script import setup as gsetup
    print("Initializing GRASS session...", flush=True)
    gs = gsetup.init(
        path=GISDBASE,
        location=LOCATION,
        mapset=MAPSET,
        grass_path=os.path.join(GRASS_BASE, "grass_launcher.py"),
    )
    print("GRASS session ready.", flush=True)

    # Enable exception raising for easier debugging
    import itzi.messenger as msgr
    msgr.raise_on_error = True

    try:
        from itzi import SimulationRunner
        print(f"Config: {CONFIG_FILE}", flush=True)
        print("Creating runner...", flush=True)
        runner = SimulationRunner()
        print("Initializing simulation...", flush=True)
        runner.initialize(CONFIG_FILE)
        print("Simulation initialized.", flush=True)

        start_time = time.time()
        print("Running simulation...", flush=True)
        runner.run()

        elapsed = timedelta(seconds=int(time.time() - start_time))
        print(f"\n  Simulation completed in {elapsed}", flush=True)

        runner.finalize()
        print("Finalized.", flush=True)
        return 0

    except Exception as e:
        print(f"\n  ERROR: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return 1
    finally:
        gs.finish()
        print("GRASS session closed.", flush=True)


if __name__ == "__main__":
    sys.exit(main())
