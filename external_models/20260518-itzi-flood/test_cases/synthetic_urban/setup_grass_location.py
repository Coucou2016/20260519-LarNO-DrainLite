#!/usr/bin/env python3
"""
Set up a GRASS GIS location and import the synthetic test data.

This script:
1. Creates a GRASS GIS location with UTM projection
2. Imports all ASCII raster files into the location
3. Creates a space-time raster dataset (STRDS) for rainfall
4. Sets up the computational region
"""
import os
import sys
import subprocess
import glob

# Paths
GRASS_BIN = r"E:\Tools\GRASS-GIS-8.4.2\grass84.bat"
GISDBASE = r"E:\Projects\20260518-itzi-flood\test_cases\synthetic_urban\grassdata"
LOCATION = "synthetic_urban"
MAPSET = "PERMANENT"
DATA_DIR = r"E:\Projects\20260518-itzi-flood\test_cases\synthetic_urban\input_data"

# UTM zone 50N (WGS84) - generic projected CRS
EPSG = "EPSG:32650"
GRASS_PROJ = f"+proj=utm +zone=50 +datum=WGS84 +units=m +no_defs"


def run_grass(cmd_list, grass_env):
    """Run a GRASS command in a given environment"""
    result = subprocess.run(cmd_list, env=grass_env, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ERROR running {' '.join(cmd_list)}:")
        print(result.stderr)
    return result


def main():
    # Ensure data directory exists
    os.makedirs(GISDBASE, exist_ok=True)

    # Check if location already exists
    if os.path.exists(os.path.join(GISDBASE, LOCATION)):
        print(f"Location {LOCATION} already exists. Skipping creation.")
    else:
        print("=== Creating GRASS location ===")
        # Create location using grass command
        env = os.environ.copy()
        env["GISDBASE"] = GISDBASE
        result = subprocess.run(
            [GRASS_BIN, "-e", "-c", EPSG, os.path.join(GISDBASE, LOCATION)],
            capture_output=True, text=True,
            env=env
        )
        if result.returncode != 0:
            print(f"Error creating location: {result.stderr}")
            print(f"Stdout: {result.stdout}")
            sys.exit(1)
        print("Location created successfully.")

    # Set up GRASS environment for subsequent commands
    grass_env = os.environ.copy()
    grass_env["GISDBASE"] = GISDBASE
    grass_env["LOCATION_NAME"] = LOCATION
    grass_env["MAPSET"] = MAPSET
    grass_env["GRASS_GUI"] = "text"
    grass_env["GRASS_OVERWRITE"] = "1"

    def grass_cmd(module, **kwargs):
        """Run a single GRASS module"""
        cmd = [GRASS_BIN, os.path.join(GISDBASE, LOCATION, MAPSET), "--exec", module]
        for k, v in kwargs.items():
            k = k.replace("_", "-")
            cmd.append(f"{k}={v}")
        result = subprocess.run(cmd, capture_output=True, text=True, env=grass_env)
        if result.returncode != 0:
            print(f"  WARNING [{module}]: {result.stderr.strip()}")
        return result

    print("\n=== Importing raster layers ===")

    # Set region to match the data
    grass_cmd("g.region", n=500, s=0, e=500, w=0, nsres=5, ewres=5)

    # Import DEM
    dem_file = os.path.join(DATA_DIR, "dem.asc")
    if os.path.exists(dem_file):
        print("  Importing DEM...")
        grass_cmd("r.in.gdal", input=dem_file, output="dem")

    # Import Manning's n
    mann_file = os.path.join(DATA_DIR, "mannings.asc")
    if os.path.exists(mann_file):
        print("  Importing Manning's n...")
        grass_cmd("r.in.gdal", input=mann_file, output="mannings")

    # Import boundary conditions
    bc_type_file = os.path.join(DATA_DIR, "bctype.asc")
    if os.path.exists(bc_type_file):
        print("  Importing boundary type...")
        grass_cmd("r.in.gdal", input=bc_type_file, output="bctype")

    bc_value_file = os.path.join(DATA_DIR, "bcvalue.asc")
    if os.path.exists(bc_value_file):
        print("  Importing boundary values...")
        grass_cmd("r.in.gdal", input=bc_value_file, output="bcvalue")

    # Import rainfall maps and create STRDS
    rain_dir = os.path.join(DATA_DIR, "rainfall")
    rain_maps = sorted(glob.glob(os.path.join(rain_dir, "rain_*.asc")))
    if rain_maps:
        print(f"\n=== Importing {len(rain_maps)} rainfall maps ===")
        rain_map_names = []
        for i, rain_file in enumerate(rain_maps):
            map_name = f"rain_{i:04d}"
            print(f"  Importing {map_name}...")
            grass_cmd("r.in.gdal", input=rain_file, output=map_name)
            rain_map_names.append(map_name)

        # Create STRDS for rainfall
        print("\n=== Creating rainfall space-time raster dataset ===")
        strds_name = "rainfall"
        # Remove existing STRDS if any
        grass_cmd("t.remove", inputs=strds_name)

        # Create STRDS
        grass_cmd("t.create",
                  output=strds_name,
                  type="strds",
                  temporaltype="relative",
                  title="Rainfall time series")

        # Register each rainfall map with relative time (5-min intervals)
        for i, map_name in enumerate(rain_map_names):
            rel_time_min = i * 5  # 5-min intervals
            grass_cmd("t.register",
                      input=strds_name,
                      type="raster",
                      map=map_name,
                      start=f"{rel_time_min}",
                      unit="minutes",
                      increment="5 minutes")

    # Set colors
    print("\n=== Setting color tables ===")
    grass_cmd("r.colors", map="dem", color="elevation")
    grass_cmd("r.colors", map="mannings", color="bgyr")

    print("\n=== GRASS GIS setup complete ===")
    print(f"GISDBASE: {GISDBASE}")
    print(f"Location: {LOCATION}")
    print(f"Mapset: {MAPSET}")


if __name__ == "__main__":
    main()
