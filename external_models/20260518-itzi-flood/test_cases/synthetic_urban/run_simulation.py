#!/usr/bin/env python3
"""
Run the ITZI flood simulation for the synthetic urban test case.

This script:
1. Generates the synthetic test data (if not already done)
2. Sets up the GRASS GIS location
3. Runs the ITZI simulation
4. Prints summary statistics
"""
import os
import sys
import subprocess
import time
from datetime import datetime, timedelta


def main():
    # Paths
    PROJECT_DIR = r"E:\Projects\20260518-itzi-flood"
    TEST_DIR = os.path.join(PROJECT_DIR, "test_cases", "synthetic_urban")
    CONFIG_FILE = os.path.join(TEST_DIR, "synthetic_urban.ini")
    DATA_DIR = os.path.join(TEST_DIR, "input_data")
    GRASS_BIN = r"E:\Tools\GRASS-GIS-8.4.2\grass84.bat"

    print("=" * 60)
    print("  ITZI Flood Simulation - Synthetic Urban Test Case")
    print("=" * 60)
    print(f"  Project: {PROJECT_DIR}")
    print(f"  Test: {TEST_DIR}")
    print()

    # Step 1: Generate test data
    print("Step 1/4: Generating synthetic test data...")
    generate_script = os.path.join(TEST_DIR, "generate_test_data.py")
    result = subprocess.run(
        [sys.executable, generate_script],
        capture_output=True, text=True, cwd=TEST_DIR
    )
    print(result.stdout)
    if result.returncode != 0:
        print("ERROR generating test data:")
        print(result.stderr)
        sys.exit(1)

    # Step 2: Set up GRASS GIS location
    print("Step 2/4: Setting up GRASS GIS location...")
    if os.path.exists(GRASS_BIN):
        setup_script = os.path.join(TEST_DIR, "setup_grass_location.py")
        result = subprocess.run(
            [sys.executable, setup_script],
            capture_output=True, text=True, cwd=TEST_DIR
        )
        print(result.stdout)
        if result.returncode != 0:
            print("ERROR setting up GRASS location:")
            print(result.stderr)
            sys.exit(1)
    else:
        print(f"  WARNING: GRASS GIS not found at {GRASS_BIN}")
        print("  Skipping GRASS location setup.")

    # Step 3: Run ITZI simulation
    print("Step 3/4: Running ITZI simulation...")
    start_time = time.time()

    # Set environment variables for ITZI
    env = os.environ.copy()
    env["GRASS_OVERWRITE"] = "1"
    env["ITZI_VERBOSE"] = "1"  # verbose mode

    result = subprocess.run(
        [sys.executable, "-m", "itzi.itzi", "run", CONFIG_FILE],
        capture_output=True, text=True, cwd=TEST_DIR, env=env
    )

    elapsed = timedelta(seconds=int(time.time() - start_time))

    # Print output
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print("ITZI messages:")
        for line in result.stderr.splitlines():
            if line.strip():
                print(f"  {line}")

    if result.returncode != 0:
        print(f"\nERROR: Simulation failed with exit code {result.returncode}")
        sys.exit(1)

    print(f"\n  Simulation completed in {elapsed}")
    print()

    # Step 4: Print summary
    print("Step 4/4: Summary")
    print("-" * 40)

    # Check for output files
    stats_file = os.path.join(TEST_DIR, "itzi_synth_urban_stats.csv")
    if os.path.exists(stats_file):
        print(f"  Mass balance statistics: {stats_file}")

    # List output map directories
    gisdb = os.path.join(TEST_DIR, "grassdata", "synthetic_urban", "PERMANENT")
    if os.path.exists(gisdb):
        print(f"  GRASS output location: {gisdb}")

    print()
    print("=" * 60)
    print("  Simulation workflow complete!")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
