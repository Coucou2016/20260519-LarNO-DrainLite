"""
Run ITZI simulation from within a GRASS GIS session.
This script is meant to be executed via grass84.py --exec.
"""
import os
import sys
import time
from datetime import timedelta

# Add ITZI to path if needed
sys.path.insert(0, r'E:\Miniconda3\Lib\site-packages')

import itzi
from itzi import SimulationRunner

def main():
    config_file = os.path.join(
        r'E:\Projects\20260518-itzi-flood\test_cases\synthetic_urban',
        'synthetic_urban.ini'
    )

    print("=" * 60)
    print("  Running ITZI Flood Simulation")
    print("=" * 60)
    print(f"  Config: {config_file}")

    start_time = time.time()

    try:
        runner = SimulationRunner()
        runner.initialize(config_file)
        runner.run()
        runner.finalize()
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

    elapsed = timedelta(seconds=int(time.time() - start_time))
    print(f"\n  Simulation completed in {elapsed}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
