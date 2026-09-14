"""ITZI runner script - executed within GRASS session via grass84.py --exec"""
import sys
sys.path.insert(0, r'E:\Tools\GRASS-GIS-8.4.2\Python312\Lib\site-packages')

import os
os.environ["GRASS_OVERWRITE"] = "1"
os.environ["ITZI_VERBOSE"] = "1"
os.environ["GRASS_VERBOSE"] = "0"

CONFIG_FILE = r"E:\Projects\20260518-itzi-flood\test_cases\synthetic_urban\synthetic_urban.ini"

print("Starting ITZI simulation...")

import itzi.messenger as msgr
msgr.raise_on_error = True

from itzi import SimulationRunner
runner = SimulationRunner()
runner.initialize(CONFIG_FILE)
print("Running simulation...")
runner.run()
print("Finalizing...")
runner.finalize()
print("ITZI simulation complete!")
