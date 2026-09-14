#!/usr/bin/env python3
"""Create a valid SWMM input file for the urban district drainage network."""
import os

DOMAIN_SIZE = 400.0
main_x = DOMAIN_SIZE / 2

main_nodes_y = [350, 290, 230, 170, 110, 50]
main_inverts = [46.0, 46.3, 46.6, 46.9, 47.2, 47.5]
branch_left = [(80, 320), (140, 320), (80, 200), (140, 200)]
branch_right = [(260, 320), (320, 320), (260, 200), (320, 200)]
all_branches = branch_left + branch_right

OUTPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "drainage_network.inp")

lines = [
    "[TITLE]",
    "Urban District Storm Drainage Network",
    "",
    "[OPTIONS]",
    "FLOW_UNITS           CMS",
    "FLOW_ROUTING         DYNWAVE",
    "START_DATE           01/01/2026",
    "START_TIME           00:00:00",
    "REPORT_START_DATE    01/01/2026",
    "REPORT_START_TIME    00:00:00",
    "END_DATE             01/01/2026",
    "END_TIME             02:00:00",
    "SWEEP_START          01/01",
    "SWEEP_END            12/31",
    "DRY_DAYS             0",
    "REPORT_STEP          00:10:00",
    "WET_STEP             00:01:00",
    "DRY_STEP             00:05:00",
    "ROUTING_STEP         0:00:05",
    "ALLOW_PONDING        NO",
    "INERTIAL_DAMPING     PARTIAL",
    "VARIABLE_STEP        0.75",
    "MINIMUM_STEP         0.5",
    "THREADS              1",
    "",
    "[EVAPORATION]",
    "CONSTANT         0.0",
    "DRY_ONLY         NO",
    "",
]

# ---- JUNCTIONS ----
lines.append("[JUNCTIONS]")
nid = 1
for y, inv in zip(main_nodes_y, main_inverts):
    lines.append(f"M_{nid:02d}        {inv:.2f}    2.0       0.0        0.0         0.0")
    nid += 1
for x, y in all_branches:
    inv = 47.5 - y * 0.005 + 0.3
    lines.append(f"B_{nid:02d}        {inv:.2f}    1.5       0.0        0.0         0.0")
    nid += 1
lines.append("")

# ---- OUTFALLS ----
lines.append("[OUTFALLS]")
lines.append("Outfall_01    46.0    FREE    NO")
lines.append("")

# ---- CONDUITS ----
lines.append("[CONDUITS]")
lines += [
    "C_M01    M_01   M_02    60      0.013      0      0       0         0",
    "C_M02    M_02   M_03    60      0.013      0      0       0         0",
    "C_M03    M_03   M_04    60      0.013      0      0       0         0",
    "C_M04    M_04   M_05    60      0.013      0      0       0         0",
    "C_M05    M_05   M_06    60      0.013      0      0       0         0",
    "C_out    M_06   Outfall_01  50   0.013      0      0       0         0",
    "C_B07    B_07   M_02    80      0.013      0      0       0         0",
    "C_B08    B_08   M_02    80      0.013      0      0       0         0",
    "C_B09    B_09   M_03    80      0.013      0      0       0         0",
    "C_B10    B_10   M_03    80      0.013      0      0       0         0",
    "C_B11    B_11   M_05    80      0.013      0      0       0         0",
    "C_B12    B_12   M_05    80      0.013      0      0       0         0",
    "C_B13    B_13   M_04    80      0.013      0      0       0         0",
    "C_B14    B_14   M_04    80      0.013      0      0       0         0",
]
lines.append("")

# ---- XSECTIONS ----
lines.append("[XSECTIONS]")
for c in ["C_M01","C_M02","C_M03","C_M04","C_M05","C_out"]:
    size = "0.8" if c == "C_out" else "0.6"
    lines.append(f"{c}        CIRCULAR    {size}    0      0      0      1")
for c in ["C_B07","C_B08","C_B09","C_B10","C_B11","C_B12","C_B13","C_B14"]:
    lines.append(f"{c}        CIRCULAR    0.4    0      0      0      1")
lines.append("")

# ---- COORDINATES (critical for surface coupling) ----
lines.append("[COORDINATES]")
for i, y in enumerate(main_nodes_y):
    lines.append(f"M_{i+1:02d}          {main_x:.1f}       {y:.1f}")
for (x, y), idx in zip(all_branches, range(7, 15)):
    lines.append(f"B_{idx:02d}          {x:.1f}       {y:.1f}")
lines.append(f"Outfall_01      {main_x:.1f}       10.0")
lines.append("")

# ---- Required sections ----
lines += [
    "[TAGS]", "", "[MAP]", "",
    "[REPORT]",
    "INPUT      NO",
    "CONTROLS   NO",
    "NODES ALL",
    "LINKS ALL",
]

with open(OUTPUT, 'w') as f:
    f.write('\n'.join(lines))

print(f"SWMM input: {OUTPUT}")
print(f"  Main manholes: 6, Branch manholes: 8, Outfall: 1")
print(f"  Conduits: 14")
