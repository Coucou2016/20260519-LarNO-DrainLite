#!/usr/bin/env python3
"""Run matched Itzi controls and native Itzi-SWMM coupling.

The three primary scenarios isolate the effects that were confounded in the
earlier dataset:

  A: n=0.015 surface only;
  B: n=0.012 around the same inlet nodes, without SWMM;
  C: scenario B plus native bidirectional Itzi-SWMM coupling.

The drainage target is C-B. The roughness-control effect is B-A.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import shutil
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CASE = ROOT / "external_models" / "20260518-itzi-flood" / "test_cases" / "shenzhen_region1"
MODEL_PATH = CASE / "run_full_domain_coupled.py"
DEFAULT_NETWORK = (
    ROOT / "extended_study" / "output" / "reviewer_major_revision_v3"
    / "network" / "swmm_bidirectional_normal_step05.inp"
)
DEFAULT_OUTPUT = (
    ROOT / "extended_study" / "output" / "reviewer_major_revision_v3"
    / "matched_physics"
)


def load_model():
    spec = importlib.util.spec_from_file_location("formal_itzi_swmm", MODEL_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def serializable(value):
    if isinstance(value, dict):
        return {key: serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def parse_swmm_report(path: Path) -> dict:
    if not path.exists():
        return {"report_found": False}
    text = path.read_text(encoding="utf-8", errors="replace")
    continuity = [float(item) for item in re.findall(r"Continuity Error \(%\) \.{2,}\s+([-+0-9.]+)", text)]

    def number(pattern: str):
        match = re.search(pattern, text, flags=re.IGNORECASE)
        return float(match.group(1)) if match else None

    return {
        "report_found": True,
        "runoff_continuity_error_pct": continuity[0] if len(continuity) > 1 else None,
        "flow_routing_continuity_error_pct": continuity[-1] if continuity else None,
        "minimum_routing_step_s": number(r"Minimum Time Step\s*:\s*([0-9.]+)"),
        "average_routing_step_s": number(r"Average Time Step\s*:\s*([0-9.]+)"),
        "maximum_routing_step_s": number(r"Maximum Time Step\s*:\s*([0-9.]+)"),
        "average_iterations_per_step": number(r"Average Iterations per Step\s*:\s*([0-9.]+)"),
        "steps_not_converging_pct": number(r"% of Steps Not Converging\s*:\s*([0-9.]+)"),
    }


def run(args: argparse.Namespace) -> None:
    model = load_model()
    network = args.network.resolve()
    if not network.exists():
        raise FileNotFoundError(network)

    dem, bldg, slc, full_shape = model.load_domain(args.domain)
    h_full, w_full = full_shape
    row_off = slc[0].start or 0
    col_off = slc[1].start or 0
    geo = dict(H_full=h_full, W_full=w_full, row_off=row_off, col_off=col_off)
    args.output.mkdir(parents=True, exist_ok=True)

    for event in args.events:
        event_dir = args.output / event
        event_dir.mkdir(parents=True, exist_ok=True)
        source_dir = Path(model.FLOOD_DIR) / event
        rainfall_full = np.load(source_dir / "rainfall.npy").astype(np.float32)
        mike_full = np.load(source_dir / "h.npy").astype(np.float32)
        if args.frames is not None:
            rainfall_full = rainfall_full[: args.frames]
            mike_full = mike_full[: args.frames]
        rainfall = rainfall_full[(slice(None),) + slc]
        mike = mike_full[(slice(None),) + slc]

        common = dict(
            dem=dem.copy(),
            bldg=bldg.copy(),
            rainfall_3d=rainfall,
            save_timeseries=True,
            infiltration_mmh=args.infiltration_mmh,
            building_rainfall_mode=args.building_rainfall_mode,
            coupling_relaxation=args.coupling_relaxation,
            coupling_damping=args.coupling_damping,
            **geo,
        )

        records = {}
        series = {}
        if "A" in args.scenarios:
            print(f"\n[{event}] A: surface n=0.015", flush=True)
            records["A"], _, series["A"] = model.run_simulation(
                label=f"{event} A", swmm_inp=None, inlet_layout_inp=None,
                inlet_manning=None, **common,
            )
        if "B" in args.scenarios:
            print(f"\n[{event}] B: matched inlet roughness, no SWMM", flush=True)
            records["B"], _, series["B"] = model.run_simulation(
                label=f"{event} B", swmm_inp=None, inlet_layout_inp=str(network),
                inlet_manning=args.inlet_manning, **common,
            )
        if "C" in args.scenarios:
            print(f"\n[{event}] C: matched roughness plus native Itzi-SWMM", flush=True)
            records["C"], _, series["C"] = model.run_simulation(
                label=f"{event} C", swmm_inp=str(network), inlet_layout_inp=str(network),
                inlet_manning=args.inlet_manning, **common,
            )

        np.save(event_dir / "rainfall.npy", rainfall)
        np.save(event_dir / "h_mike_ref.npy", mike)
        output_names = {
            "A": "h_A_surface_n015.npy",
            "B": "h_B_surface_inlet_n012.npy",
            "C": "h_C_itzi_swmm.npy",
        }
        for scenario, values in series.items():
            np.save(event_dir / output_names[scenario], values)
        swmm_states = records.get("C", {}).pop("swmm_states", None)
        if swmm_states is not None:
            np.savez_compressed(event_dir / "swmm_dynamic_states.npz", **swmm_states)
        if "A" in series and "B" in series:
            np.save(event_dir / "residual_roughness_B_minus_A.npy", series["B"] - series["A"])
        if "B" in series and "C" in series:
            np.save(event_dir / "residual_drainage_C_minus_B.npy", series["C"] - series["B"])

        report_source = network.with_suffix(".rpt")
        report_target = event_dir / "swmm_C.rpt"
        if "C" in series and report_source.exists():
            shutil.copy2(report_source, report_target)
        swmm_quality = parse_swmm_report(report_target) if "C" in series else {"report_found": False}
        metadata = {
            "event": event,
            "domain": args.domain,
            "shape": list(next(iter(series.values())).shape),
            "cell_size_m": model.CELL,
            "infiltration_mmh": args.infiltration_mmh,
            "building_rainfall_mode": args.building_rainfall_mode,
            "coupling_relaxation": args.coupling_relaxation,
            "coupling_damping": args.coupling_damping,
            "surface_manning": 0.015,
            "matched_inlet_manning": args.inlet_manning,
            "network": str(network),
            "scenario_definition": {
                "A": "surface only; n=0.015",
                "B": "surface only; n=0.012 in 3x3 inlet neighbourhoods",
                "C": "B plus native bidirectional Itzi-SWMM coupling",
                "roughness_effect": "B-A",
                "drainage_effect": "C-B",
            },
            "records": records,
            "swmm_quality": swmm_quality,
        }
        (event_dir / "metadata.json").write_text(
            json.dumps(serializable(metadata), indent=2), encoding="utf-8"
        )
        print(json.dumps(swmm_quality, indent=2), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=["sub", "full"], default="sub")
    parser.add_argument("--events", nargs="+", default=["event68"])
    parser.add_argument("--scenarios", nargs="+", choices=["A", "B", "C"], default=["A", "B", "C"])
    parser.add_argument("--network", type=Path, default=DEFAULT_NETWORK)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--infiltration-mmh", type=float, default=1.0)
    parser.add_argument("--inlet-manning", type=float, default=0.012)
    parser.add_argument(
        "--building-rainfall-mode",
        choices=["global_redistribute", "nearest_redistribute", "exclude"],
        default="global_redistribute",
    )
    parser.add_argument("--coupling-relaxation", type=float, default=0.8)
    parser.add_argument("--coupling-damping", type=float, default=0.5)
    parser.add_argument("--frames", type=int, help="Optional diagnostic truncation; omit for the formal 72-frame run")
    args = parser.parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
