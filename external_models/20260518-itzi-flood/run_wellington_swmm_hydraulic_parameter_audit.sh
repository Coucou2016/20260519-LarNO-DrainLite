#!/usr/bin/env bash
set -euo pipefail

inp="wellington_real/model/wellington_drainage.inp"
build_report="wellington_real/reports/build_swmm_from_gis.json"
pump_audit="wellington_real/reports/swmm_pump_control_audit.json"
out="wellington_real/reports/swmm_hydraulic_parameter_audit.json"
strict=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --inp)
      inp="$2"
      shift 2
      ;;
    --build-report)
      build_report="$2"
      shift 2
      ;;
    --pump-audit)
      pump_audit="$2"
      shift 2
      ;;
    --out)
      out="$2"
      shift 2
      ;;
    --strict)
      strict=1
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 64
      ;;
  esac
done

cmd=(
  python3
  benchmark_data/wellington_swmm_hydraulic_parameter_audit.py
  --inp "$inp"
  --build-report "$build_report"
  --pump-audit "$pump_audit"
  --out "$out"
)

if [[ "$strict" -eq 1 ]]; then
  cmd+=(--strict)
fi

"${cmd[@]}"
