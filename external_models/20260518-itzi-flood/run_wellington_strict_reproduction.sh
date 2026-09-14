#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

DEM=""
LINZ_API_KEY_ARG="${LINZ_API_KEY:-}"
SKIP_BUILDINGS=0
KEEP_GOING=1
RUN_GRASS=0
OFFICIAL_GATE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dem)
      DEM="$2"
      shift 2
      ;;
    --linz-api-key)
      LINZ_API_KEY_ARG="$2"
      shift 2
      ;;
    --skip-buildings)
      SKIP_BUILDINGS=1
      shift
      ;;
    --keep-going)
      KEEP_GOING=1
      shift
      ;;
    --stop-on-failure)
      KEEP_GOING=0
      shift
      ;;
    --run-grass)
      RUN_GRASS=1
      shift
      ;;
    --official-gate)
      OFFICIAL_GATE=1
      shift
      ;;
    -h|--help)
      cat <<'USAGE'
Usage:
  ./run_wellington_strict_reproduction.sh --dem /path/to/wellington_dem.tif [options]

Options:
  --linz-api-key KEY     LINZ LDS API key. Prefer LINZ_API_KEY env var.
  --skip-buildings      Skip LINZ building WFS download.
  --keep-going          Continue after failed steps and write diagnostics (default).
  --stop-on-failure     Stop at first failed step.
  --run-grass           Run benchmark_data/run_wellington_in_grass.sh after preparation.
  --official-gate       Pass --official-gate to the GRASS runner.
USAGE
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "$DEM" ]]; then
  echo "--dem is required" >&2
  exit 2
fi

if [[ ! -f "$DEM" ]]; then
  echo "DEM file not found: $DEM" >&2
  exit 2
fi

REPORTS="$ROOT/wellington_real/reports"
LOGS="$REPORTS/external_run_logs"
mkdir -p "$LOGS"
STAMP="$(date -u +%Y%m%d_%H%M%S)"
LOG="$LOGS/wellington_strict_reproduction_${STAMP}.log"

log() {
  local message="$1"
  local line
  line="[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $message"
  echo "$line" | tee -a "$LOG"
}

run_step() {
  local name="$1"
  shift
  log "START $name"
  log "CMD $*"
  "$@" 2>&1 | tee -a "$LOG"
  local code=${PIPESTATUS[0]}
  log "END $name exit_code=$code"
  if [[ $code -ne 0 && $KEEP_GOING -eq 0 ]]; then
    exit "$code"
  fi
  return "$code"
}

if [[ -n "$LINZ_API_KEY_ARG" ]]; then
  export LINZ_API_KEY="$LINZ_API_KEY_ARG"
  log "LINZ_API_KEY is set for child processes; value will not be written to command lines."
elif [[ $SKIP_BUILDINGS -eq 0 ]]; then
  log "LINZ_API_KEY is missing. Building download will fail unless --skip-buildings is used."
fi

DRIVER_ARGS=(benchmark_data/wellington_strict_reproduction_driver.py --dem "$DEM")
if [[ $KEEP_GOING -eq 1 ]]; then
  DRIVER_ARGS+=(--keep-going)
fi
if [[ $SKIP_BUILDINGS -eq 1 ]]; then
  DRIVER_ARGS+=(--skip-buildings)
fi

run_step static_script_audit python benchmark_data/static_script_audit.py
run_step runtime_dependency_audit python benchmark_data/wellington_runtime_dependency_audit.py
run_step strict_preparation_driver python "${DRIVER_ARGS[@]}"

if [[ $RUN_GRASS -eq 1 ]]; then
  GRASS_ARGS=(benchmark_data/run_wellington_in_grass.sh)
  if [[ $OFFICIAL_GATE -eq 1 ]]; then
    GRASS_ARGS+=(--official-gate)
  fi
  run_step grass_itzi_run bash "${GRASS_ARGS[@]}"
fi

run_step model_run_artifact_audit python benchmark_data/wellington_model_run_artifact_audit.py
run_step animation_sequence_audit python benchmark_data/wellington_animation_sequence_audit.py
run_step result_package_audit python benchmark_data/wellington_result_package_audit.py
run_step objective_completion_audit python benchmark_data/wellington_objective_completion_audit.py
run_step report_summary python benchmark_data/wellington_report_summary.py

log "External run finished. Log: $LOG"
