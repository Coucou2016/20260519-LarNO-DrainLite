param(
    [string]$Inp = "wellington_real/model/wellington_drainage.inp",
    [string]$BuildReport = "wellington_real/reports/build_swmm_from_gis.json",
    [string]$PumpAudit = "wellington_real/reports/swmm_pump_control_audit.json",
    [string]$Out = "wellington_real/reports/swmm_hydraulic_parameter_audit.json",
    [switch]$Strict
)

$ErrorActionPreference = "Stop"

$argsList = @(
    "benchmark_data/wellington_swmm_hydraulic_parameter_audit.py",
    "--inp", $Inp,
    "--build-report", $BuildReport,
    "--pump-audit", $PumpAudit,
    "--out", $Out
)

if ($Strict) {
    $argsList += "--strict"
}

python @argsList
exit $LASTEXITCODE
