param(
    [Parameter(Mandatory = $true)]
    [string]$Dem,

    [string]$LinzApiKey = $env:LINZ_API_KEY,

    [switch]$SkipBuildings,

    [switch]$KeepGoing = $true,

    [switch]$RunGrass,

    [switch]$OfficialGate
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

$Reports = Join-Path $Root "wellington_real\reports"
$Logs = Join-Path $Reports "external_run_logs"
New-Item -ItemType Directory -Force -Path $Logs | Out-Null

$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$LogFile = Join-Path $Logs "wellington_strict_reproduction_$Timestamp.log"

function Write-Log {
    param([string]$Message)
    $line = "[$(Get-Date -Format o)] $Message"
    Write-Host $line
    Add-Content -Path $LogFile -Value $line
}

function Run-Step {
    param(
        [string]$Name,
        [string[]]$Command
    )
    Write-Log "START $Name"
    Write-Log ("CMD " + ($Command -join " "))
    if ($Command.Length -gt 1) {
        $args = $Command[1..($Command.Length - 1)]
    }
    else {
        $args = @()
    }
    & $Command[0] @args 2>&1 | Tee-Object -FilePath $LogFile -Append
    $code = $LASTEXITCODE
    Write-Log "END $Name exit_code=$code"
    if ($code -ne 0 -and -not $KeepGoing) {
        throw "$Name failed with exit code $code"
    }
    return $code
}

if (-not (Test-Path -LiteralPath $Dem)) {
    throw "DEM file not found: $Dem"
}

if ($LinzApiKey) {
    $env:LINZ_API_KEY = $LinzApiKey
    Write-Log "LINZ_API_KEY is set for child processes; value will not be written to commands."
}
elseif (-not $SkipBuildings) {
    Write-Log "LINZ_API_KEY is missing. Use -SkipBuildings or set LINZ_API_KEY for building outlines."
}

$driverArgs = @(
    "benchmark_data\wellington_strict_reproduction_driver.py",
    "--dem", $Dem
)
if ($KeepGoing) {
    $driverArgs += "--keep-going"
}
if ($SkipBuildings) {
    $driverArgs += "--skip-buildings"
}

$driverCommand = @("python") + $driverArgs
Run-Step -Name "static_script_audit" -Command @("python", "benchmark_data\static_script_audit.py") | Out-Null
Run-Step -Name "runtime_dependency_audit" -Command @("python", "benchmark_data\wellington_runtime_dependency_audit.py") | Out-Null
Run-Step -Name "strict_preparation_driver" -Command $driverCommand | Out-Null

if ($RunGrass) {
    if (Get-Command "bash" -ErrorAction SilentlyContinue) {
        $grassArgs = @("benchmark_data/run_wellington_in_grass.sh")
        if ($OfficialGate) {
            $grassArgs += "--official-gate"
        }
        $grassCommand = @("bash") + $grassArgs
        Run-Step -Name "grass_itzi_run" -Command $grassCommand | Out-Null
    }
    else {
        Write-Log "bash was not found; skipping GRASS/ITZI run."
        if (-not $KeepGoing) {
            throw "bash was not found"
        }
    }
}

Run-Step -Name "model_run_artifact_audit" -Command @("python", "benchmark_data\wellington_model_run_artifact_audit.py") | Out-Null
Run-Step -Name "animation_sequence_audit" -Command @("python", "benchmark_data\wellington_animation_sequence_audit.py") | Out-Null
Run-Step -Name "result_package_audit" -Command @("python", "benchmark_data\wellington_result_package_audit.py") | Out-Null
Run-Step -Name "objective_completion_audit" -Command @("python", "benchmark_data\wellington_objective_completion_audit.py") | Out-Null
Run-Step -Name "report_summary" -Command @("python", "benchmark_data\wellington_report_summary.py") | Out-Null

Write-Log "External run finished. Log: $LogFile"
