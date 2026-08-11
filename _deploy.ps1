# ============================================================
# P302 AI - Build, clean, and deploy GUI package
# Usage: powershell -ExecutionPolicy Bypass -File _deploy.ps1
# ============================================================
$ErrorActionPreference = "Continue"
$PROJECT = Split-Path -Parent $MyInvocation.MyCommand.Path
$DOWNLOADS = Join-Path $env:USERPROFILE "Downloads"
$RELEASE_ROOT = Join-Path $DOWNLOADS "Microgrid_AI"
$DIST    = "$PROJECT\dist\P302_AI_GUI"
$DST     = Join-Path $RELEASE_ROOT "release_v22_flow_power_limited"
$PY      = "$PROJECT\.venv_deploy_cpu\Scripts\python.exe"
$EXPERIMENT = "v22_flow_power_limited_gpu300"
$MODEL_FILE = "best_sac_model.pth"
if (-not (Test-Path $PY)) { $PY = "py" }
$VENDOR_DIR = Join-Path $DOWNLOADS "P302_AI_V4.0\P302_AI_V4.0"
$VENDOR_EXE = Join-Path $VENDOR_DIR "P302_AI.exe"
$LOG_DIR = Join-Path $DST "results\deployment"

# --- Step 1: PyInstaller build --------------------------------
Write-Host "`n[1/4] Building with PyInstaller..." -ForegroundColor Cyan
Set-Location $PROJECT
if (Test-Path $DIST) { Remove-Item $DIST -Recurse -Force -EA SilentlyContinue }
& $PY -m PyInstaller --clean --noconfirm packaging\build_release.spec
if ($LASTEXITCODE -ne 0) {
    Write-Host "BUILD FAILED (PyInstaller exit code $LASTEXITCODE)" -ForegroundColor Red
    exit 1
}
if (-not (Test-Path "$DIST\P302_AI_GUI.exe")) {
    Write-Host "BUILD FAILED" -ForegroundColor Red; exit 1
}

# --- Step 2: Remove CUDA and unused large files ----------------
Write-Host "`n[2/4] Cleaning CUDA & unnecessary files..." -ForegroundColor Cyan
$INT = "$DIST\_internal"

# 2a. CUDA/NVIDIA DLLs are not needed for CPU-only deployment.
$cudaPatterns = @('*cuda*.dll', '*cudnn*.dll', '*cublas*.dll', '*cusparse*.dll', '*cusolver*.dll', '*curand*.dll', '*cufft*.dll', '*cupti*.dll', '*nvrtc*.dll', '*nvjitlink*.dll', '*nvToolsExt*.dll', 'caffe2_nvrtc.dll')
$removed = 0
foreach ($pat in $cudaPatterns) {
    Get-ChildItem "$INT\torch\lib" -Filter $pat -File -EA SilentlyContinue | ForEach-Object {
        Remove-Item $_.FullName -Force -EA SilentlyContinue
        $removed++
    }
}
Write-Host "  Removed $removed CUDA/NVIDIA DLLs from torch\lib"

# 2b. Optional torch submodules can be removed here if needed.
$torchRemoveDirs = @()
foreach ($d in $torchRemoveDirs) {
    $p = "$INT\torch\$d"
    if (Test-Path $p) { Remove-Item $p -Recurse -Force -EA SilentlyContinue; Write-Host "  Removed torch\$d" }
}

# 2c. Solver libraries are not used by deployment.
$topRemoveDirs = @('scs.libs','scs','clarabel')
foreach ($d in $topRemoveDirs) {
    $p = "$INT\$d"
    if (Test-Path $p) { Remove-Item $p -Recurse -Force -EA SilentlyContinue; Write-Host "  Removed $d" }
}

# 2d. Legacy packages that sometimes get pulled into builds.
$legacyPatterns = @('tensorflow*','_pywrap*','xla*','grpc*','boto*','s3transfer*','libcrypto*','libssl*','mkl_pgi*','mkl_tbb*','mkl_sequential*','mkl_vml_avx512*','mkl_vml_mc3*','mkl_avx512*','mkl_mc3*','mkl_avx.2*','mkl_avx2.2*','mkl_mc.2*','mkl_vml_avx.2*','mkl_vml_mc.2*','mkl_vml_avx2.2*','mkl_scalapack*','mkl_blacs*','mkl_cdft*')
foreach ($pat in $legacyPatterns) {
    Get-ChildItem $INT -Recurse -Filter $pat -File -EA SilentlyContinue |
        Remove-Item -Force -EA SilentlyContinue
}
$legacyDirs = @('tensorflow','tensorboard','google','keras','botocore')
foreach ($d in $legacyDirs) {
    $p = "$INT\$d"
    if (Test-Path $p) { Remove-Item $p -Recurse -Force -EA SilentlyContinue; Write-Host "  Removed $d" }
}

$cleanSize = (Get-ChildItem $INT -Recurse -File | Measure-Object -Property Length -Sum).Sum / 1MB
Write-Host "  _internal size after cleanup: $([math]::Round($cleanSize)) MB"

# --- Step 3: Copy to release directory -------------------------
Write-Host "`n[3/4] Copying to release..." -ForegroundColor Cyan
if (Test-Path $DST) {
        # If the old release is locked, rename it and continue.
    try {
        Remove-Item $DST -Recurse -Force -EA Stop
    } catch {
        $bak = "${DST}_bak_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
        Write-Host "  Old release locked, renaming to $bak" -ForegroundColor Yellow
        Rename-Item $DST $bak -Force -EA SilentlyContinue
        if (Test-Path $DST) {
            Write-Host "  Cannot rename either. Copying to release_v16 instead." -ForegroundColor Yellow
            $DST = Join-Path $RELEASE_ROOT "release_v16"
            if (Test-Path $DST) { Remove-Item $DST -Recurse -Force -EA SilentlyContinue }
        }
    }
}
Copy-Item -Recurse -Force $DIST $DST
Copy-Item "$PROJECT\load_pattern.txt" "$DST\load_pattern.txt" -Force
$SOH_MODEL_DIR = "$DST\soh_models"
New-Item -ItemType Directory -Force -Path $SOH_MODEL_DIR | Out-Null
if (Test-Path "$PROJECT\soh_models") {
    Copy-Item "$PROJECT\soh_models\*" $SOH_MODEL_DIR -Recurse -Force -EA SilentlyContinue
}
$SOH_MODEL_SRC = "$PROJECT\core\soh_predictor\model"
if (Test-Path $SOH_MODEL_SRC) {
    Copy-Item "$SOH_MODEL_SRC\*.pth" $SOH_MODEL_DIR -Force -EA SilentlyContinue
    Copy-Item "$SOH_MODEL_SRC\*.pkl" $SOH_MODEL_DIR -Force -EA SilentlyContinue
    Copy-Item "$SOH_MODEL_SRC\*.npz" $SOH_MODEL_DIR -Force -EA SilentlyContinue
}
$SOH_MODEL_PATH_DEFAULT = "$DST\soh_models"
$configGui = @{
    vendor_dir = $VENDOR_DIR
    vendor_exe = $VENDOR_EXE
    model_path = "$DST\_internal\models\$MODEL_FILE"
    soh_model_path = $SOH_MODEL_PATH_DEFAULT
    experiment_name = $EXPERIMENT
    initial_soc = 20.0
    load_count = 4
    load_power_per_unit_w = 0.1
    log_dir = $LOG_DIR
    device = "cpu"
    poll_sec = 10.0
    window_min = 15
    current_mode = "hybrid"
    manual_scenario = 3
    dry_run_enabled = $false
    pv_surplus_charge_only = $true
    cutoff_soc_fallback_enabled = $true
    cutoff_soc_fallback_percent = 20.0
    soh_prediction_enabled = $false
    soh_use_for_capacity = $false
    soh_health_protection_enabled = $false
    soh_low_voltage_v = 4.2
    soh_low_voltage_samples = 3
    soh_recover_v = 5.0
    soh_recovery_samples = 12
    use_watchdog = $true
    watchdog_interval_sec = 60
} | ConvertTo-Json
Set-Content -Path "$DST\config_gui.json" -Value $configGui -Encoding UTF8

# --- Step 4: Verify package contents --------------------------
Write-Host "`n[4/4] Verifying..." -ForegroundColor Cyan
$checks = @(
    @{Name='P302_AI_GUI.exe';      Path="$DST\P302_AI_GUI.exe"},
    @{Name='config_gui.json';      Path="$DST\config_gui.json"},
    @{Name='soh_models';           Path="$DST\soh_models"},
    @{Name='load_pattern.txt(root)';Path="$DST\load_pattern.txt"},
    @{Name='load_pattern.txt(int)'; Path="$DST\_internal\load_pattern.txt"},
    @{Name=$MODEL_FILE;             Path="$DST\_internal\models\$MODEL_FILE"},
    @{Name='experiment_config.yaml';Path="$DST\_internal\configs\experiment_config.yaml"},
    @{Name='run_deployment.py';     Path="$DST\_internal\control\run_deployment.py"},
    @{Name='microgrid_env.py';      Path="$DST\_internal\core\microgrid_env.py"}
)
$allOK = $true
foreach ($c in $checks) {
    $ok = Test-Path $c.Path
    $mark = if ($ok) { "[OK]" } else { "[MISSING]"; $allOK = $false }
    Write-Host "  $mark $($c.Name)"
}

# Verify the known deployment fixes are present in the packaged code.
$lcCount = (Select-String -Path "$DST\_internal\control\run_deployment.py" -Pattern "load_count=get_load_groups" -EA SilentlyContinue).Count
Write-Host "  load_count=get_load_groups() calls: $lcCount (should be 5)"
if ($lcCount -lt 5) { Write-Host "  WARNING: load_count fix incomplete!" -ForegroundColor Red; $allOK = $false }

$pvRatioFix = (Select-String -Path "$DST\_internal\control\run_deployment.py" -Pattern "PV_SUFFICIENT_RATIO_THRESHOLD = 0.8" -EA SilentlyContinue).Count
Write-Host "  pv ratio threshold markers: $pvRatioFix (should be >= 1)"
if ($pvRatioFix -lt 1) { Write-Host "  WARNING: deployment PV ratio logic not found!" -ForegroundColor Red; $allOK = $false }

$hybridModeHit = (Select-String -Path "$DST\_internal\control\run_deployment.py" -Pattern 'default="hybrid"' -EA SilentlyContinue).Count
Write-Host "  hybrid current-mode default markers: $hybridModeHit (should be >= 1)"
if ($hybridModeHit -lt 1) { Write-Host "  WARNING: deployment current-mode is not hybrid by default!" -ForegroundColor Red; $allOK = $false }

$sohHealthHit = (Select-String -Path "$DST\_internal\control\run_deployment.py" -Pattern "soh-health-protection" -EA SilentlyContinue).Count
Write-Host "  SoH health protection markers: $sohHealthHit (should be >= 1)"
if ($sohHealthHit -lt 1) { Write-Host "  WARNING: SoH health protection not found!" -ForegroundColor Red; $allOK = $false }

$sohModelFiles = (Get-ChildItem $SOH_MODEL_DIR -Recurse -File -Include *.pth,*.pkl,*.npz -EA SilentlyContinue).Count
Write-Host "  SoH model files: $sohModelFiles (should be >= 3)"
if ($sohModelFiles -lt 3) { Write-Host "  WARNING: SoH model files incomplete!" -ForegroundColor Red; $allOK = $false }

$pvSurplusGuardHit = (Select-String -Path "$DST\_internal\control\run_deployment.py" -Pattern "pv-surplus-charge-only" -EA SilentlyContinue).Count
Write-Host "  PV surplus charge guard markers: $pvSurplusGuardHit (should be >= 1)"
if ($pvSurplusGuardHit -lt 1) { Write-Host "  WARNING: PV surplus charging guard not found!" -ForegroundColor Red; $allOK = $false }

$flowPowerGuardHit = (Select-String -Path "$DST\_internal\control\run_deployment.py" -Pattern "flow-power-guard" -EA SilentlyContinue).Count
Write-Host "  Flow power guard markers: $flowPowerGuardHit (should be >= 1)"
if ($flowPowerGuardHit -lt 1) { Write-Host "  WARNING: Flow power guard not found!" -ForegroundColor Red; $allOK = $false }

$loadOverWarnHit = (Select-String -Path "$DST\_internal\control\run_deployment.py" -Pattern "warn_load_over_discharge_limit" -EA SilentlyContinue).Count
Write-Host "  Load-over-discharge warning markers: $loadOverWarnHit (should be >= 1; warning only)"
if ($loadOverWarnHit -lt 1) { Write-Host "  WARNING: load-over-discharge warning marker missing!" -ForegroundColor Red; $allOK = $false }

$oldLoadBlockHit = (Select-String -Path "$DST\_internal\control\run_deployment.py" -Pattern "guard_block_load_over_discharge_limit" -EA SilentlyContinue).Count
Write-Host "  Old load-over-discharge block markers: $oldLoadBlockHit (should be 0)"
if ($oldLoadBlockHit -ne 0) { Write-Host "  WARNING: old load-over-discharge block marker still present!" -ForegroundColor Red; $allOK = $false }

$restFlowHit = (Select-String -Path "$DST\_internal\control\run_deployment.py" -Pattern "FLOW_REST_PCT = 0.0" -EA SilentlyContinue).Count
Write-Host "  AI rest flow markers: $restFlowHit (should be >= 1 for 0% rest)"
if ($restFlowHit -lt 1) { Write-Host "  WARNING: AI rest flow is not 0%!" -ForegroundColor Red; $allOK = $false }

$preMeasureFlowHit = (Select-String -Path "$DST\_internal\control\run_deployment.py" -Pattern "FLOW_PRE_MEASURE_PCT = 50.0" -EA SilentlyContinue).Count
Write-Host "  AI pre-measure flow markers: $preMeasureFlowHit (should be >= 1 for 50% pre-measure)"
if ($preMeasureFlowHit -lt 1) { Write-Host "  WARNING: AI pre-measure flow is not 50%!" -ForegroundColor Red; $allOK = $false }

$recoverySecondsHit = (Select-String -Path "$DST\_internal\control\run_deployment.py" -Pattern "VOLTAGE_RECOVERY_SECONDS" -EA SilentlyContinue).Count
Write-Host "  AI voltage recovery markers: $recoverySecondsHit (should be >= 1)"
if ($recoverySecondsHit -lt 1) { Write-Host "  WARNING: AI voltage recovery wait marker missing!" -ForegroundColor Red; $allOK = $false }

$standbyIdHit = (Select-String -Path "$DST\_internal\control\run_deployment.py" -Pattern 'return "00"' -EA SilentlyContinue).Count
Write-Host "  AI zero-power PP=00 markers: $standbyIdHit (should be 0; rest/pre-measure must keep physical PP=01)"
if ($standbyIdHit -ne 0) { Write-Host "  WARNING: AI zero-power command still changes PP to 00!" -ForegroundColor Red; $allOK = $false }

$standbyModeMarkerHit = (Select-String -Path "$DST\_internal\control\run_deployment.py" -Pattern "STANDBY_SITUATION_CODE = 3|PRE_MEASURE_SITUATION_CODE = 3" -EA SilentlyContinue).Count
Write-Host "  AI mode 3 rest/pre-measure markers: $standbyModeMarkerHit (should be >= 2)"
if ($standbyModeMarkerHit -lt 2) { Write-Host "  WARNING: AI rest/pre-measure modes are not explicitly mode 3!" -ForegroundColor Red; $allOK = $false }

$aiMode4StandbyHit = (Select-String -Path "$DST\_internal\control\run_deployment.py" -Pattern "write_command_simple\(.*,\s*4,|return 4\s*#.*standby|last_sit_code = 4" -EA SilentlyContinue).Count
Write-Host "  AI mode4 standby misuse markers: $aiMode4StandbyHit (should be 0)"
if ($aiMode4StandbyHit -ne 0) { Write-Host "  WARNING: AI standby path may still use mode 4, which shuts off motor/battery!" -ForegroundColor Red; $allOK = $false }

$manualRestFlowHit = (Select-String -Path "$DST\_internal\control\solar_test_collect.py" -Pattern "FLOW_REST_PCT = 0" -EA SilentlyContinue).Count
Write-Host "  manual/solar rest flow markers: $manualRestFlowHit (should be >= 1)"
if ($manualRestFlowHit -lt 1) { Write-Host "  WARNING: manual/solar rest flow is not 0%!" -ForegroundColor Red; $allOK = $false }

$manualPreMeasureFlowHit = (Select-String -Path "$DST\_internal\control\solar_test_collect.py" -Pattern "FLOW_PRE_MEASURE_PCT = 50" -EA SilentlyContinue).Count
Write-Host "  manual/solar pre-measure flow markers: $manualPreMeasureFlowHit (should be >= 1)"
if ($manualPreMeasureFlowHit -lt 1) { Write-Host "  WARNING: manual/solar pre-measure flow is not 50%!" -ForegroundColor Red; $allOK = $false }

$manualStandbyIdHit = (Select-String -Path "$DST\_internal\control\solar_test_collect.py" -Pattern 'return "00"' -EA SilentlyContinue).Count
Write-Host "  manual/solar zero-power PP=00 markers: $manualStandbyIdHit (should be 0; rest/pre-measure must keep physical PP=01)"
if ($manualStandbyIdHit -ne 0) { Write-Host "  WARNING: manual/solar zero-power command still changes PP to 00!" -ForegroundColor Red; $allOK = $false }

$manualStandbyModeHit = (Select-String -Path "$DST\_internal\control\solar_test_collect.py" -Pattern "DEFAULT_STANDBY_SCENARIO = 3" -EA SilentlyContinue).Count
Write-Host "  manual/solar rest/pre-measure mode markers: $manualStandbyModeHit (should be >= 1; mode 3)"
if ($manualStandbyModeHit -lt 1) { Write-Host "  WARNING: manual/solar rest/pre-measure mode is not mode 3!" -ForegroundColor Red; $allOK = $false }

$manualMode4DefaultHit = (Select-String -Path "$DST\_internal\control\solar_test_collect.py" -Pattern "parser\.add_argument\('--scenario'.*default=4|scenario=4" -EA SilentlyContinue).Count
Write-Host "  manual/solar mode4 standby defaults: $manualMode4DefaultHit (should be 0)"
if ($manualMode4DefaultHit -ne 0) { Write-Host "  WARNING: manual/solar standby default may still use mode 4!" -ForegroundColor Red; $allOK = $false }

$guiManualScenarioHit = (Select-String -Path "$DST\config_gui.json" -Pattern '"manual_scenario":\s*3' -EA SilentlyContinue).Count
Write-Host "  GUI manual scenario default markers: $guiManualScenarioHit (should be >= 1)"
if ($guiManualScenarioHit -lt 1) { Write-Host "  WARNING: GUI manual/standby default is not mode 3!" -ForegroundColor Red; $allOK = $false }

$flowActionHit = (Select-String -Path "$DST\_internal\configs\experiment_config.yaml" -Pattern "use_flow_rate_action: true" -EA SilentlyContinue).Count
Write-Host "  flow action config markers: $flowActionHit (should be >= 1)"
if ($flowActionHit -lt 1) { Write-Host "  WARNING: packaged experiment_config.yaml is not flow-rate enabled!" -ForegroundColor Red; $allOK = $false }

$flowLimitHit = (Select-String -Path "$DST\_internal\configs\experiment_config.yaml" -Pattern "flow_limits_available_power: true" -EA SilentlyContinue).Count
Write-Host "  flow power limit config markers: $flowLimitHit (should be >= 1)"
if ($flowLimitHit -lt 1) { Write-Host "  WARNING: packaged experiment_config.yaml is missing flow power limit!" -ForegroundColor Red; $allOK = $false }

$loadScaleHit = (Select-String -Path "$DST\_internal\configs\experiment_config.yaml" -Pattern "deployment_group_power_kw: 0.0001" -EA SilentlyContinue).Count
Write-Host "  deployment load scale markers: $loadScaleHit (should be >= 1 for current 0.1W/group hardware)"
if ($loadScaleHit -lt 1) {
    Write-Host "  WARNING: packaged model was not trained with current 0.1W/group load scale!" -ForegroundColor Red
    $allOK = $false
}

$totalSize = (Get-ChildItem $DST -Recurse -File | Measure-Object -Property Length -Sum).Sum / 1MB
Write-Host "`n  Release: $DST"
Write-Host "  Experiment: $EXPERIMENT"
Write-Host "  Total size: $([math]::Round($totalSize)) MB"
if ($totalSize -gt 600) { Write-Host "  WARNING: Size > 600MB, check for CUDA leftovers!" -ForegroundColor Yellow }
if ($allOK) { Write-Host "`n  BUILD SUCCESS" -ForegroundColor Green }
else { Write-Host "`n  BUILD HAS WARNINGS" -ForegroundColor Yellow }
