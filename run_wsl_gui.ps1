<#
UltraGUI GPU Launcher for Windows PowerShell

Runs the PyQt6 GUI inside WSL Ubuntu through the ROCm/HIP virtual
environment, so YOLO training uses the AMD GPU instead of the Windows
CPU-only Python environment.

Usage:
  powershell -ExecutionPolicy Bypass -File .\run_wsl_gui.ps1

Optional:
  .\run_wsl_gui.ps1 -Distro Ubuntu-24.04 -ProjectPath <path-to-UltraGUI>
  .\run_wsl_gui.ps1 -ModelPath <path-to-yolo-model.pt>
  .\run_wsl_gui.ps1 -SkipGpuCheck
#>

param(
    [string]$Distro = "Ubuntu-24.04",
    [string]$ProjectPath = "",
    [string]$VenvName = ".venv-rocm",
    [string]$ModelPath = "",
    [switch]$SkipGpuCheck,
    [switch]$CheckOnly
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($ProjectPath)) {
    $ProjectPath = Split-Path -Parent $PSCommandPath
}

function Write-Step {
    param([string]$Message)
    Write-Host "[UltraGUI] $Message" -ForegroundColor Cyan
}

function Write-Ok {
    param([string]$Message)
    Write-Host "[OK] $Message" -ForegroundColor Green
}

function Write-Warn {
    param([string]$Message)
    Write-Host "[WARN] $Message" -ForegroundColor Yellow
}

function Fail {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor Red
    exit 1
}

function Convert-ToWslPath {
    param([string]$WindowsPath)

    $resolved = Resolve-Path -LiteralPath $WindowsPath -ErrorAction Stop
    $full = $resolved.Path

    if ($full -notmatch "^[A-Za-z]:\\") {
        Fail "Unsupported Windows path: $full"
    }

    $drive = $full.Substring(0, 1).ToLowerInvariant()
    $rest = $full.Substring(2).Replace("\", "/")
    return "/mnt/$drive$rest"
}

function Quote-Bash {
    param([string]$Value)
    return "'" + $Value.Replace("'", "'\''") + "'"
}

function Invoke-WslScript {
    param(
        [string]$ScriptContent,
        [string]$Label = "WSL command"
    )

    $tempBase = [System.IO.Path]::GetTempFileName()
    $tempScript = "$tempBase.sh"
    Move-Item -LiteralPath $tempBase -Destination $tempScript -Force

    try {
        $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
        $normalized = $ScriptContent -replace "`r`n", "`n"
        [System.IO.File]::WriteAllText($tempScript, $normalized, $utf8NoBom)
        $tempScriptWsl = Convert-ToWslPath $tempScript

        $oldErrorActionPreference = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        try {
            $output = & wsl.exe -d $Distro -- bash $tempScriptWsl 2>&1
            $code = $LASTEXITCODE
        }
        finally {
            $ErrorActionPreference = $oldErrorActionPreference
        }

        if ($output) {
            Write-Host ($output -join "`n")
        }

        return [int]$code
    }
    finally {
        Remove-Item -LiteralPath $tempScript -Force -ErrorAction SilentlyContinue
    }
}

Write-Step "Checking Windows prerequisites"

if (-not (Get-Command wsl.exe -ErrorAction SilentlyContinue)) {
    Fail "wsl.exe was not found. Install/enable WSL first."
}

if (-not (Test-Path -LiteralPath $ProjectPath)) {
    Fail "Project folder not found: $ProjectPath"
}

if ([string]::IsNullOrWhiteSpace($ModelPath)) {
    Write-Warn "No default model path was provided. Choose a valid .pt model inside the app."
}
elseif (-not (Test-Path -LiteralPath $ModelPath)) {
    Write-Warn "Model file not found: $ModelPath"
    Write-Warn "You can still launch the GUI, but choose a valid .pt model inside the app."
}

$distroList = & wsl.exe -l -q
if ($LASTEXITCODE -ne 0) {
    Fail "Could not list WSL distros."
}

if (-not ($distroList | Where-Object { $_.Trim() -eq $Distro })) {
    Fail "WSL distro '$Distro' was not found. Available distros:`n$($distroList -join "`n")"
}

Write-Ok "WSL distro found: $Distro"

$ProjectWsl = Convert-ToWslPath $ProjectPath
$ModelWsl = ""
if (-not [string]::IsNullOrWhiteSpace($ModelPath) -and (Test-Path -LiteralPath $ModelPath)) {
    $ModelWsl = Convert-ToWslPath $ModelPath
}

$ProjectWslQ = Quote-Bash $ProjectWsl
$VenvNameQ = Quote-Bash $VenvName
$ModelWslQ = Quote-Bash $ModelWsl

Write-Step "Checking WSL project and ROCm virtual environment"

$checkScript = @"
#!/usr/bin/env bash
set -euo pipefail

PROJECT=$ProjectWslQ
VENV=$VenvNameQ

cd "`$PROJECT"

if [ ! -d "`$VENV" ]; then
  echo "MISSING_VENV: `$PROJECT/`$VENV"
  exit 11
fi

if [ ! -x "`$VENV/bin/python" ]; then
  echo "MISSING_VENV_PYTHON: `$PROJECT/`$VENV/bin/python"
  exit 12
fi

"`$VENV/bin/python" - <<'PY'
import importlib.util
import sys

required = ["torch", "ultralytics", "PyQt6", "cv2", "matplotlib", "yaml"]
missing = [name for name in required if importlib.util.find_spec(name) is None]

if missing:
    print("MISSING_MODULES:" + ",".join(missing))
    sys.exit(13)

print("MODULES_OK")
PY
"@

$checkCode = Invoke-WslScript -ScriptContent $checkScript -Label "environment check"
if ($checkCode -ne 0) {
    Fail "WSL environment check failed. Install dependencies in $ProjectPath\$VenvName."
}

Write-Ok "WSL Python dependencies are present"

if (-not $SkipGpuCheck) {
    Write-Step "Checking ROCm GPU from PyTorch"

    $gpuScript = @"
#!/usr/bin/env bash
set -euo pipefail

PROJECT=$ProjectWslQ
VENV=$VenvNameQ

cd "`$PROJECT"

"`$VENV/bin/python" - <<'PY'
import torch

print("torch:", torch.__version__)
print("hip:", getattr(torch.version, "hip", None))
print("cuda_available:", torch.cuda.is_available())
print("device_count:", torch.cuda.device_count())

if not torch.cuda.is_available():
    raise SystemExit(20)

print("device_0:", torch.cuda.get_device_name(0))
x = torch.ones((128, 128), device="cuda")
y = (x @ x).mean()
torch.cuda.synchronize()
print("gpu_tensor_check:", float(y.cpu()))
PY
"@

    $gpuCode = Invoke-WslScript -ScriptContent $gpuScript -Label "GPU check"
    if ($gpuCode -ne 0) {
        Fail "PyTorch does not see ROCm GPU in WSL. Use -SkipGpuCheck only if you intentionally want to launch anyway."
    }

    Write-Ok "ROCm GPU is visible to PyTorch"
}
else {
    Write-Warn "GPU check skipped by user."
}

if ($CheckOnly) {
    Write-Ok "Check-only mode completed. UltraGUI was not launched."
    exit 0
}

Write-Step "Launching UltraGUI through WSLg"

$launchScript = @"
#!/usr/bin/env bash
set -euo pipefail

PROJECT=$ProjectWslQ
VENV=$VenvNameQ
MODEL=$ModelWslQ

cd "`$PROJECT"
source "`$VENV/bin/activate"

export YOLO_CONFIG_DIR="`$PROJECT/.ultralytics"
export HSA_ENABLE_DXG_DETECTION=1
export PYTHONUNBUFFERED=1

if [ -n "`$MODEL" ] && [ -f "`$MODEL" ]; then
  echo "Default model is available at: `$MODEL"
fi

python main.py
"@

$appCode = Invoke-WslScript -ScriptContent $launchScript -Label "UltraGUI"
if ($appCode -ne 0) {
    Fail "UltraGUI exited with code $appCode."
}

Write-Ok "UltraGUI closed normally."
