# Build React UI then package Windows onedir with PyInstaller.
# Creates its own dedicated build venv (.venv-build) so the build is
# self-contained and does not depend on other project environments.
param(
  # Base Python used to create the build venv (only used on first run).
  [string]$BasePython = "",
  # Force recreation of the build venv.
  [switch]$RecreateVenv
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

Write-Host "==> Building UI (Vite)"
Push-Location "$Root\ui"
if (-not (Test-Path "node_modules")) {
  npm ci
}
npm run build
if ($LASTEXITCODE -ne 0) { throw "UI build failed" }
Pop-Location

# --- Dedicated build venv -------------------------------------------------
$VenvDir = Join-Path $Root ".venv-build"
$Py = Join-Path $VenvDir "Scripts\python.exe"

if ($RecreateVenv -and (Test-Path $VenvDir)) {
  Write-Host "==> Removing existing build venv"
  Remove-Item -Recurse -Force $VenvDir
}

if (-not (Test-Path $Py)) {
  # Find a base interpreter to create the venv from.
  if (-not $BasePython) {
    $BaseCandidates = @()
    if (Get-Command py -ErrorAction SilentlyContinue) {
      $BaseCandidates += "py -3.13"
      $BaseCandidates += "py -3.12"
      $BaseCandidates += "py -3"
    }
    if (Get-Command python -ErrorAction SilentlyContinue) {
      $BaseCandidates += "python"
    }
    foreach ($c in $BaseCandidates) {
      $parts = $c -split " "
      & $parts[0] $parts[1..($parts.Length - 1)] -c "import sys; sys.exit(0)" 2>$null
      if ($LASTEXITCODE -eq 0) { $BasePython = $c; break }
    }
  }
  if (-not $BasePython) { throw "No Python found to create the build venv. Pass -BasePython <path>." }

  Write-Host "==> Creating build venv at $VenvDir (base: $BasePython)"
  $parts = $BasePython -split " "
  & $parts[0] $parts[1..($parts.Length - 1)] -m venv $VenvDir
  if ($LASTEXITCODE -ne 0 -or -not (Test-Path $Py)) { throw "Failed to create build venv" }

  & $Py -m pip install --disable-pip-version-check -q --upgrade pip
}

Write-Host "==> Using Python: $Py"

# Ensure CUDA torch (plain "pip install torch" gives a CPU-only build on Windows).
$TorchInfo = & $Py -c "import torch; print(torch.version.cuda or 'cpu')" 2>$null
if ($LASTEXITCODE -ne 0) { $TorchInfo = "missing" }
if ($TorchInfo -eq "cpu" -or $TorchInfo -eq "missing") {
  Write-Host "==> Installing CUDA torch (cu128) into venv (current: $TorchInfo)"
  if ($TorchInfo -eq "cpu") {
    & $Py -m pip uninstall -y -q torch
  }
  & $Py -m pip install --disable-pip-version-check -q torch --index-url https://download.pytorch.org/whl/cu128
  if ($LASTEXITCODE -ne 0) { throw "CUDA torch install failed" }
} else {
  Write-Host "==> torch already CUDA-enabled (cuda $TorchInfo)"
}

Write-Host "==> Installing build requirements into venv"
& $Py -m pip install --disable-pip-version-check -q -r "$Root\requirements.txt"
if ($LASTEXITCODE -ne 0) { throw "pip install failed" }

# Fail fast if the runtime stack cannot import in the build venv.
Write-Host "==> Verifying runtime imports in build venv"
& $Py -c @"
mods = [
    'torch', 'transformers', 'diffusers', 'accelerate', 'safetensors',
    'huggingface_hub', 'tokenizers', 'PIL', 'cv2', 'fastapi', 'uvicorn', 'webview',
]
missing = []
for m in mods:
    try:
        __import__(m)
    except Exception as e:
        missing.append(f'{m}: {e}')
if missing:
    raise SystemExit('Missing imports:\n' + '\n'.join(missing))
from diffusers.utils import is_transformers_available
assert is_transformers_available(), 'diffusers cannot see transformers'
print('runtime imports OK; cuda=', __import__('torch').cuda.is_available())
"@
if ($LASTEXITCODE -ne 0) { throw "Build venv is missing required packages" }

function Clear-DistOutput {
  param([string]$DistDir)

  Write-Host "==> Freeing $DistDir (kill running exe, then delete)"

  # Kill by process name and by anything launched from that folder.
  # taskkill stderr must not abort the script when the process is already gone.
  $prevEap = $ErrorActionPreference
  $ErrorActionPreference = "Continue"
  try {
    Get-Process -Name "SDAnimePose" -ErrorAction SilentlyContinue |
      Stop-Process -Force -ErrorAction SilentlyContinue
    cmd /c "taskkill /F /IM SDAnimePose.exe >nul 2>&1" | Out-Null

    # Kill anything launched FROM the dist folder, or with a file under it open
    # (e.g. notepad holding sdanime_pose.log — that alone locks the directory).
    Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
      Where-Object {
        ($_.ExecutablePath -and
         $_.ExecutablePath.StartsWith($DistDir, [System.StringComparison]::OrdinalIgnoreCase)) -or
        ($_.CommandLine -and
         $_.CommandLine.IndexOf($DistDir, [System.StringComparison]::OrdinalIgnoreCase) -ge 0)
      } |
      ForEach-Object {
        Write-Host "    killing PID $($_.ProcessId) ($($_.Name))"
        Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
        cmd /c "taskkill /F /PID $($_.ProcessId) >nul 2>&1" | Out-Null
      }
  } finally {
    $ErrorActionPreference = $prevEap
  }

  Start-Sleep -Seconds 1

  if (-not (Test-Path $DistDir)) { return }

  for ($i = 1; $i -le 8; $i++) {
    try {
      Remove-Item -LiteralPath $DistDir -Recurse -Force -ErrorAction Stop
    } catch {
      # ignore; retry below
    }
    if (-not (Test-Path $DistDir)) { return }

    # cmd rmdir often succeeds when PowerShell's Remove-Item does not
    cmd /c "rmdir /s /q `"$DistDir`" >nul 2>&1" | Out-Null
    if (-not (Test-Path $DistDir)) { return }

    Write-Host "    still locked, retry $i/8..."
    $ErrorActionPreference = "Continue"
    try {
      Get-Process -Name "SDAnimePose" -ErrorAction SilentlyContinue |
        Stop-Process -Force -ErrorAction SilentlyContinue
      cmd /c "taskkill /F /IM SDAnimePose.exe >nul 2>&1" | Out-Null
    } finally {
      $ErrorActionPreference = $prevEap
    }
    Start-Sleep -Seconds 2
  }

  # Last resort: rename so PyInstaller can create a fresh folder.
  if (Test-Path $DistDir) {
    $stash = "$DistDir.old.$(Get-Date -Format 'yyyyMMdd_HHmmss')"
    Write-Host "    could not delete; renaming to $stash"
    Rename-Item -LiteralPath $DistDir -NewName (Split-Path $stash -Leaf) -Force -ErrorAction Stop
  }
}

$DistOut = Join-Path $Root "dist\SDAnimePose"
Clear-DistOutput -DistDir $DistOut

& $Py -m PyInstaller --noconfirm --distpath "$Root\dist" --workpath "$Root\build" "$Root\packaging\sdanime_pose.spec"
if ($LASTEXITCODE -ne 0) {
  # PyInstaller itself may hit a mid-build lock; clear and retry once.
  Write-Host "==> PyInstaller failed; clearing dist and retrying once"
  Clear-DistOutput -DistDir $DistOut
  & $Py -m PyInstaller --noconfirm --distpath "$Root\dist" --workpath "$Root\build" "$Root\packaging\sdanime_pose.spec"
  if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed" }
}

$Out = Join-Path $Root "dist\SDAnimePose"
Write-Host "==> Copying data/models into package (AnythingV5 + IP-Adapter + checkpoints)"
New-Item -ItemType Directory -Force -Path "$Out\data" | Out-Null
# Prefer _internal layout (PyInstaller 6) or exe-root
$Targets = @("$Out", "$Out\_internal")
foreach ($t in $Targets) {
  if (Test-Path $t) {
    New-Item -ItemType Directory -Force -Path "$t\data\models" | Out-Null
    if (Test-Path "$Root\data\models") {
      robocopy "$Root\data\models" "$t\data\models" /E /NFL /NDL /NJH /NJS /nc /ns /np | Out-Null
    }
    if (Test-Path "$Root\data\refs") {
      New-Item -ItemType Directory -Force -Path "$t\data\refs" | Out-Null
      robocopy "$Root\data\refs" "$t\data\refs" /E /NFL /NDL /NJH /NJS /nc /ns /np | Out-Null
    }
    # Also place UI next to exe (PyInstaller puts it under _internal; path
    # resolution can pick the exe folder first because data/ lives there).
    if (Test-Path "$Root\ui\dist\index.html") {
      New-Item -ItemType Directory -Force -Path "$t\ui\dist" | Out-Null
      robocopy "$Root\ui\dist" "$t\ui\dist" /E /NFL /NDL /NJH /NJS /nc /ns /np | Out-Null
    }
  }
}

# Safety net: HF/diffusers need *.dist-info next to the package code.
Write-Host "==> Ensuring package metadata (.dist-info) is present"
$Site = Join-Path $VenvDir "Lib\site-packages"
$Internal = Join-Path $Out "_internal"
$MetaPkgs = @(
  "transformers", "diffusers", "accelerate", "safetensors",
  "huggingface_hub", "tokenizers", "torch", "numpy", "pillow", "regex"
)
foreach ($name in $MetaPkgs) {
  $src = Get-ChildItem $Site -Directory -Filter "$name-*.dist-info" -ErrorAction SilentlyContinue |
    Select-Object -First 1
  if (-not $src) {
    # Pillow installs as Pillow-*.dist-info
    $src = Get-ChildItem $Site -Directory -Filter "$($name.Substring(0,1).ToUpper() + $name.Substring(1))-*.dist-info" -ErrorAction SilentlyContinue |
      Select-Object -First 1
  }
  if ($src -and (Test-Path $Internal)) {
    $dest = Join-Path $Internal $src.Name
    if (-not (Test-Path $dest)) {
      robocopy $src.FullName $dest /E /NFL /NDL /NJH /NJS /nc /ns /np | Out-Null
      Write-Host "    copied $($src.Name)"
    }
  }
}

# Bundle OpenSeeFace next to the exe so local tracking works without monorepo paths.
# The frozen exe must NOT run facetracker.py with itself — prefer system Python via
# SDANIME_TRACKER_PYTHON, or ship facetracker.exe under tools/OpenSeeFace.
Write-Host "==> Copying OpenSeeFace tools into package"
$OsfSrc = Join-Path $Root "tools\OpenSeeFace"
if (Test-Path $OsfSrc) {
  foreach ($t in @("$Out", "$Out\_internal")) {
    if (Test-Path $t) {
      New-Item -ItemType Directory -Force -Path "$t\tools\OpenSeeFace" | Out-Null
      robocopy $OsfSrc "$t\tools\OpenSeeFace" /E /XD .git .github __pycache__ /NFL /NDL /NJH /NJS /nc /ns /np | Out-Null
    }
  }
  # Write a tiny launcher note for frozen local mode
  $note = @"
OpenSeeFace is bundled here for local tracking.

Camera *listing* uses the DirectShow DLL in-process (no extra Python).
Starting face track still needs a Python that has Pillow + OpenCV + NumPy,
or a prebuilt facetracker.exe in this folder.

Priority:
  1. Env var SDANIME_TRACKER_PYTHON=C:\path\to\python.exe
  2. Nearby .venv-build / monorepo torch_train venv (dev layouts)
  3. facetracker.exe in this folder

Server/Docker mode does not need this folder — send JSON to /api/tracking/frame.
"@
  Set-Content -Path "$Out\tools\OpenSeeFace\README_SDANIME.txt" -Value $note -Encoding UTF8
}

Write-Host "==> Done: $Out"
Write-Host "Run: $Out\SDAnimePose.exe"
Write-Host "Server/Docker: python -m backend --ui none --server-mode"
