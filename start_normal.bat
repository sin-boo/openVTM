@echo off
REM Start SDAnime Pose in normal/dev mode (API + Vite). Does NOT build an exe.
setlocal
cd /d "%~dp0"

set TF_CPP_MIN_LOG_LEVEL=3
set TF_ENABLE_ONEDNN_OPTS=0

set "PY="
if exist "%~dp0.venv-build\Scripts\python.exe" (
  set "PY=%~dp0.venv-build\Scripts\python.exe"
  echo Using .venv-build.
) else if exist "%~dp0..\pipeline\i1\torch_train\.venv\Scripts\python.exe" (
  set "PY=%~dp0..\pipeline\i1\torch_train\.venv\Scripts\python.exe"
  echo Using torch_train venv.
) else if exist "%~dp0.venv\Scripts\python.exe" (
  set "PY=%~dp0.venv\Scripts\python.exe"
  echo Using .venv.
) else (
  set "PY=python"
)

REM Prefer this same interpreter for OpenSeeFace facetracker.
set "SDANIME_TRACKER_PYTHON=%PY%"

if not exist "%~dp0data\models\AnythingV5V3_v5PrtRE.safetensors" (
  echo Model not found. Downloading...
  "%PY%" -m backend.download_model
  if errorlevel 1 goto :error
)

if not exist "%~dp0ui\node_modules" (
  echo Installing UI deps...
  pushd "%~dp0ui"
  call npm install
  if errorlevel 1 goto :error
  popd
)

echo Starting API on http://127.0.0.1:8765 ...
start "SDAnime API" cmd /c ""%PY%" -m backend --ui none"

echo Waiting for API health...
set /a _tries=0
:wait_api
set /a _tries+=1
powershell -NoProfile -Command "try { $r = Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8765/api/health -TimeoutSec 2; if ($r.StatusCode -ge 200 -and $r.StatusCode -lt 500) { exit 0 } else { exit 1 } } catch { exit 1 }" >nul 2>&1
if not errorlevel 1 goto :api_ready
if %_tries% GEQ 90 (
  echo.
  echo API did not become ready within ~3 minutes.
  echo Check the "SDAnime API" window for errors.
  goto :error
)
timeout /t 2 /nobreak >nul
goto :wait_api

:api_ready
echo API ready. Starting Vite UI on http://127.0.0.1:5173 ...
pushd "%~dp0ui"
call npm run dev
popd
goto :end

:error
echo.
echo Failed to start. Review the message above.
pause

:end
endlocal
