@echo off
REM API-only server mode for Docker / remote clients.
REM Accepts JSON tracking frames at POST /api/tracking/frame — never opens a webcam.
setlocal
cd /d "%~dp0"
set SDANIME_SERVER_MODE=1
if exist ".venv-build\Scripts\python.exe" (
  ".venv-build\Scripts\python.exe" -m backend --ui none --server-mode --host 0.0.0.0 --port 8765
) else (
  python -m backend --ui none --server-mode --host 0.0.0.0 --port 8765
)
endlocal
