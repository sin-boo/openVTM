@echo off
setlocal
cd /d "%~dp0"

echo Building SDAnimePose.exe (UI + PyInstaller onedir)...
echo This can take several minutes because torch is large.
echo.

powershell -ExecutionPolicy Bypass -File "%~dp0packaging\build.ps1"
if errorlevel 1 (
  echo.
  echo BUILD FAILED. See messages above.
  pause
  exit /b 1
)

echo.
echo Done. Run:
echo   dist\SDAnimePose\SDAnimePose.exe
echo.
pause
endlocal
