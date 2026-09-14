@echo off
setlocal
cd /d "%~dp0"
python extended_study\build_swmm_network.py
if errorlevel 1 (
  echo.
  echo Failed to build/run SWMM dynamic-wave pipe-network branch.
  pause
  exit /b 1
)
echo.
echo SWMM branch outputs:
echo extended_study\output\swmm_network
pause
