@echo off
setlocal
cd /d "%~dp0"
python extended_study\run_itzi_5m_refined_window.py
if errorlevel 1 (
  echo.
  echo Failed to run 5 m refined-grid ITZI sensitivity experiment.
  pause
  exit /b 1
)
echo.
echo 5 m refined-grid ITZI outputs:
echo extended_study\output\itzi_5m_refined
pause
