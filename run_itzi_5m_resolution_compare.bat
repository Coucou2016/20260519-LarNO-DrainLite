@echo off
setlocal
cd /d "%~dp0"
python extended_study\run_itzi_5m_resolution_compare.py
echo.
if errorlevel 1 (
  echo Failed to run 5 m resolution comparison.
  pause
  exit /b 1
)
echo 5 m resolution comparison outputs generated:
echo extended_study\output\itzi_5m_compare
pause
