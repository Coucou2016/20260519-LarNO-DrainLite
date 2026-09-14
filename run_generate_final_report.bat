@echo off
setlocal
cd /d "%~dp0"
python extended_study\generate_final_standalone_report.py
if errorlevel 1 (
  echo.
  echo Failed to generate final report.
  pause
  exit /b 1
)
echo.
echo Final standalone report generated:
echo extended_study\output\LarNO_ITZI_Final_Standalone_Report.html
pause
