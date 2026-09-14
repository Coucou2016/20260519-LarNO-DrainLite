@echo off
setlocal
cd /d "%~dp0"

echo [1/2] Building 5 m resolution comparison package...
python extended_study\run_itzi_5m_resolution_compare.py
if errorlevel 1 (
  echo Failed at 5 m resolution comparison step.
  pause
  exit /b 1
)

echo.
echo [2/4] Building/running SWMM dynamic-wave pipe-network branch...
python extended_study\build_swmm_network.py
if errorlevel 1 (
  echo Failed at SWMM pipe-network branch.
  pause
  exit /b 1
)

echo.
echo [3/4] Running 5 m refined-grid ITZI sensitivity experiment...
python extended_study\run_itzi_5m_refined_window.py
if errorlevel 1 (
  echo Failed at 5 m refined-grid ITZI step.
  pause
  exit /b 1
)

echo.
echo [4/4] Building final standalone HTML report...
python extended_study\generate_final_standalone_report.py
if errorlevel 1 (
  echo Failed at final HTML report step.
  pause
  exit /b 1
)

echo.
echo Done.
echo Final report:
echo extended_study\output\LarNO_ITZI_Final_Standalone_Report.html
pause
