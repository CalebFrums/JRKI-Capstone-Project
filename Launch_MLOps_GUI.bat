@echo off
title MBIE MLOps Pipeline Launcher
color 07
cls
echo.
echo ============================================
echo      MBIE MLOPS PIPELINE INTERFACE
echo ============================================
echo.
echo [INFO] Initializing professional interface...
echo [INFO] Loading pipeline controls...
echo [INFO] Starting application...
echo.

if exist "dist\MBIE_MLOps_Pipeline.exe" (
    echo [SUCCESS] Executable found - launching application...
    echo.
    start "" "dist\MBIE_MLOps_Pipeline.exe"
    echo [INFO] MLOps GUI launched successfully!
    echo [INFO] You can now close this terminal
    timeout /t 3 >nul
) else (
    echo [ERROR] MBIE_MLOps_Pipeline.exe not found in dist folder
    echo [INFO] Please run 'python -m PyInstaller --onefile --windowed mlops_gui.py' first
    pause
)