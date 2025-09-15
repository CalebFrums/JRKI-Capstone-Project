#!/usr/bin/env python3
"""
Build script for creating standalone executable of MBIE MLOps GUI
Creates a professional .exe file for easy deployment
"""

import os
import sys
import subprocess
import shutil

def install_pyinstaller():
    """Install PyInstaller if not available"""
    try:
        import PyInstaller
        print("[OK] PyInstaller is already installed")
        return True
    except ImportError:
        print("[INFO] Installing PyInstaller...")
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pyinstaller'])
            print("[OK] PyInstaller installed successfully")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to install PyInstaller: {e}")
            return False

def create_executable():
    """Create standalone executable"""
    if not install_pyinstaller():
        return False
    
    print("\n" + "="*60)
    print("BUILDING MBIE MLOPS GUI EXECUTABLE")
    print("="*60)
    
    # Build command for PyInstaller
    build_cmd = [
        'pyinstaller',
        '--onefile',                    # Single executable file
        '--windowed',                   # No console window (GUI only)
        '--name', 'MBIE_MLOps_Pipeline', # Executable name
        '--icon=mbie_icon.ico' if os.path.exists('mbie_icon.ico') else '',  # Icon if available
        '--add-data', 'simple_config.json;.' if os.path.exists('simple_config.json') else '',
        '--hidden-import', 'tkinter',
        '--hidden-import', 'tkinter.ttk',
        '--hidden-import', 'tkinter.scrolledtext',
        '--hidden-import', 'queue',
        '--clean',                      # Clean build
        'mlops_gui.py'
    ]
    
    # Remove empty arguments
    build_cmd = [arg for arg in build_cmd if arg]
    
    try:
        print(f"[INFO] Running: {' '.join(build_cmd)}")
        subprocess.run(build_cmd, check=True)
        
        print("\n[SUCCESS] Executable created successfully!")
        print(f"[INFO] Location: {os.path.abspath('dist/MBIE_MLOps_Pipeline.exe')}")
        
        # Create launcher script
        create_launcher()
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Build failed: {e}")
        return False
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        return False

def create_launcher():
    """Create a simple launcher script"""
    launcher_content = '''@echo off
title MBIE MLOps Pipeline Launcher
echo.
echo ========================================
echo  MBIE MLOPS PIPELINE - GAMING EDITION
echo ========================================
echo.
echo [INFO] Launching MLOps Dashboard...
echo [INFO] Please wait...
echo.

"MBIE_MLOps_Pipeline.exe"

if errorlevel 1 (
    echo.
    echo [ERROR] Application failed to start
    echo [INFO] Check that all required files are present
    pause
)
'''
    
    try:
        with open('dist/Launch_MLOps_GUI.bat', 'w') as f:
            f.write(launcher_content)
        print("[INFO] Launcher script created: Launch_MLOps_GUI.bat")
    except Exception as e:
        print(f"[WARNING] Could not create launcher: {e}")

def create_installer():
    """Create installation package"""
    installer_content = '''@echo off
title MBIE MLOps Pipeline Installer
cls
echo.
echo ==========================================
echo   MBIE MLOPS PIPELINE INSTALLATION
echo ==========================================
echo.
echo This will install the MBIE MLOps Pipeline
echo Gaming Edition to your system.
echo.
pause

echo [INFO] Creating installation directory...
mkdir "%USERPROFILE%\\Desktop\\MBIE_MLOps" 2>nul

echo [INFO] Copying application files...
copy "MBIE_MLOps_Pipeline.exe" "%USERPROFILE%\\Desktop\\MBIE_MLOps\\" >nul
copy "Launch_MLOps_GUI.bat" "%USERPROFILE%\\Desktop\\MBIE_MLOps\\" >nul

echo [INFO] Creating desktop shortcut...
echo Set oWS = WScript.CreateObject("WScript.Shell") > temp_shortcut.vbs
echo sLinkFile = "%USERPROFILE%\\Desktop\\MBIE MLOps Pipeline.lnk" >> temp_shortcut.vbs
echo Set oLink = oWS.CreateShortcut(sLinkFile) >> temp_shortcut.vbs
echo oLink.TargetPath = "%USERPROFILE%\\Desktop\\MBIE_MLOps\\MBIE_MLOps_Pipeline.exe" >> temp_shortcut.vbs
echo oLink.WorkingDirectory = "%USERPROFILE%\\Desktop\\MBIE_MLOps\\" >> temp_shortcut.vbs
echo oLink.Description = "MBIE MLOps Pipeline Gaming Edition" >> temp_shortcut.vbs
echo oLink.Save >> temp_shortcut.vbs
cscript temp_shortcut.vbs >nul 2>&1
del temp_shortcut.vbs >nul 2>&1

echo.
echo [SUCCESS] Installation completed!
echo [INFO] Desktop shortcut created
echo [INFO] Files installed to: %USERPROFILE%\\Desktop\\MBIE_MLOps
echo.
echo Press any key to launch the application...
pause >nul

start "" "%USERPROFILE%\\Desktop\\MBIE_MLOps\\MBIE_MLOps_Pipeline.exe"
'''
    
    try:
        with open('dist/Install_MLOps_GUI.bat', 'w') as f:
            f.write(installer_content)
        print("[INFO] Installer created: Install_MLOps_GUI.bat")
    except Exception as e:
        print(f"[WARNING] Could not create installer: {e}")

def main():
    """Main build process"""
    print("MBIE MLOps Pipeline - Executable Builder")
    print("=======================================")
    
    # Check if source file exists
    if not os.path.exists('mlops_gui.py'):
        print("[ERROR] mlops_gui.py not found!")
        return False
    
    # Create executable
    if create_executable():
        create_installer()
        print("\n" + "="*60)
        print("BUILD COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nFiles created:")
        print("  - dist/MBIE_MLOps_Pipeline.exe     (Main application)")
        print("  - dist/Launch_MLOps_GUI.bat        (Launcher script)")
        print("  - dist/Install_MLOps_GUI.bat       (Installer)")
        print("\nTo deploy:")
        print("  1. Copy the 'dist' folder to target machine")
        print("  2. Run 'Install_MLOps_GUI.bat' for full installation")
        print("  3. Or run 'MBIE_MLOps_Pipeline.exe' directly")
        print("\n[SUCCESS] Ready for deployment!")
        
        return True
    else:
        print("\n[ERROR] Build failed!")
        return False

if __name__ == "__main__":
    success = main()
    input("\nPress Enter to exit...")
    sys.exit(0 if success else 1)