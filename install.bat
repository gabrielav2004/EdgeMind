@echo off
echo ========================================
echo EdgeMind Installer
echo ========================================

where python >nul 2>&1
if %errorlevel% neq 0 (
    echo x python not found. please install python 3.9+
    exit /b 1
)

echo installing CPU-only torch...
pip install torch --index-url https://download.pytorch.org/whl/cpu

echo installing EdgeMind...
pip install -e .

echo.
echo ========================================
echo EdgeMind installed
echo.
echo next steps:
echo   1. edit edgemind/core/config.py
echo   2. add API key to .env file
echo   3. edgemind ingest data/docs
echo   4. edgemind interactive
echo ========================================
