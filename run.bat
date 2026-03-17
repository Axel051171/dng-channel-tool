@echo off
title DNG Channel Tool
echo ============================================
echo   DNG Channel Tool - Farbkanal-Tausch
echo   IR-Fotografie ^& Preset-Konverter
echo ============================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    python3 --version >nul 2>&1
    if errorlevel 1 (
        echo Python nicht gefunden!
        echo Bitte Python 3.10+ installieren: https://www.python.org/downloads/
        pause
        exit /b 1
    )
    set PYTHON=python3
) else (
    set PYTHON=python
)

REM Install dependencies if needed
%PYTHON% -c "import numpy, PIL, rawpy" >nul 2>&1
if errorlevel 1 (
    echo Installiere Abhaengigkeiten...
    %PYTHON% -m pip install -r requirements.txt
    echo.
)

REM Start
echo Starte DNG Channel Tool...
%PYTHON% main.py
