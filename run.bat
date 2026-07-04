@echo off
REM Convenience launcher for Windows — uses the correct Python interpreter.
REM Usage: run.bat [api|dashboard|test]

set PYTHON=python

if "%1"=="" (
    %PYTHON% main.py dashboard
) else (
    %PYTHON% main.py %1
)

if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] Application crashed with exit code %ERRORLEVEL%.
    pause
)
