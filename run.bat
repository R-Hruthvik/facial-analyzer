@echo off
REM Convenience launcher for Windows — uses the correct Python interpreter.
REM Usage: run.bat [api|dashboard|test]

set PYTHON=py -3.13

if "%1"=="" (
    %PYTHON% main.py dashboard
) else (
    %PYTHON% main.py %1
)
