@echo off
rem Default: serial tracking then braking, results/main_results/.
rem The Python entry point does not write configuration.json or summary.json.
setlocal
rem Use python from the activated environment.
set "RNDDPC_PYTHON=python"
"%RNDDPC_PYTHON%" -c "import sys" >nul 2>&1
if errorlevel 1 (
    echo Python could not be started. Activate your project environment.
    exit /b 1
)
"%RNDDPC_PYTHON%" -u "%~dp0run_rnddpc.py" %*
set "RNDDPC_EXIT=%ERRORLEVEL%"
if "%~1"=="" pause
exit /b %RNDDPC_EXIT%
