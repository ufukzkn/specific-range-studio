@echo off
setlocal

set "PROJECT_ROOT=%~dp0.."
set "PYTHON=%PROJECT_ROOT%\.venv\Scripts\python.exe"
set "SERVER=%PROJECT_ROOT%\scripts\web_app\server.py"

if not exist "%PYTHON%" (
  echo Virtual environment not found. Create it first:
  echo   python -m venv .venv
  echo   .venv\Scripts\activate
  echo   pip install -r requirements.txt
  exit /b 1
)

echo Starting Specific Range Studio...
echo URL: http://localhost:5000
"%PYTHON%" -u "%SERVER%"
