@echo off
rem Double-click to start Sports Helpers. First run creates .venv and installs packages.
cd /d "%~dp0"
where py >nul 2>nul && (set PY=py -3) || (set PY=python)
if not exist .venv\Scripts\python.exe (
  echo Creating a virtual environment in .venv ...
  %PY% -m venv .venv || (echo Python 3.11+ is needed: https://www.python.org/downloads/ & pause & exit /b 1)
)
.venv\Scripts\python.exe run.py --yes %*
pause
