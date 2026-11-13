@echo off
rem One-click local start on Windows. Creates .venv on first run.
rem Uses demo accounts and the synthetic demo MRI model: for trying the app only.
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  py -3 -m venv .venv || python -m venv .venv
  .venv\Scripts\python -m pip install --upgrade pip
  .venv\Scripts\python -m pip install -r requirements-app.txt
)
.venv\Scripts\python run_app.py seed-demo
.venv\Scripts\python run_app.py --demo-model
