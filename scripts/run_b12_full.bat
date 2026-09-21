@echo off
cd /d "%~dp0.."
.venv\Scripts\python.exe -u src\run_experiments.py --datasets cicids2017 --seeds 42 123 456 2024 7777 --models softgroup2 joint2 --epochs 100 --no-smote > results\cicids_b12_full.log 2>&1