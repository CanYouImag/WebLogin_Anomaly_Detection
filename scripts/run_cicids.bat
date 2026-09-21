@echo off
cd /d C:\Users\11831\PycharmProjects\WebLogin_Anomaly_Detection
.venv\Scripts\python.exe -u src\run_experiments.py --datasets cicids2017 --models all --seeds 42 123 456 2024 7777 --epochs 100 --no-smote > results\cicids_full.log 2> results\cicids_full_err.log