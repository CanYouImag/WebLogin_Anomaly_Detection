@echo off
cd /d C:\Users\11831\PycharmProjects\WebLogin_Anomaly_Detection
.venv\Scripts\python.exe -u src\run_experiments.py --datasets cicids2017 --models mlp two_stage mvt mvt2 --seeds 42 123 456 2024 7777 --epochs 100 --no-smote > results\cicids_followup.log 2>&1