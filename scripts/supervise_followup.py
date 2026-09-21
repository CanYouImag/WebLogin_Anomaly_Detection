"""
Supervisor: wait for the running CICIDS full-run to finish, then launch the
follow-up experiment with the fixed code (semantic grouping + focal loss).
Run detached via: cmd.exe /c scripts\supervise_followup.bat
"""
import os
import sys
import time
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.path.join(ROOT, ".venv", "Scripts", "python.exe")
LOG = os.path.join(ROOT, "results", "followup_supervisor.log")

logf = open(LOG, "w", encoding="utf-8")


def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line)
    logf.write(line + "\n")
    logf.flush()


def old_run_alive():
    import subprocess
    out = subprocess.check_output(
        ["powershell", "-NoProfile", "-Command",
         "Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*run_experiments.py*--models all*' } | "
         "Measure-Object | Select-Object -ExpandProperty Count"],
        text=True, timeout=30).strip()
    try:
        return int(out) > 0
    except ValueError:
        return False


def main():
    log("supervisor started")
    # Wait for the old full run (--models all) to fully exit.
    while True:
        try:
            alive = old_run_alive()
        except Exception as e:
            log(f"poll error: {e}; retrying")
            alive = True
        if not alive:
            break
        time.sleep(60)
    log("old run finished. Starting follow-up in 10s...")
    time.sleep(10)

    cmd = [
        PY, "-u", "src/run_experiments.py",
        "--datasets", "cicids2017",
        "--models", "mlp", "two_stage", "mvt", "mvt2",
        "--seeds", "42", "123", "456", "2024", "7777",
        "--epochs", "100",
        "--no-smote",
    ]
    out_log = os.path.join(ROOT, "results", "cicids_followup.log")
    log("RUN: " + " ".join(cmd))
    with open(out_log, "w", encoding="utf-8") as f:
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
        rc = proc.wait()
    log(f"follow-up exited rc={rc}")
    logf.close()


if __name__ == "__main__":
    main()