$ErrorActionPreference = "Stop"
$root = "C:\Users\11831\PycharmProjects\WebLogin_Anomaly_Detection"
$py = Join-Path $root ".venv\Scripts\python.exe"
$log = Join-Path $root "results\cicids_followup.log"
$oldPattern = "*run_experiments.py --datasets cicids2017 --models all*"

Write-Output "waiting for old CICIDS run to exit..."
$oldProc = $null
while ($true) {
    $oldProc = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -match 'python|cmd' -and $_.CommandLine -like $oldPattern }
    if ($oldProc) {
        Start-Sleep -Seconds 30
    } else {
        break
    }
}

Start-Sleep -Seconds 10
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$backup = Join-Path $root ("results\backup_cicids_" + $stamp)
New-Item -ItemType Directory -Path $backup -Force | Out-Null
Copy-Item (Join-Path $root "results\cicids2017") -Destination $backup -Recurse -Force
Write-Output "backed up old results to $backup"

Write-Output "starting follow-up (fixed code) at $stamp"
Set-Location $root
& $py -u src\run_experiments.py --datasets cicids2017 --models mlp two_stage mvt mvt2 --seeds 42 123 456 2024 7777 --epochs 100 --no-smote *> $log 2>&1
Write-Output ("follow-up finished at " + (Get-Date -Format 'yyyyMMdd_HHmmss'))