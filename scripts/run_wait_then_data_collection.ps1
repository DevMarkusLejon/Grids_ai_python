param(
  [int]$WaitForPid = 0,
  [double]$Hours = 14.0,
  [int]$Workers = 6,
  [int]$GamesPerBatch = 160,
  [int]$SeedBase = 960603,
  [string]$RunLabel = ""
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

New-Item -ItemType Directory -Force -Path "logs" | Out-Null
$Stamp = Get-Date -Format "yyyyMMdd-HHmmss"
if (-not $RunLabel) {
  $RunLabel = "overnight_policy_value_collection_$Stamp"
}
$LauncherLog = Join-Path "logs" "$RunLabel.launcher.log"

Start-Transcript -Path $LauncherLog | Out-Null
try {
  Write-Host "Started wait-then-collection launcher at $((Get-Date).ToString('o'))"
  Write-Host "Run label: $RunLabel"
  Write-Host "WaitForPid: $WaitForPid"
  if ($WaitForPid -gt 0) {
    $Existing = Get-Process -Id $WaitForPid -ErrorAction SilentlyContinue
    if ($Existing) {
      Write-Host "Waiting for PID $WaitForPid to exit before starting data collection."
      Wait-Process -Id $WaitForPid
      Write-Host "PID $WaitForPid exited at $((Get-Date).ToString('o'))."
    } else {
      Write-Host "PID $WaitForPid is not active; starting data collection now."
    }
  }

  if (-not (Test-Path "champion_model.txt")) {
    throw "champion_model.txt not found."
  }
  $TeacherModel = (Get-Content "champion_model.txt" -Raw).Trim()
  if (-not (Test-Path $TeacherModel)) {
    throw "Teacher model from champion_model.txt not found: $TeacherModel"
  }

  Write-Host "Starting long data collection with teacher: $TeacherModel"
  powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/run_long_policy_value_data_collection.ps1 `
    -TeacherModel $TeacherModel `
    -Hours $Hours `
    -Workers $Workers `
    -GamesPerBatch $GamesPerBatch `
    -SeedBase $SeedBase `
    -RunLabel $RunLabel
  if ($LASTEXITCODE -ne 0) {
    throw "Long data collection exited with code $LASTEXITCODE."
  }
} catch {
  $Message = "Wait-then-data-collection launcher failed: $($_.Exception.Message) run=$RunLabel"
  powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/notify_important.ps1 `
    -Title "Grids AI collection launcher failed" `
    -Message $Message `
    -Priority high `
    -Tags "warning"
  throw
} finally {
  Stop-Transcript | Out-Null
}
