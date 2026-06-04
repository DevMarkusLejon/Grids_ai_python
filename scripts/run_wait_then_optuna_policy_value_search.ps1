param(
  [string]$CollectionDir = "neural_data\overnight_policy_value_collection_20260603",
  [int]$WaitForPid = 0,
  [double]$MaxWaitHours = 16.0,
  [int]$PollSeconds = 60,
  [double]$SearchHours = 8.0,
  [int]$Trials = 40,
  [string]$Python = ".venv-gpu\Scripts\python.exe",
  [string]$ReferenceModel = "checkpoints/policy_value_torch_192_blend_20260521-073435.json",
  [int]$Workers = 6,
  [int]$ScreenGames = 48,
  [int]$FullGateGames = 192,
  [string]$RunLabel = "",
  [switch]$PromoteOnPass,
  [switch]$ValidateOnly
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

New-Item -ItemType Directory -Force -Path "logs" | Out-Null
$Stamp = Get-Date -Format "yyyyMMdd-HHmmss"
if (-not $RunLabel) {
  $RunLabel = "wait_then_optuna_policy_value_$Stamp"
}
$LauncherLog = Join-Path "logs" "$RunLabel.launcher.log"
$ManifestPath = Join-Path $CollectionDir "manifest.json"

function Invoke-Notify {
  param(
    [string]$Title,
    [string]$Message,
    [string]$Priority = "default",
    [string]$Tags = "mag"
  )
  $NotifyOutput = powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/notify_important.ps1 `
    -Title $Title `
    -Message $Message `
    -Priority $Priority `
    -Tags $Tags
  if ($LASTEXITCODE -ne 0) {
    throw "Notification failed with exit code $LASTEXITCODE. Output: $($NotifyOutput -join '; ')"
  }
  foreach ($Line in $NotifyOutput) {
    Write-Host $Line
  }
}

if ($ValidateOnly) {
  Write-Host "Validation OK."
  Write-Host "Run label: $RunLabel"
  Write-Host "Collection dir: $CollectionDir"
  Write-Host "Manifest path: $ManifestPath"
  Write-Host "Reference model: $ReferenceModel"
  Write-Host "Python: $Python"
  exit 0
}

Start-Transcript -Path $LauncherLog | Out-Null
try {
  $StartedAt = Get-Date
  $Deadline = $StartedAt.AddHours($MaxWaitHours)
  Write-Host "Started wait-then-Optuna launcher at $($StartedAt.ToString('o'))"
  Write-Host "Run label: $RunLabel"
  Write-Host "Collection dir: $CollectionDir"
  Write-Host "Manifest path: $ManifestPath"
  Write-Host "WaitForPid: $WaitForPid"

  if ($WaitForPid -gt 0) {
    $Existing = Get-Process -Id $WaitForPid -ErrorAction SilentlyContinue
    if ($Existing) {
      Write-Host "Waiting for PID $WaitForPid to exit before checking collection manifest."
      Wait-Process -Id $WaitForPid
      Write-Host "PID $WaitForPid exited at $((Get-Date).ToString('o'))."
    } else {
      Write-Host "PID $WaitForPid is not active."
    }
  }

  while (-not (Test-Path -LiteralPath $ManifestPath -PathType Leaf)) {
    if ((Get-Date) -ge $Deadline) {
      throw "Timed out waiting for collection manifest: $ManifestPath"
    }
    Write-Host "[$((Get-Date).ToString('o'))] Waiting for collection manifest: $ManifestPath"
    Start-Sleep -Seconds $PollSeconds
  }

  Write-Host "Found collection manifest at $((Get-Date).ToString('o')): $ManifestPath"
  $Args = @(
    "-NoProfile",
    "-ExecutionPolicy",
    "Bypass",
    "-File",
    "scripts/run_optuna_policy_value_search.ps1",
    "-Hours",
    "$SearchHours",
    "-Trials",
    "$Trials",
    "-Python",
    $Python,
    "-ReferenceModel",
    $ReferenceModel,
    "-Workers",
    "$Workers",
    "-ScreenGames",
    "$ScreenGames",
    "-FullGateGames",
    "$FullGateGames",
    "-RunLabel",
    "$RunLabel.search",
    "-CollectionManifest",
    $ManifestPath,
    "-NotifyOnStart"
  )
  if ($PromoteOnPass) {
    $Args += "-PromoteOnPass"
  }

  powershell.exe @Args
  if ($LASTEXITCODE -ne 0) {
    throw "Optuna wrapper exited with code $LASTEXITCODE."
  }
} catch {
  $Message = "Wait-then-Optuna launcher failed. run=$RunLabel error=$($_.Exception.Message) log=$LauncherLog"
  Invoke-Notify -Title "Grids AI wait-then-Optuna failed" -Message $Message -Priority "high" -Tags "warning"
  throw
} finally {
  Stop-Transcript | Out-Null
}
