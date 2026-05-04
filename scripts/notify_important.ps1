param(
  [string]$Title = "Grids AI",
  [string]$Message = "Important Grids AI update",
  [ValidateSet("min", "low", "default", "high", "urgent")]
  [string]$Priority = "high",
  [string]$Tags = "warning",
  [switch]$Test
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
$LocalConfig = Join-Path $Root "local_notify_config.ps1"
$LogDir = Join-Path $Root "logs"
$LogPath = Join-Path $LogDir "important_notifications.log"

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

if (Test-Path $LocalConfig) {
  . $LocalConfig
}

if ($Test) {
  $Title = "Grids AI notification test"
  $Message = "If this reached your phone, important overnight alerts are wired correctly."
  $Priority = "default"
  $Tags = "white_check_mark"
}

$line = "[$(Get-Date -Format o)] priority=$Priority title=$Title message=$Message"
Add-Content -Path $LogPath -Value $line

$notifyUrl = $env:GRIDS_NOTIFY_URL
if (-not $notifyUrl) {
  Write-Host "No GRIDS_NOTIFY_URL configured. Logged locally to $LogPath."
  Write-Host "Set GRIDS_NOTIFY_URL in local_notify_config.ps1 or as an environment variable to enable phone push."
  exit 0
}

$headers = @{
  Title = $Title
  Priority = $Priority
  Tags = $Tags
}

try {
  Invoke-RestMethod -Method Post -Uri $notifyUrl -Headers $headers -Body $Message | Out-Null
  Write-Host "Notification sent."
} catch {
  Add-Content -Path $LogPath -Value "[$(Get-Date -Format o)] send_failed=$($_.Exception.Message)"
  throw
}
