param(
  [Parameter(Mandatory = $true)]
  [string]$HostSpec,

  [string]$RemoteDir = "~/grids-ai",

  [string]$KeyPath = "",

  [string]$LocalDir = "cloud_results"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$target = Join-Path $root $LocalDir
New-Item -ItemType Directory -Force -Path $target | Out-Null

$sshArgs = @()
if ($KeyPath) {
  $sshArgs += @("-i", $KeyPath)
}

scp @sshArgs -r "${HostSpec}:$RemoteDir/checkpoints" $target
scp @sshArgs -r "${HostSpec}:$RemoteDir/neural_data" $target
scp @sshArgs -r "${HostSpec}:$RemoteDir/training_logs" $target

Write-Host "Downloaded cloud artifacts to $target"
