param(
  [Parameter(Mandatory = $true)]
  [string]$HostSpec,

  [string]$RemoteDir = "~/grids-ai",

  [string]$KeyPath = ""
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$archive = Join-Path $env:TEMP ("grids-ai-cloud-" + (Get-Date -Format "yyyyMMdd-HHmmss") + ".tar.gz")

tar `
  --exclude=".git" `
  --exclude="__pycache__" `
  --exclude="*.pyc" `
  --exclude="training_logs" `
  --exclude="neural_data" `
  --exclude="web/assets" `
  -czf $archive `
  -C $root .

$sshArgs = @()
if ($KeyPath) {
  $sshArgs += @("-i", $KeyPath)
}

ssh @sshArgs $HostSpec "mkdir -p $RemoteDir"
scp @sshArgs $archive "${HostSpec}:$RemoteDir/grids-ai.tar.gz"
ssh @sshArgs $HostSpec "cd $RemoteDir && tar -xzf grids-ai.tar.gz && chmod +x cloud/run_neural_cloud.sh"

Write-Host "Uploaded to ${HostSpec}:$RemoteDir"
Write-Host "Start training with:"
Write-Host "  ssh $($sshArgs -join ' ') $HostSpec 'cd $RemoteDir && nohup bash cloud/run_neural_cloud.sh > cloud-run.nohup.log 2>&1 &'"
