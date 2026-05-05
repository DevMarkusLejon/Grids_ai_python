param(
  [string]$DataPath = "neural_data/selfplay_shaped_1500_300hp_20260504-235049.jsonl",
  [string]$BaselineModel = "checkpoints/value_model_torch_128_shaped_1000_300hp.json",
  [string]$OpponentModel = "checkpoints/value_model_torch_192_shaped_1500_300hp_20260504-235049.json"
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$Stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$LogDir = Join-Path $Root "logs"
$ModelPath = "checkpoints/value_model_torch_128_shaped_1500_300hp_$Stamp.json"
$ReportPath = "reports/gauntlet_300hp_128_shaped_1500_$Stamp.json"
$NotifyScript = Join-Path $PSScriptRoot "notify_important.ps1"

New-Item -ItemType Directory -Force -Path $LogDir, "checkpoints", "reports" | Out-Null

Write-Host "[$(Get-Date -Format o)] Starting 300 HP follow-up training"
Write-Host "Dataset: $DataPath"
Write-Host "Model:   $ModelPath"
Write-Host "Report:  $ReportPath"

python -m grids_ai.neural train `
  --backend torch `
  --hidden-size 128 `
  --batch-size 512 `
  --validation-fraction 0.1 `
  --early-stop-patience 10 `
  --early-stop-min-delta 0.0005 `
  --data $DataPath `
  --model $ModelPath `
  --epochs 100
if ($LASTEXITCODE -ne 0) { throw "Model training failed with exit code $LASTEXITCODE." }

python -m grids_ai.neural gauntlet `
  --model $ModelPath `
  --games 8 `
  --weights trained_weights.json `
  --opponent-model $BaselineModel `
  --opponent-model $OpponentModel `
  --output $ReportPath `
  --no-auto-opponents
if ($LASTEXITCODE -ne 0) { throw "Gauntlet failed with exit code $LASTEXITCODE." }

Write-Host "[$(Get-Date -Format o)] Finished 300 HP follow-up training"

& $NotifyScript `
  -Title "Grids AI follow-up finished" `
  -Message "Finished 128-hidden follow-up on latest 300 HP data. Model: $ModelPath Report: $ReportPath" `
  -Priority high `
  -Tags "chart_with_upwards_trend"
