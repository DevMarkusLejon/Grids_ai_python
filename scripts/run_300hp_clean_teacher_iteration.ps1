param(
  [string]$BaselineModel = "checkpoints/value_model_torch_128_shaped_1000_300hp.json",
  [string]$ComparisonModel = "checkpoints/value_model_torch_192_shaped_1500_300hp_20260504-235049.json"
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$Stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$DataPath = "neural_data/selfplay_shaped_1200_clean_300hp_$Stamp.jsonl"
$ModelPath = "checkpoints/value_model_torch_128_shaped_1200_clean_300hp_$Stamp.json"
$ReportPath = "reports/gauntlet_300hp_128_shaped_1200_clean_$Stamp.json"

New-Item -ItemType Directory -Force -Path "neural_data", "checkpoints", "reports", "logs" | Out-Null

Write-Host "[$(Get-Date -Format o)] Starting cleaner 300 HP teacher iteration"
Write-Host "Dataset: $DataPath"
Write-Host "Model:   $ModelPath"
Write-Host "Report:  $ReportPath"

python -m grids_ai.neural generate `
  --target shaped `
  --exploration-rate 0.02 `
  --sampling-top-k 2 `
  --sampling-temperature 15 `
  --weights trained_weights.json `
  --games 1200 `
  --workers 5 `
  --search-width 4 `
  --search-depth 7 `
  --output $DataPath
if ($LASTEXITCODE -ne 0) { throw "Dataset generation failed with exit code $LASTEXITCODE." }

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
  --opponent-model $ComparisonModel `
  --output $ReportPath `
  --no-auto-opponents
if ($LASTEXITCODE -ne 0) { throw "Gauntlet failed with exit code $LASTEXITCODE." }

Write-Host "[$(Get-Date -Format o)] Finished cleaner 300 HP teacher iteration"
