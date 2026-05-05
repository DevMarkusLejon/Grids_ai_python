param(
  [string]$DataPath = "neural_data/selfplay_shaped_1000_300hp_20260504.jsonl",
  [string]$ChampionModel = "checkpoints/value_model_torch_128_shaped_1000_300hp.json",
  [string]$CleanModel = "checkpoints/value_model_torch_128_shaped_1200_clean_300hp_20260505-002857.json",
  [int]$Seed = 20260505
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$Stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$ModelPath = "checkpoints/value_model_torch_128_shaped_1000_300hp_seed$Seed`_$Stamp.json"
$ReportPath = "reports/gauntlet_300hp_128_shaped_1000_seed$Seed`_$Stamp.json"

New-Item -ItemType Directory -Force -Path "checkpoints", "reports", "logs" | Out-Null

Write-Host "[$(Get-Date -Format o)] Starting champion-data seed retrain"
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
  --seed $Seed `
  --data $DataPath `
  --model $ModelPath `
  --epochs 100
if ($LASTEXITCODE -ne 0) { throw "Model training failed with exit code $LASTEXITCODE." }

python -m grids_ai.neural gauntlet `
  --model $ModelPath `
  --games 8 `
  --weights trained_weights.json `
  --opponent-model $ChampionModel `
  --opponent-model $CleanModel `
  --output $ReportPath `
  --no-auto-opponents
if ($LASTEXITCODE -ne 0) { throw "Gauntlet failed with exit code $LASTEXITCODE." }

Write-Host "[$(Get-Date -Format o)] Finished champion-data seed retrain"
