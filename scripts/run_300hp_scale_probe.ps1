param(
  [double]$NeuralScale = 180.0,
  [double]$HeuristicScale = 1.0,
  [string]$ModelPath = "checkpoints/value_model_torch_128_shaped_1000_300hp.json",
  [string]$OpponentModel = "checkpoints/value_model_torch_128_shaped_1200_clean_300hp_20260505-002857.json"
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$Stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$ReportPath = "reports/gauntlet_300hp_champion_scale$NeuralScale`_$Stamp.json"

New-Item -ItemType Directory -Force -Path "reports", "logs" | Out-Null

Write-Host "[$(Get-Date -Format o)] Starting neural-scale probe"
Write-Host "Model:          $ModelPath"
Write-Host "Neural scale:   $NeuralScale"
Write-Host "Heuristic scale:$HeuristicScale"
Write-Host "Report:         $ReportPath"

python -m grids_ai.neural gauntlet `
  --model $ModelPath `
  --games 6 `
  --weights trained_weights.json `
  --opponent-model $OpponentModel `
  --neural-scale $NeuralScale `
  --heuristic-scale $HeuristicScale `
  --output $ReportPath `
  --no-auto-opponents
if ($LASTEXITCODE -ne 0) { throw "Gauntlet failed with exit code $LASTEXITCODE." }

Write-Host "[$(Get-Date -Format o)] Finished neural-scale probe"
