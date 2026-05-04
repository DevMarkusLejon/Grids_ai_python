$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$Stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$LogDir = Join-Path $Root "logs"
$DataPath = "neural_data/selfplay_shaped_1500_300hp_$Stamp.jsonl"
$ModelPath = "checkpoints/value_model_torch_192_shaped_1500_300hp_$Stamp.json"
$ReportPath = "reports/gauntlet_300hp_192_shaped_1500_$Stamp.json"

New-Item -ItemType Directory -Force -Path $LogDir, "neural_data", "checkpoints", "reports" | Out-Null

Write-Host "[$(Get-Date -Format o)] Starting 300 HP iteration"
Write-Host "Dataset: $DataPath"
Write-Host "Model:   $ModelPath"
Write-Host "Report:  $ReportPath"

python -m grids_ai.neural generate `
  --target shaped `
  --exploration-rate 0.05 `
  --sampling-top-k 4 `
  --sampling-temperature 30 `
  --weights trained_weights.json `
  --games 1500 `
  --workers 5 `
  --search-width 3 `
  --search-depth 6 `
  --output $DataPath

python -m grids_ai.neural train `
  --backend torch `
  --hidden-size 192 `
  --batch-size 512 `
  --validation-fraction 0.1 `
  --early-stop-patience 10 `
  --early-stop-min-delta 0.0005 `
  --data $DataPath `
  --model $ModelPath `
  --epochs 100

python -m grids_ai.neural gauntlet `
  --model $ModelPath `
  --games 6 `
  --weights trained_weights.json `
  --opponent-model checkpoints/value_model_torch_128_shaped_1000_300hp.json `
  --output $ReportPath `
  --no-auto-opponents

Write-Host "[$(Get-Date -Format o)] Finished 300 HP iteration"
