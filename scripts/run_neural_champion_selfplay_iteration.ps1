param(
  [string]$ChampionModel = "checkpoints/value_model_torch_128_shaped_1000_300hp.json",
  [int]$Games = 200,
  [int]$Workers = 1,
  [int]$HiddenSize = 128,
  [int]$Epochs = 40,
  [int]$BatchSize = 512,
  [int]$TeacherSearchWidth = 3,
  [int]$TeacherSearchDepth = 4,
  [int]$GateGames = 4,
  [int]$Seed = 20260520,
  [ValidateSet("outcome", "margin", "shaped")]
  [string]$Target = "shaped"
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$Stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$DataPath = "neural_data/selfplay_neural_champion_$($Games)_$($Target)_300hp_$Stamp.jsonl"
$ModelPath = "checkpoints/value_model_torch_$($HiddenSize)_neural_champion_$($Games)_$($Target)_300hp_$Stamp.json"
$ReportPath = "reports/champion_gate_neural_champion_$($HiddenSize)_$($Games)_$($Target)_$Stamp.json"
$LogPath = "logs/run_neural_champion_selfplay_$Stamp.log"

New-Item -ItemType Directory -Force -Path "neural_data", "checkpoints", "reports", "logs" | Out-Null
Start-Transcript -Path $LogPath | Out-Null

try {
  Write-Host "[$(Get-Date -Format o)] Starting neural champion self-play iteration"
  Write-Host "Champion: $ChampionModel"
  Write-Host "Target:   $Target"
  Write-Host "Dataset:  $DataPath"
  Write-Host "Model:    $ModelPath"
  Write-Host "Report:   $ReportPath"

  python -m grids_ai.neural generate `
    --teacher neural `
    --teacher-model $ChampionModel `
    --teacher-neural-search-width $TeacherSearchWidth `
    --teacher-neural-search-depth $TeacherSearchDepth `
    --target $Target `
    --weights trained_weights.json `
    --games $Games `
    --workers $Workers `
    --sample-every 2 `
    --max-examples-per-game 80 `
    --output $DataPath
  if ($LASTEXITCODE -ne 0) { throw "Neural self-play generation failed with exit code $LASTEXITCODE." }

  python -m grids_ai.neural train `
    --backend torch `
    --hidden-size $HiddenSize `
    --batch-size $BatchSize `
    --validation-fraction 0.1 `
    --early-stop-patience 8 `
    --early-stop-min-delta 0.0005 `
    --data $DataPath `
    --model $ModelPath `
    --epochs $Epochs
  if ($LASTEXITCODE -ne 0) { throw "Model training failed with exit code $LASTEXITCODE." }

  python -m grids_ai.neural champion `
    --candidate $ModelPath `
    --champion $ChampionModel `
    --games $GateGames `
    --weights trained_weights.json `
    --neural-search-width $TeacherSearchWidth `
    --neural-search-depth $TeacherSearchDepth `
    --output $ReportPath
  if ($LASTEXITCODE -ne 0) { throw "Champion gate failed with exit code $LASTEXITCODE." }

  Write-Host "[$(Get-Date -Format o)] Finished neural champion self-play iteration"
} finally {
  Stop-Transcript | Out-Null
}
