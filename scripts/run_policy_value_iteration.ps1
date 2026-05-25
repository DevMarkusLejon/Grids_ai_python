param(
  [string]$ChampionModel = "checkpoints/value_model_torch_128_shaped_1000_300hp.json",
  [string[]]$Data = @(
    "neural_data/selfplay_neural_champion_80_shaped_300hp_20260520-211300.jsonl",
    "neural_data/selfplay_neural_champion_40_shaped_300hp_20260520-131138.jsonl",
    "neural_data/selfplay_shaped_800_deterministic_300hp_20260505-014626.jsonl"
  ),
  [int]$HiddenSize = 128,
  [int]$Epochs = 18,
  [int]$BatchSize = 512,
  [int]$PerDataLimit = 8000,
  [int]$GateGames = 2,
  [int]$EvalWorkers = 1,
  [double]$PolicyScale = 18.0,
  [switch]$UseMcts,
  [int]$MctsSimulations = 24,
  [int]$MctsMaxChildren = 16,
  [int]$MctsDepth = 5
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$Stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$ModelPath = "checkpoints/policy_value_torch_$($HiddenSize)_blend_$Stamp.json"
$ReportSuffix = if ($UseMcts) { "mcts$MctsSimulations" } else { "beam" }
$ReportPath = "reports/gauntlet_policy_value_$($HiddenSize)_blend_$($ReportSuffix)_$Stamp.json"
$LogPath = "logs/run_policy_value_iteration_$Stamp.log"

New-Item -ItemType Directory -Force -Path "checkpoints", "reports", "logs" | Out-Null
Start-Transcript -Path $LogPath | Out-Null

try {
  Write-Host "[$(Get-Date -Format o)] Starting policy/value iteration"
  Write-Host "Model:  $ModelPath"
  Write-Host "Report: $ReportPath"
  Write-Host "Data:   $($Data -join ', ')"

  $trainArgs = @("-m", "grids_ai.neural", "train-policy")
  foreach ($DataPath in $Data) {
    $trainArgs += @("--data", $DataPath)
  }
  $trainArgs += @(
    "--per-data-limit", "$PerDataLimit",
    "--hidden-size", "$HiddenSize",
    "--batch-size", "$BatchSize",
    "--validation-fraction", "0.1",
    "--early-stop-patience", "5",
    "--early-stop-min-delta", "0.0005",
    "--policy-loss-weight", "0.25",
    "--model", $ModelPath,
    "--epochs", "$Epochs"
  )
  python @trainArgs
  if ($LASTEXITCODE -ne 0) { throw "Policy/value training failed with exit code $LASTEXITCODE." }

  $SearchWidth = if ($UseMcts) { "1" } else { "3" }
  $SearchDepth = if ($UseMcts) { "1" } else { "4" }

  $gauntletArgs = @(
    "-m", "grids_ai.neural", "gauntlet",
    "--model", $ModelPath,
    "--games", "$GateGames",
    "--weights", "trained_weights.json",
    "--no-auto-opponents",
    "--opponent-model", $ChampionModel,
    "--neural-search-width", $SearchWidth,
    "--neural-search-depth", $SearchDepth,
    "--policy-scale", "$PolicyScale",
    "--workers", "$EvalWorkers",
    "--output", $ReportPath
  )
  if ($UseMcts) {
    $gauntletArgs += @(
      "--mcts-simulations", "$MctsSimulations",
      "--mcts-max-children", "$MctsMaxChildren",
      "--mcts-depth", "$MctsDepth"
    )
  }
  python @gauntletArgs
  if ($LASTEXITCODE -ne 0) { throw "Policy/value gauntlet failed with exit code $LASTEXITCODE." }

  powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/update_model_registry.ps1
  if ($LASTEXITCODE -ne 0) { throw "Model registry refresh failed with exit code $LASTEXITCODE." }

  Write-Host "[$(Get-Date -Format o)] Finished policy/value iteration"
} finally {
  Stop-Transcript | Out-Null
}
