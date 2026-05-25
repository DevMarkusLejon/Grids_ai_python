param(
  [string]$ChampionModel = "checkpoints/value_model_torch_128_shaped_1000_300hp.json",
  [string]$OvernightData = "neural_data/overnight_policy_value_300hp_20260520-233904.jsonl",
  [int]$HiddenSize = 192,
  [int]$Epochs = 24,
  [int]$BatchSize = 512,
  [int]$PerDataLimit = 80000,
  [int]$GateGames = 2,
  [int]$EvalWorkers = 4,
  [double]$PolicyScale = 18.0
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$Data = @(
  $OvernightData,
  "neural_data/selfplay_policy_value_128_blend_300hp_20260520-232410.jsonl",
  "neural_data/selfplay_neural_champion_80_shaped_300hp_20260520-211300.jsonl",
  "neural_data/selfplay_shaped_800_deterministic_300hp_20260505-014626.jsonl"
)

& scripts\run_policy_value_iteration.ps1 `
  -ChampionModel $ChampionModel `
  -Data $Data `
  -HiddenSize $HiddenSize `
  -Epochs $Epochs `
  -BatchSize $BatchSize `
  -PerDataLimit $PerDataLimit `
  -GateGames $GateGames `
  -EvalWorkers $EvalWorkers `
  -PolicyScale $PolicyScale

if ($LASTEXITCODE -ne 0) {
  throw "Overnight policy/value iteration failed with exit code $LASTEXITCODE."
}
