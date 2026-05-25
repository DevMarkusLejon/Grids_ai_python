param(
  [double]$Hours = 8.0,
  [string]$Python = ".venv-gpu\Scripts\python.exe",
  [int[]]$HiddenSizes = @(384, 512, 768, 1024),
  [int]$Epochs = 90,
  [int]$BatchSize = 1024,
  [int]$PerDataLimit = 60000,
  [int]$EvalGames = 4,
  [int]$EvalWorkers = 4,
  [string]$ChampionModel = "checkpoints/value_model_torch_128_shaped_1000_300hp.json"
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

if (-not (Test-Path $Python)) {
  throw "GPU Python not found: $Python"
}

$TorchCheck = & $Python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no-cuda')"
if ($LASTEXITCODE -ne 0 -or $TorchCheck[0] -ne "True") {
  throw "CUDA PyTorch is not available in $Python. Output: $($TorchCheck -join '; ')"
}

$Data = @(
  "neural_data/overnight_policy_value_300hp_20260520-233904.jsonl",
  "neural_data/long_policy_value_collection_20260521-084951/batch_001.jsonl",
  "neural_data/long_policy_value_collection_20260521-084951/batch_002.jsonl",
  "neural_data/long_policy_value_collection_20260521-084951/batch_003.jsonl",
  "neural_data/long_policy_value_collection_20260521-084951/batch_004.jsonl",
  "neural_data/long_policy_value_collection_20260521-084951/batch_005.jsonl",
  "neural_data/selfplay_policy_value_128_blend_300hp_20260520-232410.jsonl",
  "neural_data/selfplay_neural_champion_80_shaped_300hp_20260520-211300.jsonl",
  "neural_data/selfplay_shaped_800_deterministic_300hp_20260505-014626.jsonl"
)

$Missing = $Data | Where-Object { -not (Test-Path $_) }
if ($Missing) {
  throw "Missing training data: $($Missing -join ', ')"
}

$Stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$RunLabel = "long_gpu_policy_value_training_$Stamp"
$LogPath = Join-Path "logs" "$RunLabel.log"
$ManifestPath = Join-Path "reports" "$RunLabel.manifest.json"

New-Item -ItemType Directory -Force -Path "checkpoints", "reports", "logs" | Out-Null
Start-Transcript -Path $LogPath | Out-Null

$StartedAt = Get-Date
$Deadline = $StartedAt.AddHours($Hours)
$Runs = @()
$Index = 0

try {
  Write-Host "Started long GPU policy/value training at $($StartedAt.ToString('o'))"
  Write-Host "Deadline: $($Deadline.ToString('o'))"
  Write-Host "GPU: $($TorchCheck[1])"
  Write-Host "Python: $Python"
  Write-Host "Data files: $($Data -join ', ')"

  while ((Get-Date) -lt $Deadline) {
    $HiddenSize = $HiddenSizes[$Index % $HiddenSizes.Count]
    $RunStamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $ModelPath = "checkpoints/policy_value_gpu_$($HiddenSize)_blend_$RunStamp.json"
    $ReportPath = "reports/gauntlet_policy_value_gpu_$($HiddenSize)_blend_$RunStamp.json"
    $RunStarted = Get-Date

    Write-Host "[$($RunStarted.ToString('o'))] Training hidden_size=$HiddenSize model=$ModelPath"
    $TrainArgs = @("-m", "grids_ai.neural", "train-policy")
    foreach ($DataPath in $Data) {
      $TrainArgs += @("--data", $DataPath)
    }
    $TrainArgs += @(
      "--per-data-limit", "$PerDataLimit",
      "--hidden-size", "$HiddenSize",
      "--batch-size", "$BatchSize",
      "--device", "cuda",
      "--validation-fraction", "0.1",
      "--early-stop-patience", "12",
      "--early-stop-min-delta", "0.0005",
      "--policy-loss-weight", "0.25",
      "--model", $ModelPath,
      "--epochs", "$Epochs"
    )
    & $Python @TrainArgs
    if ($LASTEXITCODE -ne 0) {
      throw "GPU policy/value training failed with exit code $LASTEXITCODE."
    }

    Write-Host "[$((Get-Date).ToString('o'))] Evaluating $ModelPath"
    & $Python -m grids_ai.neural champion `
      --candidate $ModelPath `
      --champion $ChampionModel `
      --games $EvalGames `
      --workers $EvalWorkers `
      --weights trained_weights.json `
      --neural-search-width 3 `
      --neural-search-depth 4 `
      --policy-scale 18 `
      --opponent-model checkpoints/policy_value_torch_192_blend_20260521-073435.json `
      --opponent-model checkpoints/policy_value_torch_128_blend_20260520-231637.json `
      --output $ReportPath
    if ($LASTEXITCODE -ne 0) {
      throw "GPU policy/value evaluation failed with exit code $LASTEXITCODE."
    }

    powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/update_model_registry.ps1
    if ($LASTEXITCODE -ne 0) {
      throw "Model registry refresh failed with exit code $LASTEXITCODE."
    }

    $Report = Get-Content $ReportPath -Raw | ConvertFrom-Json
    $Runs += [pscustomobject]@{
      hidden_size = $HiddenSize
      model = $ModelPath
      report = $ReportPath
      started_at = $RunStarted.ToString("o")
      finished_at = (Get-Date).ToString("o")
      overall_score = $Report.overall.score_rate
      wins = $Report.overall.wins
      losses = $Report.overall.losses
      head_to_head_score = $Report.champion_decision.head_to_head_score_rate
      promote = $Report.champion_decision.promote
    }
    $Index += 1
  }

  $FinishedAt = Get-Date
  $Manifest = [pscustomobject]@{
    started_at = $StartedAt.ToString("o")
    finished_at = $FinishedAt.ToString("o")
    target_hours = $Hours
    elapsed_hours = [math]::Round(($FinishedAt - $StartedAt).TotalHours, 3)
    python = $Python
    gpu = $TorchCheck[1]
    epochs = $Epochs
    batch_size = $BatchSize
    per_data_limit = $PerDataLimit
    eval_games_per_side = $EvalGames
    eval_workers = $EvalWorkers
    runs = $Runs
  }
  $Manifest | ConvertTo-Json -Depth 6 | Set-Content -Path $ManifestPath -Encoding UTF8
  $Best = $Runs | Sort-Object -Property overall_score, head_to_head_score -Descending | Select-Object -First 1
  $Message = "Long GPU training finished. runs=$($Runs.Count) best_overall=$($Best.overall_score) best_h2h=$($Best.head_to_head_score) best_model=$($Best.model) manifest=$ManifestPath"
  powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/notify_important.ps1 `
    -Title "Grids AI GPU training finished" `
    -Message $Message `
    -Priority high `
    -Tags "chart_with_upwards_trend"
  Write-Host $Message
} catch {
  $Message = "Long GPU training failed: $($_.Exception.Message) manifest=$ManifestPath"
  powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/notify_important.ps1 `
    -Title "Grids AI GPU training failed" `
    -Message $Message `
    -Priority high `
    -Tags "warning"
  throw
} finally {
  Stop-Transcript | Out-Null
}
