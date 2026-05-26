param(
  [double]$Hours = 8.0,
  [string]$Python = ".venv-gpu\Scripts\python.exe",
  [string]$ChampionPointer = "champion_model.txt",
  [string]$InitialCandidate = "checkpoints/policy_value_gpu_384_blend_20260522-000734.json",
  [int]$HiddenSize = 384,
  [int]$Epochs = 90,
  [int]$BatchSize = 1024,
  [int]$PerDataLimit = 60000,
  [int]$ConfirmGames = 32,
  [int]$EvalWorkers = 6,
  [int]$Seed = 20260525,
  [double]$PolicyScale = 18.0,
  [double]$MinHeadToHeadScore = 0.55,
  [double]$MinOverallScore = 0.60,
  [double]$MinHeadToHeadLowerBound = 0.50
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

if (-not (Test-Path $ChampionPointer)) {
  Set-Content -Path $ChampionPointer -Value "checkpoints/value_model_torch_128_shaped_1000_300hp.json" -Encoding ASCII
}

$ChampionModel = (Get-Content $ChampionPointer -Raw).Trim()
if (-not (Test-Path $ChampionModel)) {
  throw "Current champion model does not exist: $ChampionModel"
}

if (-not (Test-Path $Python)) {
  $Python = "python"
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

$MissingData = $Data | Where-Object { -not (Test-Path $_) }
if ($MissingData) {
  throw "Missing training data: $($MissingData -join ', ')"
}

$OpponentModels = @(
  "checkpoints/policy_value_torch_192_blend_20260521-073435.json"
) | Where-Object { Test-Path $_ }

New-Item -ItemType Directory -Force -Path "checkpoints", "reports", "logs" | Out-Null

$Stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$RunLabel = "champion_push_$Stamp"
$LogPath = Join-Path "logs" "$RunLabel.log"
$ManifestPath = Join-Path "reports" "$RunLabel.manifest.json"
$StartedAt = Get-Date
$Deadline = $StartedAt.AddHours($Hours)
$Attempts = @()

Start-Transcript -Path $LogPath | Out-Null

function Invoke-Notify {
  param(
    [string]$Title,
    [string]$Message,
    [string]$Priority = "high",
    [string]$Tags = "chart_with_upwards_trend"
  )
  $NotifyOutput = powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/notify_important.ps1 `
    -Title $Title `
    -Message $Message `
    -Priority $Priority `
    -Tags $Tags
  if ($LASTEXITCODE -ne 0) {
    throw "Notification failed with exit code $LASTEXITCODE. Output: $($NotifyOutput -join '; ')"
  }
  foreach ($Line in $NotifyOutput) {
    Write-Host $Line
  }
}

function Invoke-RegistryRefresh {
  param([string]$Champion)
  $RegistryOutput = powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/update_model_registry.ps1 -Champion $Champion
  if ($LASTEXITCODE -ne 0) {
    throw "Model registry refresh failed with exit code $LASTEXITCODE. Output: $($RegistryOutput -join '; ')"
  }
  foreach ($Line in $RegistryOutput) {
    Write-Host $Line
  }
}

function Invoke-ChampionGate {
  param(
    [string]$Candidate,
    [int]$AttemptIndex,
    [int]$GateSeed
  )

  if (-not (Test-Path $Candidate)) {
    throw "Candidate model does not exist: $Candidate"
  }

  $CandidateName = [System.IO.Path]::GetFileNameWithoutExtension($Candidate)
  $SafeName = $CandidateName -replace '[^A-Za-z0-9_.-]', '_'
  $ReportPath = "reports/champion_push_$($AttemptIndex)_$($SafeName)_$(Get-Date -Format 'yyyyMMdd-HHmmss').json"

  $Args = @(
    "-m", "grids_ai.neural", "champion",
    "--candidate", $Candidate,
    "--champion", $ChampionModel,
    "--games", "$ConfirmGames",
    "--seed", "$GateSeed",
    "--weights", "trained_weights.json",
    "--neural-search-width", "3",
    "--neural-search-depth", "4",
    "--policy-scale", "$PolicyScale",
    "--workers", "$EvalWorkers",
    "--only-neural-opponents",
    "--min-head-to-head-score", "$MinHeadToHeadScore",
    "--min-overall-score", "$MinOverallScore",
    "--min-head-to-head-lower-bound", "$MinHeadToHeadLowerBound",
    "--output", $ReportPath
  )
  foreach ($Model in $OpponentModels) {
    if ([System.IO.Path]::GetFullPath($Model) -ne [System.IO.Path]::GetFullPath($ChampionModel)) {
      $Args += @("--opponent-model", $Model)
    }
  }

  Write-Host "[$((Get-Date).ToString('o'))] Champion gate attempt=$AttemptIndex candidate=$Candidate report=$ReportPath"
  & $Python @Args
  if ($LASTEXITCODE -ne 0) {
    throw "Champion gate failed with exit code $LASTEXITCODE."
  }

  $Report = Get-Content $ReportPath -Raw | ConvertFrom-Json
  $Decision = $Report.champion_decision
  $Attempt = [pscustomobject]@{
    attempt = $AttemptIndex
    candidate = $Candidate
    report = $ReportPath
    seed = $GateSeed
    overall_score = $Report.overall.score_rate
    wins = $Report.overall.wins
    losses = $Report.overall.losses
    head_to_head_score = $Decision.head_to_head_score_rate
    head_to_head_games = $Decision.head_to_head_games
    head_to_head_lower_bound = $Decision.head_to_head_lower_bound
    promote = [bool]$Decision.promote
    finished_at = (Get-Date).ToString("o")
  }
  $script:Attempts += $Attempt

  if ($Attempt.promote) {
    Set-Content -Path $ChampionPointer -Value $Candidate -Encoding ASCII
    Invoke-RegistryRefresh -Champion $Candidate
    $Message = "New champion confirmed: $Candidate overall=$($Attempt.wins)-$($Attempt.losses) score=$([math]::Round($Attempt.overall_score, 3)) h2h=$([math]::Round($Attempt.head_to_head_score, 3)) lower=$([math]::Round($Attempt.head_to_head_lower_bound, 3)) report=$ReportPath"
    Invoke-Notify -Title "Grids AI new champion" -Message $Message -Priority "high" -Tags "trophy"
    Write-Host $Message
    return [bool]$true
  }

  Invoke-RegistryRefresh -Champion $ChampionModel
  Write-Host "[$((Get-Date).ToString('o'))] Candidate held. overall=$($Attempt.wins)-$($Attempt.losses) h2h=$($Attempt.head_to_head_score) lower=$($Attempt.head_to_head_lower_bound)"
  return [bool]$false
}

try {
  Write-Host "Started champion push at $($StartedAt.ToString('o'))"
  Write-Host "Deadline: $($Deadline.ToString('o'))"
  Write-Host "Champion: $ChampionModel"
  Write-Host "Python: $Python"
  Write-Host "Workers: $EvalWorkers"

  $AttemptIndex = 1
  if (Test-Path $InitialCandidate) {
    $Promoted = Invoke-ChampionGate -Candidate $InitialCandidate -AttemptIndex $AttemptIndex -GateSeed $Seed
    if ($Promoted -eq $true) {
      $FinishedAt = Get-Date
      $Manifest = [pscustomobject]@{
        started_at = $StartedAt.ToString("o")
        finished_at = $FinishedAt.ToString("o")
        elapsed_hours = [math]::Round(($FinishedAt - $StartedAt).TotalHours, 3)
        champion_before = $ChampionModel
        champion_after = (Get-Content $ChampionPointer -Raw).Trim()
        attempts = $Attempts
      }
      $Manifest | ConvertTo-Json -Depth 6 | Set-Content -Path $ManifestPath -Encoding UTF8
      exit 0
    }
    $AttemptIndex += 1
  }

  while ((Get-Date) -lt $Deadline) {
    $RunStamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $TrainSeed = $Seed + (1000 * $AttemptIndex)
    $ModelPath = "checkpoints/policy_value_gpu_$($HiddenSize)_champion_push_seed$($TrainSeed)_$RunStamp.json"

    Write-Host "[$((Get-Date).ToString('o'))] Training candidate attempt=$AttemptIndex seed=$TrainSeed model=$ModelPath"
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
      "--seed", "$TrainSeed",
      "--model", $ModelPath,
      "--epochs", "$Epochs"
    )
    & $Python @TrainArgs
    if ($LASTEXITCODE -ne 0) {
      throw "Policy/value training failed with exit code $LASTEXITCODE."
    }

    $Promoted = Invoke-ChampionGate -Candidate $ModelPath -AttemptIndex $AttemptIndex -GateSeed ($Seed + (50000 * $AttemptIndex))
    if ($Promoted -eq $true) {
      break
    }
    $AttemptIndex += 1
  }

  $FinishedAt = Get-Date
  $ChampionAfter = (Get-Content $ChampionPointer -Raw).Trim()
  $Best = $Attempts | Sort-Object -Property overall_score, head_to_head_score, head_to_head_lower_bound -Descending | Select-Object -First 1
  $Manifest = [pscustomobject]@{
    started_at = $StartedAt.ToString("o")
    finished_at = $FinishedAt.ToString("o")
    target_hours = $Hours
    elapsed_hours = [math]::Round(($FinishedAt - $StartedAt).TotalHours, 3)
    champion_before = $ChampionModel
    champion_after = $ChampionAfter
    attempts = $Attempts
  }
  $Manifest | ConvertTo-Json -Depth 6 | Set-Content -Path $ManifestPath -Encoding UTF8

  if ($ChampionAfter -eq $ChampionModel) {
    $Message = "Champion push ended without promotion. attempts=$($Attempts.Count) best=$($Best.candidate) overall=$($Best.wins)-$($Best.losses) h2h=$([math]::Round($Best.head_to_head_score, 3)) lower=$([math]::Round($Best.head_to_head_lower_bound, 3)) manifest=$ManifestPath"
    Invoke-Notify -Title "Grids AI champion push complete" -Message $Message -Priority "default" -Tags "mag"
    Write-Host $Message
  }
} catch {
  $Message = "Champion push failed: $($_.Exception.Message) log=$LogPath manifest=$ManifestPath"
  Invoke-Notify -Title "Grids AI champion push failed" -Message $Message -Priority "high" -Tags "warning"
  throw
} finally {
  Stop-Transcript | Out-Null
}
