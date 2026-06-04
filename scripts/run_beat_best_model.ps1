param(
  [double]$Hours = 8.0,
  [string]$Python = ".venv-gpu\Scripts\python.exe",
  [string]$ReferenceModel = "checkpoints/policy_value_torch_192_blend_20260521-073435.json",
  [object[]]$HiddenSizes = @("384", "512", "768"),
  [string]$ExistingTeacherData = "",
  [int]$TeacherGames = 240,
  [int]$GenerateWorkers = 6,
  [int]$EvalWorkers = 6,
  [int]$GateGames = 64,
  [int]$Epochs = 90,
  [int]$BatchSize = 1024,
  [int]$PerDataLimit = 70000,
  [int]$Seed = 20260602,
  [double]$PolicyScale = 18.0,
  [double]$MinHeadToHeadScore = 0.55,
  [double]$MinHeadToHeadLowerBound = 0.50,
  [switch]$ValidateOnly
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

if (-not (Test-Path $ReferenceModel)) {
  throw "Reference model not found: $ReferenceModel"
}

if (-not (Test-Path $Python)) {
  $Python = "python"
}

$HiddenSizes = @(
  $HiddenSizes |
    ForEach-Object { "$_" -split "," } |
    ForEach-Object { $_.Trim() } |
    Where-Object { $_.Length -gt 0 } |
    ForEach-Object { [int]$_ }
)
if ($HiddenSizes.Count -eq 0) {
  throw "At least one hidden size is required."
}
foreach ($HiddenSize in $HiddenSizes) {
  if ($HiddenSize -lt 16 -or $HiddenSize -gt 4096) {
    throw "Hidden size looks invalid: $HiddenSize"
  }
}
if ($ValidateOnly) {
  Write-Host "Validation OK. HiddenSizes=$($HiddenSizes -join ',') ExistingTeacherData=$ExistingTeacherData"
  exit 0
}

$Stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$RunLabel = "beat_best_$Stamp"
$LogPath = Join-Path "logs" "$RunLabel.log"
$ManifestPath = Join-Path "reports" "$RunLabel.manifest.json"
$StartedAt = Get-Date
$Deadline = $StartedAt.AddHours($Hours)
$Attempts = @()

New-Item -ItemType Directory -Force -Path "neural_data", "checkpoints", "reports", "logs" | Out-Null
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

function Get-ReportAttempt {
  param(
    [string]$Candidate,
    [string]$ReportPath,
    [int]$AttemptIndex,
    [int]$HiddenSize
  )
  $Report = Get-Content $ReportPath -Raw | ConvertFrom-Json
  $Decision = $Report.champion_decision
  [pscustomobject]@{
    attempt = $AttemptIndex
    hidden_size = $HiddenSize
    candidate = $Candidate
    report = $ReportPath
    overall_score = $Report.overall.score_rate
    wins = $Report.overall.wins
    losses = $Report.overall.losses
    head_to_head_score = $Decision.head_to_head_score_rate
    head_to_head_games = $Decision.head_to_head_games
    head_to_head_lower_bound = $Decision.head_to_head_lower_bound
    promote = [bool]$Decision.promote
    finished_at = (Get-Date).ToString("o")
  }
}

try {
  Write-Host "Started beat-best run at $($StartedAt.ToString('o'))"
  Write-Host "Deadline: $($Deadline.ToString('o'))"
  Write-Host "Reference model: $ReferenceModel"
  Write-Host "Python: $Python"

  if ($ExistingTeacherData) {
    if (-not (Test-Path $ExistingTeacherData)) {
      throw "Existing teacher data not found: $ExistingTeacherData"
    }
    $TeacherData = $ExistingTeacherData
    Write-Host "[$((Get-Date).ToString('o'))] Reusing 192-teacher data: $TeacherData"
  } else {
    $TeacherData = "neural_data/teacher_192_policy_value_$Stamp.jsonl"
    Write-Host "[$((Get-Date).ToString('o'))] Generating 192-teacher data: $TeacherData"
    & $Python -m grids_ai.neural generate `
      --teacher neural `
      --teacher-model $ReferenceModel `
      --teacher-neural-search-width 3 `
      --teacher-neural-search-depth 4 `
      --target shaped `
      --weights trained_weights.json `
      --games $TeacherGames `
      --workers $GenerateWorkers `
      --sample-every 1 `
      --max-examples-per-game 120 `
      --output $TeacherData
    if ($LASTEXITCODE -ne 0) {
      throw "192-teacher data generation failed with exit code $LASTEXITCODE."
    }
  }

  $Data = @(
    $TeacherData,
    "neural_data/overnight_policy_value_300hp_20260520-233904.jsonl",
    "neural_data/long_policy_value_collection_20260521-084951/batch_001.jsonl",
    "neural_data/long_policy_value_collection_20260521-084951/batch_002.jsonl",
    "neural_data/long_policy_value_collection_20260521-084951/batch_003.jsonl",
    "neural_data/long_policy_value_collection_20260521-084951/batch_004.jsonl",
    "neural_data/long_policy_value_collection_20260521-084951/batch_005.jsonl",
    "neural_data/selfplay_policy_value_128_blend_300hp_20260520-232410.jsonl"
  ) | Where-Object { Test-Path $_ }

  $AttemptIndex = 1
  while ((Get-Date) -lt $Deadline) {
    $HiddenSize = $HiddenSizes[($AttemptIndex - 1) % $HiddenSizes.Count]
    $RunStamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $TrainSeed = $Seed + ($AttemptIndex * 1009)
    $GateSeed = $Seed + ($AttemptIndex * 50021)
    $ModelPath = "checkpoints/policy_value_gpu_$($HiddenSize)_beat_192_seed$($TrainSeed)_$RunStamp.json"
    $ReportPath = "reports/beat_192_$($AttemptIndex)_policy_value_gpu_$($HiddenSize)_seed$($TrainSeed)_$RunStamp.json"

    Write-Host "[$((Get-Date).ToString('o'))] Training attempt=$AttemptIndex hidden=$HiddenSize seed=$TrainSeed"
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
      "--policy-loss-weight", "0.35",
      "--seed", "$TrainSeed",
      "--model", $ModelPath,
      "--epochs", "$Epochs"
    )
    & $Python @TrainArgs
    if ($LASTEXITCODE -ne 0) {
      throw "Policy/value training failed with exit code $LASTEXITCODE."
    }

    Write-Host "[$((Get-Date).ToString('o'))] Evaluating against reference model"
    & $Python -m grids_ai.neural champion `
      --candidate $ModelPath `
      --champion $ReferenceModel `
      --games $GateGames `
      --seed $GateSeed `
      --weights trained_weights.json `
      --neural-search-width 3 `
      --neural-search-depth 4 `
      --policy-scale $PolicyScale `
      --workers $EvalWorkers `
      --only-neural-opponents `
      --min-head-to-head-score $MinHeadToHeadScore `
      --min-overall-score $MinHeadToHeadScore `
      --min-head-to-head-lower-bound $MinHeadToHeadLowerBound `
      --output $ReportPath
    if ($LASTEXITCODE -ne 0) {
      throw "Reference gate failed with exit code $LASTEXITCODE."
    }

    $Attempt = Get-ReportAttempt -Candidate $ModelPath -ReportPath $ReportPath -AttemptIndex $AttemptIndex -HiddenSize $HiddenSize
    $Attempts += $Attempt
    Write-Host "[$((Get-Date).ToString('o'))] attempt=$AttemptIndex score=$($Attempt.head_to_head_score) lower=$($Attempt.head_to_head_lower_bound) promote=$($Attempt.promote)"

    if ($Attempt.promote) {
      Set-Content -Path "champion_model.txt" -Value $ModelPath -Encoding ASCII
      Set-Content -Path "scripts/play_strongest.cmd" -Value @(
        "@echo off",
        "cd /d ""%~dp0\..""",
        "python -m grids_ai.cli --blue human --red neural --model $ModelPath --policy-scale $PolicyScale --neural-search-width 3 --neural-search-depth 4 %*"
      ) -Encoding ASCII
      Invoke-RegistryRefresh -Champion $ModelPath
      $Message = "New best model beat 192 reference: $ModelPath score=$([math]::Round($Attempt.head_to_head_score, 3)) lower=$([math]::Round($Attempt.head_to_head_lower_bound, 3)) report=$ReportPath"
      Invoke-Notify -Title "Grids AI new best model" -Message $Message -Priority "high" -Tags "trophy"
      break
    }

    $AttemptIndex += 1
  }

  $FinishedAt = Get-Date
  $Best = $Attempts | Sort-Object -Property head_to_head_score, head_to_head_lower_bound -Descending | Select-Object -First 1
  $Manifest = [pscustomobject]@{
    started_at = $StartedAt.ToString("o")
    finished_at = $FinishedAt.ToString("o")
    target_hours = $Hours
    elapsed_hours = [math]::Round(($FinishedAt - $StartedAt).TotalHours, 3)
    reference_model = $ReferenceModel
    teacher_data = $TeacherData
    attempts = $Attempts
  }
  $Manifest | ConvertTo-Json -Depth 6 | Set-Content -Path $ManifestPath -Encoding UTF8

  if (-not ($Attempts | Where-Object { $_.promote })) {
    $Message = "Beat-best run ended without beating 192. attempts=$($Attempts.Count) best=$($Best.candidate) score=$([math]::Round($Best.head_to_head_score, 3)) lower=$([math]::Round($Best.head_to_head_lower_bound, 3)) manifest=$ManifestPath"
    Invoke-Notify -Title "Grids AI beat-best run complete" -Message $Message -Priority "default" -Tags "mag"
    Write-Host $Message
  }
} catch {
  $Message = "Beat-best run failed: $($_.Exception.Message) log=$LogPath manifest=$ManifestPath"
  Invoke-Notify -Title "Grids AI beat-best failed" -Message $Message -Priority "high" -Tags "warning"
  throw
} finally {
  Stop-Transcript | Out-Null
}
