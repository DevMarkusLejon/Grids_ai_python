param(
  [int]$WaitForPid = 0,
  [double]$Hours = 8.0,
  [string]$Python = ".venv-gpu\Scripts\python.exe",
  [string]$ReferenceModel = "checkpoints/policy_value_torch_192_blend_20260521-073435.json",
  [string]$TeacherData = "neural_data/teacher_192_policy_value_20260602-234805.jsonl",
  [int]$Workers = 0,
  [int]$ScreenGames = 48,
  [int]$FullGateGames = 192,
  [double]$MinScreenScore = 0.52,
  [double]$PolicyScale = 18.0,
  [string]$ExistingFreshData = "",
  [switch]$SkipCalibration
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

if (-not (Test-Path $Python)) {
  $Python = "python"
}
if (-not (Test-Path $ReferenceModel)) {
  throw "Reference model not found: $ReferenceModel"
}
if (-not (Test-Path $TeacherData)) {
  throw "Teacher data not found: $TeacherData"
}
if ($Workers -lt 1) {
  $Logical = [int](Get-CimInstance Win32_ComputerSystem).NumberOfLogicalProcessors
  $Workers = [Math]::Max(1, [Math]::Min($Logical, 8))
}

$Stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$RunLabel = "new_champion_supervisor_$Stamp"
$LogPath = Join-Path "logs" "$RunLabel.log"
$ManifestPath = Join-Path "reports" "$RunLabel.manifest.json"
$StartedAt = Get-Date
$Deadline = $StartedAt.AddHours($Hours)
$Attempts = @()

New-Item -ItemType Directory -Force -Path "checkpoints", "reports", "logs", "neural_data" | Out-Null
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

function Get-Decision {
  param([string]$ReportPath)
  $Report = Get-Content $ReportPath -Raw | ConvertFrom-Json
  return $Report.champion_decision
}

function Confirm-Promotion {
  param(
    [string]$Candidate,
    [string]$Report,
    [object]$Decision
  )
  if (-not [bool]$Decision.promote) {
    return $false
  }
  $AlreadyChampion = $false
  if (Test-Path "champion_model.txt") {
    $AlreadyChampion = ((Get-Content "champion_model.txt" -Raw).Trim() -eq $Candidate)
  }
  $AlreadyNotified = $false
  if ($AlreadyChampion -and (Test-Path "logs/important_notifications.log")) {
    $AlreadyNotified = (Select-String -Path "logs/important_notifications.log" -SimpleMatch -Pattern $Candidate -Quiet)
  }
  Set-Content -Path "champion_model.txt" -Value $Candidate -Encoding ASCII
  Set-Content -Path "scripts/play_strongest.cmd" -Value @(
    "@echo off",
    "cd /d ""%~dp0\..""",
    "python -m grids_ai.cli --blue human --red neural --model $Candidate --policy-scale $PolicyScale --neural-search-width 3 --neural-search-depth 4 %*"
  ) -Encoding ASCII
  Invoke-RegistryRefresh -Champion $Candidate
  $Message = "New Grids AI champion: $Candidate score=$([Math]::Round([double]$Decision.head_to_head_score_rate, 3)) lower=$([Math]::Round([double]$Decision.head_to_head_lower_bound, 3)) report=$Report"
  if ($AlreadyChampion -and $AlreadyNotified) {
    Write-Host "Promotion was already applied and notified; skipping duplicate phone notification."
  } else {
    Invoke-Notify -Title "Grids AI new champion" -Message $Message -Priority "high" -Tags "trophy"
  }
  Write-Host $Message
  return $true
}

function Find-ExistingPromotion {
  $Reports = Get-ChildItem reports -Filter "*.json" -File -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending
  foreach ($Report in $Reports) {
    try {
      $Payload = Get-Content $Report.FullName -Raw | ConvertFrom-Json
      $Decision = $Payload.champion_decision
      if ($null -eq $Decision) {
        continue
      }
      if ([bool]$Decision.promote -and [string]$Decision.champion_model -eq $ReferenceModel) {
        return [pscustomobject]@{
          candidate = [string]$Decision.candidate_model
          report = $Report.FullName
          decision = $Decision
        }
      }
    } catch {
      continue
    }
  }
  return $null
}

try {
  Write-Host "Started new champion supervisor at $($StartedAt.ToString('o'))"
  Write-Host "Deadline: $($Deadline.ToString('o'))"
  Write-Host "Reference model: $ReferenceModel"
  Write-Host "Workers: $Workers"
  Write-Host "Python: $Python"

  if ($WaitForPid -gt 0) {
    $Existing = Get-Process -Id $WaitForPid -ErrorAction SilentlyContinue
    if ($Existing) {
      Write-Host "Waiting for existing run PID $WaitForPid to exit."
      Wait-Process -Id $WaitForPid
      Write-Host "Existing run PID $WaitForPid exited at $((Get-Date).ToString('o'))."
    }
  }

  $ExistingPromotion = Find-ExistingPromotion
  if ($ExistingPromotion) {
    Write-Host "Found existing promotion report: $($ExistingPromotion.report)"
    if (Confirm-Promotion -Candidate $ExistingPromotion.candidate -Report $ExistingPromotion.report -Decision $ExistingPromotion.decision) {
      return
    }
  }

  $CalibrationReport = $null
  if ($SkipCalibration) {
    Write-Host "[$((Get-Date).ToString('o'))] Skipping unchanged-copy calibration by request."
  } else {
    $CalibrationCandidate = "checkpoints/policy_value_torch_192_blend_calibration_$Stamp.json"
    $CalibrationReport = "reports/calibration_192_copy_vs_reference_$Stamp.json"
    Copy-Item -LiteralPath $ReferenceModel -Destination $CalibrationCandidate -Force
    Write-Host "[$((Get-Date).ToString('o'))] Running unchanged-copy calibration."
    & $Python -m grids_ai.neural champion `
      --candidate $CalibrationCandidate `
      --champion $ReferenceModel `
      --games 48 `
      --seed 20270603 `
      --weights trained_weights.json `
      --neural-search-width 3 `
      --neural-search-depth 4 `
      --policy-scale $PolicyScale `
      --workers $Workers `
      --only-neural-opponents `
      --min-head-to-head-score 0.55 `
      --min-overall-score 0.55 `
      --min-head-to-head-lower-bound 0 `
      --output $CalibrationReport
    if ($LASTEXITCODE -ne 0) {
      throw "Calibration champion run failed with exit code $LASTEXITCODE."
    }
    $CalibrationDecision = Get-Decision -ReportPath $CalibrationReport
    Write-Host "[$((Get-Date).ToString('o'))] calibration score=$($CalibrationDecision.head_to_head_score_rate) lower=$($CalibrationDecision.head_to_head_lower_bound)"
  }

  if ((Get-Date) -ge $Deadline) {
    throw "Deadline reached after calibration; no follow-up training started."
  }

  if ($ExistingFreshData) {
    if (-not (Test-Path $ExistingFreshData)) {
      throw "Existing fresh data not found: $ExistingFreshData"
    }
    $FreshData = $ExistingFreshData
    Write-Host "[$((Get-Date).ToString('o'))] Reusing existing fresh neural-teacher data: $FreshData"
  } else {
    $FreshData = "neural_data/selfplay_192_teacher_refresh_$Stamp.jsonl"
    Write-Host "[$((Get-Date).ToString('o'))] Generating fresh neural-teacher self-play data: $FreshData"
    & $Python -m grids_ai.neural generate `
      --output $FreshData `
      --games 160 `
      --seed 20270631 `
      --weights trained_weights.json `
      --teacher neural `
      --teacher-model $ReferenceModel `
      --teacher-neural-search-width 3 `
      --teacher-neural-search-depth 4 `
      --target shaped `
      --sample-every 2 `
      --max-examples-per-game 90 `
      --exploration-rate 0.04 `
      --sampling-top-k 3 `
      --sampling-temperature 0.30 `
      --workers $Workers
    if ($LASTEXITCODE -ne 0) {
      throw "Fresh self-play generation failed with exit code $LASTEXITCODE."
    }
  }

  $Data = @(
    $FreshData,
    $FreshData,
    $FreshData,
    $TeacherData,
    "neural_data/overnight_policy_value_300hp_20260520-233904.jsonl",
    "neural_data/long_policy_value_collection_20260521-084951/batch_001.jsonl",
    "neural_data/long_policy_value_collection_20260521-084951/batch_002.jsonl",
    "neural_data/long_policy_value_collection_20260521-084951/batch_003.jsonl",
    "neural_data/long_policy_value_collection_20260521-084951/batch_004.jsonl",
    "neural_data/long_policy_value_collection_20260521-084951/batch_005.jsonl",
    "neural_data/selfplay_policy_value_128_blend_300hp_20260520-232410.jsonl"
  ) | Where-Object { Test-Path $_ }

  $Configs = @(
    [pscustomobject]@{ hidden = 384; policy = 0.0; lr = 0.000001; epochs = 4; limit = 60000; patience = 2 },
    [pscustomobject]@{ hidden = 384; policy = 0.001; lr = 0.000001; epochs = 4; limit = 60000; patience = 2 },
    [pscustomobject]@{ hidden = 384; policy = 0.0; lr = 0.000003; epochs = 5; limit = 60000; patience = 2 },
    [pscustomobject]@{ hidden = 512; policy = 0.0; lr = 0.000001; epochs = 4; limit = 60000; patience = 2 },
    [pscustomobject]@{ hidden = 512; policy = 0.001; lr = 0.000001; epochs = 4; limit = 60000; patience = 2 },
    [pscustomobject]@{ hidden = 384; policy = 0.005; lr = 0.000001; epochs = 6; limit = 90000; patience = 2 }
  )

  $AttemptIndex = 1
  while ((Get-Date) -lt $Deadline) {
    $Config = $Configs[($AttemptIndex - 1) % $Configs.Count]
    $RunStamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $TrainSeed = 20270603 + ($AttemptIndex * 7919)
    $PolicyTag = ("p{0:0.###}" -f $Config.policy).Replace(".", "p")
    $LrTag = ("lr{0:0.#####}" -f $Config.lr).Replace(".", "p")
    $ModelPath = "checkpoints/policy_value_$($Config.hidden)_widen_refresh_$($PolicyTag)_$($LrTag)_seed$($TrainSeed)_$RunStamp.json"
    $ScreenReport = "reports/refresh_$($Config.hidden)_screen_$($AttemptIndex)_$($PolicyTag)_$($LrTag)_seed$($TrainSeed)_$RunStamp.json"
    $FullReport = "reports/refresh_$($Config.hidden)_full_$($AttemptIndex)_$($PolicyTag)_$($LrTag)_seed$($TrainSeed)_$RunStamp.json"

    Write-Host "[$((Get-Date).ToString('o'))] Training widened refresh attempt=$AttemptIndex hidden=$($Config.hidden) policy=$($Config.policy) lr=$($Config.lr) epochs=$($Config.epochs) seed=$TrainSeed"
    $TrainArgs = @("-m", "grids_ai.neural", "train-policy")
    foreach ($DataPath in $Data) {
      $TrainArgs += @("--data", $DataPath)
    }
    $TrainArgs += @(
      "--init-model", $ReferenceModel,
      "--freeze-init-model",
      "--per-data-limit", "$($Config.limit)",
      "--hidden-size", "$($Config.hidden)",
      "--batch-size", "1024",
      "--device", "cuda",
      "--validation-fraction", "0.1",
      "--early-stop-patience", "$($Config.patience)",
      "--learning-rate", "$($Config.lr)",
      "--policy-loss-weight", "$($Config.policy)",
      "--seed", "$TrainSeed",
      "--model", $ModelPath,
      "--epochs", "$($Config.epochs)"
    )
    & $Python @TrainArgs
    if ($LASTEXITCODE -ne 0) {
      throw "Refresh fine-tune training failed with exit code $LASTEXITCODE."
    }

    $ScreenSeed = 20270603 + ($AttemptIndex * 41017)
    Write-Host "[$((Get-Date).ToString('o'))] Screening $ModelPath"
    & $Python -m grids_ai.neural champion `
      --candidate $ModelPath `
      --champion $ReferenceModel `
      --games $ScreenGames `
      --seed $ScreenSeed `
      --weights trained_weights.json `
      --neural-search-width 3 `
      --neural-search-depth 4 `
      --policy-scale $PolicyScale `
      --workers $Workers `
      --only-neural-opponents `
      --min-head-to-head-score 0.55 `
      --min-overall-score 0.55 `
      --min-head-to-head-lower-bound 0 `
      --output $ScreenReport
    if ($LASTEXITCODE -ne 0) {
      throw "Refresh screen failed with exit code $LASTEXITCODE."
    }

    $ScreenDecision = Get-Decision -ReportPath $ScreenReport
    $Attempt = [pscustomobject]@{
      attempt = $AttemptIndex
      candidate = $ModelPath
      hidden_size = $Config.hidden
      policy_loss_weight = $Config.policy
      learning_rate = $Config.lr
      seed = $TrainSeed
      screen_report = $ScreenReport
      screen_score = $ScreenDecision.head_to_head_score_rate
      screen_lower_bound = $ScreenDecision.head_to_head_lower_bound
      full_report = $null
      full_score = $null
      full_lower_bound = $null
      promote = $false
      finished_at = (Get-Date).ToString("o")
    }
    Write-Host "[$((Get-Date).ToString('o'))] screen score=$($Attempt.screen_score) lower=$($Attempt.screen_lower_bound)"

    if ([double]$Attempt.screen_score -ge $MinScreenScore) {
      $FullSeed = 20270603 + ($AttemptIndex * 99089)
      Write-Host "[$((Get-Date).ToString('o'))] Full gate for $ModelPath"
      & $Python -m grids_ai.neural champion `
        --candidate $ModelPath `
        --champion $ReferenceModel `
        --games $FullGateGames `
        --seed $FullSeed `
        --weights trained_weights.json `
        --neural-search-width 3 `
        --neural-search-depth 4 `
        --policy-scale $PolicyScale `
        --workers $Workers `
        --only-neural-opponents `
        --min-head-to-head-score 0.55 `
        --min-overall-score 0.55 `
        --min-head-to-head-lower-bound 0.50 `
        --output $FullReport
      if ($LASTEXITCODE -ne 0) {
        throw "Refresh full gate failed with exit code $LASTEXITCODE."
      }

      $FullDecision = Get-Decision -ReportPath $FullReport
      $Attempt.full_report = $FullReport
      $Attempt.full_score = $FullDecision.head_to_head_score_rate
      $Attempt.full_lower_bound = $FullDecision.head_to_head_lower_bound
      $Attempt.promote = [bool]$FullDecision.promote
      Write-Host "[$((Get-Date).ToString('o'))] full score=$($Attempt.full_score) lower=$($Attempt.full_lower_bound) promote=$($Attempt.promote)"
      $Attempts += $Attempt

      if (Confirm-Promotion -Candidate $ModelPath -Report $FullReport -Decision $FullDecision) {
        break
      }
    } else {
      $Attempts += $Attempt
    }

    $AttemptIndex += 1
  }

  $FinishedAt = Get-Date
  $Manifest = [pscustomobject]@{
    started_at = $StartedAt.ToString("o")
    finished_at = $FinishedAt.ToString("o")
    target_hours = $Hours
    elapsed_hours = [Math]::Round(($FinishedAt - $StartedAt).TotalHours, 3)
    reference_model = $ReferenceModel
    teacher_data = $TeacherData
    fresh_data = $FreshData
    calibration_report = $CalibrationReport
    screen_games_per_side = $ScreenGames
    full_gate_games_per_side = $FullGateGames
    min_screen_score = $MinScreenScore
    workers = $Workers
    data = $Data
    attempts = $Attempts
  }
  $Manifest | ConvertTo-Json -Depth 7 | Set-Content -Path $ManifestPath -Encoding UTF8

  if (-not ($Attempts | Where-Object { $_.promote })) {
    $Best = $Attempts | Sort-Object -Property @{ Expression = { if ($_.full_score -ne $null) { $_.full_score } else { $_.screen_score } } }, screen_score -Descending | Select-Object -First 1
    if ($Best) {
      $Message = "New champion supervisor ended without promotion. attempts=$($Attempts.Count) best=$($Best.candidate) screen=$([Math]::Round([double]$Best.screen_score, 3)) full=$Best.full_score manifest=$ManifestPath"
    } else {
      $Message = "New champion supervisor ended without promotion before training attempts. manifest=$ManifestPath"
    }
    Invoke-Notify -Title "Grids AI supervisor complete" -Message $Message -Priority "default" -Tags "mag"
    Write-Host $Message
  }
} catch {
  $Message = "New champion supervisor failed: $($_.Exception.Message) log=$LogPath manifest=$ManifestPath"
  Invoke-Notify -Title "Grids AI supervisor failed" -Message $Message -Priority "high" -Tags "warning"
  throw
} finally {
  Stop-Transcript | Out-Null
}
