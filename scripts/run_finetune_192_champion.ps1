param(
  [double]$Hours = 6.0,
  [string]$Python = ".venv-gpu\Scripts\python.exe",
  [string]$ReferenceModel = "checkpoints/policy_value_torch_192_blend_20260521-073435.json",
  [string]$TeacherData = "neural_data/teacher_192_policy_value_20260602-234805.jsonl",
  [int]$ScreenGames = 24,
  [int]$FullGateGames = 96,
  [int]$EvalWorkers = 6,
  [int]$BatchSize = 1024,
  [int]$PerDataLimit = 30000,
  [int]$Seed = 20260603,
  [double]$PolicyScale = 18.0,
  [double]$MinScreenScore = 0.52
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

$Stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$RunLabel = "finetune_192_champion_$Stamp"
$LogPath = Join-Path "logs" "$RunLabel.log"
$ManifestPath = Join-Path "reports" "$RunLabel.manifest.json"
$StartedAt = Get-Date
$Deadline = $StartedAt.AddHours($Hours)
$Attempts = @()

New-Item -ItemType Directory -Force -Path "checkpoints", "reports", "logs" | Out-Null
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

try {
  Write-Host "Started 192 champion fine-tune run at $($StartedAt.ToString('o'))"
  Write-Host "Deadline: $($Deadline.ToString('o'))"
  Write-Host "Reference/init model: $ReferenceModel"
  Write-Host "Teacher data: $TeacherData"
  Write-Host "Python: $Python"

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

  $Configs = @(
    [pscustomobject]@{ policy = 0.02; lr = 0.00001; epochs = 4; patience = 0 },
    [pscustomobject]@{ policy = 0.02; lr = 0.00002; epochs = 6; patience = 0 },
    [pscustomobject]@{ policy = 0.05; lr = 0.00001; epochs = 4; patience = 0 },
    [pscustomobject]@{ policy = 0.05; lr = 0.00002; epochs = 6; patience = 0 },
    [pscustomobject]@{ policy = 0.01; lr = 0.00002; epochs = 6; patience = 0 },
    [pscustomobject]@{ policy = 0.05; lr = 0.00005; epochs = 6; patience = 0 }
  )

  $AttemptIndex = 1
  while ((Get-Date) -lt $Deadline) {
    $Config = $Configs[($AttemptIndex - 1) % $Configs.Count]
    $RunStamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $TrainSeed = $Seed + ($AttemptIndex * 7919)
    $PolicyTag = ("p{0:0.###}" -f $Config.policy).Replace(".", "p")
    $LrTag = ("lr{0:0.#####}" -f $Config.lr).Replace(".", "p")
    $ModelPath = "checkpoints/policy_value_192_finetune_$($PolicyTag)_$($LrTag)_seed$($TrainSeed)_$RunStamp.json"
    $ScreenReport = "reports/finetune_192_screen_$($AttemptIndex)_$($PolicyTag)_$($LrTag)_seed$($TrainSeed)_$RunStamp.json"
    $FullReport = "reports/finetune_192_full_$($AttemptIndex)_$($PolicyTag)_$($LrTag)_seed$($TrainSeed)_$RunStamp.json"

    Write-Host "[$((Get-Date).ToString('o'))] Fine-tuning attempt=$AttemptIndex policy=$($Config.policy) lr=$($Config.lr) seed=$TrainSeed"
    $TrainArgs = @("-m", "grids_ai.neural", "train-policy")
    foreach ($DataPath in $Data) {
      $TrainArgs += @("--data", $DataPath)
    }
    $TrainArgs += @(
      "--init-model", $ReferenceModel,
      "--per-data-limit", "$PerDataLimit",
      "--hidden-size", "192",
      "--batch-size", "$BatchSize",
      "--device", "cuda",
      "--validation-fraction", "0.1",
      "--early-stop-patience", "$($Config.patience)",
      "--early-stop-min-delta", "0.0002",
      "--learning-rate", "$($Config.lr)",
      "--policy-loss-weight", "$($Config.policy)",
      "--seed", "$TrainSeed",
      "--model", $ModelPath,
      "--epochs", "$($Config.epochs)"
    )
    & $Python @TrainArgs
    if ($LASTEXITCODE -ne 0) {
      throw "Fine-tune training failed with exit code $LASTEXITCODE."
    }

    $ScreenSeed = $Seed + ($AttemptIndex * 41017)
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
      --workers $EvalWorkers `
      --only-neural-opponents `
      --min-head-to-head-score 0.55 `
      --min-overall-score 0.55 `
      --min-head-to-head-lower-bound 0 `
      --output $ScreenReport
    if ($LASTEXITCODE -ne 0) {
      throw "Fine-tune screen failed with exit code $LASTEXITCODE."
    }

    $ScreenDecision = Get-Decision -ReportPath $ScreenReport
    $Attempt = [pscustomobject]@{
      attempt = $AttemptIndex
      candidate = $ModelPath
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
      $FullSeed = $Seed + ($AttemptIndex * 99089)
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
        --workers $EvalWorkers `
        --only-neural-opponents `
        --min-head-to-head-score 0.55 `
        --min-overall-score 0.55 `
        --min-head-to-head-lower-bound 0.50 `
        --output $FullReport
      if ($LASTEXITCODE -ne 0) {
        throw "Fine-tune full gate failed with exit code $LASTEXITCODE."
      }

      $FullDecision = Get-Decision -ReportPath $FullReport
      $Attempt.full_report = $FullReport
      $Attempt.full_score = $FullDecision.head_to_head_score_rate
      $Attempt.full_lower_bound = $FullDecision.head_to_head_lower_bound
      $Attempt.promote = [bool]$FullDecision.promote
      Write-Host "[$((Get-Date).ToString('o'))] full score=$($Attempt.full_score) lower=$($Attempt.full_lower_bound) promote=$($Attempt.promote)"

      if ($Attempt.promote) {
        Set-Content -Path "champion_model.txt" -Value $ModelPath -Encoding ASCII
        Set-Content -Path "scripts/play_strongest.cmd" -Value @(
          "@echo off",
          "cd /d ""%~dp0\..""",
          "python -m grids_ai.cli --blue human --red neural --model $ModelPath --policy-scale $PolicyScale --neural-search-width 3 --neural-search-depth 4 %*"
        ) -Encoding ASCII
        Invoke-RegistryRefresh -Champion $ModelPath
        $Message = "New Grids AI champion from fine-tune: $ModelPath score=$([math]::Round($Attempt.full_score, 3)) lower=$([math]::Round($Attempt.full_lower_bound, 3)) report=$FullReport"
        Invoke-Notify -Title "Grids AI new champion" -Message $Message -Priority "high" -Tags "trophy"
        $Attempts += $Attempt
        break
      }
    }

    $Attempts += $Attempt
    $AttemptIndex += 1
  }

  $FinishedAt = Get-Date
  $Manifest = [pscustomobject]@{
    started_at = $StartedAt.ToString("o")
    finished_at = $FinishedAt.ToString("o")
    target_hours = $Hours
    elapsed_hours = [math]::Round(($FinishedAt - $StartedAt).TotalHours, 3)
    reference_model = $ReferenceModel
    teacher_data = $TeacherData
    screen_games_per_side = $ScreenGames
    full_gate_games_per_side = $FullGateGames
    min_screen_score = $MinScreenScore
    eval_workers = $EvalWorkers
    data = $Data
    attempts = $Attempts
  }
  $Manifest | ConvertTo-Json -Depth 7 | Set-Content -Path $ManifestPath -Encoding UTF8

  if (-not ($Attempts | Where-Object { $_.promote })) {
    $Best = $Attempts | Sort-Object -Property @{ Expression = { if ($_.full_score -ne $null) { $_.full_score } else { $_.screen_score } } }, screen_score -Descending | Select-Object -First 1
    $Message = "192 fine-tune run ended without promotion. attempts=$($Attempts.Count) best=$($Best.candidate) screen=$([math]::Round([double]$Best.screen_score, 3)) full=$Best.full_score manifest=$ManifestPath"
    Invoke-Notify -Title "Grids AI fine-tune complete" -Message $Message -Priority "default" -Tags "mag"
    Write-Host $Message
  }
} catch {
  $Message = "192 fine-tune run failed: $($_.Exception.Message) log=$LogPath manifest=$ManifestPath"
  Invoke-Notify -Title "Grids AI fine-tune failed" -Message $Message -Priority "high" -Tags "warning"
  throw
} finally {
  Stop-Transcript | Out-Null
}
