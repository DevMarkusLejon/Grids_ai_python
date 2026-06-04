param(
  [double]$Hours = 8.0,
  [int]$Trials = 40,
  [string]$Python = ".venv-gpu\Scripts\python.exe",
  [string]$ReferenceModel = "checkpoints/policy_value_torch_192_blend_20260521-073435.json",
  [int]$Workers = 6,
  [int]$ScreenGames = 48,
  [int]$FullGateGames = 192,
  [string]$RunLabel = "",
  [string]$StudyName = "",
  [string]$Storage = "",
  [string[]]$ExtraData = @(),
  [string[]]$CollectionDir = @(),
  [string[]]$CollectionManifest = @(),
  [switch]$IncludeIncompleteCollections,
  [switch]$SkipLatestCompletedCollection,
  [switch]$ValidateOnly,
  [switch]$NotifyOnStart,
  [switch]$NoNotifyOnComplete,
  [switch]$PromoteOnPass
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

New-Item -ItemType Directory -Force -Path "logs", "reports", "optuna" | Out-Null
$Stamp = Get-Date -Format "yyyyMMdd-HHmmss"
if (-not $RunLabel) {
  $RunLabel = "optuna_policy_value_search_$Stamp"
}
if (-not $StudyName) {
  $StudyName = $RunLabel
}
if (-not $Storage) {
  $Storage = "sqlite:///optuna/$RunLabel.db"
}
$LogPath = Join-Path "logs" "$RunLabel.log"
$StartedAt = Get-Date

function Invoke-Notify {
  param(
    [string]$Title,
    [string]$Message,
    [string]$Priority = "default",
    [string]$Tags = "mag"
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

if (-not (Test-Path $Python)) {
  $Python = "python"
}

$TrainingData = New-Object System.Collections.Generic.List[string]

function Add-TrainingDataFile {
  param([string]$Path)
  if (-not $Path) {
    return
  }
  $Resolved = Resolve-Path -LiteralPath $Path -ErrorAction SilentlyContinue
  if (-not $Resolved) {
    return
  }
  $Normalized = $Resolved.Path
  if (-not $TrainingData.Contains($Normalized)) {
    [void]$TrainingData.Add($Normalized)
  }
}

function Add-TrainingDataDirectory {
  param([string]$Path)
  if (-not (Test-Path -LiteralPath $Path -PathType Container)) {
    return
  }
  $ManifestPath = Join-Path $Path "manifest.json"
  if (Test-Path -LiteralPath $ManifestPath -PathType Leaf) {
    Add-TrainingDataManifest $ManifestPath
    return
  }
  if (-not $IncludeIncompleteCollections) {
    Write-Warning "Skipping collection directory without manifest.json: $Path"
    return
  }
  Get-ChildItem -LiteralPath $Path -Filter "*.jsonl" -File |
    Sort-Object Name |
    ForEach-Object { Add-TrainingDataFile $_.FullName }
}

function Add-TrainingDataManifest {
  param([string]$Path)
  if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
    return
  }
  $ManifestDir = Split-Path -Parent (Resolve-Path -LiteralPath $Path).Path
  $Manifest = Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json
  foreach ($Item in @($Manifest.files)) {
    if (-not $Item.file) {
      continue
    }
    $DataPath = [string]$Item.file
    if (-not [System.IO.Path]::IsPathRooted($DataPath)) {
      $RepoRelative = Join-Path $Root $DataPath
      $ManifestRelative = Join-Path $ManifestDir $DataPath
      if (Test-Path -LiteralPath $RepoRelative) {
        $DataPath = $RepoRelative
      } elseif (Test-Path -LiteralPath $ManifestRelative) {
        $DataPath = $ManifestRelative
      }
    }
    Add-TrainingDataFile $DataPath
  }
}

@(
  "neural_data/selfplay_192_teacher_refresh_20260603-102147.jsonl",
  "neural_data/teacher_192_policy_value_20260602-234805.jsonl",
  "neural_data/overnight_policy_value_300hp_20260520-233904.jsonl",
  "neural_data/long_policy_value_collection_20260521-084951/batch_001.jsonl",
  "neural_data/long_policy_value_collection_20260521-084951/batch_002.jsonl",
  "neural_data/long_policy_value_collection_20260521-084951/batch_003.jsonl",
  "neural_data/long_policy_value_collection_20260521-084951/batch_004.jsonl",
  "neural_data/long_policy_value_collection_20260521-084951/batch_005.jsonl",
  "neural_data/selfplay_policy_value_128_blend_300hp_20260520-232410.jsonl"
) | ForEach-Object { Add-TrainingDataFile $_ }

foreach ($Path in $ExtraData) {
  Add-TrainingDataFile $Path
}

foreach ($Path in $CollectionManifest) {
  Add-TrainingDataManifest $Path
}

foreach ($Path in $CollectionDir) {
  Add-TrainingDataDirectory $Path
}

if (-not $SkipLatestCompletedCollection) {
  $LatestManifest = Get-ChildItem -LiteralPath "neural_data" -Directory -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -match "policy_value_collection" -and (Test-Path -LiteralPath (Join-Path $_.FullName "manifest.json")) } |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1
  if ($LatestManifest) {
    Add-TrainingDataManifest (Join-Path $LatestManifest.FullName "manifest.json")
  }
}

if (-not (Test-Path $ReferenceModel)) {
  throw "Reference model not found: $ReferenceModel"
}
if ($TrainingData.Count -lt 1) {
  throw "No training data files found."
}

if ($ValidateOnly) {
  Write-Host "Validation OK."
  Write-Host "Run label: $RunLabel"
  Write-Host "Log path: $LogPath"
  Write-Host "Study name: $StudyName"
  Write-Host "Storage: $Storage"
  Write-Host "Python: $Python"
  Write-Host "Reference model: $ReferenceModel"
  Write-Host "Training files: $($TrainingData.Count)"
  foreach ($Path in $TrainingData) {
    Write-Host "  $Path"
  }
  exit 0
}

$Args = @(
  "scripts\optuna_policy_value_search.py",
  "--reference-model", $ReferenceModel,
  "--study-name", $StudyName,
  "--storage", $Storage,
  "--trials", "$Trials",
  "--timeout-hours", "$Hours",
  "--python", $Python,
  "--device", "cuda",
  "--workers", "$Workers",
  "--screen-games", "$ScreenGames",
  "--full-gate-games", "$FullGateGames",
  "--full-gate-score", "0.54",
  "--hidden-sizes", "192,384,512",
  "--policy-loss-weight-choices", "0,0.001,0.005,0.01,0.02",
  "--learning-rate-min", "0.000001",
  "--learning-rate-max", "0.00005",
  "--epoch-choices", "3,4,5,6",
  "--per-data-limit-choices", "30000,60000,90000",
  "--primary-repeat-choices", "1,2,3"
)

foreach ($Path in $TrainingData) {
  $Args += @("--data", $Path)
}
if ($PromoteOnPass) {
  $Args += "--promote-on-pass"
}

Start-Transcript -Path $LogPath | Out-Null
try {
  Write-Host "Started Optuna policy/value search at $($StartedAt.ToString('o'))"
  Write-Host "Run label: $RunLabel"
  Write-Host "Study name: $StudyName"
  Write-Host "Storage: $Storage"
  Write-Host "Reference model: $ReferenceModel"
  Write-Host "Python: $Python"
  Write-Host "Hours: $Hours Trials: $Trials Workers: $Workers ScreenGames: $ScreenGames FullGateGames: $FullGateGames"
  Write-Host "Training files: $($TrainingData.Count)"
  foreach ($Path in $TrainingData) {
    Write-Host "  $Path"
  }
  if ($NotifyOnStart) {
    Invoke-Notify `
      -Title "Grids AI Optuna search started" `
      -Message "Optuna search started. run=$RunLabel hours=$Hours trials=$Trials reference=$ReferenceModel log=$LogPath" `
      -Priority "default" `
      -Tags "hourglass_flowing_sand"
  }

  & $Python @Args
  if ($LASTEXITCODE -ne 0) {
    throw "Optuna policy/value search failed with exit code $LASTEXITCODE."
  }

  $SummaryPath = Join-Path "reports\optuna" "$($StudyName)_summary.json"
  $FinishedAt = Get-Date
  if (-not $NoNotifyOnComplete) {
    $Message = "Optuna search complete. run=$RunLabel elapsed_hours=$([math]::Round(($FinishedAt - $StartedAt).TotalHours, 2)) summary=$SummaryPath log=$LogPath"
    if (Test-Path $SummaryPath) {
      try {
        $Summary = Get-Content $SummaryPath -Raw | ConvertFrom-Json
        $BestAttrs = $Summary.best_user_attrs
        $BestModel = if ($BestAttrs -and $BestAttrs.candidate_model) { $BestAttrs.candidate_model } else { "" }
        $BestScore = if ($Summary.best_value -ne $null) { [math]::Round([double]$Summary.best_value, 3) } else { "n/a" }
        $Message = "$Message best_score=$BestScore best_model=$BestModel"
      } catch {
        Write-Warning "Could not parse Optuna summary: $($_.Exception.Message)"
      }
    }
    Invoke-Notify -Title "Grids AI Optuna search complete" -Message $Message -Priority "default" -Tags "mag"
  }
} catch {
  $Message = "Optuna search failed. run=$RunLabel error=$($_.Exception.Message) log=$LogPath"
  Invoke-Notify -Title "Grids AI Optuna search failed" -Message $Message -Priority "high" -Tags "warning"
  throw
} finally {
  Stop-Transcript | Out-Null
}
