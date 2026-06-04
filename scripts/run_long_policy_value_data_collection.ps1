param(
  [string]$TeacherModel = "checkpoints/policy_value_torch_192_blend_20260521-073435.json",
  [double]$Hours = 8.0,
  [int]$Workers = 6,
  [int]$GamesPerBatch = 120,
  [int]$SeedBase = 930000,
  [string]$RunLabel = ""
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

if (-not (Test-Path $TeacherModel)) {
  throw "Teacher model not found: $TeacherModel"
}

$Stamp = Get-Date -Format "yyyyMMdd-HHmmss"
if (-not $RunLabel) {
  $RunLabel = "long_policy_value_collection_$Stamp"
}

$RunDir = Join-Path "neural_data" $RunLabel
$LogPath = Join-Path "logs" "$RunLabel.log"
$ManifestPath = Join-Path $RunDir "manifest.json"

New-Item -ItemType Directory -Force -Path $RunDir, "logs" | Out-Null
Start-Transcript -Path $LogPath | Out-Null

$StartedAt = Get-Date
$Deadline = $StartedAt.AddHours($Hours)
$Files = @()
$Batch = 1

try {
  Write-Host "Started long policy/value data collection at $($StartedAt.ToString('o'))"
  Write-Host "Deadline: $($Deadline.ToString('o'))"
  Write-Host "Teacher: $TeacherModel"
  Write-Host "Run directory: $RunDir"
  Write-Host "Workers: $Workers"
  Write-Host "Games per batch: $GamesPerBatch"

  $StartMessage = "Long data collection started. teacher=$TeacherModel target_hours=$Hours deadline=$($Deadline.ToString('o')) dir=$RunDir"
  powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/notify_important.ps1 `
    -Title "Grids AI data collection started" `
    -Message $StartMessage `
    -Priority default `
    -Tags "hourglass_flowing_sand"

  while ((Get-Date) -lt $Deadline) {
    $Output = Join-Path $RunDir ("batch_{0:D3}.jsonl" -f $Batch)
    $Seed = $SeedBase + $Batch
    Write-Host "[$((Get-Date).ToString('o'))] Starting batch $Batch seed=$Seed output=$Output"
    python -m grids_ai.neural generate `
      --output $Output `
      --games $GamesPerBatch `
      --seed $Seed `
      --teacher neural `
      --teacher-model $TeacherModel `
      --target shaped `
      --teacher-neural-search-width 3 `
      --teacher-neural-search-depth 4 `
      --sample-every 2 `
      --max-examples-per-game 80 `
      --workers $Workers `
      --quiet
    if ($LASTEXITCODE -ne 0) {
      throw "Batch $Batch failed with exit code $LASTEXITCODE."
    }

    $Lines = (Get-Content $Output | Measure-Object -Line).Lines
    $Files += [pscustomobject]@{
      batch = $Batch
      file = $Output
      examples = $Lines
      seed = $Seed
      completed_at = (Get-Date).ToString("o")
    }
    $TotalExamples = ($Files | Measure-Object -Property examples -Sum).Sum
    Write-Host "[$((Get-Date).ToString('o'))] Completed batch $Batch examples=$Lines total_examples=$TotalExamples"
    $Batch += 1
  }

  $FinishedAt = Get-Date
  $TotalExamples = ($Files | Measure-Object -Property examples -Sum).Sum
  $Manifest = [pscustomobject]@{
    started_at = $StartedAt.ToString("o")
    finished_at = $FinishedAt.ToString("o")
    target_hours = $Hours
    elapsed_hours = [math]::Round(($FinishedAt - $StartedAt).TotalHours, 3)
    teacher_model = $TeacherModel
    teacher = "policy_value_neural_beam_w3_d4"
    target = "shaped"
    workers = $Workers
    games_per_batch = $GamesPerBatch
    sample_every = 2
    max_examples_per_game = 80
    total_batches = $Files.Count
    total_examples = $TotalExamples
    files = $Files
  }
  $Manifest | ConvertTo-Json -Depth 6 | Set-Content -Path $ManifestPath -Encoding UTF8

  $Message = "Long data collection finished. examples=$TotalExamples batches=$($Files.Count) elapsed_hours=$([math]::Round(($FinishedAt - $StartedAt).TotalHours, 2)) dir=$RunDir"
  powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/notify_important.ps1 `
    -Title "Grids AI data collection finished" `
    -Message $Message `
    -Priority high `
    -Tags "chart_with_upwards_trend"
  Write-Host $Message
} catch {
  $Message = "Long data collection failed: $($_.Exception.Message) dir=$RunDir"
  powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/notify_important.ps1 `
    -Title "Grids AI data collection failed" `
    -Message $Message `
    -Priority high `
    -Tags "warning"
  throw
} finally {
  Stop-Transcript | Out-Null
}
