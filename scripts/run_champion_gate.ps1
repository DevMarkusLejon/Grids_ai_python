param(
  [Parameter(Mandatory=$true)]
  [string]$CandidateModel,
  [string]$ChampionModel = "checkpoints/value_model_torch_128_shaped_1000_300hp.json",
  [string[]]$OpponentModel = @(),
  [int]$Games = 4,
  [int]$Seed = 20260520,
  [int]$NeuralSearchWidth = 3,
  [int]$NeuralSearchDepth = 4,
  [double]$MinHeadToHeadScore = 0.55,
  [double]$MinOverallScore = 0.60,
  [double]$MinHeadToHeadLowerBound = 0.50
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$Stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$CandidateName = [System.IO.Path]::GetFileNameWithoutExtension($CandidateModel)
$SafeName = $CandidateName -replace '[^A-Za-z0-9_.-]', '_'
$ReportPath = "reports/champion_gate_$SafeName`_$Stamp.json"

New-Item -ItemType Directory -Force -Path "reports" | Out-Null

$Args = @(
  "-m", "grids_ai.neural", "champion",
  "--candidate", $CandidateModel,
  "--champion", $ChampionModel,
  "--games", $Games,
  "--seed", $Seed,
  "--weights", "trained_weights.json",
  "--neural-search-width", $NeuralSearchWidth,
  "--neural-search-depth", $NeuralSearchDepth,
  "--min-head-to-head-score", $MinHeadToHeadScore,
  "--min-overall-score", $MinOverallScore,
  "--min-head-to-head-lower-bound", $MinHeadToHeadLowerBound,
  "--output", $ReportPath
)

foreach ($Model in $OpponentModel) {
  $Args += @("--opponent-model", $Model)
}

Write-Host "[$(Get-Date -Format o)] Starting champion gate"
Write-Host "Candidate: $CandidateModel"
Write-Host "Champion:  $ChampionModel"
Write-Host "Report:    $ReportPath"

python @Args
if ($LASTEXITCODE -ne 0) { throw "Champion gate failed with exit code $LASTEXITCODE." }

Write-Host "[$(Get-Date -Format o)] Finished champion gate"
