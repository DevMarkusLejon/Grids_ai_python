param(
  [string]$Output = "web/assets/model-registry.json",
  [string]$Champion = ""
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

if (-not $Champion) {
  $Pointer = "champion_model.txt"
  if (Test-Path $Pointer) {
    $Champion = (Get-Content $Pointer -Raw).Trim()
  } else {
    $Champion = "checkpoints/value_model_torch_128_shaped_1000_300hp.json"
  }
}

python -m grids_ai.model_registry --champion $Champion --output $Output
if ($LASTEXITCODE -ne 0) {
  throw "Model registry update failed with exit code $LASTEXITCODE."
}
