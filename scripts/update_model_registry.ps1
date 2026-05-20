param(
  [string]$Output = "web/assets/model-registry.json",
  [string]$Champion = "checkpoints/value_model_torch_128_shaped_1000_300hp.json"
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

python -m grids_ai.model_registry --champion $Champion --output $Output
if ($LASTEXITCODE -ne 0) {
  throw "Model registry update failed with exit code $LASTEXITCODE."
}
