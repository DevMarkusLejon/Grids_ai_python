@echo off
cd /d "%~dp0\.."
python -m grids_ai.cli --blue human --red neural --model checkpoints/policy_value_torch_192_blend_20260521-073435.json --policy-scale 18 --neural-search-width 3 --neural-search-depth 4 %*
