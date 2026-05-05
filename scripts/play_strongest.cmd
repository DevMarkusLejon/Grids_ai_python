@echo off
cd /d "%~dp0\.."
python -m grids_ai.cli --blue human --red neural --model checkpoints/value_model_torch_128_shaped_1000_300hp.json --neural-search-width 3 --neural-search-depth 4 %*
