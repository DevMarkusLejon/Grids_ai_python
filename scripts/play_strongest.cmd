@echo off
cd /d "%~dp0\.."
python -m grids_ai.cli --blue human --red neural --model checkpoints/value_model_torch_128_shaped_1000.json %*
