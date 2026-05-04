# Cloud Training Setup

This project is cloud-ready for the current neural pipeline, but the best first target is a
many-core CPU machine rather than an expensive GPU. The current bottleneck is self-play game
simulation and search. GPUs become important after the value/policy network moves to PyTorch or a
batched neural evaluator.

## Recommendation

Use a short-lived SSH-accessible instance with at least:

- 16+ vCPU
- 32+ GB RAM
- 30+ GB disk
- Ubuntu or a Python-ready container image

RunPod, Vast.ai, Lambda, or EC2 all work if they expose SSH. For this stage, choose CPU count and
stable storage over raw GPU model.

## Quick Start On A Remote Instance

Upload this repo from Windows PowerShell:

```powershell
.\cloud\upload_to_cloud.ps1 -HostSpec "user@host" -RemoteDir "~/grids-ai"
```

Start training on the remote box:

```bash
cd ~/grids-ai
nohup bash cloud/run_neural_cloud.sh > cloud-run.nohup.log 2>&1 &
tail -f training_logs/neural-*.out.log
```

Use more or fewer workers by setting environment variables:

```bash
GAMES=800 WORKERS=31 HIDDEN_SIZE=128 EPOCHS=20 bash cloud/run_neural_cloud.sh
```

Download artifacts afterward:

```powershell
.\cloud\download_from_cloud.ps1 -HostSpec "user@host" -RemoteDir "~/grids-ai"
```

## Docker

Build locally or on a cloud host:

```bash
docker build -f cloud/Dockerfile -t grids-ai-cloud .
docker run --rm -it grids-ai-cloud bash cloud/run_neural_cloud.sh
```

For GPU hosts this image still runs CPU mode. That is intentional for now; the next architecture
step is adding a PyTorch backend and then a GPU image.

## Files Produced

- `neural_data/selfplay-<run>.jsonl`: self-play training examples
- `checkpoints/value_model-<run>.json`: trained value model
- `training_logs/neural-<run>.out.log`: progress log
- `training_logs/neural-<run>.eval.json`: evaluation result
- `training_logs/neural-<run>.manifest.json`: run configuration

Delete remote instances or pods when done. Stopped cloud instances can still charge storage fees
depending on provider.
