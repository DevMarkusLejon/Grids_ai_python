# Provider Notes

Current project stage:

- Expensive step: self-play simulation and search.
- Cheap step: dependency-free value-network training.
- Best first cloud target: many CPU cores with reliable storage.
- GPU target later: after adding a PyTorch policy/value network or batched neural evaluator.

## Practical Provider Ranking For This Project

1. **Vast.ai**: likely cheapest for experiments if you are comfortable with marketplace hosts.
   Pick high vCPU count, enough RAM, and good host reliability. Delete instances when done because
   storage can continue billing.
2. **RunPod**: easiest pod-style workflow and good when you later want GPUs. Secure Cloud is more
   predictable; Community/Spot is cheaper but can be interrupted.
3. **Lambda Cloud**: clean managed GPU instances, but its pricing is more GPU-focused and usually
   less compelling for this current CPU-heavy phase.
4. **AWS EC2**: robust and scriptable. Use Spot for cheap CPU experiments, but expect more setup and
   more pricing footnotes.

## Suggested First Cloud Run

Use a 16-32 vCPU Ubuntu instance. After upload:

```bash
cd ~/grids-ai
GAMES=800 WORKERS=31 HIDDEN_SIZE=128 EPOCHS=20 bash cloud/run_neural_cloud.sh
tail -f training_logs/neural-*.out.log
```

If the provider only has 8 vCPU:

```bash
GAMES=400 WORKERS=7 HIDDEN_SIZE=96 EPOCHS=16 bash cloud/run_neural_cloud.sh
```

## When To Rent A GPU

Rent a GPU once one of these is true:

- the value/policy model moves to PyTorch or JAX
- inference is batched inside search/MCTS
- model training time dominates self-play generation time

Before that, a GPU may sit mostly idle while Python simulates games on CPU.
