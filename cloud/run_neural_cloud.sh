#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${RUN_ID:-cloud-$(date +%Y%m%d-%H%M%S)}"
TEACHER="${TEACHER:-checkpoints/heuristic-benchmark-long-20260504-155255.gen_015.json}"
GAMES="${GAMES:-400}"
WORKERS="${WORKERS:-$(python - <<'PY'
import os
print(max(1, (os.cpu_count() or 2) - 1))
PY
)}"
SEARCH_WIDTH="${SEARCH_WIDTH:-3}"
SEARCH_DEPTH="${SEARCH_DEPTH:-6}"
SAMPLE_EVERY="${SAMPLE_EVERY:-2}"
MAX_EXAMPLES_PER_GAME="${MAX_EXAMPLES_PER_GAME:-80}"
HIDDEN_SIZE="${HIDDEN_SIZE:-96}"
EPOCHS="${EPOCHS:-16}"
LEARNING_RATE="${LEARNING_RATE:-0.003}"
EVAL_GAMES="${EVAL_GAMES:-24}"

mkdir -p neural_data checkpoints training_logs

DATASET="neural_data/selfplay-${RUN_ID}.jsonl"
MODEL="checkpoints/value_model-${RUN_ID}.json"
EVAL="training_logs/neural-${RUN_ID}.eval.json"
LOG="training_logs/neural-${RUN_ID}.out.log"
MANIFEST="training_logs/neural-${RUN_ID}.manifest.json"

cat > "${MANIFEST}" <<JSON
{
  "run_id": "${RUN_ID}",
  "teacher": "${TEACHER}",
  "dataset": "${DATASET}",
  "model": "${MODEL}",
  "evaluation": "${EVAL}",
  "log": "${LOG}",
  "games": ${GAMES},
  "workers": ${WORKERS},
  "search_width": ${SEARCH_WIDTH},
  "search_depth": ${SEARCH_DEPTH},
  "sample_every": ${SAMPLE_EVERY},
  "max_examples_per_game": ${MAX_EXAMPLES_PER_GAME},
  "hidden_size": ${HIDDEN_SIZE},
  "epochs": ${EPOCHS},
  "learning_rate": ${LEARNING_RATE},
  "eval_games": ${EVAL_GAMES}
}
JSON

{
  echo "[cloud] run_id=${RUN_ID}"
  echo "[cloud] teacher=${TEACHER}"
  echo "[cloud] workers=${WORKERS}"
  python -m grids_ai.neural generate \
    --weights "${TEACHER}" \
    --games "${GAMES}" \
    --workers "${WORKERS}" \
    --search-width "${SEARCH_WIDTH}" \
    --search-depth "${SEARCH_DEPTH}" \
    --sample-every "${SAMPLE_EVERY}" \
    --max-examples-per-game "${MAX_EXAMPLES_PER_GAME}" \
    --output "${DATASET}"
  python -m grids_ai.neural train \
    --data "${DATASET}" \
    --model "${MODEL}" \
    --hidden-size "${HIDDEN_SIZE}" \
    --epochs "${EPOCHS}" \
    --learning-rate "${LEARNING_RATE}"
  python -m grids_ai.neural evaluate \
    --model "${MODEL}" \
    --games "${EVAL_GAMES}" \
    --weights "${TEACHER}" | tee "${EVAL}"
  echo "[cloud] complete"
} 2>&1 | tee "${LOG}"
