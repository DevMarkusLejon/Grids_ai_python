# Grids AI

`grids-ai` is a small Python re-creation of the core gameplay loop from the GameMaker project in `Grids 1.5.7.gmx.zip`, rewritten around a headless rules engine so it is easy to play from the terminal and train bots with self-play.

## What It Keeps

- A `10x7` tactical board with commanders on opposite sides.
- Separate unit and item decks.
- A shared action-point economy inspired by the original game.
- Deployment from your side of the board.
- Unit actions for movement, attacks, and healing.
- A manual draw action that costs action points.
- Commander defeat as the main win condition.

## What It Simplifies

- No networking, menus, editor, or graphical effects.
- A smaller roster focused on the core loop.
- Manhattan-range targeting instead of the original presentation-heavy targeting helpers.
- A text UI instead of sprite rendering.

## Roster

The default roster is intentionally compact:

- `Commander`: the win-condition unit.
- `Warrior`: durable melee bruiser.
- `Archer`: long-range backline damage.
- `Healer`: ranged healing support.
- `Assassin`: cheap melee skirmisher.
- `Viking`: heavy frontliner.

Items:

- `Fireball`: direct damage to an enemy unit.
- `Strength Tonic`: permanent attack buff for a friendly unit.

## Project Layout

- `grids_ai/data.py`: unit, item, deck, and map definitions.
- `grids_ai/engine.py`: the rules engine and ASCII board rendering.
- `grids_ai/bots.py`: random and heuristic bots, including a short turn planner for tactical combos.
- `grids_ai/training.py`: evolutionary self-play trainer.
- `grids_ai/encoding.py`: fixed state/action encoders for neural policy and value experiments.
- `grids_ai/neural.py`: dependency-free self-play dataset generation and value-network training.
- `grids_ai/cli.py`: terminal interface for human or bot players.

## Running

Install Python 3.10+ and then:

```bash
python -m grids_ai.cli --blue human --red heuristic
```

Useful CLI commands:

- `show`: print the board, hands, and turn status.
- `actions`: list legal actions for the current player.
- `do <index>`: apply a numbered legal action.
- `auto`: let the heuristic bot play the rest of this turn.
- `help`: print the command help.
- `quit`: exit.

You can also run bot-vs-bot matches:

```bash
python -m grids_ai.cli --blue heuristic --red random
```

## Browser Version

The browser prototype now lives in the repo at [`web/index.html`](./web/index.html). It is a static
front-end version of the game with a default AI-vs-AI spectator mode and optional click-to-select
or drag-and-drop controls for playing blue.

- Open `web/index.html` in a browser.
- Use `Watch AI` to spectate autonomous play.
- Use `Play Blue` to take over the blue side.
- Click or drag blue hand cards onto highlighted targets.
- Click or drag blue units to move, attack, or heal.
- Use the control buttons to draw cards or end the turn.
- The red side is controlled by an in-browser heuristic AI using the latest bundled trained weights
  plus a short same-turn beam search.

For GitHub Pages, publish the repository root. The root [`index.html`](./index.html) redirects to
the static web app under `web/`, [`.nojekyll`](./.nojekyll) keeps Pages from applying Jekyll
processing to the assets, and [`.github/workflows/pages.yml`](./.github/workflows/pages.yml)
deploys the static site when the `main` branch is pushed with Pages set to GitHub Actions.

To spectate more comfortably, add a delay between bot actions. In bot-vs-bot matches the CLI will
refresh the screen with a cleaner spectator view instead of endlessly scrolling:

```bash
python -m grids_ai.cli --blue heuristic --red heuristic --weights trained_weights.json --delay 0.4
```

## Training

The trainer uses dependency-free evolutionary self-play to improve the heuristic bot's evaluation weights. It samples candidate weight sets, plays paired-seed matches, and keeps candidates only when they clear a promotion margin.
During training, the terminal now shows a live per-generation progress bar so long runs have visible feedback.
Candidates are scored on a mix of win/loss, final margin, and win speed, and they are evaluated
against a rolling pool of recent champions, fixed benchmark archetypes, and a random bot.
The score now also rewards preserving your commander, keeping more units alive, controlling more
board space, finishing with more resources in hand, creating commander threats, avoiding lethal
counter-threats, making moves that enable attacks, and staying consistent across evaluations.
Games that end on the turn-limit tiebreak get an explicit penalty to discourage stalling.
Terminal win/loss weights are fixed during mutation so evolution focuses on tactical preferences
instead of drifting the meaning of victory.

Example:

```bash
python -m grids_ai.training --generations 20 --population 10 --games 4 --output trained_weights.json
```

To widen the benchmark pool:

```bash
python -m grids_ai.training --champion-pool-size 5 --games 8 --output trained_weights.json
```

To train with the same short planner used during play, raise the training search width. This is
stronger but more expensive:

```bash
python -m grids_ai.training --ai-search-width 3 --ai-search-depth 6 --games 4 --output trained_weights.json
```

Continue training from a previous run:

```bash
python -m grids_ai.training --resume-from trained_weights.json --generations 10 --output trained_weights_v2.json
```

You can also write resumable checkpoints during training. The `latest` checkpoint is updated every
generation, and numbered snapshots are kept on your chosen interval:

```bash
python -m grids_ai.training --checkpoint-prefix checkpoints/run --checkpoint-every 5 --output trained_weights.json
```

That produces files like `checkpoints/run.latest.json` and `checkpoints/run.gen_005.json`, and you
can resume from either one with `--resume-from`.

Training now stops early if the champion reaches the theoretical maximum score for the current
evaluation setup, since no later candidate can beat that score without changing the config.

Use the trained weights in the CLI:

```bash
python -m grids_ai.cli --blue human --red heuristic --weights trained_weights.json
```

## Neural Value Experiments

The first neural layer is intentionally small and dependency-free so it works on plain Python. It
uses the current heuristic/planner to generate self-play states, trains a compact tanh MLP to predict
the eventual winner from each side-to-move position, and can use that model as a value evaluator.

Inspect the encoder dimensions:

```bash
python -m grids_ai.neural inspect
```

Generate a small self-play dataset:

```bash
python -m grids_ai.neural generate --weights checkpoints/run.latest.json --games 20 --output neural_data/selfplay.jsonl
```

Train a value model:

```bash
python -m grids_ai.neural train --data neural_data/selfplay.jsonl --model checkpoints/value_model.json --epochs 8
```

Smoke-test the neural value bot against random play:

```bash
python -m grids_ai.neural evaluate --model checkpoints/value_model.json --games 8 --weights checkpoints/run.latest.json
```

This is not an AlphaZero system yet. It is the first reusable pipeline piece: stable encodings,
self-play data, model checkpoints, and a neural value evaluator that can later be replaced by a
PyTorch CNN or policy/value network.

For longer runs on a remote machine, see [`cloud/README.md`](./cloud/README.md). The current cloud
setup is CPU-first and parallelizes self-play with `--workers`; GPU rental becomes more useful once
the model backend moves to PyTorch or batched neural inference.

## Tests

The test suite uses the standard library:

```bash
python -m unittest discover -s tests
```

## Rule Mapping Notes

This version is inspired by the original source files:

- `scr_createVars.gml`: action economy, hand sizes, turn structure.
- `scr_createGrid.gml` and `scr_placeObstacles.gml`: board size, commanders, and map shape.
- `scr_playerState.gml`, `scr_move.gml`, and `scr_attack.gml`: select/move/attack loop.
- `scr_handDeployUnit.gml` and `scr_drawObject.gml`: deploy and draw flow.
- `scr_endTurn.gml`: turn reset behavior.

The Python version keeps the spirit of those systems while trimming the feature surface enough to make AI training practical.
