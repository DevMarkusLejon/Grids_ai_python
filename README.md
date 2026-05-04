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
- `grids_ai/bots.py`: random and heuristic bots.
- `grids_ai/training.py`: evolutionary self-play trainer.
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

To spectate more comfortably, add a delay between bot actions. In bot-vs-bot matches the CLI will
refresh the screen with a cleaner spectator view instead of endlessly scrolling:

```bash
python -m grids_ai.cli --blue heuristic --red heuristic --weights trained_weights.json --delay 0.4
```

## Training

The trainer uses dependency-free evolutionary self-play to improve the heuristic bot's evaluation weights. It samples candidate weight sets, plays matches, and keeps the best-performing policy.
During training, the terminal now shows a live per-generation progress bar so long runs have visible feedback.
Candidates are scored on a mix of win/loss, final margin, and win speed, and they are evaluated
against a rolling pool of recent champions plus a random bot.
The score now also rewards preserving your commander, keeping more units alive, controlling more
board space, finishing with more resources in hand, and staying consistent across evaluations.
Games that end on the turn-limit tiebreak get an explicit penalty to discourage stalling.

Example:

```bash
python -m grids_ai.training --generations 20 --population 10 --games 4 --output trained_weights.json
```

To widen the benchmark pool:

```bash
python -m grids_ai.training --champion-pool-size 5 --games 8 --output trained_weights.json
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

## Tests

The test suite uses the standard library:

```bash
python -m unittest
```

## Rule Mapping Notes

This version is inspired by the original source files:

- `scr_createVars.gml`: action economy, hand sizes, turn structure.
- `scr_createGrid.gml` and `scr_placeObstacles.gml`: board size, commanders, and map shape.
- `scr_playerState.gml`, `scr_move.gml`, and `scr_attack.gml`: select/move/attack loop.
- `scr_handDeployUnit.gml` and `scr_drawObject.gml`: deploy and draw flow.
- `scr_endTurn.gml`: turn reset behavior.

The Python version keeps the spirit of those systems while trimming the feature surface enough to make AI training practical.
