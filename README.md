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

## Training

The trainer uses dependency-free evolutionary self-play to improve the heuristic bot's evaluation weights. It samples candidate weight sets, plays matches, and keeps the best-performing policy.

Example:

```bash
python -m grids_ai.training --generations 20 --population 10 --games 4 --output trained_weights.json
```

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
