from __future__ import annotations

from dataclasses import dataclass
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import math
import os
import random
import time
from typing import Iterable, Sequence

from .bots import DEFAULT_WEIGHTS, Bot, HeuristicBot, RandomBot, load_weights
from .data import Side
from .encoding import encode_action_index, encode_state_vector, encoded_state_size
from .engine import Action, GameState, new_game
from .training import play_match


MODEL_VERSION = 1


@dataclass(frozen=True)
class TrainingExample:
    features: list[float]
    value: float
    action_index: int | None = None
    side: str | None = None
    winner: str | None = None
    half_turns: int | None = None


class ValueNetwork:
    """Small dependency-free tanh MLP for value prediction."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 48,
        seed: int = 0,
        *,
        w1: list[list[float]] | None = None,
        b1: list[float] | None = None,
        w2: list[float] | None = None,
        b2: float = 0.0,
    ) -> None:
        self.input_size = input_size
        self.hidden_size = hidden_size
        rng = random.Random(seed)
        scale = 1.0 / math.sqrt(max(input_size, 1))
        self.w1 = w1 or [
            [rng.uniform(-scale, scale) for _ in range(input_size)]
            for _ in range(hidden_size)
        ]
        self.b1 = b1 or [0.0 for _ in range(hidden_size)]
        self.w2 = w2 or [rng.uniform(-scale, scale) for _ in range(hidden_size)]
        self.b2 = b2

    def forward(self, features: Sequence[float]) -> tuple[float, list[float]]:
        hidden: list[float] = []
        for row, bias in zip(self.w1, self.b1):
            activation = bias
            for weight, value in zip(row, features):
                activation += weight * value
            hidden.append(math.tanh(activation))
        output = self.b2
        for weight, value in zip(self.w2, hidden):
            output += weight * value
        return math.tanh(output), hidden

    def predict(self, features: Sequence[float]) -> float:
        prediction, _ = self.forward(features)
        return prediction

    def predict_state_for_player(self, state: GameState, player: Side) -> float:
        if state.is_done:
            if state.winner is None:
                return 0.0
            return 1.0 if state.winner is player else -1.0
        prediction = self.predict(encode_state_vector(state))
        return prediction if state.current_side is player else -prediction

    def train_step(self, example: TrainingExample, learning_rate: float) -> float:
        prediction, hidden = self.forward(example.features)
        target = max(-1.0, min(1.0, example.value))
        error = prediction - target
        loss = error * error
        output_grad = 2.0 * error * (1.0 - prediction * prediction)

        old_w2 = list(self.w2)
        for index in range(self.hidden_size):
            self.w2[index] -= learning_rate * output_grad * hidden[index]
        self.b2 -= learning_rate * output_grad

        for hidden_index in range(self.hidden_size):
            hidden_grad = output_grad * old_w2[hidden_index] * (1.0 - hidden[hidden_index] * hidden[hidden_index])
            if hidden_grad == 0.0:
                continue
            row = self.w1[hidden_index]
            for feature_index, value in enumerate(example.features):
                if value:
                    row[feature_index] -= learning_rate * hidden_grad * value
            self.b1[hidden_index] -= learning_rate * hidden_grad
        return loss

    def fit(
        self,
        examples: Sequence[TrainingExample],
        *,
        epochs: int = 8,
        learning_rate: float = 0.003,
        seed: int = 0,
        progress: bool = False,
    ) -> list[float]:
        if not examples:
            raise ValueError("Cannot train a value network with no examples.")
        rng = random.Random(seed)
        history: list[float] = []
        indices = list(range(len(examples)))
        started_at = time.perf_counter()
        for epoch in range(1, epochs + 1):
            epoch_started = time.perf_counter()
            rng.shuffle(indices)
            total_loss = 0.0
            for index in indices:
                total_loss += self.train_step(examples[index], learning_rate)
            average_loss = total_loss / len(examples)
            history.append(average_loss)
            if progress:
                elapsed = time.perf_counter() - started_at
                epoch_seconds = time.perf_counter() - epoch_started
                print(
                    f"[neural train] epoch {epoch}/{epochs} "
                    f"loss={average_loss:.5f} "
                    f"epoch={epoch_seconds:.1f}s elapsed={elapsed:.1f}s",
                    flush=True,
                )
        return history

    def to_dict(self, metadata: dict[str, object] | None = None) -> dict[str, object]:
        payload: dict[str, object] = {
            "version": MODEL_VERSION,
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "w1": self.w1,
            "b1": self.b1,
            "w2": self.w2,
            "b2": self.b2,
        }
        if metadata:
            payload["metadata"] = metadata
        return payload

    def save(self, path: str, metadata: dict[str, object] | None = None) -> None:
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(metadata), handle)

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "ValueNetwork":
        return cls(
            input_size=int(data["input_size"]),
            hidden_size=int(data["hidden_size"]),
            w1=[[float(value) for value in row] for row in data["w1"]],
            b1=[float(value) for value in data["b1"]],
            w2=[float(value) for value in data["w2"]],
            b2=float(data["b2"]),
        )

    @classmethod
    def load(cls, path: str) -> "ValueNetwork":
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return cls.from_dict(data)


class NeuralValueBot(Bot):
    def __init__(
        self,
        model: ValueNetwork,
        *,
        fallback_weights: dict[str, float] | None = None,
        seed: int | None = None,
        neural_scale: float = 120.0,
        heuristic_scale: float = 1.0,
    ) -> None:
        self.model = model
        self.heuristic = HeuristicBot(fallback_weights or dict(DEFAULT_WEIGHTS), seed=seed, search_width=1)
        self.rng = random.Random(seed)
        self.neural_scale = neural_scale
        self.heuristic_scale = heuristic_scale

    def choose_action(self, state: GameState) -> Action:
        legal = state.legal_actions()
        if not legal:
            raise ValueError("No legal actions available.")
        player = state.current_side
        scored: list[tuple[float, float, Action]] = []
        for action in legal:
            after = state.clone()
            after.apply_unchecked(action)
            heuristic_score = self.heuristic.score_action(state, after, action, player)
            neural_score = self.model.predict_state_for_player(after, player) * self.neural_scale
            scored.append((heuristic_score * self.heuristic_scale + neural_score, self.rng.random(), action))
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return scored[0][2]


def winner_value(winner: Side | None, side: Side) -> float:
    if winner is None:
        return 0.0
    return 1.0 if winner is side else -1.0


def example_to_json(example: TrainingExample) -> str:
    return json.dumps(
        {
            "features": example.features,
            "value": example.value,
            "action_index": example.action_index,
            "side": example.side,
            "winner": example.winner,
            "half_turns": example.half_turns,
        },
        separators=(",", ":"),
    )


def example_from_json(line: str) -> TrainingExample:
    data = json.loads(line)
    return TrainingExample(
        features=[float(value) for value in data["features"]],
        value=float(data["value"]),
        action_index=data.get("action_index"),
        side=data.get("side"),
        winner=data.get("winner"),
        half_turns=data.get("half_turns"),
    )


def load_examples(path: str, *, limit: int | None = None) -> list[TrainingExample]:
    examples: list[TrainingExample] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            examples.append(example_from_json(line))
            if limit is not None and len(examples) >= limit:
                break
    return examples


def write_examples(path: str, examples: Iterable[TrainingExample]) -> int:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    count = 0
    with open(path, "w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(example_to_json(example))
            handle.write("\n")
            count += 1
    return count


def generate_game_examples(
    *,
    seed: int,
    weights: dict[str, float],
    map_name: str = "plains",
    search_width: int = 3,
    search_depth: int | None = 6,
    sample_every: int = 2,
    max_examples: int = 80,
) -> list[TrainingExample]:
    state = new_game(seed=seed, map_name=map_name)
    bots = {
        Side.BLUE: HeuristicBot(weights, seed=seed + 1, search_width=search_width, search_depth=search_depth),
        Side.RED: HeuristicBot(weights, seed=seed + 2, search_width=search_width, search_depth=search_depth),
    }
    pending: list[tuple[list[float], int, Side, int]] = []
    steps = 0
    turn_safety = state.config.max_half_turns * 8

    while not state.is_done and steps < turn_safety:
        bot = bots[state.current_side]
        action = bot.choose_action(state)
        if steps % max(sample_every, 1) == 0 and len(pending) < max_examples:
            pending.append(
                (
                    encode_state_vector(state),
                    encode_action_index(state, action),
                    state.current_side,
                    state.half_turns_played,
                )
            )
        state.apply_unchecked(action)
        steps += 1

    if not state.is_done:
        state._resolve_timeout_winner()

    return [
        TrainingExample(
            features=features,
            value=winner_value(state.winner, side),
            action_index=action_index,
            side=side.value,
            winner=state.winner.value if state.winner else None,
            half_turns=half_turns,
        )
        for features, action_index, side, half_turns in pending
    ]


def generate_game_examples_task(task: tuple[int, dict[str, float], str, int, int | None, int, int]) -> list[TrainingExample]:
    seed, weights, map_name, search_width, search_depth, sample_every, max_examples_per_game = task
    return generate_game_examples(
        seed=seed,
        weights=weights,
        map_name=map_name,
        search_width=search_width,
        search_depth=search_depth,
        sample_every=sample_every,
        max_examples=max_examples_per_game,
    )


def generate_self_play_dataset(
    *,
    output_path: str,
    games: int = 20,
    seed: int = 11,
    weights: dict[str, float] | None = None,
    map_name: str = "plains",
    search_width: int = 3,
    search_depth: int | None = 6,
    sample_every: int = 2,
    max_examples_per_game: int = 80,
    progress: bool = False,
    workers: int = 1,
) -> int:
    chosen_weights = dict(DEFAULT_WEIGHTS)
    if weights:
        chosen_weights.update(weights)
    directory = os.path.dirname(output_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    count = 0
    started_at = time.perf_counter()
    if progress:
        print(
            f"[neural generate] writing {games} games to {output_path} "
            f"(search_width={search_width}, search_depth={search_depth})",
            flush=True,
        )
    tasks = [
        (
            seed + game_index,
            chosen_weights,
            map_name,
            search_width,
            search_depth,
            sample_every,
            max_examples_per_game,
        )
        for game_index in range(games)
    ]

    def write_game_examples(handle, game_index: int, game_started: float, examples: list[TrainingExample]) -> None:
        nonlocal count
        for example in examples:
            handle.write(example_to_json(example))
            handle.write("\n")
        handle.flush()
        count += len(examples)
        if progress:
            elapsed = time.perf_counter() - started_at
            completed = game_index + 1
            rate = completed / max(elapsed, 0.001)
            remaining = (games - completed) / max(rate, 0.001)
            print(
                f"[neural generate] game {completed}/{games} "
                f"examples={count} last={len(examples)} "
                f"game={time.perf_counter() - game_started:.1f}s eta={remaining:.0f}s",
                flush=True,
            )

    with open(output_path, "w", encoding="utf-8") as handle:
        if workers <= 1:
            for game_index, task in enumerate(tasks):
                game_started = time.perf_counter()
                examples = generate_game_examples_task(task)
                write_game_examples(handle, game_index, game_started, examples)
        else:
            completed_games = 0
            with ProcessPoolExecutor(max_workers=workers) as executor:
                future_to_start = {
                    executor.submit(generate_game_examples_task, task): time.perf_counter()
                    for task in tasks
                }
                for future in as_completed(future_to_start):
                    game_started = future_to_start[future]
                    examples = future.result()
                    completed_games += 1
                    write_game_examples(handle, completed_games - 1, game_started, examples)
    if progress:
        print(f"[neural generate] complete examples={count} elapsed={time.perf_counter() - started_at:.1f}s", flush=True)
    return count


def train_value_model(
    *,
    dataset_path: str,
    model_path: str,
    hidden_size: int = 48,
    epochs: int = 8,
    learning_rate: float = 0.003,
    seed: int = 13,
    limit: int | None = None,
    progress: bool = False,
) -> list[float]:
    examples = load_examples(dataset_path, limit=limit)
    if not examples:
        raise ValueError(f"No training examples found in {dataset_path}.")
    if progress:
        print(
            f"[neural train] loaded examples={len(examples)} "
            f"input_size={len(examples[0].features)} hidden_size={hidden_size}",
            flush=True,
        )
    model = ValueNetwork(len(examples[0].features), hidden_size=hidden_size, seed=seed)
    history = model.fit(examples, epochs=epochs, learning_rate=learning_rate, seed=seed, progress=progress)
    model.save(
        model_path,
        metadata={
            "dataset": dataset_path,
            "examples": len(examples),
            "epochs": epochs,
            "learning_rate": learning_rate,
            "loss_history": history,
        },
    )
    return history


def evaluate_value_bot(
    *,
    model_path: str,
    games: int = 6,
    seed: int = 23,
    weights: dict[str, float] | None = None,
) -> dict[str, object]:
    model = ValueNetwork.load(model_path)
    fallback_weights = dict(DEFAULT_WEIGHTS)
    if weights:
        fallback_weights.update(weights)
    wins = {Side.BLUE.value: 0, Side.RED.value: 0, "none": 0}
    half_turns: list[int] = []
    for game_index in range(games):
        state = play_match(
            NeuralValueBot(model, fallback_weights=fallback_weights, seed=seed + game_index),
            RandomBot(seed=seed + 1000 + game_index),
            seed=seed + 2000 + game_index,
        )
        wins[state.winner.value if state.winner else "none"] += 1
        half_turns.append(state.half_turns_played)
    return {
        "wins": wins,
        "average_half_turns": sum(half_turns) / max(len(half_turns), 1),
        "games": games,
        "input_size": model.input_size,
        "hidden_size": model.hidden_size,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate and train neural value models for Grids AI.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="Generate a self-play JSONL dataset.")
    generate.add_argument("--output", default="neural_data/selfplay.jsonl")
    generate.add_argument("--games", type=int, default=20)
    generate.add_argument("--seed", type=int, default=11)
    generate.add_argument("--weights", help="Optional heuristic weights JSON to seed self-play.")
    generate.add_argument("--map", default="plains", choices=["plains", "desert"])
    generate.add_argument("--search-width", type=int, default=3)
    generate.add_argument("--search-depth", type=int, default=6)
    generate.add_argument("--sample-every", type=int, default=2)
    generate.add_argument("--max-examples-per-game", type=int, default=80)
    generate.add_argument("--workers", type=int, default=1, help="Parallel self-play worker processes.")
    generate.add_argument("--quiet", action="store_true", help="Suppress lightweight per-game progress output.")

    train = subparsers.add_parser("train", help="Train a small dependency-free value network.")
    train.add_argument("--data", default="neural_data/selfplay.jsonl")
    train.add_argument("--model", default="checkpoints/value_model.json")
    train.add_argument("--hidden-size", type=int, default=48)
    train.add_argument("--epochs", type=int, default=8)
    train.add_argument("--learning-rate", type=float, default=0.003)
    train.add_argument("--seed", type=int, default=13)
    train.add_argument("--limit", type=int)
    train.add_argument("--quiet", action="store_true", help="Suppress lightweight per-epoch progress output.")

    evaluate = subparsers.add_parser("evaluate", help="Evaluate a value model against random play.")
    evaluate.add_argument("--model", default="checkpoints/value_model.json")
    evaluate.add_argument("--games", type=int, default=6)
    evaluate.add_argument("--seed", type=int, default=23)
    evaluate.add_argument("--weights", help="Optional fallback heuristic weights JSON.")

    inspect = subparsers.add_parser("inspect", help="Print encoder dimensions for a new game.")
    inspect.add_argument("--seed", type=int, default=1)
    inspect.add_argument("--map", default="plains", choices=["plains", "desert"])

    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "generate":
        if args.workers < 1:
            raise SystemExit("--workers must be at least 1.")
        weights = load_weights(args.weights) if args.weights else None
        count = generate_self_play_dataset(
            output_path=args.output,
            games=args.games,
            seed=args.seed,
            weights=weights,
            map_name=args.map,
            search_width=args.search_width,
            search_depth=args.search_depth,
            sample_every=args.sample_every,
            max_examples_per_game=args.max_examples_per_game,
            progress=not args.quiet,
            workers=args.workers,
        )
        print(f"Wrote {count} examples to {args.output}")
        return 0
    if args.command == "train":
        history = train_value_model(
            dataset_path=args.data,
            model_path=args.model,
            hidden_size=args.hidden_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            seed=args.seed,
            limit=args.limit,
            progress=not args.quiet,
        )
        print(f"Saved value model to {args.model}")
        print("Loss history: " + ", ".join(f"{loss:.4f}" for loss in history))
        return 0
    if args.command == "evaluate":
        weights = load_weights(args.weights) if args.weights else None
        result = evaluate_value_bot(model_path=args.model, games=args.games, seed=args.seed, weights=weights)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "inspect":
        state = new_game(seed=args.seed, map_name=args.map)
        print(
            json.dumps(
                {
                    "state_vector_size": encoded_state_size(state),
                    "legal_actions": len(state.legal_actions()),
                    "sample_legal_action_indices": [
                        encode_action_index(state, action) for action in state.legal_actions()[:10]
                    ],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
