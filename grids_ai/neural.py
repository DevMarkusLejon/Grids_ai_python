from __future__ import annotations

from dataclasses import dataclass
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import math
import os
import random
from pathlib import Path
import time
from typing import Callable, Iterable, Sequence

from .bots import DEFAULT_WEIGHTS, Bot, HeuristicBot, RandomBot, load_weights
from .data import Side
from .encoding import encode_action_index, encode_state_vector, encoded_state_size
from .engine import Action, GameState, new_game
from .training import play_match


MODEL_VERSION = 1
SparseFeatures = tuple[tuple[int, float], ...]
TrainBackend = str
TargetMode = str


def nonzero_features(features: Sequence[float]) -> SparseFeatures:
    return tuple((index, float(value)) for index, value in enumerate(features) if value)


@dataclass(frozen=True)
class TrainingExample:
    features: list[float]
    value: float
    action_index: int | None = None
    side: str | None = None
    winner: str | None = None
    half_turns: int | None = None
    sparse_features: SparseFeatures | None = None


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

    def forward_sparse(self, features: SparseFeatures) -> tuple[float, list[float]]:
        hidden: list[float] = []
        for row, bias in zip(self.w1, self.b1):
            activation = bias
            for feature_index, value in features:
                activation += row[feature_index] * value
            hidden.append(math.tanh(activation))
        output = self.b2
        for weight, value in zip(self.w2, hidden):
            output += weight * value
        return math.tanh(output), hidden

    def forward(self, features: Sequence[float]) -> tuple[float, list[float]]:
        return self.forward_sparse(nonzero_features(features))

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
        sparse_features = example.sparse_features or nonzero_features(example.features)
        prediction, hidden = self.forward_sparse(sparse_features)
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
            for feature_index, value in sparse_features:
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

    @classmethod
    def from_weights(
        cls,
        *,
        w1: Sequence[Sequence[float]],
        b1: Sequence[float],
        w2: Sequence[float],
        b2: float,
    ) -> "ValueNetwork":
        return cls(
            input_size=len(w1[0]) if w1 else 0,
            hidden_size=len(w1),
            w1=[[float(value) for value in row] for row in w1],
            b1=[float(value) for value in b1],
            w2=[float(value) for value in w2],
            b2=float(b2),
        )

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
        search_width: int = 1,
        search_depth: int | None = 1,
    ) -> None:
        self.model = model
        self.heuristic = HeuristicBot(fallback_weights or dict(DEFAULT_WEIGHTS), seed=seed, search_width=1)
        self.rng = random.Random(seed)
        self.neural_scale = neural_scale
        self.heuristic_scale = heuristic_scale
        self.search_width = max(1, search_width)
        self.search_depth = search_depth

    def score_action(self, before: GameState, after: GameState, action: Action, player: Side) -> float:
        heuristic_score = self.heuristic.score_action(before, after, action, player)
        neural_score = self.model.predict_state_for_player(after, player) * self.neural_scale
        return heuristic_score * self.heuristic_scale + neural_score

    def choose_greedy_action(self, state: GameState, legal: list[Action], player: Side) -> Action:
        scored: list[tuple[float, float, Action]] = []
        for action in legal:
            after = state.clone()
            after.apply_unchecked(action)
            scored.append((self.score_action(state, after, action, player), self.rng.random(), action))
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return scored[0][2]

    def choose_planned_action(self, state: GameState, legal: list[Action], player: Side) -> Action:
        depth_limit = self.search_depth or state.config.max_actions + 1
        depth_limit = max(1, depth_limit)
        width = max(1, self.search_width)
        decay = 0.92

        frontier: list[tuple[float, float, Action, GameState]] = []
        all_paths: list[tuple[float, float, Action, GameState]] = []
        for action in legal:
            after = state.clone()
            after.apply_unchecked(action)
            path = (self.score_action(state, after, action, player), self.rng.random(), action, after)
            frontier.append(path)
            all_paths.append(path)

        frontier.sort(key=lambda item: (item[0], item[1]), reverse=True)
        frontier = frontier[:width]

        for depth in range(1, depth_limit):
            expanded: list[tuple[float, float, Action, GameState]] = []
            for cumulative_score, tie_breaker, first_action, branch_state in frontier:
                if branch_state.is_done or branch_state.current_side is not player:
                    continue
                for action in branch_state.legal_actions():
                    after = branch_state.clone()
                    after.apply_unchecked(action)
                    step_score = self.score_action(branch_state, after, action, player)
                    path = (
                        cumulative_score + step_score * (decay**depth),
                        tie_breaker,
                        first_action,
                        after,
                    )
                    expanded.append(path)
                    all_paths.append(path)
            if not expanded:
                break
            expanded.sort(key=lambda item: (item[0], item[1]), reverse=True)
            frontier = expanded[:width]

        all_paths.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return all_paths[0][2]

    def choose_action(self, state: GameState) -> Action:
        legal = state.legal_actions()
        if not legal:
            raise ValueError("No legal actions available.")
        player = state.current_side
        if self.search_width <= 1 and (self.search_depth is None or self.search_depth <= 1):
            return self.choose_greedy_action(state, legal, player)
        return self.choose_planned_action(state, legal, player)


class ExploratoryHeuristicBot(Bot):
    """Heuristic teacher with optional top-k sampling for dataset diversity."""

    def __init__(
        self,
        weights: dict[str, float],
        *,
        seed: int,
        search_width: int,
        search_depth: int | None,
        exploration_rate: float = 0.0,
        sampling_top_k: int = 1,
        sampling_temperature: float = 0.0,
    ) -> None:
        self.base = HeuristicBot(weights, seed=seed, search_width=search_width, search_depth=search_depth)
        self.rng = random.Random(seed)
        self.exploration_rate = max(0.0, min(1.0, exploration_rate))
        self.sampling_top_k = max(1, sampling_top_k)
        self.sampling_temperature = max(0.0, sampling_temperature)

    def choose_action(self, state: GameState) -> Action:
        legal = state.legal_actions()
        if not legal:
            raise ValueError("No legal actions available.")
        if self.exploration_rate > 0.0 and self.rng.random() < self.exploration_rate:
            return self.rng.choice(legal)
        if self.sampling_top_k <= 1 or self.sampling_temperature <= 0.0:
            return self.base.choose_action(state)

        player = state.current_side
        scored: list[tuple[float, float, Action]] = []
        for action in legal:
            after = state.clone()
            after.apply_unchecked(action)
            score = self.base.score_action(state, after, action, player)
            scored.append((score, self.rng.random(), action))
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        top_actions = scored[: self.sampling_top_k]
        max_score = top_actions[0][0]
        weights = [math.exp((score - max_score) / self.sampling_temperature) for score, _, _ in top_actions]
        total = sum(weights)
        pick = self.rng.random() * total
        cumulative = 0.0
        for weight, (_, _, action) in zip(weights, top_actions):
            cumulative += weight
            if cumulative >= pick:
                return action
        return top_actions[-1][2]


BotFactory = Callable[[int], Bot]


@dataclass(frozen=True)
class GauntletOpponent:
    name: str
    kind: str
    make_bot: BotFactory
    metadata: dict[str, object] | None = None


def winner_value(winner: Side | None, side: Side) -> float:
    if winner is None:
        return 0.0
    return 1.0 if winner is side else -1.0


def clamp_value(value: float) -> float:
    return max(-1.0, min(1.0, value))


def final_margin_value(state: GameState, side: Side) -> float:
    return clamp_value(state.side_score(side) / 250.0)


def speed_value(state: GameState, side: Side) -> float:
    if state.winner is None:
        return 0.0
    remaining = 1.0 - min(state.half_turns_played, state.config.max_half_turns) / max(state.config.max_half_turns, 1)
    return remaining if state.winner is side else -remaining


def target_value(state: GameState, side: Side, mode: TargetMode = "outcome") -> float:
    outcome = winner_value(state.winner, side)
    if mode == "outcome":
        return outcome
    if mode == "margin":
        return final_margin_value(state, side)
    if mode == "shaped":
        return clamp_value(outcome * 0.65 + final_margin_value(state, side) * 0.25 + speed_value(state, side) * 0.10)
    raise ValueError(f"Unknown target mode: {mode}")


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
    features = [float(value) for value in data["features"]]
    return TrainingExample(
        features=features,
        value=float(data["value"]),
        action_index=data.get("action_index"),
        side=data.get("side"),
        winner=data.get("winner"),
        half_turns=data.get("half_turns"),
        sparse_features=nonzero_features(features),
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


def split_examples(
    examples: Sequence[TrainingExample],
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[list[TrainingExample], list[TrainingExample]]:
    if validation_fraction <= 0.0 or len(examples) < 2:
        return list(examples), []
    if validation_fraction >= 1.0:
        raise ValueError("--validation-fraction must be less than 1.")
    indices = list(range(len(examples)))
    random.Random(seed).shuffle(indices)
    validation_count = max(1, int(round(len(examples) * validation_fraction)))
    validation_count = min(validation_count, len(examples) - 1)
    validation_indices = set(indices[:validation_count])
    training = [example for index, example in enumerate(examples) if index not in validation_indices]
    validation = [example for index, example in enumerate(examples) if index in validation_indices]
    return training, validation


def mean_squared_error(model: ValueNetwork, examples: Sequence[TrainingExample]) -> float:
    if not examples:
        return 0.0
    total = 0.0
    for example in examples:
        target = max(-1.0, min(1.0, example.value))
        error = model.predict(example.features) - target
        total += error * error
    return total / len(examples)


def clone_value_model(model: ValueNetwork) -> ValueNetwork:
    return ValueNetwork.from_dict(model.to_dict())


def resolve_train_backend(backend: TrainBackend) -> TrainBackend:
    if backend != "auto":
        return backend
    try:
        import torch  # noqa: F401

        return "torch"
    except ImportError:
        pass
    try:
        import numpy  # noqa: F401

        return "numpy"
    except ImportError:
        return "python"


def require_numpy():
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "NumPy backend requested but numpy is not installed. "
            'Install with: pip install -e ".[neural]"'
        ) from exc
    return np


def require_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch backend requested but torch is not installed. "
            'Install with: pip install -e ".[neural]"'
        ) from exc
    return torch


def examples_to_arrays(examples: Sequence[TrainingExample], np):
    features = np.asarray([example.features for example in examples], dtype=np.float32)
    targets = np.asarray([max(-1.0, min(1.0, example.value)) for example in examples], dtype=np.float32)
    return features, targets


def adam_update(param, grad, first_moment, second_moment, step: int, learning_rate: float, np) -> None:
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8
    first_moment *= beta1
    first_moment += (1.0 - beta1) * grad
    second_moment *= beta2
    second_moment += (1.0 - beta2) * (grad * grad)
    corrected_first = first_moment / (1.0 - beta1**step)
    corrected_second = second_moment / (1.0 - beta2**step)
    param -= learning_rate * corrected_first / (np.sqrt(corrected_second) + epsilon)


def train_python_value_model(
    examples: Sequence[TrainingExample],
    *,
    validation_examples: Sequence[TrainingExample],
    hidden_size: int,
    epochs: int,
    learning_rate: float,
    seed: int,
    early_stop_patience: int,
    early_stop_min_delta: float,
    progress: bool,
) -> tuple[ValueNetwork, list[float], dict[str, object]]:
    model = ValueNetwork(len(examples[0].features), hidden_size=hidden_size, seed=seed)
    rng = random.Random(seed)
    indices = list(range(len(examples)))
    history: list[float] = []
    validation_history: list[float] = []
    best_model = clone_value_model(model)
    best_validation_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0
    started_at = time.perf_counter()

    for epoch in range(1, epochs + 1):
        epoch_started = time.perf_counter()
        rng.shuffle(indices)
        total_loss = 0.0
        for index in indices:
            total_loss += model.train_step(examples[index], learning_rate)
        average_loss = total_loss / len(examples)
        history.append(average_loss)

        validation_loss = mean_squared_error(model, validation_examples) if validation_examples else None
        if validation_loss is not None:
            validation_history.append(validation_loss)
            if validation_loss < best_validation_loss - early_stop_min_delta:
                best_validation_loss = validation_loss
                best_epoch = epoch
                best_model = clone_value_model(model)
                stale_epochs = 0
            else:
                stale_epochs += 1

        if progress:
            elapsed = time.perf_counter() - started_at
            epoch_seconds = time.perf_counter() - epoch_started
            validation_part = f" val_loss={validation_loss:.5f}" if validation_loss is not None else ""
            print(
                f"[neural train] epoch {epoch}/{epochs} "
                f"loss={average_loss:.5f}{validation_part} "
                f"epoch={epoch_seconds:.1f}s elapsed={elapsed:.1f}s",
                flush=True,
            )

        if validation_examples and early_stop_patience > 0 and stale_epochs >= early_stop_patience:
            if progress:
                print(
                    f"[neural train] early stop at epoch {epoch}; "
                    f"best_epoch={best_epoch} best_val_loss={best_validation_loss:.5f}",
                    flush=True,
                )
            break

    if validation_examples:
        model = best_model
    metadata = {
        "training_backend": "sparse-python",
        "validation_loss_history": validation_history,
        "best_epoch": best_epoch or len(history),
        "best_validation_loss": best_validation_loss if validation_examples else None,
    }
    return model, history, metadata


def train_numpy_value_model(
    examples: Sequence[TrainingExample],
    *,
    validation_examples: Sequence[TrainingExample],
    hidden_size: int,
    epochs: int,
    learning_rate: float,
    seed: int,
    batch_size: int,
    early_stop_patience: int,
    early_stop_min_delta: float,
    progress: bool,
) -> tuple[ValueNetwork, list[float], dict[str, object]]:
    np = require_numpy()
    features, targets = examples_to_arrays(examples, np)
    validation_arrays = examples_to_arrays(validation_examples, np) if validation_examples else None
    input_size = int(features.shape[1])
    rng = np.random.default_rng(seed)
    scale = 1.0 / math.sqrt(max(input_size, 1))
    w1 = rng.uniform(-scale, scale, size=(hidden_size, input_size)).astype(np.float32)
    b1 = np.zeros(hidden_size, dtype=np.float32)
    w2 = rng.uniform(-scale, scale, size=hidden_size).astype(np.float32)
    b2 = np.asarray(0.0, dtype=np.float32)

    moments = {
        "w1": (np.zeros_like(w1), np.zeros_like(w1)),
        "b1": (np.zeros_like(b1), np.zeros_like(b1)),
        "w2": (np.zeros_like(w2), np.zeros_like(w2)),
        "b2": (np.zeros_like(b2), np.zeros_like(b2)),
    }
    indices = np.arange(len(examples))
    history: list[float] = []
    validation_history: list[float] = []
    best_weights = (
        w1.copy(),
        b1.copy(),
        w2.copy(),
        np.asarray(b2).copy(),
    )
    best_validation_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0
    step = 0
    started_at = time.perf_counter()
    batch_size = max(1, batch_size)

    for epoch in range(1, epochs + 1):
        epoch_started = time.perf_counter()
        rng.shuffle(indices)
        total_loss = 0.0
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            x_batch = features[batch_indices]
            y_batch = targets[batch_indices]
            z1 = x_batch @ w1.T + b1
            hidden = np.tanh(z1)
            raw_output = hidden @ w2 + b2
            prediction = np.tanh(raw_output)
            error = prediction - y_batch
            total_loss += float(np.sum(error * error))

            batch_count = float(len(batch_indices))
            output_grad = (2.0 / batch_count) * error * (1.0 - prediction * prediction)
            grad_w2 = output_grad @ hidden
            grad_b2 = np.asarray(np.sum(output_grad), dtype=np.float32)
            hidden_grad = output_grad[:, None] * w2[None, :] * (1.0 - hidden * hidden)
            grad_w1 = hidden_grad.T @ x_batch
            grad_b1 = np.sum(hidden_grad, axis=0)

            step += 1
            adam_update(w1, grad_w1, moments["w1"][0], moments["w1"][1], step, learning_rate, np)
            adam_update(b1, grad_b1, moments["b1"][0], moments["b1"][1], step, learning_rate, np)
            adam_update(w2, grad_w2, moments["w2"][0], moments["w2"][1], step, learning_rate, np)
            adam_update(b2, grad_b2, moments["b2"][0], moments["b2"][1], step, learning_rate, np)

        average_loss = total_loss / len(examples)
        history.append(average_loss)
        validation_loss = None
        if validation_arrays is not None:
            val_features, val_targets = validation_arrays
            val_hidden = np.tanh(val_features @ w1.T + b1)
            val_prediction = np.tanh(val_hidden @ w2 + b2)
            val_error = val_prediction - val_targets
            validation_loss = float(np.mean(val_error * val_error))
            validation_history.append(validation_loss)
            if validation_loss < best_validation_loss - early_stop_min_delta:
                best_validation_loss = validation_loss
                best_epoch = epoch
                best_weights = (
                    w1.copy(),
                    b1.copy(),
                    w2.copy(),
                    np.asarray(b2).copy(),
                )
                stale_epochs = 0
            else:
                stale_epochs += 1
        if progress:
            elapsed = time.perf_counter() - started_at
            epoch_seconds = time.perf_counter() - epoch_started
            validation_part = f" val_loss={validation_loss:.5f}" if validation_loss is not None else ""
            print(
                f"[neural train] epoch {epoch}/{epochs} "
                f"loss={average_loss:.5f}{validation_part} "
                f"epoch={epoch_seconds:.1f}s elapsed={elapsed:.1f}s",
                flush=True,
            )
        if validation_arrays is not None and early_stop_patience > 0 and stale_epochs >= early_stop_patience:
            if progress:
                print(
                    f"[neural train] early stop at epoch {epoch}; "
                    f"best_epoch={best_epoch} best_val_loss={best_validation_loss:.5f}",
                    flush=True,
                )
            break

    if validation_arrays is not None:
        w1, b1, w2, b2 = best_weights
    model = ValueNetwork.from_weights(
        w1=w1.astype(float).tolist(),
        b1=b1.astype(float).tolist(),
        w2=w2.astype(float).tolist(),
        b2=float(b2),
    )
    metadata = {
        "training_backend": "numpy",
        "batch_size": batch_size,
        "optimizer": "adam",
        "validation_loss_history": validation_history,
        "best_epoch": best_epoch or len(history),
        "best_validation_loss": best_validation_loss if validation_examples else None,
    }
    return model, history, metadata


def train_torch_value_model(
    examples: Sequence[TrainingExample],
    *,
    validation_examples: Sequence[TrainingExample],
    hidden_size: int,
    epochs: int,
    learning_rate: float,
    seed: int,
    batch_size: int,
    device: str,
    early_stop_patience: int,
    early_stop_min_delta: float,
    progress: bool,
) -> tuple[ValueNetwork, list[float], dict[str, object]]:
    torch = require_torch()
    if device == "auto":
        chosen_device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        chosen_device = device
    target_device = torch.device(chosen_device)
    torch.manual_seed(seed)

    features = torch.tensor([example.features for example in examples], dtype=torch.float32, device=target_device)
    targets = torch.tensor(
        [max(-1.0, min(1.0, example.value)) for example in examples],
        dtype=torch.float32,
        device=target_device,
    )
    if validation_examples:
        validation_features = torch.tensor(
            [example.features for example in validation_examples],
            dtype=torch.float32,
            device=target_device,
        )
        validation_targets = torch.tensor(
            [max(-1.0, min(1.0, example.value)) for example in validation_examples],
            dtype=torch.float32,
            device=target_device,
        )
    else:
        validation_features = None
        validation_targets = None
    input_size = int(features.shape[1])

    class TorchValueModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.hidden = torch.nn.Linear(input_size, hidden_size)
            self.output = torch.nn.Linear(hidden_size, 1)
            scale = 1.0 / math.sqrt(max(input_size, 1))
            with torch.no_grad():
                self.hidden.weight.uniform_(-scale, scale)
                self.hidden.bias.zero_()
                self.output.weight.uniform_(-scale, scale)
                self.output.bias.zero_()

        def forward(self, x):
            hidden = torch.tanh(self.hidden(x))
            return torch.tanh(self.output(hidden)).squeeze(-1)

    torch_model = TorchValueModel().to(target_device)
    optimizer = torch.optim.Adam(torch_model.parameters(), lr=learning_rate)
    history: list[float] = []
    validation_history: list[float] = []
    best_state = {name: value.detach().clone() for name, value in torch_model.state_dict().items()}
    best_validation_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0
    rng = random.Random(seed)
    indices = list(range(len(examples)))
    started_at = time.perf_counter()
    batch_size = max(1, batch_size)

    for epoch in range(1, epochs + 1):
        epoch_started = time.perf_counter()
        rng.shuffle(indices)
        total_loss = 0.0
        for start in range(0, len(indices), batch_size):
            batch_indices = torch.tensor(indices[start : start + batch_size], dtype=torch.long, device=target_device)
            prediction = torch_model(features.index_select(0, batch_indices))
            target = targets.index_select(0, batch_indices)
            loss = torch.mean((prediction - target) ** 2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * len(batch_indices)

        average_loss = total_loss / len(examples)
        history.append(average_loss)
        validation_loss = None
        if validation_features is not None and validation_targets is not None:
            with torch.no_grad():
                validation_prediction = torch_model(validation_features)
                validation_loss = float(torch.mean((validation_prediction - validation_targets) ** 2).detach().cpu())
            validation_history.append(validation_loss)
            if validation_loss < best_validation_loss - early_stop_min_delta:
                best_validation_loss = validation_loss
                best_epoch = epoch
                best_state = {name: value.detach().clone() for name, value in torch_model.state_dict().items()}
                stale_epochs = 0
            else:
                stale_epochs += 1
        if progress:
            elapsed = time.perf_counter() - started_at
            epoch_seconds = time.perf_counter() - epoch_started
            validation_part = f" val_loss={validation_loss:.5f}" if validation_loss is not None else ""
            print(
                f"[neural train] epoch {epoch}/{epochs} "
                f"loss={average_loss:.5f}{validation_part} "
                f"epoch={epoch_seconds:.1f}s elapsed={elapsed:.1f}s",
                flush=True,
            )
        if validation_features is not None and early_stop_patience > 0 and stale_epochs >= early_stop_patience:
            if progress:
                print(
                    f"[neural train] early stop at epoch {epoch}; "
                    f"best_epoch={best_epoch} best_val_loss={best_validation_loss:.5f}",
                    flush=True,
                )
            break

    if validation_features is not None:
        torch_model.load_state_dict(best_state)
    with torch.no_grad():
        model = ValueNetwork.from_weights(
            w1=torch_model.hidden.weight.detach().cpu().tolist(),
            b1=torch_model.hidden.bias.detach().cpu().tolist(),
            w2=torch_model.output.weight.detach().cpu().reshape(-1).tolist(),
            b2=float(torch_model.output.bias.detach().cpu().reshape(-1)[0]),
        )
    metadata = {
        "training_backend": "torch",
        "batch_size": batch_size,
        "optimizer": "adam",
        "device": str(target_device),
        "torch_version": str(torch.__version__),
        "validation_loss_history": validation_history,
        "best_epoch": best_epoch or len(history),
        "best_validation_loss": best_validation_loss if validation_examples else None,
    }
    return model, history, metadata


def generate_game_examples(
    *,
    seed: int,
    weights: dict[str, float],
    map_name: str = "plains",
    search_width: int = 3,
    search_depth: int | None = 6,
    sample_every: int = 2,
    max_examples: int = 80,
    target_mode: TargetMode = "outcome",
    exploration_rate: float = 0.0,
    sampling_top_k: int = 1,
    sampling_temperature: float = 0.0,
) -> list[TrainingExample]:
    state = new_game(seed=seed, map_name=map_name)
    bots = {
        Side.BLUE: ExploratoryHeuristicBot(
            weights,
            seed=seed + 1,
            search_width=search_width,
            search_depth=search_depth,
            exploration_rate=exploration_rate,
            sampling_top_k=sampling_top_k,
            sampling_temperature=sampling_temperature,
        ),
        Side.RED: ExploratoryHeuristicBot(
            weights,
            seed=seed + 2,
            search_width=search_width,
            search_depth=search_depth,
            exploration_rate=exploration_rate,
            sampling_top_k=sampling_top_k,
            sampling_temperature=sampling_temperature,
        ),
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
            value=target_value(state, side, target_mode),
            action_index=action_index,
            side=side.value,
            winner=state.winner.value if state.winner else None,
            half_turns=half_turns,
        )
        for features, action_index, side, half_turns in pending
    ]


def generate_game_examples_task(
    task: tuple[int, dict[str, float], str, int, int | None, int, int, TargetMode, float, int, float],
) -> list[TrainingExample]:
    (
        seed,
        weights,
        map_name,
        search_width,
        search_depth,
        sample_every,
        max_examples_per_game,
        target_mode,
        exploration_rate,
        sampling_top_k,
        sampling_temperature,
    ) = task
    return generate_game_examples(
        seed=seed,
        weights=weights,
        map_name=map_name,
        search_width=search_width,
        search_depth=search_depth,
        sample_every=sample_every,
        max_examples=max_examples_per_game,
        target_mode=target_mode,
        exploration_rate=exploration_rate,
        sampling_top_k=sampling_top_k,
        sampling_temperature=sampling_temperature,
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
    target_mode: TargetMode = "outcome",
    exploration_rate: float = 0.0,
    sampling_top_k: int = 1,
    sampling_temperature: float = 0.0,
) -> int:
    if target_mode not in {"outcome", "margin", "shaped"}:
        raise ValueError("--target must be one of: outcome, margin, shaped.")
    if exploration_rate < 0.0 or exploration_rate > 1.0:
        raise ValueError("--exploration-rate must be between 0 and 1.")
    if sampling_top_k < 1:
        raise ValueError("--sampling-top-k must be at least 1.")
    if sampling_temperature < 0.0:
        raise ValueError("--sampling-temperature must be zero or greater.")
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
            f"(search_width={search_width}, search_depth={search_depth}, target={target_mode}, "
            f"exploration={exploration_rate}, top_k={sampling_top_k}, temperature={sampling_temperature})",
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
            target_mode,
            exploration_rate,
            sampling_top_k,
            sampling_temperature,
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
    backend: TrainBackend = "auto",
    batch_size: int = 256,
    device: str = "auto",
    validation_fraction: float = 0.1,
    early_stop_patience: int = 0,
    early_stop_min_delta: float = 0.0,
) -> list[float]:
    examples = load_examples(dataset_path, limit=limit)
    if not examples:
        raise ValueError(f"No training examples found in {dataset_path}.")
    chosen_backend = resolve_train_backend(backend)
    if batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")
    if early_stop_patience < 0:
        raise ValueError("--early-stop-patience must be zero or greater.")
    if early_stop_min_delta < 0:
        raise ValueError("--early-stop-min-delta must be zero or greater.")
    training_examples, validation_examples = split_examples(
        examples,
        validation_fraction=validation_fraction,
        seed=seed + 1009,
    )
    if progress:
        nonzero_counts = [len(example.sparse_features or nonzero_features(example.features)) for example in examples]
        print(
            f"[neural train] loaded examples={len(examples)} "
            f"train={len(training_examples)} validation={len(validation_examples)} "
            f"input_size={len(examples[0].features)} hidden_size={hidden_size} "
            f"avg_nonzero={sum(nonzero_counts) / len(nonzero_counts):.1f} "
            f"backend={chosen_backend}",
            flush=True,
        )
    if chosen_backend == "python":
        model, history, backend_metadata = train_python_value_model(
            training_examples,
            validation_examples=validation_examples,
            hidden_size=hidden_size,
            epochs=epochs,
            learning_rate=learning_rate,
            seed=seed,
            early_stop_patience=early_stop_patience,
            early_stop_min_delta=early_stop_min_delta,
            progress=progress,
        )
    elif chosen_backend == "numpy":
        model, history, backend_metadata = train_numpy_value_model(
            training_examples,
            validation_examples=validation_examples,
            hidden_size=hidden_size,
            epochs=epochs,
            learning_rate=learning_rate,
            seed=seed,
            batch_size=batch_size,
            early_stop_patience=early_stop_patience,
            early_stop_min_delta=early_stop_min_delta,
            progress=progress,
        )
    elif chosen_backend == "torch":
        model, history, backend_metadata = train_torch_value_model(
            training_examples,
            validation_examples=validation_examples,
            hidden_size=hidden_size,
            epochs=epochs,
            learning_rate=learning_rate,
            seed=seed,
            batch_size=batch_size,
            device=device,
            early_stop_patience=early_stop_patience,
            early_stop_min_delta=early_stop_min_delta,
            progress=progress,
        )
    else:
        raise ValueError(f"Unknown training backend: {backend}")

    metadata: dict[str, object] = {
        "dataset": dataset_path,
        "examples": len(examples),
        "training_examples": len(training_examples),
        "validation_examples": len(validation_examples),
        "validation_fraction": validation_fraction,
        "early_stop_patience": early_stop_patience,
        "early_stop_min_delta": early_stop_min_delta,
        "epochs": epochs,
        "completed_epochs": len(history),
        "learning_rate": learning_rate,
        "loss_history": history,
    }
    metadata.update(backend_metadata)
    model.save(
        model_path,
        metadata=metadata,
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


def side_result(state: GameState, side: Side) -> int:
    if state.winner is side:
        return 1
    if state.winner is None:
        return 0
    return -1


def discover_neural_opponent_models(candidate_model_path: str, *, max_models: int = 4) -> list[str]:
    candidate = os.path.abspath(candidate_model_path)
    checkpoints_dir = Path("checkpoints")
    if not checkpoints_dir.exists():
        return []
    patterns = ["value_model*.json", "value_model_*.json", "value_model-*.json"]
    found: dict[str, Path] = {}
    for pattern in patterns:
        for path in checkpoints_dir.glob(pattern):
            absolute = os.path.abspath(str(path))
            if absolute == candidate:
                continue
            found[absolute] = path
    ordered = sorted(found.values(), key=lambda path: path.stat().st_mtime, reverse=True)
    return [str(path) for path in ordered[:max_models]]


def build_gauntlet_opponents(
    *,
    fallback_weights: dict[str, float],
    heuristic_weights_path: str | None = None,
    neural_model_paths: Sequence[str] = (),
    include_baseline_opponents: bool = True,
    heuristic_search_width: int = 3,
    heuristic_search_depth: int | None = 6,
    neural_scale: float = 120.0,
    heuristic_scale: float = 1.0,
    neural_search_width: int = 1,
    neural_search_depth: int | None = 1,
) -> list[GauntletOpponent]:
    opponents: list[GauntletOpponent] = []

    if include_baseline_opponents:
        opponents.extend(
            [
                GauntletOpponent(
                    name="random",
                    kind="random",
                    make_bot=lambda seed: RandomBot(seed=seed),
                ),
                GauntletOpponent(
                    name=f"default_heuristic_w{heuristic_search_width}_d{heuristic_search_depth or 'full'}",
                    kind="heuristic",
                    make_bot=lambda seed: HeuristicBot(
                        dict(DEFAULT_WEIGHTS),
                        seed=seed,
                        search_width=heuristic_search_width,
                        search_depth=heuristic_search_depth,
                    ),
                    metadata={"weights": "default"},
                ),
            ]
        )

    if include_baseline_opponents and heuristic_weights_path and os.path.exists(heuristic_weights_path):
        trained_weights = load_weights(heuristic_weights_path)
        opponents.append(
            GauntletOpponent(
                name=f"trained_heuristic:{os.path.basename(heuristic_weights_path)}",
                kind="heuristic",
                make_bot=lambda seed, weights=trained_weights: HeuristicBot(
                    dict(weights),
                    seed=seed,
                    search_width=heuristic_search_width,
                    search_depth=heuristic_search_depth,
                ),
                metadata={"weights": heuristic_weights_path},
            )
        )

    for path in neural_model_paths:
        if not os.path.exists(path):
            continue
        model = ValueNetwork.load(path)
        opponents.append(
            GauntletOpponent(
                name=f"neural:{os.path.basename(path)}",
                kind="neural",
                make_bot=lambda seed, loaded=model: NeuralValueBot(
                    loaded,
                    fallback_weights=fallback_weights,
                    seed=seed,
                    neural_scale=neural_scale,
                    heuristic_scale=heuristic_scale,
                    search_width=neural_search_width,
                    search_depth=neural_search_depth,
                ),
                metadata={
                    "model": path,
                    "hidden_size": model.hidden_size,
                    "neural_scale": neural_scale,
                    "heuristic_scale": heuristic_scale,
                    "search_width": neural_search_width,
                    "search_depth": neural_search_depth,
                },
            )
        )
    return opponents


def run_value_gauntlet(
    *,
    model_path: str,
    games: int = 8,
    seed: int = 101,
    weights: dict[str, float] | None = None,
    weights_path: str | None = None,
    neural_opponent_models: Sequence[str] = (),
    auto_neural_opponents: bool = True,
    include_baseline_opponents: bool = True,
    map_name: str = "plains",
    heuristic_search_width: int = 3,
    heuristic_search_depth: int | None = 6,
    neural_scale: float = 120.0,
    heuristic_scale: float = 1.0,
    neural_search_width: int = 1,
    neural_search_depth: int | None = 1,
    progress: bool = False,
) -> dict[str, object]:
    if games < 1:
        raise ValueError("--games must be at least 1.")
    model = ValueNetwork.load(model_path)
    fallback_weights = dict(DEFAULT_WEIGHTS)
    if weights:
        fallback_weights.update(weights)

    opponent_model_paths = list(neural_opponent_models)
    if auto_neural_opponents:
        for path in discover_neural_opponent_models(model_path):
            if path not in opponent_model_paths:
                opponent_model_paths.append(path)

    opponents = build_gauntlet_opponents(
        fallback_weights=fallback_weights,
        heuristic_weights_path=weights_path,
        neural_model_paths=opponent_model_paths,
        include_baseline_opponents=include_baseline_opponents,
        heuristic_search_width=heuristic_search_width,
        heuristic_search_depth=heuristic_search_depth,
        neural_scale=neural_scale,
        heuristic_scale=heuristic_scale,
        neural_search_width=neural_search_width,
        neural_search_depth=neural_search_depth,
    )

    results: list[dict[str, object]] = []
    overall_wins = 0
    overall_draws = 0
    overall_losses = 0
    overall_half_turns: list[int] = []
    started_at = time.perf_counter()

    for opponent_index, opponent in enumerate(opponents):
        wins = 0
        draws = 0
        losses = 0
        blue_wins = 0
        red_wins = 0
        half_turns: list[int] = []
        for game_index in range(games):
            game_seed = seed + opponent_index * 10000 + game_index
            subject_blue = NeuralValueBot(
                model,
                fallback_weights=fallback_weights,
                seed=game_seed + 1,
                neural_scale=neural_scale,
                heuristic_scale=heuristic_scale,
                search_width=neural_search_width,
                search_depth=neural_search_depth,
            )
            opponent_red = opponent.make_bot(game_seed + 2)
            blue_state = play_match(subject_blue, opponent_red, seed=game_seed, map_name=map_name)
            result = side_result(blue_state, Side.BLUE)
            wins += 1 if result == 1 else 0
            draws += 1 if result == 0 else 0
            losses += 1 if result == -1 else 0
            blue_wins += 1 if result == 1 else 0
            half_turns.append(blue_state.half_turns_played)

            opponent_blue = opponent.make_bot(game_seed + 1002)
            subject_red = NeuralValueBot(
                model,
                fallback_weights=fallback_weights,
                seed=game_seed + 1001,
                neural_scale=neural_scale,
                heuristic_scale=heuristic_scale,
                search_width=neural_search_width,
                search_depth=neural_search_depth,
            )
            red_state = play_match(opponent_blue, subject_red, seed=game_seed, map_name=map_name)
            result = side_result(red_state, Side.RED)
            wins += 1 if result == 1 else 0
            draws += 1 if result == 0 else 0
            losses += 1 if result == -1 else 0
            red_wins += 1 if result == 1 else 0
            half_turns.append(red_state.half_turns_played)

        played = wins + draws + losses
        overall_wins += wins
        overall_draws += draws
        overall_losses += losses
        overall_half_turns.extend(half_turns)
        row: dict[str, object] = {
            "opponent": opponent.name,
            "kind": opponent.kind,
            "games_per_side": games,
            "total_games": played,
            "wins": wins,
            "draws": draws,
            "losses": losses,
            "score_rate": (wins + 0.5 * draws) / max(played, 1),
            "blue_wins": blue_wins,
            "red_wins": red_wins,
            "average_half_turns": sum(half_turns) / max(len(half_turns), 1),
        }
        if opponent.metadata:
            row["metadata"] = opponent.metadata
        results.append(row)
        if progress:
            print(
                f"[neural gauntlet] {opponent.name}: "
                f"{wins}W-{draws}D-{losses}L "
                f"score_rate={row['score_rate']:.3f} "
                f"avg_half_turns={row['average_half_turns']:.1f}",
                flush=True,
            )

    total_games = overall_wins + overall_draws + overall_losses
    return {
        "model": model_path,
        "input_size": model.input_size,
        "hidden_size": model.hidden_size,
        "map": map_name,
        "games_per_side": games,
        "total_games": total_games,
        "neural_scale": neural_scale,
        "heuristic_scale": heuristic_scale,
        "neural_search_width": neural_search_width,
        "neural_search_depth": neural_search_depth,
        "overall": {
            "wins": overall_wins,
            "draws": overall_draws,
            "losses": overall_losses,
            "score_rate": (overall_wins + 0.5 * overall_draws) / max(total_games, 1),
            "average_half_turns": sum(overall_half_turns) / max(len(overall_half_turns), 1),
        },
        "opponents": results,
        "elapsed_seconds": time.perf_counter() - started_at,
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
    generate.add_argument(
        "--target",
        default="outcome",
        choices=["outcome", "margin", "shaped"],
        help="Training target to write: terminal outcome, final margin, or blended shaped value.",
    )
    generate.add_argument(
        "--exploration-rate",
        type=float,
        default=0.0,
        help="Chance that the teacher chooses a random legal action while generating data.",
    )
    generate.add_argument(
        "--sampling-top-k",
        type=int,
        default=1,
        help="Sample among the top immediate heuristic actions when temperature is above zero.",
    )
    generate.add_argument(
        "--sampling-temperature",
        type=float,
        default=0.0,
        help="Softmax temperature for top-k teacher sampling. Zero keeps deterministic planner play.",
    )
    generate.add_argument("--quiet", action="store_true", help="Suppress lightweight per-game progress output.")

    train = subparsers.add_parser("train", help="Train a value network.")
    train.add_argument("--data", default="neural_data/selfplay.jsonl")
    train.add_argument("--model", default="checkpoints/value_model.json")
    train.add_argument("--hidden-size", type=int, default=48)
    train.add_argument("--epochs", type=int, default=8)
    train.add_argument("--learning-rate", type=float, default=0.003)
    train.add_argument("--seed", type=int, default=13)
    train.add_argument("--limit", type=int)
    train.add_argument(
        "--backend",
        default="auto",
        choices=["auto", "python", "numpy", "torch"],
        help="Training backend. Auto prefers PyTorch, then NumPy, then sparse Python.",
    )
    train.add_argument("--batch-size", type=int, default=256, help="Mini-batch size for NumPy/PyTorch training.")
    train.add_argument("--device", default="auto", help="PyTorch device, for example auto, cpu, or cuda.")
    train.add_argument("--validation-fraction", type=float, default=0.1, help="Holdout fraction for validation loss.")
    train.add_argument(
        "--early-stop-patience",
        type=int,
        default=0,
        help="Stop after this many epochs without validation improvement. Disabled at 0.",
    )
    train.add_argument(
        "--early-stop-min-delta",
        type=float,
        default=0.0,
        help="Minimum validation-loss improvement needed to reset early stopping.",
    )
    train.add_argument("--quiet", action="store_true", help="Suppress lightweight per-epoch progress output.")

    evaluate = subparsers.add_parser("evaluate", help="Evaluate a value model against random play.")
    evaluate.add_argument("--model", default="checkpoints/value_model.json")
    evaluate.add_argument("--games", type=int, default=6)
    evaluate.add_argument("--seed", type=int, default=23)
    evaluate.add_argument("--weights", help="Optional fallback heuristic weights JSON.")

    gauntlet = subparsers.add_parser("gauntlet", help="Evaluate a value model against harder opponent pools.")
    gauntlet.add_argument("--model", default="checkpoints/value_model.json")
    gauntlet.add_argument("--games", type=int, default=8, help="Games per side against each opponent.")
    gauntlet.add_argument("--seed", type=int, default=101)
    gauntlet.add_argument("--weights", help="Optional trained heuristic weights JSON.")
    gauntlet.add_argument("--map", default="plains", choices=["plains", "desert"])
    gauntlet.add_argument("--heuristic-search-width", type=int, default=3)
    gauntlet.add_argument("--heuristic-search-depth", type=int, default=6)
    gauntlet.add_argument("--neural-scale", type=float, default=120.0)
    gauntlet.add_argument("--heuristic-scale", type=float, default=1.0)
    gauntlet.add_argument("--neural-search-width", type=int, default=1)
    gauntlet.add_argument("--neural-search-depth", type=int, default=1)
    gauntlet.add_argument(
        "--opponent-model",
        action="append",
        default=[],
        help="Additional neural checkpoint to include as an opponent. Can be passed more than once.",
    )
    gauntlet.add_argument("--no-auto-opponents", action="store_true", help="Do not auto-discover older checkpoints.")
    gauntlet.add_argument(
        "--only-neural-opponents",
        action="store_true",
        help="Skip random and heuristic baselines; useful for faster checkpoint head-to-heads.",
    )
    gauntlet.add_argument("--output", help="Optional JSON file for gauntlet results.")
    gauntlet.add_argument("--quiet", action="store_true", help="Suppress per-opponent progress output.")

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
            target_mode=args.target,
            exploration_rate=args.exploration_rate,
            sampling_top_k=args.sampling_top_k,
            sampling_temperature=args.sampling_temperature,
        )
        print(f"Wrote {count} examples to {args.output}")
        return 0
    if args.command == "train":
        try:
            history = train_value_model(
                dataset_path=args.data,
                model_path=args.model,
                hidden_size=args.hidden_size,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                seed=args.seed,
                limit=args.limit,
                progress=not args.quiet,
                backend=args.backend,
                batch_size=args.batch_size,
                device=args.device,
                validation_fraction=args.validation_fraction,
                early_stop_patience=args.early_stop_patience,
                early_stop_min_delta=args.early_stop_min_delta,
            )
        except RuntimeError as exc:
            raise SystemExit(str(exc)) from exc
        print(f"Saved value model to {args.model}")
        print("Loss history: " + ", ".join(f"{loss:.4f}" for loss in history))
        return 0
    if args.command == "evaluate":
        weights = load_weights(args.weights) if args.weights else None
        result = evaluate_value_bot(model_path=args.model, games=args.games, seed=args.seed, weights=weights)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "gauntlet":
        weights = load_weights(args.weights) if args.weights else None
        result = run_value_gauntlet(
            model_path=args.model,
            games=args.games,
            seed=args.seed,
            weights=weights,
            weights_path=args.weights,
            neural_opponent_models=args.opponent_model,
            auto_neural_opponents=not args.no_auto_opponents,
            include_baseline_opponents=not args.only_neural_opponents,
            map_name=args.map,
            heuristic_search_width=args.heuristic_search_width,
            heuristic_search_depth=args.heuristic_search_depth,
            neural_scale=args.neural_scale,
            heuristic_scale=args.heuristic_scale,
            neural_search_width=args.neural_search_width,
            neural_search_depth=args.neural_search_depth,
            progress=not args.quiet,
        )
        payload = json.dumps(result, indent=2, sort_keys=True)
        if args.output:
            directory = os.path.dirname(args.output)
            if directory:
                os.makedirs(directory, exist_ok=True)
            with open(args.output, "w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.write("\n")
        print(payload)
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
