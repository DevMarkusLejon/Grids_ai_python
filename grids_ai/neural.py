from __future__ import annotations

from dataclasses import dataclass, field
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
from .encoding import action_space_for_state, encode_action_index, encode_state_vector, encoded_state_size, legal_action_indices
from .engine import Action, GameState, new_game
from .training import play_match


MODEL_VERSION = 1
SparseFeatures = tuple[tuple[int, float], ...]
TrainBackend = str
TargetMode = str
TeacherMode = str


def nonzero_features(features: Sequence[float]) -> SparseFeatures:
    return tuple((index, float(value)) for index, value in enumerate(features) if value)


@dataclass(frozen=True)
class TrainingExample:
    features: list[float]
    value: float
    action_index: int | None = None
    legal_action_indices: tuple[int, ...] | None = None
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

    def _numpy_weights(self):
        cache = getattr(self, "_numpy_cache", None)
        if cache is not None:
            return cache
        try:
            import numpy as np
        except ImportError:
            self._numpy_cache = False
            return False
        cache = (
            np.asarray(self.w1, dtype=np.float32),
            np.asarray(self.b1, dtype=np.float32),
            np.asarray(self.w2, dtype=np.float32),
            np.asarray(self.b2, dtype=np.float32),
        )
        self._numpy_cache = cache
        return cache

    def predict_many(self, feature_batch: Sequence[Sequence[float]]) -> list[float]:
        if not feature_batch:
            return []
        numpy_weights = self._numpy_weights()
        if numpy_weights:
            try:
                import numpy as np

                w1, b1, w2, b2 = numpy_weights
                features = np.asarray(feature_batch, dtype=np.float32)
                hidden = np.tanh(features @ w1.T + b1)
                predictions = np.tanh(hidden @ w2 + b2)
                return [float(value) for value in predictions.tolist()]
            except (TypeError, ValueError):
                pass
        return [self.predict(features) for features in feature_batch]

    def predict_state_for_player(self, state: GameState, player: Side) -> float:
        if state.is_done:
            if state.winner is None:
                return 0.0
            return 1.0 if state.winner is player else -1.0
        prediction = self.predict(encode_state_vector(state))
        return prediction if state.current_side is player else -prediction

    def predict_states_for_player(self, states: Sequence[GameState], player: Side) -> list[float]:
        values: list[float | None] = []
        feature_batch: list[list[float]] = []
        feature_indices: list[int] = []
        for state in states:
            if state.is_done:
                if state.winner is None:
                    values.append(0.0)
                else:
                    values.append(1.0 if state.winner is player else -1.0)
                continue
            values.append(None)
            feature_indices.append(len(values) - 1)
            feature_batch.append(encode_state_vector(state))

        predictions = self.predict_many(feature_batch)
        for index, prediction in zip(feature_indices, predictions):
            state = states[index]
            values[index] = prediction if state.current_side is player else -prediction
        return [float(value) for value in values]

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


class PolicyValueNetwork:
    """Shared-trunk policy/value MLP exported as plain JSON for local inference."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        action_size: int,
        *,
        w1: list[list[float]],
        b1: list[float],
        value_w: list[float],
        value_b: float,
        policy_w: list[list[float]],
        policy_b: list[float],
    ) -> None:
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.action_size = action_size
        self.w1 = w1
        self.b1 = b1
        self.value_w = value_w
        self.value_b = value_b
        self.policy_w = policy_w
        self.policy_b = policy_b

    def _numpy_weights(self):
        cache = getattr(self, "_numpy_cache", None)
        if cache is not None:
            return cache
        try:
            import numpy as np
        except ImportError:
            self._numpy_cache = False
            return False
        cache = (
            np.asarray(self.w1, dtype=np.float32),
            np.asarray(self.b1, dtype=np.float32),
            np.asarray(self.value_w, dtype=np.float32),
            np.asarray(self.value_b, dtype=np.float32),
            np.asarray(self.policy_w, dtype=np.float32),
            np.asarray(self.policy_b, dtype=np.float32),
        )
        self._numpy_cache = cache
        return cache

    def _hidden(self, features: Sequence[float]) -> list[float]:
        hidden: list[float] = []
        for row, bias in zip(self.w1, self.b1):
            activation = bias
            for feature_index, value in enumerate(features):
                if value:
                    activation += row[feature_index] * value
            hidden.append(math.tanh(activation))
        return hidden

    def predict_value(self, features: Sequence[float]) -> float:
        numpy_weights = self._numpy_weights()
        if numpy_weights:
            try:
                import numpy as np

                w1, b1, value_w, value_b, _, _ = numpy_weights
                feature_array = np.asarray(features, dtype=np.float32)
                hidden = np.tanh(w1 @ feature_array + b1)
                return float(np.tanh(hidden @ value_w + value_b))
            except (TypeError, ValueError):
                pass
        hidden = self._hidden(features)
        output = self.value_b
        for weight, value in zip(self.value_w, hidden):
            output += weight * value
        return math.tanh(output)

    def predict_values(self, feature_batch: Sequence[Sequence[float]]) -> list[float]:
        if not feature_batch:
            return []
        numpy_weights = self._numpy_weights()
        if numpy_weights:
            try:
                import numpy as np

                w1, b1, value_w, value_b, _, _ = numpy_weights
                features = np.asarray(feature_batch, dtype=np.float32)
                hidden = np.tanh(features @ w1.T + b1)
                values = np.tanh(hidden @ value_w + value_b)
                return [float(value) for value in values.tolist()]
            except (TypeError, ValueError):
                pass
        return [self.predict_value(features) for features in feature_batch]

    def predict_policy_logits(self, features: Sequence[float], action_indices: Sequence[int]) -> list[float]:
        if not action_indices:
            return []
        numpy_weights = self._numpy_weights()
        if numpy_weights:
            try:
                import numpy as np

                w1, b1, _, _, policy_w, policy_b = numpy_weights
                feature_array = np.asarray(features, dtype=np.float32)
                hidden = np.tanh(w1 @ feature_array + b1)
                indices = np.asarray(action_indices, dtype=np.int64)
                logits = policy_w[indices] @ hidden + policy_b[indices]
                return [float(value) for value in logits.tolist()]
            except (TypeError, ValueError, IndexError):
                pass
        hidden = self._hidden(features)
        logits: list[float] = []
        for action_index in action_indices:
            if action_index < 0 or action_index >= self.action_size:
                logits.append(float("-inf"))
                continue
            output = self.policy_b[action_index]
            row = self.policy_w[action_index]
            for weight, value in zip(row, hidden):
                output += weight * value
            logits.append(output)
        return logits

    def action_priors(self, state: GameState, legal: Sequence[Action]) -> list[float]:
        if not legal:
            return []
        action_indices = [encode_action_index(state, action) for action in legal]
        logits = self.predict_policy_logits(encode_state_vector(state), action_indices)
        finite_logits = [value for value in logits if math.isfinite(value)]
        if not finite_logits:
            return [1.0 / len(legal) for _ in legal]
        maximum = max(finite_logits)
        exp_values = [math.exp(max(-60.0, min(60.0, value - maximum))) if math.isfinite(value) else 0.0 for value in logits]
        total = sum(exp_values)
        if total <= 0.0:
            return [1.0 / len(legal) for _ in legal]
        return [value / total for value in exp_values]

    def predict_state_for_player(self, state: GameState, player: Side) -> float:
        if state.is_done:
            if state.winner is None:
                return 0.0
            return 1.0 if state.winner is player else -1.0
        prediction = self.predict_value(encode_state_vector(state))
        return prediction if state.current_side is player else -prediction

    def predict_states_for_player(self, states: Sequence[GameState], player: Side) -> list[float]:
        values: list[float | None] = []
        feature_batch: list[list[float]] = []
        feature_indices: list[int] = []
        for state in states:
            if state.is_done:
                if state.winner is None:
                    values.append(0.0)
                else:
                    values.append(1.0 if state.winner is player else -1.0)
                continue
            values.append(None)
            feature_indices.append(len(values) - 1)
            feature_batch.append(encode_state_vector(state))
        predictions = self.predict_values(feature_batch)
        for index, prediction in zip(feature_indices, predictions):
            state = states[index]
            values[index] = prediction if state.current_side is player else -prediction
        return [float(value) for value in values]

    def to_dict(self, metadata: dict[str, object] | None = None) -> dict[str, object]:
        payload: dict[str, object] = {
            "version": MODEL_VERSION,
            "kind": "policy_value",
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "action_size": self.action_size,
            "w1": self.w1,
            "b1": self.b1,
            "value_w": self.value_w,
            "value_b": self.value_b,
            "policy_w": self.policy_w,
            "policy_b": self.policy_b,
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
    def from_dict(cls, data: dict[str, object]) -> "PolicyValueNetwork":
        return cls(
            input_size=int(data["input_size"]),
            hidden_size=int(data["hidden_size"]),
            action_size=int(data["action_size"]),
            w1=[[float(value) for value in row] for row in data["w1"]],
            b1=[float(value) for value in data["b1"]],
            value_w=[float(value) for value in data["value_w"]],
            value_b=float(data["value_b"]),
            policy_w=[[float(value) for value in row] for row in data["policy_w"]],
            policy_b=[float(value) for value in data["policy_b"]],
        )

    @classmethod
    def load(cls, path: str) -> "PolicyValueNetwork":
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return cls.from_dict(data)


NeuralModel = ValueNetwork | PolicyValueNetwork


def load_neural_model(path: str) -> NeuralModel:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if data.get("kind") == "policy_value" or "policy_w" in data:
        return PolicyValueNetwork.from_dict(data)
    return ValueNetwork.from_dict(data)


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

    def score_action_batch(
        self,
        candidates: Sequence[tuple[GameState, GameState, Action]],
        player: Side,
    ) -> list[float]:
        if not candidates:
            return []
        after_states = [after for _, after, _ in candidates]
        neural_scores = self.model.predict_states_for_player(after_states, player)
        scores: list[float] = []
        for (before, after, action), neural_score in zip(candidates, neural_scores):
            heuristic_score = self.heuristic.score_action(before, after, action, player)
            scores.append(heuristic_score * self.heuristic_scale + neural_score * self.neural_scale)
        return scores

    def choose_greedy_action(self, state: GameState, legal: list[Action], player: Side) -> Action:
        candidates: list[tuple[GameState, GameState, Action]] = []
        for action in legal:
            after = state.clone()
            after.apply_unchecked(action)
            candidates.append((state, after, action))
        scored = [
            (score, self.rng.random(), action)
            for score, (_, _, action) in zip(self.score_action_batch(candidates, player), candidates)
        ]
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return scored[0][2]

    def choose_planned_action(self, state: GameState, legal: list[Action], player: Side) -> Action:
        depth_limit = self.search_depth or state.config.max_actions + 1
        depth_limit = max(1, depth_limit)
        width = max(1, self.search_width)
        decay = 0.92

        initial_candidates: list[tuple[GameState, GameState, Action]] = []
        for action in legal:
            after = state.clone()
            after.apply_unchecked(action)
            initial_candidates.append((state, after, action))

        frontier: list[tuple[float, float, Action, GameState]] = []
        all_paths: list[tuple[float, float, Action, GameState]] = []
        for score, (_, after, action) in zip(self.score_action_batch(initial_candidates, player), initial_candidates):
            path = (score, self.rng.random(), action, after)
            frontier.append(path)
            all_paths.append(path)

        frontier.sort(key=lambda item: (item[0], item[1]), reverse=True)
        frontier = frontier[:width]

        for depth in range(1, depth_limit):
            candidate_rows: list[tuple[float, float, Action, GameState, GameState, Action]] = []
            for cumulative_score, tie_breaker, first_action, branch_state in frontier:
                if branch_state.is_done or branch_state.current_side is not player:
                    continue
                for action in branch_state.legal_actions():
                    after = branch_state.clone()
                    after.apply_unchecked(action)
                    candidate_rows.append((cumulative_score, tie_breaker, first_action, branch_state, after, action))
            step_scores = self.score_action_batch(
                [(branch_state, after, action) for _, _, _, branch_state, after, action in candidate_rows],
                player,
            )
            expanded: list[tuple[float, float, Action, GameState]] = []
            for step_score, (cumulative_score, tie_breaker, first_action, _, after, _) in zip(step_scores, candidate_rows):
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


class PolicyValueBot(NeuralValueBot):
    """Policy/value bot that uses the policy head as a legal-action prior."""

    def __init__(
        self,
        model: PolicyValueNetwork,
        *,
        fallback_weights: dict[str, float] | None = None,
        seed: int | None = None,
        neural_scale: float = 120.0,
        heuristic_scale: float = 1.0,
        policy_scale: float = 18.0,
        search_width: int = 1,
        search_depth: int | None = 1,
    ) -> None:
        super().__init__(
            model,
            fallback_weights=fallback_weights,
            seed=seed,
            neural_scale=neural_scale,
            heuristic_scale=heuristic_scale,
            search_width=search_width,
            search_depth=search_depth,
        )
        self.model = model
        self.policy_scale = policy_scale

    def score_action_batch(
        self,
        candidates: Sequence[tuple[GameState, GameState, Action]],
        player: Side,
    ) -> list[float]:
        scores = super().score_action_batch(candidates, player)
        priors_by_state: dict[int, dict[Action, float]] = {}
        for index, (before, _, action) in enumerate(candidates):
            cache_key = id(before)
            action_priors = priors_by_state.get(cache_key)
            if action_priors is None:
                legal = before.legal_actions()
                priors = self.model.action_priors(before, legal)
                action_priors = {legal_action: prior for legal_action, prior in zip(legal, priors)}
                priors_by_state[cache_key] = action_priors
            scores[index] += self.policy_scale * action_priors.get(action, 0.0)
        return scores


@dataclass
class MCTSChild:
    action: Action
    node: "MCTSNode"
    prior: float
    visits: int = 0
    value_sum: float = 0.0

    @property
    def value(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0


@dataclass
class MCTSNode:
    state: GameState
    visits: int = 0
    children: list[MCTSChild] = field(default_factory=list)
    expanded: bool = False


class PolicyValueMCTSBot(PolicyValueBot):
    """Bounded same-turn MCTS guided by policy priors and neural value leaves."""

    def __init__(
        self,
        model: PolicyValueNetwork,
        *,
        fallback_weights: dict[str, float] | None = None,
        seed: int | None = None,
        neural_scale: float = 120.0,
        heuristic_scale: float = 1.0,
        policy_scale: float = 18.0,
        search_width: int = 1,
        search_depth: int | None = 1,
        simulations: int = 64,
        exploration: float = 1.25,
        max_children: int = 24,
        mcts_depth: int = 7,
    ) -> None:
        super().__init__(
            model,
            fallback_weights=fallback_weights,
            seed=seed,
            neural_scale=neural_scale,
            heuristic_scale=heuristic_scale,
            policy_scale=policy_scale,
            search_width=search_width,
            search_depth=search_depth,
        )
        self.simulations = max(1, simulations)
        self.exploration = exploration
        self.max_children = max(1, max_children)
        self.mcts_depth = max(1, mcts_depth)

    def evaluate_leaf(self, state: GameState, player: Side) -> float:
        return self.model.predict_state_for_player(state, player)

    def expand_node(self, node: MCTSNode, player: Side, depth: int) -> float:
        state = node.state
        if state.is_done or state.current_side is not player or depth >= self.mcts_depth:
            node.expanded = True
            return self.evaluate_leaf(state, player)

        legal = state.legal_actions()
        priors = self.model.action_priors(state, legal)
        candidates: list[tuple[GameState, GameState, Action]] = []
        for action in legal:
            after = state.clone()
            after.apply_unchecked(action)
            candidates.append((state, after, action))
        scores = self.score_action_batch(candidates, player)
        rows = [
            {
                "action": action,
                "after": after,
                "prior": prior,
                "score": score,
            }
            for (_, after, action), prior, score in zip(candidates, priors, scores)
        ]
        chosen_by_action: dict[Action, dict[str, object]] = {}
        for row in sorted(rows, key=lambda item: float(item["prior"]), reverse=True)[: self.max_children]:
            chosen_by_action[row["action"]] = row
        for row in sorted(rows, key=lambda item: float(item["score"]), reverse=True)[: self.max_children]:
            if len(chosen_by_action) >= self.max_children and row["action"] not in chosen_by_action:
                continue
            chosen_by_action[row["action"]] = row
        end_turn = next((row for row in rows if row["action"].kind == "end_turn"), None)
        if end_turn is not None and all(action.kind != "end_turn" for action in chosen_by_action):
            if len(chosen_by_action) >= self.max_children:
                weakest = min(chosen_by_action.values(), key=lambda item: float(item["prior"]) + float(item["score"]) / 1000.0)
                chosen_by_action.pop(weakest["action"])
            chosen_by_action[end_turn["action"]] = end_turn

        node.children = []
        for row in chosen_by_action.values():
            child = MCTSChild(
                action=row["action"],
                node=MCTSNode(row["after"]),
                prior=float(row["prior"]),
                visits=1,
                value_sum=math.tanh(float(row["score"]) / 200.0),
            )
            node.children.append(child)
        node.expanded = True
        return self.evaluate_leaf(state, player)

    def select_child(self, node: MCTSNode) -> MCTSChild:
        parent_visits = max(1, node.visits)
        exploration = self.exploration * math.sqrt(parent_visits)
        return max(
            node.children,
            key=lambda child: (
                child.value + exploration * child.prior / (1 + child.visits),
                self.rng.random(),
            ),
        )

    def choose_mcts_action(self, state: GameState, legal: list[Action], player: Side) -> Action:
        root = MCTSNode(state.clone())
        self.expand_node(root, player, 0)
        if not root.children:
            return legal[0]

        for _ in range(self.simulations):
            node = root
            visited_nodes = [root]
            visited_children: list[MCTSChild] = []
            depth = 0
            while node.expanded and node.children and depth < self.mcts_depth:
                child = self.select_child(node)
                visited_children.append(child)
                node = child.node
                visited_nodes.append(node)
                depth += 1
                if node.state.is_done or node.state.current_side is not player:
                    break
            value = self.expand_node(node, player, depth)
            for visited_node in visited_nodes:
                visited_node.visits += 1
            for child in visited_children:
                child.visits += 1
                child.value_sum += value

        # Visit-count selection over-favors high-prior "end turn" in this same-turn planner.
        # Use the backed-up root value as the final choice while visits still guide exploration.
        best = max(root.children, key=lambda child: (child.value, child.visits, child.prior, self.rng.random()))
        return best.action

    def choose_action(self, state: GameState) -> Action:
        legal = state.legal_actions()
        if not legal:
            raise ValueError("No legal actions available.")
        return self.choose_mcts_action(state, legal, state.current_side)


def make_neural_bot(
    model: NeuralModel,
    *,
    fallback_weights: dict[str, float] | None = None,
    seed: int | None = None,
    neural_scale: float = 120.0,
    heuristic_scale: float = 1.0,
    policy_scale: float = 18.0,
    search_width: int = 1,
    search_depth: int | None = 1,
    mcts_simulations: int = 0,
    mcts_exploration: float = 1.25,
    mcts_max_children: int = 24,
    mcts_depth: int = 7,
) -> Bot:
    if isinstance(model, PolicyValueNetwork):
        if mcts_simulations > 0:
            return PolicyValueMCTSBot(
                model,
                fallback_weights=fallback_weights,
                seed=seed,
                neural_scale=neural_scale,
                heuristic_scale=heuristic_scale,
                policy_scale=policy_scale,
                search_width=search_width,
                search_depth=search_depth,
                simulations=mcts_simulations,
                exploration=mcts_exploration,
                max_children=mcts_max_children,
                mcts_depth=mcts_depth,
            )
        return PolicyValueBot(
            model,
            fallback_weights=fallback_weights,
            seed=seed,
            neural_scale=neural_scale,
            heuristic_scale=heuristic_scale,
            policy_scale=policy_scale,
            search_width=search_width,
            search_depth=search_depth,
        )
    return NeuralValueBot(
        model,
        fallback_weights=fallback_weights,
        seed=seed,
        neural_scale=neural_scale,
        heuristic_scale=heuristic_scale,
        search_width=search_width,
        search_depth=search_depth,
    )


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
_GAUNTLET_MODEL_CACHE: dict[str, NeuralModel] = {}


@dataclass(frozen=True)
class GauntletOpponent:
    name: str
    kind: str
    make_bot: BotFactory
    metadata: dict[str, object] | None = None


def _cached_gauntlet_model(path: str) -> NeuralModel:
    model = _GAUNTLET_MODEL_CACHE.get(path)
    if model is None:
        model = load_neural_model(path)
        _GAUNTLET_MODEL_CACHE[path] = model
    return model


def _gauntlet_opponent_spec(opponent: GauntletOpponent) -> dict[str, object]:
    metadata = dict(opponent.metadata or {})
    spec: dict[str, object] = {
        "name": opponent.name,
        "kind": opponent.kind,
        "metadata": metadata,
    }
    if opponent.kind == "heuristic":
        weights_path = metadata.get("weights")
        if isinstance(weights_path, str) and weights_path != "default" and os.path.exists(weights_path):
            spec["weights"] = load_weights(weights_path)
        else:
            spec["weights"] = dict(DEFAULT_WEIGHTS)
    elif opponent.kind == "neural":
        model_path = metadata.get("model")
        if isinstance(model_path, str):
            spec["model"] = model_path
    return spec


def _make_gauntlet_opponent_bot(
    spec: dict[str, object],
    *,
    seed: int,
    fallback_weights: dict[str, float],
    heuristic_search_width: int,
    heuristic_search_depth: int | None,
    neural_scale: float,
    heuristic_scale: float,
    policy_scale: float,
    neural_search_width: int,
    neural_search_depth: int | None,
    mcts_simulations: int,
    mcts_exploration: float,
    mcts_max_children: int,
    mcts_depth: int,
) -> Bot:
    kind = spec["kind"]
    if kind == "random":
        return RandomBot(seed=seed)
    if kind == "heuristic":
        weights = spec.get("weights")
        if not isinstance(weights, dict):
            weights = dict(DEFAULT_WEIGHTS)
        return HeuristicBot(
            dict(weights),
            seed=seed,
            search_width=heuristic_search_width,
            search_depth=heuristic_search_depth,
        )
    if kind == "neural":
        model_path = spec.get("model")
        if not isinstance(model_path, str):
            raise ValueError(f"Neural opponent is missing a model path: {spec.get('name')}")
        return make_neural_bot(
            _cached_gauntlet_model(model_path),
            fallback_weights=fallback_weights,
            seed=seed,
            neural_scale=neural_scale,
            heuristic_scale=heuristic_scale,
            policy_scale=policy_scale,
            search_width=neural_search_width,
            search_depth=neural_search_depth,
            mcts_simulations=mcts_simulations,
            mcts_exploration=mcts_exploration,
            mcts_max_children=mcts_max_children,
            mcts_depth=mcts_depth,
        )
    raise ValueError(f"Unknown gauntlet opponent kind: {kind}")


def _play_gauntlet_pair_task(task: dict[str, object]) -> dict[str, object]:
    model_path = str(task["model_path"])
    game_seed = int(task["game_seed"])
    map_name = str(task["map_name"])
    fallback_weights = dict(task["fallback_weights"])  # type: ignore[arg-type]
    spec = dict(task["opponent_spec"])  # type: ignore[arg-type]
    neural_scale = float(task["neural_scale"])
    heuristic_scale = float(task["heuristic_scale"])
    policy_scale = float(task["policy_scale"])
    neural_search_width = int(task["neural_search_width"])
    neural_search_depth = task["neural_search_depth"]
    neural_search_depth = int(neural_search_depth) if neural_search_depth is not None else None
    heuristic_search_width = int(task["heuristic_search_width"])
    heuristic_search_depth = task["heuristic_search_depth"]
    heuristic_search_depth = int(heuristic_search_depth) if heuristic_search_depth is not None else None
    mcts_simulations = int(task["mcts_simulations"])
    mcts_exploration = float(task["mcts_exploration"])
    mcts_max_children = int(task["mcts_max_children"])
    mcts_depth = int(task["mcts_depth"])
    model = _cached_gauntlet_model(model_path)

    subject_blue = make_neural_bot(
        model,
        fallback_weights=fallback_weights,
        seed=game_seed + 1,
        neural_scale=neural_scale,
        heuristic_scale=heuristic_scale,
        policy_scale=policy_scale,
        search_width=neural_search_width,
        search_depth=neural_search_depth,
        mcts_simulations=mcts_simulations,
        mcts_exploration=mcts_exploration,
        mcts_max_children=mcts_max_children,
        mcts_depth=mcts_depth,
    )
    opponent_red = _make_gauntlet_opponent_bot(
        spec,
        seed=game_seed + 2,
        fallback_weights=fallback_weights,
        heuristic_search_width=heuristic_search_width,
        heuristic_search_depth=heuristic_search_depth,
        neural_scale=neural_scale,
        heuristic_scale=heuristic_scale,
        policy_scale=policy_scale,
        neural_search_width=neural_search_width,
        neural_search_depth=neural_search_depth,
        mcts_simulations=mcts_simulations,
        mcts_exploration=mcts_exploration,
        mcts_max_children=mcts_max_children,
        mcts_depth=mcts_depth,
    )
    blue_state = play_match(subject_blue, opponent_red, seed=game_seed, map_name=map_name)
    blue_result = side_result(blue_state, Side.BLUE)

    opponent_blue = _make_gauntlet_opponent_bot(
        spec,
        seed=game_seed + 1002,
        fallback_weights=fallback_weights,
        heuristic_search_width=heuristic_search_width,
        heuristic_search_depth=heuristic_search_depth,
        neural_scale=neural_scale,
        heuristic_scale=heuristic_scale,
        policy_scale=policy_scale,
        neural_search_width=neural_search_width,
        neural_search_depth=neural_search_depth,
        mcts_simulations=mcts_simulations,
        mcts_exploration=mcts_exploration,
        mcts_max_children=mcts_max_children,
        mcts_depth=mcts_depth,
    )
    subject_red = make_neural_bot(
        model,
        fallback_weights=fallback_weights,
        seed=game_seed + 1001,
        neural_scale=neural_scale,
        heuristic_scale=heuristic_scale,
        policy_scale=policy_scale,
        search_width=neural_search_width,
        search_depth=neural_search_depth,
        mcts_simulations=mcts_simulations,
        mcts_exploration=mcts_exploration,
        mcts_max_children=mcts_max_children,
        mcts_depth=mcts_depth,
    )
    red_state = play_match(opponent_blue, subject_red, seed=game_seed, map_name=map_name)
    red_result = side_result(red_state, Side.RED)
    return {
        "wins": (1 if blue_result == 1 else 0) + (1 if red_result == 1 else 0),
        "draws": (1 if blue_result == 0 else 0) + (1 if red_result == 0 else 0),
        "losses": (1 if blue_result == -1 else 0) + (1 if red_result == -1 else 0),
        "blue_wins": 1 if blue_result == 1 else 0,
        "red_wins": 1 if red_result == 1 else 0,
        "half_turns": [blue_state.half_turns_played, red_state.half_turns_played],
    }


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
            "legal_action_indices": example.legal_action_indices,
            "side": example.side,
            "winner": example.winner,
            "half_turns": example.half_turns,
        },
        separators=(",", ":"),
    )


def example_from_json(line: str) -> TrainingExample:
    data = json.loads(line)
    features = [float(value) for value in data["features"]]
    raw_legal_action_indices = data.get("legal_action_indices")
    parsed_legal_action_indices = (
        tuple(int(index) for index in raw_legal_action_indices)
        if raw_legal_action_indices is not None
        else None
    )
    return TrainingExample(
        features=features,
        value=float(data["value"]),
        action_index=data.get("action_index"),
        legal_action_indices=parsed_legal_action_indices,
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


def load_examples_from_paths(
    paths: Sequence[str],
    *,
    limit: int | None = None,
    per_dataset_limit: int | None = None,
) -> list[TrainingExample]:
    examples: list[TrainingExample] = []
    for path in paths:
        remaining = None if limit is None else limit - len(examples)
        if remaining is not None and remaining <= 0:
            break
        path_limit = per_dataset_limit if remaining is None else min(remaining, per_dataset_limit or remaining)
        examples.extend(load_examples(path, limit=path_limit))
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
    teacher: TeacherMode = "heuristic",
    teacher_model_path: str | None = None,
    teacher_neural_scale: float = 120.0,
    teacher_heuristic_scale: float = 1.0,
    teacher_neural_search_width: int = 3,
    teacher_neural_search_depth: int | None = 4,
) -> list[TrainingExample]:
    state = new_game(seed=seed, map_name=map_name)
    if teacher == "heuristic":
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
    elif teacher == "neural":
        if not teacher_model_path:
            raise ValueError("--teacher-model is required when --teacher neural.")
        model = load_neural_model(teacher_model_path)
        bots = {
            Side.BLUE: make_neural_bot(
                model,
                fallback_weights=weights,
                seed=seed + 1,
                neural_scale=teacher_neural_scale,
                heuristic_scale=teacher_heuristic_scale,
                search_width=teacher_neural_search_width,
                search_depth=teacher_neural_search_depth,
            ),
            Side.RED: make_neural_bot(
                model,
                fallback_weights=weights,
                seed=seed + 2,
                neural_scale=teacher_neural_scale,
                heuristic_scale=teacher_heuristic_scale,
                search_width=teacher_neural_search_width,
                search_depth=teacher_neural_search_depth,
            ),
        }
    else:
        raise ValueError("--teacher must be one of: heuristic, neural.")
    pending: list[tuple[list[float], int, tuple[int, ...], Side, int]] = []
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
                    tuple(legal_action_indices(state)),
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
            legal_action_indices=legal_indices,
            side=side.value,
            winner=state.winner.value if state.winner else None,
            half_turns=half_turns,
        )
        for features, action_index, legal_indices, side, half_turns in pending
    ]


def generate_game_examples_task(
    task: tuple[
        int,
        dict[str, float],
        str,
        int,
        int | None,
        int,
        int,
        TargetMode,
        float,
        int,
        float,
        TeacherMode,
        str | None,
        float,
        float,
        int,
        int | None,
    ],
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
        teacher,
        teacher_model_path,
        teacher_neural_scale,
        teacher_heuristic_scale,
        teacher_neural_search_width,
        teacher_neural_search_depth,
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
        teacher=teacher,
        teacher_model_path=teacher_model_path,
        teacher_neural_scale=teacher_neural_scale,
        teacher_heuristic_scale=teacher_heuristic_scale,
        teacher_neural_search_width=teacher_neural_search_width,
        teacher_neural_search_depth=teacher_neural_search_depth,
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
    teacher: TeacherMode = "heuristic",
    teacher_model_path: str | None = None,
    teacher_neural_scale: float = 120.0,
    teacher_heuristic_scale: float = 1.0,
    teacher_neural_search_width: int = 3,
    teacher_neural_search_depth: int | None = 4,
) -> int:
    if target_mode not in {"outcome", "margin", "shaped"}:
        raise ValueError("--target must be one of: outcome, margin, shaped.")
    if teacher not in {"heuristic", "neural"}:
        raise ValueError("--teacher must be one of: heuristic, neural.")
    if teacher == "neural" and not teacher_model_path:
        raise ValueError("--teacher-model is required when --teacher neural.")
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
            f"teacher={teacher}, exploration={exploration_rate}, top_k={sampling_top_k}, "
            f"temperature={sampling_temperature})",
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
            teacher,
            teacher_model_path,
            teacher_neural_scale,
            teacher_heuristic_scale,
            teacher_neural_search_width,
            teacher_neural_search_depth,
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
    dataset_path: str | Sequence[str],
    model_path: str,
    hidden_size: int = 48,
    epochs: int = 8,
    learning_rate: float = 0.003,
    seed: int = 13,
    limit: int | None = None,
    per_dataset_limit: int | None = None,
    progress: bool = False,
    backend: TrainBackend = "auto",
    batch_size: int = 256,
    device: str = "auto",
    validation_fraction: float = 0.1,
    early_stop_patience: int = 0,
    early_stop_min_delta: float = 0.0,
) -> list[float]:
    dataset_paths = [dataset_path] if isinstance(dataset_path, str) else list(dataset_path)
    if not dataset_paths:
        raise ValueError("At least one training dataset is required.")
    if per_dataset_limit is not None and per_dataset_limit < 1:
        raise ValueError("--per-data-limit must be at least 1.")
    examples = load_examples_from_paths(dataset_paths, limit=limit, per_dataset_limit=per_dataset_limit)
    if not examples:
        raise ValueError(f"No training examples found in {dataset_paths}.")
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
            f"datasets={len(dataset_paths)} "
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
        "dataset": dataset_paths[0] if len(dataset_paths) == 1 else dataset_paths,
        "examples": len(examples),
        "per_dataset_limit": per_dataset_limit,
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


def train_torch_policy_value_model(
    examples: Sequence[TrainingExample],
    *,
    validation_examples: Sequence[TrainingExample],
    hidden_size: int,
    action_size: int,
    epochs: int,
    learning_rate: float,
    seed: int,
    batch_size: int,
    device: str,
    value_loss_weight: float,
    policy_loss_weight: float,
    init_model: PolicyValueNetwork | None,
    freeze_init_model: bool,
    early_stop_patience: int,
    early_stop_min_delta: float,
    progress: bool,
) -> tuple[PolicyValueNetwork, list[float], dict[str, object]]:
    torch = require_torch()
    if device == "auto":
        chosen_device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        chosen_device = device
    target_device = torch.device(chosen_device)
    torch.manual_seed(seed)

    features = torch.tensor([example.features for example in examples], dtype=torch.float32, device=target_device)
    value_targets = torch.tensor(
        [max(-1.0, min(1.0, example.value)) for example in examples],
        dtype=torch.float32,
        device=target_device,
    )
    action_targets = torch.tensor(
        [int(example.action_index) for example in examples],
        dtype=torch.long,
        device=target_device,
    )
    training_legal_action_indices = [example.legal_action_indices for example in examples]
    if validation_examples:
        validation_features = torch.tensor(
            [example.features for example in validation_examples],
            dtype=torch.float32,
            device=target_device,
        )
        validation_values = torch.tensor(
            [max(-1.0, min(1.0, example.value)) for example in validation_examples],
            dtype=torch.float32,
            device=target_device,
        )
        validation_actions = torch.tensor(
            [int(example.action_index) for example in validation_examples],
            dtype=torch.long,
            device=target_device,
        )
        validation_legal_action_indices = [example.legal_action_indices for example in validation_examples]
    else:
        validation_features = None
        validation_values = None
        validation_actions = None
        validation_legal_action_indices = None

    input_size = int(features.shape[1])

    class TorchPolicyValueModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.hidden = torch.nn.Linear(input_size, hidden_size)
            self.value = torch.nn.Linear(hidden_size, 1)
            self.policy = torch.nn.Linear(hidden_size, action_size)
            scale = 1.0 / math.sqrt(max(input_size, 1))
            with torch.no_grad():
                self.hidden.weight.uniform_(-scale, scale)
                self.hidden.bias.zero_()
                self.value.weight.uniform_(-scale, scale)
                self.value.bias.zero_()
                self.policy.weight.uniform_(-scale, scale)
                self.policy.bias.zero_()

        def forward(self, x):
            hidden = torch.tanh(self.hidden(x))
            return torch.tanh(self.value(hidden)).squeeze(-1), self.policy(hidden)

    torch_model = TorchPolicyValueModel().to(target_device)
    init_hidden_size = None
    if init_model is not None:
        if init_model.input_size != input_size:
            raise ValueError(f"--init-model input_size={init_model.input_size} does not match training input_size={input_size}.")
        if init_model.hidden_size > hidden_size:
            raise ValueError(
                f"--init-model hidden_size={init_model.hidden_size} cannot be copied into smaller --hidden-size={hidden_size}."
            )
        if init_model.action_size != action_size:
            raise ValueError(f"--init-model action_size={init_model.action_size} does not match action_size={action_size}.")
        init_hidden_size = init_model.hidden_size
        with torch.no_grad():
            torch_model.hidden.weight[:init_hidden_size].copy_(
                torch.tensor(init_model.w1, dtype=torch.float32, device=target_device)
            )
            torch_model.hidden.bias[:init_hidden_size].copy_(
                torch.tensor(init_model.b1, dtype=torch.float32, device=target_device)
            )
            torch_model.value.weight[:, :init_hidden_size].copy_(
                torch.tensor([init_model.value_w], dtype=torch.float32, device=target_device)
            )
            torch_model.value.bias.copy_(torch.tensor([init_model.value_b], dtype=torch.float32, device=target_device))
            torch_model.policy.weight[:, :init_hidden_size].copy_(
                torch.tensor(init_model.policy_w, dtype=torch.float32, device=target_device)
            )
            torch_model.policy.bias.copy_(torch.tensor(init_model.policy_b, dtype=torch.float32, device=target_device))
            if init_hidden_size < hidden_size:
                torch_model.value.weight[:, init_hidden_size:].zero_()
                torch_model.policy.weight[:, init_hidden_size:].zero_()
    optimizer = torch.optim.Adam(torch_model.parameters(), lr=learning_rate)
    batch_size = max(1, batch_size)
    rng = random.Random(seed)
    indices = list(range(len(examples)))
    history: list[float] = []
    value_history: list[float] = []
    policy_history: list[float] = []
    validation_history: list[float] = []
    validation_value_history: list[float] = []
    validation_policy_history: list[float] = []
    validation_accuracy_history: list[float] = []
    best_state = {name: value.detach().clone() for name, value in torch_model.state_dict().items()}
    best_validation_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0
    started_at = time.perf_counter()
    masked_policy_examples = sum(1 for indices in training_legal_action_indices if indices)
    masked_validation_policy_examples = (
        sum(1 for indices in validation_legal_action_indices if indices)
        if validation_legal_action_indices is not None
        else 0
    )

    def policy_cross_entropy(
        logits,
        targets,
        batch_legal_action_indices: Sequence[tuple[int, ...] | None],
    ):
        if not batch_legal_action_indices or not any(indices for indices in batch_legal_action_indices):
            return torch.nn.functional.cross_entropy(logits, targets), torch.argmax(logits, dim=1)
        mask = torch.ones(logits.shape, dtype=torch.bool, device=logits.device)
        any_masked = False
        for row, legal_indices in enumerate(batch_legal_action_indices):
            if not legal_indices:
                continue
            valid_indices = {
                int(index)
                for index in legal_indices
                if 0 <= int(index) < action_size
            }
            valid_indices.add(int(targets[row].detach().cpu()))
            if not valid_indices:
                continue
            mask[row].zero_()
            index_tensor = torch.tensor(sorted(valid_indices), dtype=torch.long, device=logits.device)
            mask[row, index_tensor] = True
            any_masked = True
        if not any_masked:
            return torch.nn.functional.cross_entropy(logits, targets), torch.argmax(logits, dim=1)
        masked_logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        return torch.nn.functional.cross_entropy(masked_logits, targets), torch.argmax(masked_logits, dim=1)

    def evaluate_validation() -> tuple[float, float, float, float] | None:
        if validation_features is None or validation_values is None or validation_actions is None:
            return None
        total_value_loss = 0.0
        total_policy_loss = 0.0
        correct = 0
        count = 0
        with torch.no_grad():
            for start in range(0, len(validation_examples), batch_size):
                end = start + batch_size
                values, logits = torch_model(validation_features[start:end])
                batch_values = validation_values[start:end]
                batch_actions = validation_actions[start:end]
                batch_legal_indices = validation_legal_action_indices[start:end] if validation_legal_action_indices else []
                value_loss = torch.mean((values - batch_values) ** 2)
                policy_loss, predictions = policy_cross_entropy(logits, batch_actions, batch_legal_indices)
                batch_count = int(len(batch_actions))
                total_value_loss += float(value_loss.detach().cpu()) * batch_count
                total_policy_loss += float(policy_loss.detach().cpu()) * batch_count
                correct += int((predictions == batch_actions).sum().detach().cpu())
                count += batch_count
        value_loss = total_value_loss / max(count, 1)
        policy_loss = total_policy_loss / max(count, 1)
        total_loss = value_loss * value_loss_weight + policy_loss * policy_loss_weight
        accuracy = correct / max(count, 1)
        return total_loss, value_loss, policy_loss, accuracy

    for epoch in range(1, epochs + 1):
        epoch_started = time.perf_counter()
        rng.shuffle(indices)
        total_loss = 0.0
        total_value_loss = 0.0
        total_policy_loss = 0.0
        for start in range(0, len(indices), batch_size):
            batch_example_indices = indices[start : start + batch_size]
            batch_indices = torch.tensor(batch_example_indices, dtype=torch.long, device=target_device)
            values, logits = torch_model(features.index_select(0, batch_indices))
            batch_values = value_targets.index_select(0, batch_indices)
            batch_actions = action_targets.index_select(0, batch_indices)
            batch_legal_indices = [training_legal_action_indices[index] for index in batch_example_indices]
            value_loss = torch.mean((values - batch_values) ** 2)
            policy_loss, _ = policy_cross_entropy(logits, batch_actions, batch_legal_indices)
            loss = value_loss * value_loss_weight + policy_loss * policy_loss_weight
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if freeze_init_model and init_hidden_size is not None:
                torch_model.hidden.weight.grad[:init_hidden_size].zero_()
                torch_model.hidden.bias.grad[:init_hidden_size].zero_()
                torch_model.value.weight.grad[:, :init_hidden_size].zero_()
                torch_model.value.bias.grad.zero_()
                torch_model.policy.weight.grad[:, :init_hidden_size].zero_()
                torch_model.policy.bias.grad.zero_()
            optimizer.step()
            batch_count = len(batch_indices)
            total_loss += float(loss.detach().cpu()) * batch_count
            total_value_loss += float(value_loss.detach().cpu()) * batch_count
            total_policy_loss += float(policy_loss.detach().cpu()) * batch_count

        average_loss = total_loss / len(examples)
        average_value_loss = total_value_loss / len(examples)
        average_policy_loss = total_policy_loss / len(examples)
        history.append(average_loss)
        value_history.append(average_value_loss)
        policy_history.append(average_policy_loss)
        validation_metrics = evaluate_validation()
        validation_part = ""
        if validation_metrics is not None:
            validation_loss, validation_value_loss, validation_policy_loss, validation_accuracy = validation_metrics
            validation_history.append(validation_loss)
            validation_value_history.append(validation_value_loss)
            validation_policy_history.append(validation_policy_loss)
            validation_accuracy_history.append(validation_accuracy)
            validation_part = (
                f" val_loss={validation_loss:.5f}"
                f" val_value={validation_value_loss:.5f}"
                f" val_policy={validation_policy_loss:.5f}"
                f" val_acc={validation_accuracy:.3f}"
            )
            if validation_loss < best_validation_loss - early_stop_min_delta:
                best_validation_loss = validation_loss
                best_epoch = epoch
                best_state = {name: value.detach().clone() for name, value in torch_model.state_dict().items()}
                stale_epochs = 0
            else:
                stale_epochs += 1

        if progress:
            elapsed = time.perf_counter() - started_at
            print(
                f"[policy train] epoch {epoch}/{epochs} "
                f"loss={average_loss:.5f} value={average_value_loss:.5f} "
                f"policy={average_policy_loss:.5f}{validation_part} "
                f"epoch={time.perf_counter() - epoch_started:.1f}s elapsed={elapsed:.1f}s",
                flush=True,
            )

        if validation_metrics is not None and early_stop_patience > 0 and stale_epochs >= early_stop_patience:
            if progress:
                print(
                    f"[policy train] early stop at epoch {epoch}; "
                    f"best_epoch={best_epoch} best_val_loss={best_validation_loss:.5f}",
                    flush=True,
                )
            break

    if validation_examples:
        torch_model.load_state_dict(best_state)
    with torch.no_grad():
        model = PolicyValueNetwork(
            input_size=input_size,
            hidden_size=hidden_size,
            action_size=action_size,
            w1=torch_model.hidden.weight.detach().cpu().tolist(),
            b1=torch_model.hidden.bias.detach().cpu().tolist(),
            value_w=torch_model.value.weight.detach().cpu().reshape(-1).tolist(),
            value_b=float(torch_model.value.bias.detach().cpu().reshape(-1)[0]),
            policy_w=torch_model.policy.weight.detach().cpu().tolist(),
            policy_b=torch_model.policy.bias.detach().cpu().tolist(),
        )
    metadata = {
        "training_backend": "torch-policy-value",
        "batch_size": batch_size,
        "optimizer": "adam",
        "device": str(target_device),
        "torch_version": str(torch.__version__),
        "initialized_from_model": init_model is not None,
        "init_hidden_size": init_hidden_size,
        "frozen_init_model": bool(freeze_init_model and init_model is not None),
        "masked_policy_examples": masked_policy_examples,
        "masked_validation_policy_examples": masked_validation_policy_examples,
        "value_loss_weight": value_loss_weight,
        "policy_loss_weight": policy_loss_weight,
        "value_loss_history": value_history,
        "policy_loss_history": policy_history,
        "validation_loss_history": validation_history,
        "validation_value_loss_history": validation_value_history,
        "validation_policy_loss_history": validation_policy_history,
        "validation_policy_accuracy_history": validation_accuracy_history,
        "best_epoch": best_epoch or len(history),
        "best_validation_loss": best_validation_loss if validation_examples else None,
        "best_validation_policy_accuracy": max(validation_accuracy_history) if validation_accuracy_history else None,
    }
    return model, history, metadata


def train_policy_value_model(
    *,
    dataset_path: str | Sequence[str],
    model_path: str,
    hidden_size: int = 128,
    action_size: int | None = None,
    epochs: int = 12,
    learning_rate: float = 0.003,
    seed: int = 37,
    limit: int | None = None,
    per_dataset_limit: int | None = None,
    progress: bool = False,
    batch_size: int = 256,
    device: str = "auto",
    validation_fraction: float = 0.1,
    early_stop_patience: int = 0,
    early_stop_min_delta: float = 0.0,
    value_loss_weight: float = 1.0,
    policy_loss_weight: float = 0.25,
    map_name: str = "plains",
    init_model_path: str | None = None,
    freeze_init_model: bool = False,
) -> list[float]:
    dataset_paths = [dataset_path] if isinstance(dataset_path, str) else list(dataset_path)
    if not dataset_paths:
        raise ValueError("At least one training dataset is required.")
    if per_dataset_limit is not None and per_dataset_limit < 1:
        raise ValueError("--per-data-limit must be at least 1.")
    if batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")
    if early_stop_patience < 0:
        raise ValueError("--early-stop-patience must be zero or greater.")
    if early_stop_min_delta < 0:
        raise ValueError("--early-stop-min-delta must be zero or greater.")
    if value_loss_weight <= 0.0 or policy_loss_weight < 0.0:
        raise ValueError("Value loss weight must be positive and policy loss weight must be non-negative.")

    inferred_action_size = action_size or action_space_for_state(new_game(seed=seed, map_name=map_name)).size
    init_model: PolicyValueNetwork | None = None
    if init_model_path:
        loaded_init_model = load_neural_model(init_model_path)
        if not isinstance(loaded_init_model, PolicyValueNetwork):
            raise ValueError("--init-model must point to a policy/value model.")
        init_model = loaded_init_model
    all_examples = load_examples_from_paths(dataset_paths, limit=limit, per_dataset_limit=per_dataset_limit)
    examples = [
        example
        for example in all_examples
        if example.action_index is not None and 0 <= int(example.action_index) < inferred_action_size
    ]
    if not examples:
        raise ValueError(f"No policy-labelled examples found in {dataset_paths}.")
    training_examples, validation_examples = split_examples(
        examples,
        validation_fraction=validation_fraction,
        seed=seed + 2003,
    )
    if progress:
        print(
            f"[policy train] loaded examples={len(examples)} "
            f"datasets={len(dataset_paths)} train={len(training_examples)} "
            f"validation={len(validation_examples)} input_size={len(examples[0].features)} "
            f"hidden_size={hidden_size} action_size={inferred_action_size}",
            flush=True,
        )

    model, history, backend_metadata = train_torch_policy_value_model(
        training_examples,
        validation_examples=validation_examples,
        hidden_size=hidden_size,
        action_size=inferred_action_size,
        epochs=epochs,
        learning_rate=learning_rate,
        seed=seed,
        batch_size=batch_size,
        device=device,
        value_loss_weight=value_loss_weight,
        policy_loss_weight=policy_loss_weight,
        init_model=init_model,
        freeze_init_model=freeze_init_model,
        early_stop_patience=early_stop_patience,
        early_stop_min_delta=early_stop_min_delta,
        progress=progress,
    )
    metadata: dict[str, object] = {
        "dataset": dataset_paths[0] if len(dataset_paths) == 1 else dataset_paths,
        "examples": len(examples),
        "raw_examples": len(all_examples),
        "per_dataset_limit": per_dataset_limit,
        "training_examples": len(training_examples),
        "validation_examples": len(validation_examples),
        "validation_fraction": validation_fraction,
        "early_stop_patience": early_stop_patience,
        "early_stop_min_delta": early_stop_min_delta,
        "epochs": epochs,
        "completed_epochs": len(history),
        "learning_rate": learning_rate,
        "loss_history": history,
        "action_size": inferred_action_size,
        "map": map_name,
        "init_model": init_model_path,
        "freeze_init_model": freeze_init_model,
    }
    metadata.update(backend_metadata)
    model.save(model_path, metadata=metadata)
    return history


def evaluate_value_bot(
    *,
    model_path: str,
    games: int = 6,
    seed: int = 23,
    weights: dict[str, float] | None = None,
) -> dict[str, object]:
    model = load_neural_model(model_path)
    fallback_weights = dict(DEFAULT_WEIGHTS)
    if weights:
        fallback_weights.update(weights)
    wins = {Side.BLUE.value: 0, Side.RED.value: 0, "none": 0}
    half_turns: list[int] = []
    for game_index in range(games):
        state = play_match(
            make_neural_bot(model, fallback_weights=fallback_weights, seed=seed + game_index),
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
        "model_kind": "policy_value" if isinstance(model, PolicyValueNetwork) else "value",
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
    policy_scale: float = 18.0,
    neural_search_width: int = 1,
    neural_search_depth: int | None = 1,
    mcts_simulations: int = 0,
    mcts_exploration: float = 1.25,
    mcts_max_children: int = 24,
    mcts_depth: int = 7,
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
        model = load_neural_model(path)
        opponents.append(
            GauntletOpponent(
                name=f"neural:{os.path.basename(path)}",
                kind="neural",
                make_bot=lambda seed, loaded=model: make_neural_bot(
                    loaded,
                    fallback_weights=fallback_weights,
                    seed=seed,
                    neural_scale=neural_scale,
                    heuristic_scale=heuristic_scale,
                    policy_scale=policy_scale,
                    search_width=neural_search_width,
                    search_depth=neural_search_depth,
                    mcts_simulations=mcts_simulations,
                    mcts_exploration=mcts_exploration,
                    mcts_max_children=mcts_max_children,
                    mcts_depth=mcts_depth,
                ),
                metadata={
                    "model": path,
                    "hidden_size": model.hidden_size,
                    "model_kind": "policy_value" if isinstance(model, PolicyValueNetwork) else "value",
                    "action_size": model.action_size if isinstance(model, PolicyValueNetwork) else None,
                    "neural_scale": neural_scale,
                    "heuristic_scale": heuristic_scale,
                    "policy_scale": policy_scale if isinstance(model, PolicyValueNetwork) else None,
                    "search_width": neural_search_width,
                    "search_depth": neural_search_depth,
                    "mcts_simulations": mcts_simulations if isinstance(model, PolicyValueNetwork) else 0,
                    "mcts_max_children": mcts_max_children if isinstance(model, PolicyValueNetwork) else None,
                    "mcts_depth": mcts_depth if isinstance(model, PolicyValueNetwork) else None,
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
    policy_scale: float = 18.0,
    neural_search_width: int = 1,
    neural_search_depth: int | None = 1,
    mcts_simulations: int = 0,
    mcts_exploration: float = 1.25,
    mcts_max_children: int = 24,
    mcts_depth: int = 7,
    workers: int = 1,
    progress: bool = False,
) -> dict[str, object]:
    if games < 1:
        raise ValueError("--games must be at least 1.")
    if workers < 1:
        raise ValueError("--workers must be at least 1.")
    model = load_neural_model(model_path)
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
        policy_scale=policy_scale,
        neural_search_width=neural_search_width,
        neural_search_depth=neural_search_depth,
        mcts_simulations=mcts_simulations,
        mcts_exploration=mcts_exploration,
        mcts_max_children=mcts_max_children,
        mcts_depth=mcts_depth,
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
        if workers == 1:
            for game_index in range(games):
                game_seed = seed + opponent_index * 10000 + game_index
                subject_blue = make_neural_bot(
                    model,
                    fallback_weights=fallback_weights,
                    seed=game_seed + 1,
                    neural_scale=neural_scale,
                    heuristic_scale=heuristic_scale,
                    policy_scale=policy_scale,
                    search_width=neural_search_width,
                    search_depth=neural_search_depth,
                    mcts_simulations=mcts_simulations,
                    mcts_exploration=mcts_exploration,
                    mcts_max_children=mcts_max_children,
                    mcts_depth=mcts_depth,
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
                subject_red = make_neural_bot(
                    model,
                    fallback_weights=fallback_weights,
                    seed=game_seed + 1001,
                    neural_scale=neural_scale,
                    heuristic_scale=heuristic_scale,
                    policy_scale=policy_scale,
                    search_width=neural_search_width,
                    search_depth=neural_search_depth,
                    mcts_simulations=mcts_simulations,
                    mcts_exploration=mcts_exploration,
                    mcts_max_children=mcts_max_children,
                    mcts_depth=mcts_depth,
                )
                red_state = play_match(opponent_blue, subject_red, seed=game_seed, map_name=map_name)
                result = side_result(red_state, Side.RED)
                wins += 1 if result == 1 else 0
                draws += 1 if result == 0 else 0
                losses += 1 if result == -1 else 0
                red_wins += 1 if result == 1 else 0
                half_turns.append(red_state.half_turns_played)
        else:
            opponent_spec = _gauntlet_opponent_spec(opponent)
            tasks = [
                {
                    "model_path": model_path,
                    "opponent_spec": opponent_spec,
                    "game_seed": seed + opponent_index * 10000 + game_index,
                    "map_name": map_name,
                    "fallback_weights": fallback_weights,
                    "heuristic_search_width": heuristic_search_width,
                    "heuristic_search_depth": heuristic_search_depth,
                    "neural_scale": neural_scale,
                    "heuristic_scale": heuristic_scale,
                    "policy_scale": policy_scale,
                    "neural_search_width": neural_search_width,
                    "neural_search_depth": neural_search_depth,
                    "mcts_simulations": mcts_simulations,
                    "mcts_exploration": mcts_exploration,
                    "mcts_max_children": mcts_max_children,
                    "mcts_depth": mcts_depth,
                }
                for game_index in range(games)
            ]
            with ProcessPoolExecutor(max_workers=min(workers, games)) as executor:
                futures = [executor.submit(_play_gauntlet_pair_task, task) for task in tasks]
                for future in as_completed(futures):
                    paired_result = future.result()
                    wins += int(paired_result["wins"])
                    draws += int(paired_result["draws"])
                    losses += int(paired_result["losses"])
                    blue_wins += int(paired_result["blue_wins"])
                    red_wins += int(paired_result["red_wins"])
                    half_turns.extend(int(value) for value in paired_result["half_turns"])

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
        "model_kind": "policy_value" if isinstance(model, PolicyValueNetwork) else "value",
        "action_size": model.action_size if isinstance(model, PolicyValueNetwork) else None,
        "map": map_name,
        "games_per_side": games,
        "total_games": total_games,
        "neural_scale": neural_scale,
        "heuristic_scale": heuristic_scale,
        "policy_scale": policy_scale if isinstance(model, PolicyValueNetwork) else None,
        "neural_search_width": neural_search_width,
        "neural_search_depth": neural_search_depth,
        "mcts_simulations": mcts_simulations if isinstance(model, PolicyValueNetwork) else 0,
        "mcts_exploration": mcts_exploration if isinstance(model, PolicyValueNetwork) else None,
        "mcts_max_children": mcts_max_children if isinstance(model, PolicyValueNetwork) else None,
        "mcts_depth": mcts_depth if isinstance(model, PolicyValueNetwork) else None,
        "workers": workers,
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


def wilson_lower_bound(score_rate: float, total_games: int, *, z: float = 1.96) -> float:
    if total_games <= 0:
        return 0.0
    denominator = 1.0 + z * z / total_games
    center = score_rate + z * z / (2.0 * total_games)
    margin = z * math.sqrt((score_rate * (1.0 - score_rate) + z * z / (4.0 * total_games)) / total_games)
    return max(0.0, (center - margin) / denominator)


def run_champion_gauntlet(
    *,
    candidate_model_path: str,
    champion_model_path: str,
    games: int = 8,
    seed: int = 101,
    weights: dict[str, float] | None = None,
    weights_path: str | None = None,
    extra_opponent_models: Sequence[str] = (),
    include_baseline_opponents: bool = True,
    map_name: str = "plains",
    heuristic_search_width: int = 3,
    heuristic_search_depth: int | None = 6,
    neural_scale: float = 120.0,
    heuristic_scale: float = 1.0,
    policy_scale: float = 18.0,
    neural_search_width: int = 1,
    neural_search_depth: int | None = 1,
    mcts_simulations: int = 0,
    mcts_exploration: float = 1.25,
    mcts_max_children: int = 24,
    mcts_depth: int = 7,
    min_head_to_head_score: float = 0.55,
    min_overall_score: float = 0.60,
    min_head_to_head_lower_bound: float = 0.50,
    workers: int = 1,
    progress: bool = False,
) -> dict[str, object]:
    if os.path.abspath(candidate_model_path) == os.path.abspath(champion_model_path):
        raise ValueError("Candidate and champion model paths must be different.")

    opponent_models = [champion_model_path]
    for path in extra_opponent_models:
        if os.path.abspath(path) != os.path.abspath(champion_model_path):
            opponent_models.append(path)

    result = run_value_gauntlet(
        model_path=candidate_model_path,
        games=games,
        seed=seed,
        weights=weights,
        weights_path=weights_path,
        neural_opponent_models=opponent_models,
        auto_neural_opponents=False,
        include_baseline_opponents=include_baseline_opponents,
        map_name=map_name,
        heuristic_search_width=heuristic_search_width,
        heuristic_search_depth=heuristic_search_depth,
        neural_scale=neural_scale,
        heuristic_scale=heuristic_scale,
        policy_scale=policy_scale,
        neural_search_width=neural_search_width,
        neural_search_depth=neural_search_depth,
        mcts_simulations=mcts_simulations,
        mcts_exploration=mcts_exploration,
        mcts_max_children=mcts_max_children,
        mcts_depth=mcts_depth,
        workers=workers,
        progress=progress,
    )

    champion_basename = os.path.basename(champion_model_path)
    champion_row = next(
        (
            row
            for row in result["opponents"]
            if row.get("kind") == "neural" and str(row.get("opponent", "")).endswith(champion_basename)
        ),
        None,
    )
    if champion_row is None:
        raise ValueError(f"Champion opponent was not evaluated: {champion_model_path}")

    head_to_head_score = float(champion_row["score_rate"])
    head_to_head_games = int(champion_row["total_games"])
    head_to_head_lower_bound = wilson_lower_bound(head_to_head_score, head_to_head_games)
    overall_score = float(result["overall"]["score_rate"])
    promote = (
        head_to_head_score >= min_head_to_head_score
        and overall_score >= min_overall_score
        and head_to_head_lower_bound >= min_head_to_head_lower_bound
    )
    result["champion_decision"] = {
        "candidate_model": candidate_model_path,
        "champion_model": champion_model_path,
        "promote": promote,
        "reason": "candidate cleared promotion thresholds" if promote else "candidate did not clear promotion thresholds",
        "head_to_head_score_rate": head_to_head_score,
        "head_to_head_games": head_to_head_games,
        "head_to_head_lower_bound": head_to_head_lower_bound,
        "overall_score_rate": overall_score,
        "thresholds": {
            "min_head_to_head_score": min_head_to_head_score,
            "min_overall_score": min_overall_score,
            "min_head_to_head_lower_bound": min_head_to_head_lower_bound,
        },
    }
    return result


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
    generate.add_argument(
        "--teacher",
        default="heuristic",
        choices=["heuristic", "neural"],
        help="Teacher used to play self-play games.",
    )
    generate.add_argument(
        "--teacher-model",
        help="Neural value model JSON used when --teacher neural.",
    )
    generate.add_argument("--teacher-neural-scale", type=float, default=120.0)
    generate.add_argument("--teacher-heuristic-scale", type=float, default=1.0)
    generate.add_argument("--teacher-neural-search-width", type=int, default=3)
    generate.add_argument("--teacher-neural-search-depth", type=int, default=4)
    generate.add_argument("--quiet", action="store_true", help="Suppress lightweight per-game progress output.")

    train = subparsers.add_parser("train", help="Train a value network.")
    train.add_argument(
        "--data",
        action="append",
        default=None,
        help="Training JSONL dataset. Can be passed more than once to blend datasets.",
    )
    train.add_argument("--model", default="checkpoints/value_model.json")
    train.add_argument("--hidden-size", type=int, default=48)
    train.add_argument("--epochs", type=int, default=8)
    train.add_argument("--learning-rate", type=float, default=0.003)
    train.add_argument("--seed", type=int, default=13)
    train.add_argument("--limit", type=int)
    train.add_argument("--per-data-limit", type=int, help="Maximum examples to load from each --data file.")
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

    train_policy = subparsers.add_parser("train-policy", help="Train a shared policy/value network.")
    train_policy.add_argument(
        "--data",
        action="append",
        default=None,
        help="Training JSONL dataset with action_index labels. Can be passed more than once.",
    )
    train_policy.add_argument("--model", default="checkpoints/policy_value_model.json")
    train_policy.add_argument("--hidden-size", type=int, default=128)
    train_policy.add_argument("--action-size", type=int, help="Policy action-space size. Defaults to the current map.")
    train_policy.add_argument("--epochs", type=int, default=12)
    train_policy.add_argument("--learning-rate", type=float, default=0.003)
    train_policy.add_argument("--seed", type=int, default=37)
    train_policy.add_argument("--limit", type=int)
    train_policy.add_argument("--per-data-limit", type=int, help="Maximum examples to load from each --data file.")
    train_policy.add_argument("--batch-size", type=int, default=256)
    train_policy.add_argument("--device", default="auto", help="PyTorch device, for example auto, cpu, or cuda.")
    train_policy.add_argument("--validation-fraction", type=float, default=0.1)
    train_policy.add_argument("--early-stop-patience", type=int, default=0)
    train_policy.add_argument("--early-stop-min-delta", type=float, default=0.0)
    train_policy.add_argument("--value-loss-weight", type=float, default=1.0)
    train_policy.add_argument("--policy-loss-weight", type=float, default=0.25)
    train_policy.add_argument("--init-model", help="Optional policy/value checkpoint to fine-tune from.")
    train_policy.add_argument(
        "--freeze-init-model",
        action="store_true",
        help="When widening from --init-model, freeze copied weights and train only added residual capacity.",
    )
    train_policy.add_argument("--map", default="plains", choices=["plains", "desert"])
    train_policy.add_argument("--quiet", action="store_true", help="Suppress lightweight per-epoch progress output.")

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
    gauntlet.add_argument("--policy-scale", type=float, default=18.0)
    gauntlet.add_argument("--neural-search-width", type=int, default=1)
    gauntlet.add_argument("--neural-search-depth", type=int, default=1)
    gauntlet.add_argument("--mcts-simulations", type=int, default=0)
    gauntlet.add_argument("--mcts-exploration", type=float, default=1.25)
    gauntlet.add_argument("--mcts-max-children", type=int, default=24)
    gauntlet.add_argument("--mcts-depth", type=int, default=7)
    gauntlet.add_argument("--workers", type=int, default=1, help="Parallel paired-game workers for gauntlet evaluation.")
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

    champion = subparsers.add_parser(
        "champion",
        help="Evaluate a candidate model and decide whether it should replace the current champion.",
    )
    champion.add_argument("--candidate", required=True, help="Candidate neural value model JSON.")
    champion.add_argument(
        "--champion",
        default="checkpoints/value_model_torch_128_shaped_1000_300hp.json",
        help="Current champion neural value model JSON.",
    )
    champion.add_argument("--games", type=int, default=8, help="Games per side against each opponent.")
    champion.add_argument("--seed", type=int, default=101)
    champion.add_argument("--weights", help="Optional trained heuristic weights JSON.")
    champion.add_argument("--map", default="plains", choices=["plains", "desert"])
    champion.add_argument("--heuristic-search-width", type=int, default=3)
    champion.add_argument("--heuristic-search-depth", type=int, default=6)
    champion.add_argument("--neural-scale", type=float, default=120.0)
    champion.add_argument("--heuristic-scale", type=float, default=1.0)
    champion.add_argument("--policy-scale", type=float, default=18.0)
    champion.add_argument("--neural-search-width", type=int, default=1)
    champion.add_argument("--neural-search-depth", type=int, default=1)
    champion.add_argument("--mcts-simulations", type=int, default=0)
    champion.add_argument("--mcts-exploration", type=float, default=1.25)
    champion.add_argument("--mcts-max-children", type=int, default=24)
    champion.add_argument("--mcts-depth", type=int, default=7)
    champion.add_argument("--workers", type=int, default=1, help="Parallel paired-game workers for champion evaluation.")
    champion.add_argument(
        "--opponent-model",
        action="append",
        default=[],
        help="Additional neural checkpoint to include after the champion. Can be passed more than once.",
    )
    champion.add_argument(
        "--only-neural-opponents",
        action="store_true",
        help="Skip random and heuristic baselines; useful for faster checkpoint head-to-heads.",
    )
    champion.add_argument("--min-head-to-head-score", type=float, default=0.55)
    champion.add_argument("--min-overall-score", type=float, default=0.60)
    champion.add_argument("--min-head-to-head-lower-bound", type=float, default=0.50)
    champion.add_argument("--output", help="Optional JSON file for champion decision results.")
    champion.add_argument("--quiet", action="store_true", help="Suppress per-opponent progress output.")

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
            teacher=args.teacher,
            teacher_model_path=args.teacher_model,
            teacher_neural_scale=args.teacher_neural_scale,
            teacher_heuristic_scale=args.teacher_heuristic_scale,
            teacher_neural_search_width=args.teacher_neural_search_width,
            teacher_neural_search_depth=args.teacher_neural_search_depth,
        )
        print(f"Wrote {count} examples to {args.output}")
        return 0
    if args.command == "train":
        try:
            history = train_value_model(
                dataset_path=args.data or ["neural_data/selfplay.jsonl"],
                model_path=args.model,
                hidden_size=args.hidden_size,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                seed=args.seed,
                limit=args.limit,
                per_dataset_limit=args.per_data_limit,
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
    if args.command == "train-policy":
        try:
            history = train_policy_value_model(
                dataset_path=args.data or ["neural_data/selfplay.jsonl"],
                model_path=args.model,
                hidden_size=args.hidden_size,
                action_size=args.action_size,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                seed=args.seed,
                limit=args.limit,
                per_dataset_limit=args.per_data_limit,
                progress=not args.quiet,
                batch_size=args.batch_size,
                device=args.device,
                validation_fraction=args.validation_fraction,
                early_stop_patience=args.early_stop_patience,
                early_stop_min_delta=args.early_stop_min_delta,
                value_loss_weight=args.value_loss_weight,
                policy_loss_weight=args.policy_loss_weight,
                map_name=args.map,
                init_model_path=args.init_model,
                freeze_init_model=args.freeze_init_model,
            )
        except RuntimeError as exc:
            raise SystemExit(str(exc)) from exc
        print(f"Saved policy/value model to {args.model}")
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
            policy_scale=args.policy_scale,
            neural_search_width=args.neural_search_width,
            neural_search_depth=args.neural_search_depth,
            mcts_simulations=args.mcts_simulations,
            mcts_exploration=args.mcts_exploration,
            mcts_max_children=args.mcts_max_children,
            mcts_depth=args.mcts_depth,
            workers=args.workers,
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
    if args.command == "champion":
        weights = load_weights(args.weights) if args.weights else None
        result = run_champion_gauntlet(
            candidate_model_path=args.candidate,
            champion_model_path=args.champion,
            games=args.games,
            seed=args.seed,
            weights=weights,
            weights_path=args.weights,
            extra_opponent_models=args.opponent_model,
            include_baseline_opponents=not args.only_neural_opponents,
            map_name=args.map,
            heuristic_search_width=args.heuristic_search_width,
            heuristic_search_depth=args.heuristic_search_depth,
            neural_scale=args.neural_scale,
            heuristic_scale=args.heuristic_scale,
            policy_scale=args.policy_scale,
            neural_search_width=args.neural_search_width,
            neural_search_depth=args.neural_search_depth,
            mcts_simulations=args.mcts_simulations,
            mcts_exploration=args.mcts_exploration,
            mcts_max_children=args.mcts_max_children,
            mcts_depth=args.mcts_depth,
            min_head_to_head_score=args.min_head_to_head_score,
            min_overall_score=args.min_overall_score,
            min_head_to_head_lower_bound=args.min_head_to_head_lower_bound,
            workers=args.workers,
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
                    "action_space_size": action_space_for_state(state).size,
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
