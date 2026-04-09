from __future__ import annotations

from dataclasses import dataclass
import json
import random

from .data import AttackRole, Side
from .engine import Action, GameState


DEFAULT_WEIGHTS: dict[str, float] = {
    "bias": 0.0,
    "enemy_commander_delta": 14.0,
    "own_commander_delta": -16.0,
    "enemy_unit_delta": 48.0,
    "own_unit_delta": -52.0,
    "enemy_total_hp_delta": 1.7,
    "own_total_hp_delta": -1.3,
    "forward_pressure_delta": 2.0,
    "hand_delta": 0.8,
    "deploy": 6.0,
    "move": 1.0,
    "attack": 4.0,
    "heal": 3.5,
    "item": 3.0,
    "draw_unit": 1.8,
    "draw_item": 1.0,
    "end_turn": -2.5,
    "remaining_ap": 0.2,
    "win": 10000.0,
    "loss": -10000.0,
}


class Bot:
    def choose_action(self, state: GameState) -> Action:
        raise NotImplementedError


class RandomBot(Bot):
    def __init__(self, seed: int | None = None) -> None:
        self.rng = random.Random(seed)

    def choose_action(self, state: GameState) -> Action:
        legal = state.legal_actions()
        if not legal:
            raise ValueError("No legal actions available.")
        return self.rng.choice(legal)


@dataclass
class HeuristicBot(Bot):
    weights: dict[str, float]
    seed: int | None = None

    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed)

    def choose_action(self, state: GameState) -> Action:
        player = state.current_side
        legal = state.legal_actions()
        if not legal:
            raise ValueError("No legal actions available.")

        scored_actions: list[tuple[float, float, Action]] = []
        for action in legal:
            after = state.clone()
            after.apply(action)
            score = self.score_action(state, after, action, player)
            scored_actions.append((score, self.rng.random(), action))

        scored_actions.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return scored_actions[0][2]

    def score_action(self, before: GameState, after: GameState, action: Action, player: Side) -> float:
        enemy = player.other()
        features = extract_features(before, after, action, player)
        score = 0.0
        for name, value in features.items():
            score += self.weights.get(name, 0.0) * value

        if after.is_done:
            score += self.weights.get("win", 0.0) if after.winner is player else self.weights.get("loss", 0.0)
        elif commander_hp(after, enemy) <= 40:
            score += 25.0
        return score


def extract_features(
    before: GameState,
    after: GameState,
    action: Action,
    player: Side,
) -> dict[str, float]:
    enemy = player.other()
    before_enemy_commander_hp = commander_hp(before, enemy)
    before_own_commander_hp = commander_hp(before, player)
    after_enemy_commander_hp = commander_hp(after, enemy)
    after_own_commander_hp = commander_hp(after, player)
    features: dict[str, float] = {
        "bias": 1.0,
        "enemy_commander_delta": before_enemy_commander_hp - after_enemy_commander_hp,
        "own_commander_delta": before_own_commander_hp - after_own_commander_hp,
        "enemy_unit_delta": len(before.units_for_side(enemy)) - len(after.units_for_side(enemy)),
        "own_unit_delta": len(before.units_for_side(player)) - len(after.units_for_side(player)),
        "enemy_total_hp_delta": before.total_hp(enemy) - after.total_hp(enemy),
        "own_total_hp_delta": before.total_hp(player) - after.total_hp(player),
        "forward_pressure_delta": after.forward_pressure(player) - before.forward_pressure(player),
        "hand_delta": len(after.hands[player]) - len(before.hands[player]),
        "deploy": 1.0 if action.kind == "deploy" else 0.0,
        "move": 1.0 if action.kind == "move" else 0.0,
        "attack": 1.0 if action.kind == "attack" else 0.0,
        "heal": 1.0
        if action.kind == "attack" and before.units[action.unit_id].role is AttackRole.HEALER
        else 0.0,
        "item": 1.0 if action.kind == "play_item" else 0.0,
        "draw_unit": 1.0 if action.kind == "draw_unit" else 0.0,
        "draw_item": 1.0 if action.kind == "draw_item" else 0.0,
        "end_turn": 1.0 if action.kind == "end_turn" else 0.0,
        "remaining_ap": after.actions_left,
    }
    return features


def commander_hp(state: GameState, side: Side) -> int:
    for unit in state.units_for_side(side):
        if unit.is_commander:
            return unit.hp
    return 0


def load_weights(path: str) -> dict[str, float]:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if "weights" in data:
        return {key: float(value) for key, value in data["weights"].items()}
    return {key: float(value) for key, value in data.items()}


def save_weights(path: str, weights: dict[str, float], metadata: dict[str, object] | None = None) -> None:
    payload: dict[str, object] = {"weights": weights}
    if metadata:
        payload["metadata"] = metadata
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
