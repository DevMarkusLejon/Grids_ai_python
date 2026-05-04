from __future__ import annotations

from dataclasses import dataclass
import json
import os
import random

from .data import AttackRole, ITEM_BLUEPRINTS, Side
from .engine import Action, GameState, Unit


DEFAULT_WEIGHTS: dict[str, float] = {
    "bias": 0.0,
    "enemy_commander_delta": 14.0,
    "own_commander_delta": -16.0,
    "enemy_unit_delta": 48.0,
    "own_unit_delta": -52.0,
    "enemy_total_hp_delta": 1.7,
    "own_total_hp_delta": -1.3,
    "enemy_unit_value_delta": 0.20,
    "own_unit_value_delta": -0.25,
    "forward_pressure_delta": 2.0,
    "commander_distance_delta": 5.0,
    "enemy_commander_threat_delta": 2.2,
    "own_commander_threat_delta": -2.8,
    "lethal_threat": 180.0,
    "own_lethal_risk": -220.0,
    "move_enables_attack": 16.0,
    "effective_healing": 1.6,
    "overkill_damage": -0.35,
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
    search_width: int = 5
    search_depth: int | None = None

    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed)

    def choose_action(self, state: GameState) -> Action:
        player = state.current_side
        legal = state.legal_actions()
        if not legal:
            raise ValueError("No legal actions available.")

        if self.search_width <= 1:
            return self.choose_greedy_action(state, legal, player)
        return self.choose_planned_action(state, legal, player)

    def choose_greedy_action(self, state: GameState, legal: list[Action], player: Side) -> Action:
        scored_actions: list[tuple[float, float, Action]] = []
        for action in legal:
            after = state.clone()
            after.apply_unchecked(action)
            score = self.score_action(state, after, action, player)
            scored_actions.append((score, self.rng.random(), action))
        scored_actions.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return scored_actions[0][2]

    def choose_planned_action(self, state: GameState, legal: list[Action], player: Side) -> Action:
        depth_limit = self.search_depth or state.config.max_actions + 1
        depth_limit = max(1, depth_limit)
        width = max(1, self.search_width)
        decay = 0.92

        # (score, tie_breaker, first_action, simulated_state)
        frontier: list[tuple[float, float, Action, GameState]] = []
        all_paths: list[tuple[float, float, Action, GameState]] = []
        for action in legal:
            after = state.clone()
            after.apply_unchecked(action)
            score = self.score_action(state, after, action, player)
            path = (score, self.rng.random(), action, after)
            frontier.append(path)
            all_paths.append(path)

        frontier.sort(key=lambda item: (item[0], item[1]), reverse=True)
        frontier = frontier[:width]

        for depth in range(1, depth_limit):
            expanded: list[tuple[float, float, Action, GameState]] = []
            for cumulative_score, tie_breaker, first_action, branch_state in frontier:
                if branch_state.is_done or branch_state.current_side is not player:
                    continue
                next_actions = branch_state.legal_actions()
                for action in next_actions:
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
    before_enemy_threat = commander_threat(before, player, enemy)
    after_enemy_threat = commander_threat(after, player, enemy)
    before_own_threat = commander_threat(before, enemy, player)
    after_own_threat = commander_threat(after, enemy, player)
    features: dict[str, float] = {
        "bias": 1.0,
        "enemy_commander_delta": before_enemy_commander_hp - after_enemy_commander_hp,
        "own_commander_delta": before_own_commander_hp - after_own_commander_hp,
        "enemy_unit_delta": len(before.units_for_side(enemy)) - len(after.units_for_side(enemy)),
        "own_unit_delta": len(before.units_for_side(player)) - len(after.units_for_side(player)),
        "enemy_total_hp_delta": before.total_hp(enemy) - after.total_hp(enemy),
        "own_total_hp_delta": before.total_hp(player) - after.total_hp(player),
        "enemy_unit_value_delta": total_unit_value(before, enemy) - total_unit_value(after, enemy),
        "own_unit_value_delta": total_unit_value(before, player) - total_unit_value(after, player),
        "forward_pressure_delta": after.forward_pressure(player) - before.forward_pressure(player),
        "commander_distance_delta": commander_distance(before, player) - commander_distance(after, player),
        "enemy_commander_threat_delta": after_enemy_threat - before_enemy_threat,
        "own_commander_threat_delta": after_own_threat - before_own_threat,
        "lethal_threat": 1.0
        if not after.is_done and after_enemy_threat >= after_enemy_commander_hp > 0
        else 0.0,
        "own_lethal_risk": 1.0
        if not after.is_done and after_own_threat >= after_own_commander_hp > 0
        else 0.0,
        "move_enables_attack": 1.0 if move_enables_attack(before, after, action, player) else 0.0,
        "effective_healing": effective_healing(before, after, action),
        "overkill_damage": overkill_damage(before, action),
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


def unit_value(unit: Unit) -> float:
    if unit.is_commander:
        return 420.0 + unit.hp
    return (
        unit.max_hp * 0.35
        + unit.hp * 0.25
        + unit.damage * 1.4
        + unit.move_range * 5.0
        + unit.attack_range * 6.0
    )


def total_unit_value(state: GameState, side: Side) -> float:
    return sum(unit_value(unit) for unit in state.units_for_side(side))


def commander_distance(state: GameState, side: Side) -> float:
    enemy_commander = state.find_commander(side.other())
    if enemy_commander is None:
        return 0.0
    distances = [
        state.distance(unit.coord, enemy_commander.coord)
        for unit in state.units_for_side(side)
        if not unit.is_commander
    ]
    if not distances:
        return float(state.map_def.width + state.map_def.height)
    return float(min(distances))


def commander_threat(state: GameState, attacker_side: Side, defender_side: Side) -> float:
    commander = state.find_commander(defender_side)
    if commander is None:
        return 0.0
    attacks: list[float] = []
    for unit in state.units_for_side(attacker_side):
        if unit.role is AttackRole.HEALER or unit.attacked_this_turn:
            continue
        if state.distance(unit.coord, commander.coord) <= unit.attack_range:
            attacks.append(float(unit.damage))
    for card in state.hands[attacker_side]:
        if not card.is_item:
            continue
        item = ITEM_BLUEPRINTS[card.key]
        if item.effect == "damage_enemy" and state.actions_left >= item.cost:
            attacks.append(float(item.power))
    attacks.sort(reverse=True)
    return sum(attacks[: max(state.actions_left, 0)])


def move_enables_attack(before: GameState, after: GameState, action: Action, player: Side) -> bool:
    if action.kind != "move" or action.unit_id is None:
        return False
    before_unit = before.units[action.unit_id]
    if before_unit.role is AttackRole.HEALER or before_unit.attacked_this_turn:
        return False
    after_unit = after.units[action.unit_id]
    return any(
        after.distance(after_unit.coord, enemy.coord) <= after_unit.attack_range
        for enemy in after.units_for_side(player.other())
    )


def effective_healing(before: GameState, after: GameState, action: Action) -> float:
    if action.kind != "attack" or action.unit_id is None or action.target_unit_id is None:
        return 0.0
    attacker = before.units[action.unit_id]
    if attacker.role is not AttackRole.HEALER:
        return 0.0
    before_target = before.units[action.target_unit_id]
    after_target = after.units.get(action.target_unit_id)
    if after_target is None:
        return 0.0
    return max(0.0, float(after_target.hp - before_target.hp))


def overkill_damage(before: GameState, action: Action) -> float:
    if action.target_unit_id is None:
        return 0.0
    target = before.units[action.target_unit_id]
    damage = 0.0
    if action.kind == "attack" and action.unit_id is not None:
        attacker = before.units[action.unit_id]
        if attacker.role is AttackRole.HEALER:
            return 0.0
        damage = float(attacker.damage)
    elif action.kind == "play_item" and action.hand_index is not None:
        card = before.hands[before.current_side][action.hand_index]
        if not card.is_item:
            return 0.0
        item = ITEM_BLUEPRINTS[card.key]
        if item.effect != "damage_enemy":
            return 0.0
        damage = float(item.power)
    return max(0.0, damage - float(target.hp))


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
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
