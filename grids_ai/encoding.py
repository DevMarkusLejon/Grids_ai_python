from __future__ import annotations

from dataclasses import dataclass

from .data import Side, UNIT_BLUEPRINTS
from .engine import Action, GameState, Unit


UNIT_KEYS = ("commander", "warrior", "archer", "healer", "assassin", "viking")
ACTION_KINDS = ("end_turn", "draw_unit", "draw_item", "move", "attack", "deploy", "play_item")


@dataclass(frozen=True)
class EncodedActionSpace:
    width: int
    height: int
    max_hand_size: int

    @property
    def cells(self) -> int:
        return self.width * self.height

    @property
    def move_offset(self) -> int:
        return 3

    @property
    def attack_offset(self) -> int:
        return self.move_offset + self.cells * self.cells

    @property
    def deploy_offset(self) -> int:
        return self.attack_offset + self.cells * self.cells

    @property
    def item_offset(self) -> int:
        return self.deploy_offset + self.max_hand_size * self.cells

    @property
    def size(self) -> int:
        return self.item_offset + self.max_hand_size * self.cells


def action_space_for_state(state: GameState) -> EncodedActionSpace:
    return EncodedActionSpace(
        width=state.map_def.width,
        height=state.map_def.height,
        max_hand_size=state.config.max_hand_size,
    )


def cell_index(state: GameState, coord: tuple[int, int]) -> int:
    x, y = coord
    return y * state.map_def.width + x


def unit_cell_index(state: GameState, unit_id: int | None) -> int:
    if unit_id is None:
        raise ValueError("Action is missing a unit id.")
    return cell_index(state, state.units[unit_id].coord)


def target_cell_index(state: GameState, target_unit_id: int | None) -> int:
    if target_unit_id is None:
        raise ValueError("Action is missing a target unit id.")
    return cell_index(state, state.units[target_unit_id].coord)


def encode_action_index(state: GameState, action: Action) -> int:
    space = action_space_for_state(state)
    if action.kind == "end_turn":
        return 0
    if action.kind == "draw_unit":
        return 1
    if action.kind == "draw_item":
        return 2
    if action.kind == "move":
        if action.destination is None:
            raise ValueError("Move action is missing a destination.")
        source = unit_cell_index(state, action.unit_id)
        destination = cell_index(state, action.destination)
        return space.move_offset + source * space.cells + destination
    if action.kind == "attack":
        source = unit_cell_index(state, action.unit_id)
        target = target_cell_index(state, action.target_unit_id)
        return space.attack_offset + source * space.cells + target
    if action.kind == "deploy":
        if action.hand_index is None or action.destination is None:
            raise ValueError("Deploy action is missing a hand index or destination.")
        destination = cell_index(state, action.destination)
        return space.deploy_offset + action.hand_index * space.cells + destination
    if action.kind == "play_item":
        if action.hand_index is None:
            raise ValueError("Item action is missing a hand index.")
        target = target_cell_index(state, action.target_unit_id)
        return space.item_offset + action.hand_index * space.cells + target
    raise ValueError(f"Unknown action kind: {action.kind}")


def legal_action_indices(state: GameState) -> list[int]:
    return [encode_action_index(state, action) for action in state.legal_actions()]


def unit_kind_value(unit: Unit) -> float:
    return (UNIT_KEYS.index(unit.blueprint_key) + 1) / len(UNIT_KEYS)


def side_sign(side: Side) -> float:
    return 1.0 if side is Side.BLUE else -1.0


def encode_state_vector(state: GameState) -> list[float]:
    """Compact neural input vector using board cells plus global game context."""
    width = state.map_def.width
    height = state.map_def.height
    cells = width * height
    channels = 13
    values = [0.0] * (cells * channels)

    def set_cell(channel: int, coord: tuple[int, int], value: float) -> None:
        values[channel * cells + cell_index(state, coord)] = value

    for coord in state.map_def.blockers:
        set_cell(0, coord, 1.0)
    for coord in state.map_def.blue_deploy:
        set_cell(1, coord, 1.0)
    for coord in state.map_def.red_deploy:
        set_cell(2, coord, 1.0)

    for unit in state.units.values():
        offset = 3 if unit.side is Side.BLUE else 8
        coord = unit.coord
        set_cell(offset + 0, coord, unit_kind_value(unit))
        set_cell(offset + 1, coord, max(0.0, min(1.0, unit.hp / max(unit.max_hp, 1))))
        set_cell(offset + 2, coord, max(0.0, min(1.0, unit.damage / 100.0)))
        set_cell(offset + 3, coord, 1.0 if unit.attacked_this_turn else 0.0)
        set_cell(offset + 4, coord, 1.0 if unit.is_commander else 0.0)

    max_half_turns = max(state.config.max_half_turns, 1)
    max_actions = max(state.config.max_actions, 1)
    max_hand = max(state.config.max_hand_size, 1)
    max_unit_deck = max(len(state.unit_decks[Side.BLUE]) + len(state.hands[Side.BLUE]), 1)
    max_item_deck = max(len(state.item_decks[Side.BLUE]) + len(state.hands[Side.BLUE]), 1)

    global_values = [
        side_sign(state.current_side),
        state.actions_left / max_actions,
        min(state.half_turns_played / max_half_turns, 1.0),
        len(state.hands[Side.BLUE]) / max_hand,
        len(state.hands[Side.RED]) / max_hand,
        len(state.unit_decks[Side.BLUE]) / max_unit_deck,
        len(state.unit_decks[Side.RED]) / max_unit_deck,
        len(state.item_decks[Side.BLUE]) / max_item_deck,
        len(state.item_decks[Side.RED]) / max_item_deck,
        state.commander_hp(Side.BLUE) / max(state.commander_max_hp(Side.BLUE), 1),
        state.commander_hp(Side.RED) / max(state.commander_max_hp(Side.RED), 1),
        state.forward_pressure(Side.BLUE) / max(width * 5, 1),
        state.forward_pressure(Side.RED) / max(width * 5, 1),
        len(state.units_for_side(Side.BLUE)) / 8.0,
        len(state.units_for_side(Side.RED)) / 8.0,
    ]
    return values + global_values


def encoded_state_size(state: GameState) -> int:
    return len(encode_state_vector(state))


def encode_state_planes(state: GameState) -> list[list[list[float]]]:
    """Board-shaped planes for future CNN/PyTorch experiments."""
    vector = encode_state_vector(state)
    cells = state.map_def.width * state.map_def.height
    board_values = vector[: cells * 13]
    planes: list[list[list[float]]] = []
    for channel in range(13):
        plane: list[list[float]] = []
        start = channel * cells
        for y in range(state.map_def.height):
            row_start = start + y * state.map_def.width
            plane.append(board_values[row_start : row_start + state.map_def.width])
        planes.append(plane)
    return planes
