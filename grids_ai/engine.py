from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import random
from typing import Iterable

from .data import (
    AttackRole,
    Card,
    CardKind,
    Coord,
    DEFAULT_ITEM_DECK,
    DEFAULT_UNIT_DECK,
    ITEM_BLUEPRINTS,
    MAPS,
    MapDefinition,
    Side,
    UNIT_BLUEPRINTS,
)


@dataclass(frozen=True)
class GameConfig:
    max_actions: int = 7
    max_hand_size: int = 7
    starting_hand_size: int = 5
    draw_cost_unit: int = 1
    draw_cost_item: int = 1
    max_half_turns: int = 80


@dataclass
class Unit:
    unit_id: int
    blueprint_key: str
    side: Side
    x: int
    y: int
    hp: int
    damage_bonus: int = 0
    attacked_this_turn: bool = False

    @property
    def coord(self) -> Coord:
        return (self.x, self.y)

    @property
    def blueprint(self):
        return UNIT_BLUEPRINTS[self.blueprint_key]

    @property
    def name(self) -> str:
        return self.blueprint.name

    @property
    def max_hp(self) -> int:
        return self.blueprint.max_hp

    @property
    def damage(self) -> int:
        return self.blueprint.damage + self.damage_bonus

    @property
    def move_range(self) -> int:
        return self.blueprint.move_range

    @property
    def attack_range(self) -> int:
        return self.blueprint.attack_range

    @property
    def role(self) -> AttackRole:
        return self.blueprint.role

    @property
    def is_commander(self) -> bool:
        return self.blueprint_key == "commander"

    def clone(self) -> "Unit":
        return Unit(
            unit_id=self.unit_id,
            blueprint_key=self.blueprint_key,
            side=self.side,
            x=self.x,
            y=self.y,
            hp=self.hp,
            damage_bonus=self.damage_bonus,
            attacked_this_turn=self.attacked_this_turn,
        )


@dataclass(frozen=True)
class Action:
    kind: str
    hand_index: int | None = None
    unit_id: int | None = None
    target_unit_id: int | None = None
    destination: Coord | None = None


@dataclass
class GameState:
    map_def: MapDefinition
    config: GameConfig
    current_side: Side
    turn_number: int
    actions_left: int
    units: dict[int, Unit]
    hands: dict[Side, list[Card]]
    unit_decks: dict[Side, list[str]]
    item_decks: dict[Side, list[str]]
    next_unit_id: int
    winner: Side | None = None
    winner_reason: str = ""
    log: list[str] = field(default_factory=list)
    half_turns_played: int = 0

    def clone(self) -> "GameState":
        return GameState(
            map_def=self.map_def,
            config=self.config,
            current_side=self.current_side,
            turn_number=self.turn_number,
            actions_left=self.actions_left,
            units={unit_id: unit.clone() for unit_id, unit in self.units.items()},
            hands={
                Side.BLUE: list(self.hands[Side.BLUE]),
                Side.RED: list(self.hands[Side.RED]),
            },
            unit_decks={
                Side.BLUE: list(self.unit_decks[Side.BLUE]),
                Side.RED: list(self.unit_decks[Side.RED]),
            },
            item_decks={
                Side.BLUE: list(self.item_decks[Side.BLUE]),
                Side.RED: list(self.item_decks[Side.RED]),
            },
            next_unit_id=self.next_unit_id,
            winner=self.winner,
            winner_reason=self.winner_reason,
            log=list(self.log),
            half_turns_played=self.half_turns_played,
        )

    @property
    def is_done(self) -> bool:
        return self.winner is not None

    def legal_actions(self) -> list[Action]:
        if self.is_done:
            return []

        actions: list[Action] = [Action(kind="end_turn")]
        hand = self.hands[self.current_side]

        if (
            self.actions_left >= self.config.draw_cost_unit
            and len(hand) < self.config.max_hand_size
            and self.unit_decks[self.current_side]
        ):
            actions.append(Action(kind="draw_unit"))

        if (
            self.actions_left >= self.config.draw_cost_item
            and len(hand) < self.config.max_hand_size
            and self.item_decks[self.current_side]
        ):
            actions.append(Action(kind="draw_item"))

        for hand_index, card in enumerate(hand):
            if card.is_unit:
                blueprint = UNIT_BLUEPRINTS[card.key]
                if self.actions_left < blueprint.deploy_cost:
                    continue
                for destination in self.available_deploy_cells(self.current_side):
                    actions.append(
                        Action(
                            kind="deploy",
                            hand_index=hand_index,
                            destination=destination,
                        )
                    )
            else:
                item = ITEM_BLUEPRINTS[card.key]
                if self.actions_left < item.cost:
                    continue
                if item.effect == "damage_enemy":
                    for target in self.units_for_side(self.current_side.other()):
                        actions.append(
                            Action(
                                kind="play_item",
                                hand_index=hand_index,
                                target_unit_id=target.unit_id,
                            )
                        )
                elif item.effect == "buff_friendly_damage":
                    for target in self.units_for_side(self.current_side):
                        actions.append(
                            Action(
                                kind="play_item",
                                hand_index=hand_index,
                                target_unit_id=target.unit_id,
                            )
                        )

        if self.actions_left <= 0:
            return actions

        for unit in self.units_for_side(self.current_side):
            for destination in sorted(self.reachable_cells(unit)):
                actions.append(
                    Action(
                        kind="move",
                        unit_id=unit.unit_id,
                        destination=destination,
                    )
                )

            if unit.attacked_this_turn:
                continue

            if unit.role is AttackRole.HEALER:
                targets = [
                    other
                    for other in self.units_for_side(self.current_side)
                    if other.hp < other.max_hp
                    and self.distance(unit.coord, other.coord) <= unit.attack_range
                    and other.unit_id != unit.unit_id
                ]
            else:
                targets = [
                    other
                    for other in self.units_for_side(self.current_side.other())
                    if self.distance(unit.coord, other.coord) <= unit.attack_range
                ]

            for target in targets:
                actions.append(
                    Action(
                        kind="attack",
                        unit_id=unit.unit_id,
                        target_unit_id=target.unit_id,
                    )
                )

        return actions

    def apply(self, action: Action) -> None:
        if self.is_done:
            raise ValueError("The game is already finished.")

        legal = self.legal_actions()
        if action not in legal:
            raise ValueError(f"Illegal action: {action}")

        self.apply_unchecked(action)

    def apply_unchecked(self, action: Action) -> None:
        """Apply an already-validated action for fast AI simulation."""
        if self.is_done:
            raise ValueError("The game is already finished.")

        if action.kind == "draw_unit":
            self._draw_from_deck(CardKind.UNIT)
            return

        if action.kind == "draw_item":
            self._draw_from_deck(CardKind.ITEM)
            return

        if action.kind == "deploy":
            self._deploy_from_hand(action.hand_index, action.destination)
            return

        if action.kind == "move":
            self._move_unit(action.unit_id, action.destination)
            return

        if action.kind == "attack":
            self._attack_unit(action.unit_id, action.target_unit_id)
            return

        if action.kind == "play_item":
            self._play_item(action.hand_index, action.target_unit_id)
            return

        if action.kind == "end_turn":
            self._end_turn()
            return

        raise ValueError(f"Unknown action kind: {action.kind}")

    def _draw_from_deck(self, kind: CardKind) -> None:
        if kind is CardKind.UNIT:
            card_key = self.unit_decks[self.current_side].pop(0)
            self.hands[self.current_side].append(Card(kind=CardKind.UNIT, key=card_key))
            self.actions_left -= self.config.draw_cost_unit
            self.log.append(f"{self.current_side.short} drew unit {UNIT_BLUEPRINTS[card_key].name}.")
        else:
            card_key = self.item_decks[self.current_side].pop(0)
            self.hands[self.current_side].append(Card(kind=CardKind.ITEM, key=card_key))
            self.actions_left -= self.config.draw_cost_item
            self.log.append(f"{self.current_side.short} drew item {ITEM_BLUEPRINTS[card_key].name}.")

    def _deploy_from_hand(self, hand_index: int | None, destination: Coord | None) -> None:
        assert hand_index is not None
        assert destination is not None
        card = self.hands[self.current_side][hand_index]
        blueprint = UNIT_BLUEPRINTS[card.key]
        self.actions_left -= blueprint.deploy_cost
        self.hands[self.current_side].pop(hand_index)
        self._spawn_unit(card.key, self.current_side, destination)
        self.log.append(f"{self.current_side.short} deployed {blueprint.name} at {destination}.")

    def _move_unit(self, unit_id: int | None, destination: Coord | None) -> None:
        assert unit_id is not None
        assert destination is not None
        unit = self.units[unit_id]
        start = unit.coord
        unit.x, unit.y = destination
        self.actions_left -= 1
        self.log.append(f"{self.current_side.short} moved {unit.name} from {start} to {destination}.")

    def _attack_unit(self, unit_id: int | None, target_unit_id: int | None) -> None:
        assert unit_id is not None
        assert target_unit_id is not None
        attacker = self.units[unit_id]
        target = self.units[target_unit_id]
        self.actions_left -= 1
        attacker.attacked_this_turn = True

        if attacker.role is AttackRole.HEALER:
            before = target.hp
            target.hp = min(target.max_hp, target.hp + attacker.damage)
            healed = target.hp - before
            self.log.append(
                f"{self.current_side.short} healed {target.name} for {healed} using {attacker.name}."
            )
            return

        damage = attacker.damage
        target.hp -= damage
        self.log.append(
            f"{self.current_side.short} attacked {target.name} with {attacker.name} for {damage}."
        )
        if target.hp <= 0:
            self._remove_unit(target.unit_id)
            return

        knockback_destination = self._knockback_destination(attacker.coord, target.coord)
        if knockback_destination is not None:
            start = target.coord
            target.x, target.y = knockback_destination
            self.log.append(f"{target.name} was knocked back from {start} to {knockback_destination}.")

    def _play_item(self, hand_index: int | None, target_unit_id: int | None) -> None:
        assert hand_index is not None
        assert target_unit_id is not None
        card = self.hands[self.current_side].pop(hand_index)
        item = ITEM_BLUEPRINTS[card.key]
        target = self.units[target_unit_id]
        self.actions_left -= item.cost

        if item.effect == "damage_enemy":
            target.hp -= item.power
            self.log.append(
                f"{self.current_side.short} cast {item.name} on {target.name} for {item.power} damage."
            )
            if target.hp <= 0:
                self._remove_unit(target.unit_id)
        elif item.effect == "buff_friendly_damage":
            target.damage_bonus += item.power
            self.log.append(
                f"{self.current_side.short} used {item.name} on {target.name}, gaining +{item.power} damage."
            )
        else:
            raise ValueError(f"Unsupported item effect: {item.effect}")

    def _end_turn(self) -> None:
        self.current_side = self.current_side.other()
        self.turn_number += 1
        self.half_turns_played += 1
        self.actions_left = self.config.max_actions
        for unit in self.units_for_side(self.current_side):
            unit.attacked_this_turn = False
        self.log.append(f"It is now {self.current_side.short}'s turn.")

        if self.half_turns_played >= self.config.max_half_turns and not self.is_done:
            self._resolve_timeout_winner()

    def _resolve_timeout_winner(self) -> None:
        blue_score = self.side_score(Side.BLUE)
        red_score = self.side_score(Side.RED)
        if blue_score > red_score:
            self.winner = Side.BLUE
        elif red_score > blue_score:
            self.winner = Side.RED
        else:
            self.winner = self.current_side.other()
        self.winner_reason = "score advantage after turn limit"
        self.log.append(f"{self.winner.short} wins on score after the turn limit.")

    def _remove_unit(self, unit_id: int) -> None:
        unit = self.units.pop(unit_id)
        self.log.append(f"{unit.name} was defeated.")
        if unit.is_commander:
            self.winner = unit.side.other()
            self.winner_reason = f"{unit.side.value} commander defeated"
            self.log.append(f"{self.winner.short} wins by defeating the enemy commander.")

    def _spawn_unit(self, blueprint_key: str, side: Side, destination: Coord) -> Unit:
        blueprint = UNIT_BLUEPRINTS[blueprint_key]
        unit = Unit(
            unit_id=self.next_unit_id,
            blueprint_key=blueprint.key,
            side=side,
            x=destination[0],
            y=destination[1],
            hp=blueprint.max_hp,
        )
        self.units[unit.unit_id] = unit
        self.next_unit_id += 1
        return unit

    def units_for_side(self, side: Side) -> list[Unit]:
        return sorted(
            (unit for unit in self.units.values() if unit.side is side),
            key=lambda unit: (unit.y, unit.x, unit.unit_id),
        )

    def available_deploy_cells(self, side: Side) -> list[Coord]:
        deploy_zone = self.map_def.blue_deploy if side is Side.BLUE else self.map_def.red_deploy
        return [
            coord
            for coord in sorted(deploy_zone)
            if coord not in self.map_def.blockers and self.unit_at(coord) is None
        ]

    def unit_at(self, coord: Coord) -> Unit | None:
        for unit in self.units.values():
            if unit.coord == coord:
                return unit
        return None

    def _knockback_destination(self, attacker: Coord, target: Coord) -> Coord | None:
        dx = target[0] - attacker[0]
        dy = target[1] - attacker[1]
        destination = (target[0] + dx, target[1] + dy)
        if not self.map_def.in_bounds(destination):
            return None
        if destination in self.map_def.blockers:
            return None
        if self.unit_at(destination) is not None:
            return None
        return destination

    def reachable_cells(self, unit: Unit) -> set[Coord]:
        frontier: deque[tuple[Coord, int]] = deque([(unit.coord, 0)])
        seen: set[Coord] = {unit.coord}
        reachable: set[Coord] = set()

        while frontier:
            coord, distance = frontier.popleft()
            if distance == unit.move_range:
                continue
            for neighbour in self.neighbours(coord):
                if neighbour in seen:
                    continue
                seen.add(neighbour)
                if not self.map_def.in_bounds(neighbour):
                    continue
                if neighbour in self.map_def.blockers:
                    continue
                occupant = self.unit_at(neighbour)
                if occupant is not None and occupant.unit_id != unit.unit_id:
                    continue
                reachable.add(neighbour)
                frontier.append((neighbour, distance + 1))

        reachable.discard(unit.coord)
        return reachable

    def neighbours(self, coord: Coord) -> Iterable[Coord]:
        x, y = coord
        yield (x + 1, y)
        yield (x - 1, y)
        yield (x, y + 1)
        yield (x, y - 1)

    @staticmethod
    def distance(a: Coord, b: Coord) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def find_commander(self, side: Side) -> Unit | None:
        for unit in self.units_for_side(side):
            if unit.is_commander:
                return unit
        return None

    def commander(self, side: Side) -> Unit:
        commander = self.find_commander(side)
        if commander is not None:
            return commander
        raise ValueError(f"No commander found for {side}.")

    def commander_hp(self, side: Side) -> int:
        commander = self.find_commander(side)
        if commander is None:
            return 0
        return commander.hp

    def commander_max_hp(self, side: Side) -> int:
        commander = self.find_commander(side)
        if commander is None:
            return UNIT_BLUEPRINTS["commander"].max_hp
        return commander.max_hp

    def total_hp(self, side: Side) -> int:
        return sum(unit.hp for unit in self.units_for_side(side))

    def side_score(self, side: Side) -> float:
        enemy = side.other()
        own_units = self.units_for_side(side)
        enemy_units = self.units_for_side(enemy)
        score = 0.0
        score += self.commander_hp(side) * 5.0
        score -= self.commander_hp(enemy) * 5.0
        score += len(own_units) * 8.0
        score -= len(enemy_units) * 8.0
        score += self.total_hp(side) * 0.25
        score -= self.total_hp(enemy) * 0.25
        score += self.forward_pressure(side) * 2.0
        score -= self.forward_pressure(enemy) * 2.0
        return score

    def forward_pressure(self, side: Side) -> float:
        total = 0.0
        for unit in self.units_for_side(side):
            if unit.is_commander:
                continue
            progress = unit.x if side is Side.BLUE else (self.map_def.width - 1 - unit.x)
            center_bias = self.map_def.height / 2 - abs(unit.y - (self.map_def.height - 1) / 2)
            total += progress + center_bias * 0.25
        return total

    def render(self) -> str:
        grid: list[list[str]] = [["." for _ in range(self.map_def.width)] for _ in range(self.map_def.height)]
        for x, y in self.map_def.blockers:
            grid[y][x] = "#"
        for x, y in self.map_def.blue_deploy:
            if grid[y][x] == ".":
                grid[y][x] = ":"
        for x, y in self.map_def.red_deploy:
            if grid[y][x] == ".":
                grid[y][x] = ";"
        for unit in self.units.values():
            glyph = unit.blueprint.glyph
            grid[unit.y][unit.x] = glyph if unit.side is Side.BLUE else glyph.lower()

        header = "    " + " ".join(f"{x:>2}" for x in range(self.map_def.width))
        lines = [header]
        for y, row in enumerate(grid):
            lines.append(f"{y:>2} | " + "  ".join(row))

        status = (
            f"Turn {self.turn_number} | Side {self.current_side.short} | "
            f"AP {self.actions_left}/{self.config.max_actions}"
        )
        lines.append(status)
        lines.append(
            f"Blue commander HP: {self.commander_hp(Side.BLUE)}/{self.commander_max_hp(Side.BLUE)} | "
            f"Red commander HP: {self.commander_hp(Side.RED)}/{self.commander_max_hp(Side.RED)}"
        )
        return "\n".join(lines)

    def hand_summary(self, side: Side | None = None) -> str:
        player = self.current_side if side is None else side
        hand = self.hands[player]
        if not hand:
            return f"{player.short} hand: <empty>"
        chunks = [f"{player.short} hand:"]
        for index, card in enumerate(hand):
            if card.is_unit:
                blueprint = UNIT_BLUEPRINTS[card.key]
                chunks.append(
                    f"[{index}] {blueprint.name} (unit, cost {blueprint.deploy_cost}, "
                    f"hp {blueprint.max_hp}, dmg {blueprint.damage}, "
                    f"move {blueprint.move_range}, range {blueprint.attack_range})"
                )
            else:
                item = ITEM_BLUEPRINTS[card.key]
                chunks.append(f"[{index}] {item.name} (item, cost {item.cost})")
        return "\n".join(chunks)

    def unit_summary(self) -> str:
        lines = ["Units:"]
        for side in (Side.BLUE, Side.RED):
            for unit in self.units_for_side(side):
                tag = unit.blueprint.name
                if unit.attacked_this_turn:
                    tag += " [spent]"
                if unit.damage_bonus:
                    tag += f" [+{unit.damage_bonus} dmg]"
                lines.append(
                    f"{unit.unit_id}: {side.short} {tag} at {unit.coord} "
                    f"HP {unit.hp}/{unit.max_hp}"
                )
        return "\n".join(lines)

    def describe_action(self, action: Action) -> str:
        if action.kind == "end_turn":
            return "End turn"
        if action.kind == "draw_unit":
            return "Draw a unit card (cost 1 AP)"
        if action.kind == "draw_item":
            return "Draw an item card (cost 1 AP)"
        if action.kind == "deploy":
            card = self.hands[self.current_side][action.hand_index]
            blueprint = UNIT_BLUEPRINTS[card.key]
            return f"Deploy {blueprint.name} to {action.destination} (cost {blueprint.deploy_cost})"
        if action.kind == "move":
            unit = self.units[action.unit_id]
            return f"Move {unit.name}#{unit.unit_id} to {action.destination} (cost 1)"
        if action.kind == "attack":
            attacker = self.units[action.unit_id]
            target = self.units[action.target_unit_id]
            verb = "Heal" if attacker.role is AttackRole.HEALER else "Attack"
            return f"{verb} {target.name}#{target.unit_id} with {attacker.name}#{attacker.unit_id} (cost 1)"
        if action.kind == "play_item":
            card = self.hands[self.current_side][action.hand_index]
            item = ITEM_BLUEPRINTS[card.key]
            target = self.units[action.target_unit_id]
            return f"Use {item.name} on {target.name}#{target.unit_id} (cost {item.cost})"
        return repr(action)


def draw_starting_hand(
    rng: random.Random,
    unit_deck: list[str],
    item_deck: list[str],
    starting_hand_size: int,
) -> list[Card]:
    hand: list[Card] = []
    while len(hand) < starting_hand_size and (unit_deck or item_deck):
        draw_unit = bool(unit_deck) and (not item_deck or rng.random() < 0.5)
        if draw_unit:
            hand.append(Card(kind=CardKind.UNIT, key=unit_deck.pop(0)))
        else:
            hand.append(Card(kind=CardKind.ITEM, key=item_deck.pop(0)))
    return hand


def new_game(
    seed: int | None = None,
    map_name: str = "plains",
    config: GameConfig | None = None,
) -> GameState:
    rng = random.Random(seed)
    cfg = config or GameConfig()
    map_def = MAPS[map_name]

    blue_unit_deck = list(DEFAULT_UNIT_DECK)
    red_unit_deck = list(DEFAULT_UNIT_DECK)
    blue_item_deck = list(DEFAULT_ITEM_DECK)
    red_item_deck = list(DEFAULT_ITEM_DECK)
    rng.shuffle(blue_unit_deck)
    rng.shuffle(red_unit_deck)
    rng.shuffle(blue_item_deck)
    rng.shuffle(red_item_deck)

    hands = {
        Side.BLUE: draw_starting_hand(rng, blue_unit_deck, blue_item_deck, cfg.starting_hand_size),
        Side.RED: draw_starting_hand(rng, red_unit_deck, red_item_deck, cfg.starting_hand_size),
    }

    state = GameState(
        map_def=map_def,
        config=cfg,
        current_side=Side.BLUE,
        turn_number=1,
        actions_left=cfg.max_actions,
        units={},
        hands=hands,
        unit_decks={Side.BLUE: blue_unit_deck, Side.RED: red_unit_deck},
        item_decks={Side.BLUE: blue_item_deck, Side.RED: red_item_deck},
        next_unit_id=1,
        log=[f"Started a new {map_def.name} match."],
    )

    state._spawn_unit("commander", Side.BLUE, map_def.blue_commander)
    state._spawn_unit("commander", Side.RED, map_def.red_commander)
    state.log.append("Both commanders are on the field.")
    return state
