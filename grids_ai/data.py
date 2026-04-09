from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


Coord = tuple[int, int]


class Side(str, Enum):
    BLUE = "blue"
    RED = "red"

    @property
    def short(self) -> str:
        return "B" if self is Side.BLUE else "R"

    def other(self) -> "Side":
        return Side.RED if self is Side.BLUE else Side.BLUE


class AttackRole(str, Enum):
    MELEE = "melee"
    RANGED = "ranged"
    HEALER = "healer"


class CardKind(str, Enum):
    UNIT = "unit"
    ITEM = "item"


@dataclass(frozen=True)
class UnitBlueprint:
    key: str
    name: str
    glyph: str
    max_hp: int
    damage: int
    move_range: int
    attack_range: int
    deploy_cost: int
    role: AttackRole
    can_deploy: bool = True


@dataclass(frozen=True)
class ItemBlueprint:
    key: str
    name: str
    cost: int
    effect: str
    power: int
    description: str


@dataclass(frozen=True)
class Card:
    kind: CardKind
    key: str

    @property
    def is_unit(self) -> bool:
        return self.kind is CardKind.UNIT

    @property
    def is_item(self) -> bool:
        return self.kind is CardKind.ITEM


@dataclass(frozen=True)
class MapDefinition:
    name: str
    width: int
    height: int
    blockers: frozenset[Coord]
    blue_commander: Coord
    red_commander: Coord
    blue_deploy: frozenset[Coord]
    red_deploy: frozenset[Coord]

    def in_bounds(self, coord: Coord) -> bool:
        x, y = coord
        return 0 <= x < self.width and 0 <= y < self.height


UNIT_BLUEPRINTS: dict[str, UnitBlueprint] = {
    "commander": UnitBlueprint(
        key="commander",
        name="Commander",
        glyph="C",
        max_hp=300,
        damage=20,
        move_range=2,
        attack_range=1,
        deploy_cost=0,
        role=AttackRole.MELEE,
        can_deploy=False,
    ),
    "warrior": UnitBlueprint(
        key="warrior",
        name="Warrior",
        glyph="W",
        max_hp=100,
        damage=40,
        move_range=2,
        attack_range=1,
        deploy_cost=2,
        role=AttackRole.MELEE,
    ),
    "archer": UnitBlueprint(
        key="archer",
        name="Archer",
        glyph="A",
        max_hp=80,
        damage=20,
        move_range=2,
        attack_range=4,
        deploy_cost=2,
        role=AttackRole.RANGED,
    ),
    "healer": UnitBlueprint(
        key="healer",
        name="Healer",
        glyph="H",
        max_hp=80,
        damage=30,
        move_range=3,
        attack_range=3,
        deploy_cost=2,
        role=AttackRole.HEALER,
    ),
    "assassin": UnitBlueprint(
        key="assassin",
        name="Assassin",
        glyph="S",
        max_hp=70,
        damage=35,
        move_range=3,
        attack_range=1,
        deploy_cost=1,
        role=AttackRole.MELEE,
    ),
    "viking": UnitBlueprint(
        key="viking",
        name="Viking",
        glyph="V",
        max_hp=110,
        damage=30,
        move_range=2,
        attack_range=1,
        deploy_cost=2,
        role=AttackRole.MELEE,
    ),
}


ITEM_BLUEPRINTS: dict[str, ItemBlueprint] = {
    "fireball": ItemBlueprint(
        key="fireball",
        name="Fireball",
        cost=1,
        effect="damage_enemy",
        power=30,
        description="Deal 30 damage to an enemy unit anywhere on the board.",
    ),
    "strength_tonic": ItemBlueprint(
        key="strength_tonic",
        name="Strength Tonic",
        cost=2,
        effect="buff_friendly_damage",
        power=10,
        description="Give a friendly unit +10 permanent damage.",
    ),
}


def deployment_columns(width: int, height: int) -> tuple[frozenset[Coord], frozenset[Coord]]:
    blue = frozenset((0, y) for y in range(height))
    red = frozenset((width - 1, y) for y in range(height))
    return blue, red


def make_maps() -> dict[str, MapDefinition]:
    width = 10
    height = 7
    blue_deploy, red_deploy = deployment_columns(width, height)
    return {
        "plains": MapDefinition(
            name="plains",
            width=width,
            height=height,
            blockers=frozenset({(6, 1), (3, 5), (5, 2), (4, 4)}),
            blue_commander=(1, 3),
            red_commander=(8, 3),
            blue_deploy=blue_deploy,
            red_deploy=red_deploy,
        ),
        "desert": MapDefinition(
            name="desert",
            width=width,
            height=height,
            blockers=frozenset({(3, 4), (4, 5), (5, 1), (6, 2)}),
            blue_commander=(1, 3),
            red_commander=(8, 3),
            blue_deploy=blue_deploy,
            red_deploy=red_deploy,
        ),
    }


MAPS = make_maps()


DEFAULT_UNIT_DECK: list[str] = ["warrior", "archer"] * 3 + ["healer", "viking", "assassin"] * 2
DEFAULT_ITEM_DECK: list[str] = ["fireball", "strength_tonic"] * 2
