from __future__ import annotations

import unittest

from grids_ai.bots import DEFAULT_WEIGHTS, HeuristicBot
from grids_ai.data import Card, CardKind, Side
from grids_ai.engine import Action, new_game


class GameStateTests(unittest.TestCase):
    def test_new_game_spawns_commanders_and_starting_hands(self) -> None:
        state = new_game(seed=1)
        self.assertEqual(state.commander(Side.BLUE).coord, (1, 3))
        self.assertEqual(state.commander(Side.RED).coord, (8, 3))
        self.assertEqual(len(state.hands[Side.BLUE]), 5)
        self.assertEqual(len(state.hands[Side.RED]), 5)
        self.assertEqual(state.actions_left, 7)

    def test_deploy_consumes_ap_and_places_unit(self) -> None:
        state = new_game(seed=1)
        state.hands[Side.BLUE] = [Card(kind=CardKind.UNIT, key="warrior")]
        action = Action(kind="deploy", hand_index=0, destination=(0, 0))
        state.apply(action)
        deployed = [unit for unit in state.units_for_side(Side.BLUE) if unit.blueprint_key == "warrior"]
        self.assertEqual(len(deployed), 1)
        self.assertEqual(deployed[0].coord, (0, 0))
        self.assertEqual(state.actions_left, 5)

    def test_attack_can_defeat_commander(self) -> None:
        state = new_game(seed=1)
        blue = state._spawn_unit("warrior", Side.BLUE, (7, 3))
        red_commander = state.commander(Side.RED)
        red_commander.hp = 40
        attack = Action(kind="attack", unit_id=blue.unit_id, target_unit_id=red_commander.unit_id)
        state.apply(attack)
        self.assertTrue(state.is_done)
        self.assertEqual(state.winner, Side.BLUE)

    def test_end_turn_switches_side_and_refreshes_ap(self) -> None:
        state = new_game(seed=1)
        state.actions_left = 2
        state.apply(Action(kind="end_turn"))
        self.assertEqual(state.current_side, Side.RED)
        self.assertEqual(state.actions_left, state.config.max_actions)

    def test_heuristic_bot_chooses_legal_action(self) -> None:
        state = new_game(seed=4)
        bot = HeuristicBot(dict(DEFAULT_WEIGHTS))
        action = bot.choose_action(state)
        self.assertIn(action, state.legal_actions())


if __name__ == "__main__":
    unittest.main()
