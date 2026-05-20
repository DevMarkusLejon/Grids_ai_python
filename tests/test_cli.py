from __future__ import annotations

import os
import tempfile
import unittest

from grids_ai.cli import build_bot, parse_args
from grids_ai.encoding import action_space_for_state
from grids_ai.engine import new_game
from grids_ai.neural import PolicyValueBot, PolicyValueMCTSBot, PolicyValueNetwork


def make_tiny_policy_value_model() -> PolicyValueNetwork:
    state = new_game(seed=1)
    action_size = action_space_for_state(state).size
    hidden_size = 2
    return PolicyValueNetwork(
        input_size=925,
        hidden_size=hidden_size,
        action_size=action_size,
        w1=[[0.0 for _ in range(925)] for _ in range(hidden_size)],
        b1=[0.0 for _ in range(hidden_size)],
        value_w=[0.0 for _ in range(hidden_size)],
        value_b=0.0,
        policy_w=[[0.0 for _ in range(hidden_size)] for _ in range(action_size)],
        policy_b=[0.0 for _ in range(action_size)],
    )


class CliTests(unittest.TestCase):
    def test_parse_args_accepts_delay(self) -> None:
        args = parse_args(["--blue", "heuristic", "--red", "random", "--delay", "0.25"])
        self.assertEqual(args.blue, "heuristic")
        self.assertEqual(args.red, "random")
        self.assertEqual(args.delay, 0.25)

    def test_parse_args_accepts_neural_model(self) -> None:
        args = parse_args(["--blue", "human", "--red", "neural", "--model", "checkpoints/model.json"])
        self.assertEqual(args.red, "neural")
        self.assertEqual(args.model, "checkpoints/model.json")

    def test_parse_args_accepts_policy_and_mcts_options(self) -> None:
        args = parse_args(
            [
                "--red",
                "neural",
                "--policy-scale",
                "21",
                "--mcts-simulations",
                "16",
                "--mcts-max-children",
                "12",
                "--mcts-depth",
                "5",
            ]
        )
        self.assertEqual(args.policy_scale, 21.0)
        self.assertEqual(args.mcts_simulations, 16)
        self.assertEqual(args.mcts_max_children, 12)
        self.assertEqual(args.mcts_depth, 5)

    def test_build_bot_returns_none_for_human(self) -> None:
        self.assertIsNone(build_bot("human"))

    def test_build_bot_loads_policy_value_model(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, "policy_value.json")
            make_tiny_policy_value_model().save(model_path)

            bot = build_bot(
                "neural",
                model_path=model_path,
                neural_search_width=1,
                neural_search_depth=1,
                policy_scale=12.0,
            )

            self.assertIsInstance(bot, PolicyValueBot)

    def test_build_bot_can_enable_policy_value_mcts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, "policy_value.json")
            make_tiny_policy_value_model().save(model_path)

            bot = build_bot(
                "neural",
                model_path=model_path,
                neural_search_width=1,
                neural_search_depth=1,
                mcts_simulations=4,
                mcts_max_children=8,
            )

            self.assertIsInstance(bot, PolicyValueMCTSBot)


if __name__ == "__main__":
    unittest.main()
