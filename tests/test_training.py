from __future__ import annotations

import random
from unittest.mock import patch
import unittest

from grids_ai.bots import DEFAULT_WEIGHTS
from grids_ai.data import Side
from grids_ai.engine import new_game
from grids_ai.training import (
    TrainingConfig,
    consistency_penalty,
    evaluate_candidate,
    match_score,
    max_candidate_score,
    mutated_weights,
    parse_args,
    train,
)


class TrainingTests(unittest.TestCase):
    def test_max_candidate_score_scales_with_games_and_pool(self) -> None:
        self.assertEqual(max_candidate_score(TrainingConfig(games_per_candidate=4), benchmark_count=1), 24.0)
        self.assertEqual(max_candidate_score(TrainingConfig(games_per_candidate=4), benchmark_count=3), 56.0)

    def test_parse_args_accepts_resume_from(self) -> None:
        args = parse_args(
            [
                "--resume-from",
                "saved.json",
                "--champion-pool-size",
                "7",
                "--checkpoint-prefix",
                "runs/checkpoint",
                "--checkpoint-every",
                "3",
                "--output",
                "next.json",
            ]
        )
        self.assertEqual(args.resume_from, "saved.json")
        self.assertEqual(args.champion_pool_size, 7)
        self.assertEqual(args.checkpoint_prefix, "runs/checkpoint")
        self.assertEqual(args.checkpoint_every, 3)
        self.assertEqual(args.output, "next.json")

    def test_mutation_keeps_terminal_outcomes_fixed(self) -> None:
        mutated = mutated_weights(random.Random(1), dict(DEFAULT_WEIGHTS), mutation_scale=50.0)
        self.assertEqual(mutated["win"], DEFAULT_WEIGHTS["win"])
        self.assertEqual(mutated["loss"], DEFAULT_WEIGHTS["loss"])

    def test_train_can_start_from_supplied_weights(self) -> None:
        starting = dict(DEFAULT_WEIGHTS)
        starting["attack"] += 9.5

        result = train(
            TrainingConfig(
                generations=0,
                population=1,
                games_per_candidate=1,
                mutation_scale=0.5,
                seed=3,
                starting_weights=starting,
                use_fixed_benchmarks=False,
            )
        )

        self.assertEqual(result.champion, starting)
        self.assertEqual(result.history, [])

    def test_train_merges_supplied_weights_over_defaults(self) -> None:
        result = train(
            TrainingConfig(
                generations=0,
                population=1,
                games_per_candidate=1,
                mutation_scale=0.5,
                seed=3,
                starting_weights={"attack": 42.0},
                use_fixed_benchmarks=False,
            )
        )

        self.assertEqual(result.champion["attack"], 42.0)
        self.assertEqual(result.champion["move"], DEFAULT_WEIGHTS["move"])

    def test_match_score_rewards_faster_wins(self) -> None:
        fast_win = new_game(seed=1)
        fast_win.winner = Side.BLUE
        fast_win.half_turns_played = 8
        fast_win.commander(Side.RED).hp = 0

        slow_win = new_game(seed=1)
        slow_win.winner = Side.BLUE
        slow_win.half_turns_played = slow_win.config.max_half_turns
        slow_win.commander(Side.RED).hp = 0

        self.assertGreater(match_score(fast_win), match_score(slow_win))

    def test_match_score_rewards_larger_margin(self) -> None:
        dominant_win = new_game(seed=1)
        dominant_win.winner = Side.BLUE
        dominant_win.half_turns_played = 10
        dominant_win.commander(Side.BLUE).hp = 145
        dominant_win.commander(Side.RED).hp = 120

        narrow_win = new_game(seed=1)
        narrow_win.winner = Side.BLUE
        narrow_win.half_turns_played = 10
        narrow_win.commander(Side.BLUE).hp = 140
        narrow_win.commander(Side.RED).hp = 130

        self.assertGreater(match_score(dominant_win), match_score(narrow_win))

    def test_match_score_rewards_commander_preservation(self) -> None:
        healthy_commander = new_game(seed=1)
        healthy_commander.winner = Side.BLUE
        healthy_commander.half_turns_played = 10
        healthy_commander.commander(Side.BLUE).hp = 180
        healthy_commander.commander(Side.RED).hp = 120

        battered_commander = new_game(seed=1)
        battered_commander.winner = Side.BLUE
        battered_commander.half_turns_played = 10
        battered_commander.commander(Side.BLUE).hp = 120
        battered_commander.commander(Side.RED).hp = 120

        self.assertGreater(match_score(healthy_commander), match_score(battered_commander))

    def test_match_score_rewards_material_advantage(self) -> None:
        material_edge = new_game(seed=1)
        material_edge.winner = Side.BLUE
        material_edge.half_turns_played = 10
        material_edge._spawn_unit("warrior", Side.BLUE, (2, 2))

        no_material_edge = new_game(seed=1)
        no_material_edge.winner = Side.BLUE
        no_material_edge.half_turns_played = 10

        self.assertGreater(match_score(material_edge), match_score(no_material_edge))

    def test_match_score_rewards_board_control(self) -> None:
        strong_position = new_game(seed=1)
        strong_position.winner = Side.BLUE
        strong_position.half_turns_played = 10
        strong_position._spawn_unit("warrior", Side.BLUE, (7, 3))

        passive_position = new_game(seed=1)
        passive_position.winner = Side.BLUE
        passive_position.half_turns_played = 10
        passive_position._spawn_unit("warrior", Side.BLUE, (2, 3))

        self.assertGreater(match_score(strong_position), match_score(passive_position))

    def test_match_score_rewards_resource_efficiency(self) -> None:
        full_hand = new_game(seed=1)
        full_hand.winner = Side.BLUE
        full_hand.half_turns_played = 10

        empty_hand = new_game(seed=1)
        empty_hand.winner = Side.BLUE
        empty_hand.half_turns_played = 10
        empty_hand.hands[Side.BLUE] = []

        self.assertGreater(match_score(full_hand), match_score(empty_hand))

    def test_match_score_penalizes_timeout_games(self) -> None:
        decisive_win = new_game(seed=1)
        decisive_win.winner = Side.BLUE
        decisive_win.half_turns_played = decisive_win.config.max_half_turns

        timeout_win = new_game(seed=1)
        timeout_win.winner = Side.BLUE
        timeout_win.half_turns_played = timeout_win.config.max_half_turns
        timeout_win.winner_reason = "score advantage after turn limit"

        self.assertGreater(match_score(decisive_win), match_score(timeout_win))

    def test_consistency_penalty_grows_with_variance(self) -> None:
        self.assertEqual(consistency_penalty([1.0, 1.0, 1.0]), 0.0)
        self.assertGreater(consistency_penalty([2.0, -2.0, 2.0, -2.0]), 0.0)

    def test_evaluate_candidate_uses_every_benchmark_in_pool(self) -> None:
        config = TrainingConfig(games_per_candidate=2, seed=5)
        benchmark_pool = [dict(DEFAULT_WEIGHTS), dict(DEFAULT_WEIGHTS)]
        finished = new_game(seed=1)
        finished.winner = Side.BLUE
        finished.commander(Side.RED).hp = 0

        with patch("grids_ai.training.play_match", return_value=finished) as play_match_mock:
            evaluate_candidate(dict(DEFAULT_WEIGHTS), benchmark_pool, config, seed_offset=0)

        self.assertEqual(play_match_mock.call_count, 12)

    def test_train_writes_latest_checkpoint_each_generation(self) -> None:
        config = TrainingConfig(
            generations=3,
            population=1,
            games_per_candidate=1,
            seed=3,
            checkpoint_prefix="checkpoints/run",
            checkpoint_every=2,
            use_fixed_benchmarks=False,
            holdout_games=0,
        )

        with patch("grids_ai.training.save_weights") as save_mock:
            result = train(config)

        self.assertEqual(len(result.history), 3)
        paths = [call.args[0] for call in save_mock.call_args_list]
        self.assertEqual(
            paths,
            [
                "checkpoints/run.latest.json",
                "checkpoints/run.latest.json",
                "checkpoints/run.gen_002.json",
                "checkpoints/run.latest.json",
            ],
        )
        latest_metadata = save_mock.call_args_list[-1].kwargs["metadata"]
        self.assertEqual(latest_metadata["checkpoint_kind"], "latest")
        self.assertEqual(latest_metadata["completed_generations"], 3)

    def test_train_writes_snapshot_when_stopping_early(self) -> None:
        config = TrainingConfig(
            generations=5,
            population=1,
            games_per_candidate=2,
            seed=3,
            checkpoint_prefix="checkpoints/early",
            checkpoint_every=1,
            use_fixed_benchmarks=False,
            holdout_games=0,
        )
        ceiling = max_candidate_score(config, benchmark_count=1)

        with (
            patch("grids_ai.training.evaluate_candidate", return_value=ceiling),
            patch("grids_ai.training.save_weights") as save_mock,
        ):
            result = train(config)

        self.assertTrue(result.stopped_early)
        checkpoint_kinds = [call.kwargs["metadata"]["checkpoint_kind"] for call in save_mock.call_args_list]
        self.assertEqual(checkpoint_kinds, ["latest", "snapshot"])
        self.assertTrue(save_mock.call_args_list[0].kwargs["metadata"]["stopped_early"])

    def test_train_stops_early_at_max_score(self) -> None:
        config = TrainingConfig(
            generations=5,
            population=1,
            games_per_candidate=2,
            seed=3,
            use_fixed_benchmarks=False,
            holdout_games=0,
        )
        ceiling = max_candidate_score(config, benchmark_count=1)

        with patch("grids_ai.training.evaluate_candidate", return_value=ceiling):
            result = train(config)

        self.assertEqual(result.champion, DEFAULT_WEIGHTS)
        self.assertEqual(result.history, [(1, ceiling)])
        self.assertTrue(result.stopped_early)


if __name__ == "__main__":
    unittest.main()
