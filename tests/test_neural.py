from __future__ import annotations

import os
import json
import tempfile
import unittest

from grids_ai.neural import (
    PolicyValueMCTSBot,
    PolicyValueNetwork,
    TrainingExample,
    ValueNetwork,
    generate_self_play_dataset,
    load_examples,
    load_examples_from_paths,
    resolve_train_backend,
    run_champion_gauntlet,
    run_value_gauntlet,
    split_examples,
    target_value,
    train_policy_value_model,
    train_value_model,
)
from grids_ai.encoding import action_space_for_state
from grids_ai.data import Side
from grids_ai.engine import new_game


class NeuralTests(unittest.TestCase):
    def test_value_network_can_save_and_load_predictions(self) -> None:
        model = ValueNetwork(input_size=3, hidden_size=4, seed=1)
        features = [1.0, 0.0, -1.0]
        before = model.predict(features)

        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "value.json")
            model.save(path)
            loaded = ValueNetwork.load(path)

        self.assertAlmostEqual(before, loaded.predict(features), places=10)

    def test_value_network_predict_many_matches_scalar_predictions(self) -> None:
        model = ValueNetwork(input_size=3, hidden_size=4, seed=1)
        examples = [
            [1.0, 0.0, -1.0],
            [0.0, 0.5, 1.0],
            [-0.25, 1.0, 0.0],
        ]

        batch_predictions = model.predict_many(examples)
        scalar_predictions = [model.predict(features) for features in examples]

        self.assertEqual(len(batch_predictions), len(scalar_predictions))
        for batch, scalar in zip(batch_predictions, scalar_predictions):
            self.assertAlmostEqual(batch, scalar, places=6)

    def test_value_network_training_reduces_single_example_loss(self) -> None:
        example = TrainingExample(features=[1.0, 0.5, -0.25], value=1.0)
        model = ValueNetwork(input_size=3, hidden_size=4, seed=2)
        first_loss = (model.predict(example.features) - example.value) ** 2

        model.fit([example], epochs=8, learning_rate=0.05, seed=2)
        final_loss = (model.predict(example.features) - example.value) ** 2

        self.assertLess(final_loss, first_loss)

    def test_policy_value_network_can_score_legal_actions(self) -> None:
        state = new_game(seed=2)
        action_size = action_space_for_state(state).size
        model = PolicyValueNetwork(
            input_size=925,
            hidden_size=3,
            action_size=action_size,
            w1=[[0.0 for _ in range(925)] for _ in range(3)],
            b1=[0.0, 0.1, -0.1],
            value_w=[0.1, -0.2, 0.3],
            value_b=0.0,
            policy_w=[[0.0, 0.0, 0.0] for _ in range(action_size)],
            policy_b=[0.0 for _ in range(action_size)],
        )
        legal = state.legal_actions()
        priors = model.action_priors(state, legal)

        self.assertEqual(len(priors), len(legal))
        self.assertAlmostEqual(sum(priors), 1.0, places=6)
        self.assertIn(PolicyValueMCTSBot(model, simulations=2, max_children=4, mcts_depth=2).choose_action(state), legal)

    def test_self_play_dataset_can_train_tiny_model(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = os.path.join(temp_dir, "selfplay.jsonl")
            model_path = os.path.join(temp_dir, "model.json")
            count = generate_self_play_dataset(
                output_path=data_path,
                games=1,
                seed=3,
                search_width=1,
                search_depth=2,
                sample_every=25,
                max_examples_per_game=3,
            )

            self.assertGreater(count, 0)
            examples = load_examples(data_path)
            self.assertEqual(len(examples), count)
            self.assertTrue(all(example.legal_action_indices for example in examples))
            self.assertTrue(all(example.action_index in example.legal_action_indices for example in examples))

            history = train_value_model(
                dataset_path=data_path,
                model_path=model_path,
                hidden_size=4,
                epochs=1,
                learning_rate=0.01,
                backend="python",
                validation_fraction=0.0,
            )

            self.assertEqual(len(history), 1)
            self.assertTrue(os.path.exists(model_path))

    def test_self_play_dataset_can_use_neural_teacher(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            teacher_path = os.path.join(temp_dir, "teacher.json")
            data_path = os.path.join(temp_dir, "neural_selfplay.jsonl")
            ValueNetwork(input_size=925, hidden_size=4, seed=4).save(teacher_path)

            count = generate_self_play_dataset(
                output_path=data_path,
                games=1,
                seed=4,
                search_width=1,
                search_depth=1,
                sample_every=25,
                max_examples_per_game=3,
                teacher="neural",
                teacher_model_path=teacher_path,
                teacher_neural_search_width=1,
                teacher_neural_search_depth=1,
            )

            self.assertGreater(count, 0)
            self.assertEqual(len(load_examples(data_path)), count)

    def test_train_value_model_records_validation_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = os.path.join(temp_dir, "examples.jsonl")
            model_path = os.path.join(temp_dir, "model.json")
            examples = [
                TrainingExample(features=[float(index % 2), 1.0, 0.0], value=1.0 if index % 2 else -1.0)
                for index in range(10)
            ]
            with open(data_path, "w", encoding="utf-8") as handle:
                for example in examples:
                    handle.write(json.dumps({"features": example.features, "value": example.value}))
                    handle.write("\n")

            train_value_model(
                dataset_path=data_path,
                model_path=model_path,
                hidden_size=4,
                epochs=2,
                learning_rate=0.01,
                backend="python",
                validation_fraction=0.2,
            )

            with open(model_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)

        metadata = payload["metadata"]
        self.assertEqual(metadata["training_examples"], 8)
        self.assertEqual(metadata["validation_examples"], 2)
        self.assertEqual(len(metadata["validation_loss_history"]), 2)

    def test_training_can_blend_multiple_datasets(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            first_path = os.path.join(temp_dir, "first.jsonl")
            second_path = os.path.join(temp_dir, "second.jsonl")
            model_path = os.path.join(temp_dir, "model.json")
            for path, offset in [(first_path, 0), (second_path, 10)]:
                with open(path, "w", encoding="utf-8") as handle:
                    for index in range(4):
                        value = 1.0 if (index + offset) % 2 else -1.0
                        features = [float((index + offset) % 3), 1.0, 0.0]
                        handle.write(json.dumps({"features": features, "value": value}))
                        handle.write("\n")

            examples = load_examples_from_paths([first_path, second_path])
            train_value_model(
                dataset_path=[first_path, second_path],
                model_path=model_path,
                hidden_size=4,
                epochs=1,
                learning_rate=0.01,
                backend="python",
                validation_fraction=0.25,
            )

            with open(model_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)

        self.assertEqual(len(examples), 8)
        self.assertEqual(payload["metadata"]["examples"], 8)
        self.assertEqual(payload["metadata"]["dataset"], [first_path, second_path])

    def test_policy_value_training_writes_policy_model(self) -> None:
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("PyTorch is not installed.")

        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = os.path.join(temp_dir, "examples.jsonl")
            model_path = os.path.join(temp_dir, "policy_value.json")
            with open(data_path, "w", encoding="utf-8") as handle:
                for index in range(10):
                    handle.write(
                        json.dumps(
                            {
                                "features": [float(index % 2), 1.0, 0.0],
                                "value": 1.0 if index % 2 else -1.0,
                                "action_index": index % 4,
                            }
                        )
                    )
                    handle.write("\n")

            history = train_policy_value_model(
                dataset_path=data_path,
                model_path=model_path,
                hidden_size=4,
                action_size=4,
                epochs=1,
                batch_size=4,
                validation_fraction=0.2,
            )

            with open(model_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)

        self.assertEqual(len(history), 1)
        self.assertEqual(payload["kind"], "policy_value")
        self.assertEqual(payload["action_size"], 4)
        self.assertIn("validation_policy_accuracy_history", payload["metadata"])

    def test_policy_value_training_masks_illegal_action_logits(self) -> None:
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("PyTorch is not installed.")

        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = os.path.join(temp_dir, "masked_examples.jsonl")
            model_path = os.path.join(temp_dir, "policy_value.json")
            with open(data_path, "w", encoding="utf-8") as handle:
                for index in range(12):
                    action_index = 3 + (index % 2)
                    handle.write(
                        json.dumps(
                            {
                                "features": [1.0, float(index % 2), 0.0],
                                "value": 0.0,
                                "action_index": action_index,
                                "legal_action_indices": [3, 4],
                            }
                        )
                    )
                    handle.write("\n")

            history = train_policy_value_model(
                dataset_path=data_path,
                model_path=model_path,
                hidden_size=4,
                action_size=1000,
                epochs=1,
                batch_size=4,
                device="cpu",
                validation_fraction=0.0,
                value_loss_weight=1.0,
                policy_loss_weight=1.0,
            )

            with open(model_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)

        metadata = payload["metadata"]
        self.assertEqual(len(history), 1)
        self.assertEqual(metadata["masked_policy_examples"], 12)
        self.assertLess(metadata["policy_loss_history"][0], 2.5)

    def test_multi_dataset_loader_can_cap_each_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            first_path = os.path.join(temp_dir, "first.jsonl")
            second_path = os.path.join(temp_dir, "second.jsonl")
            for path in [first_path, second_path]:
                with open(path, "w", encoding="utf-8") as handle:
                    for index in range(5):
                        handle.write(json.dumps({"features": [float(index)], "value": 0.0}))
                        handle.write("\n")

            examples = load_examples_from_paths([first_path, second_path], per_dataset_limit=2)

        self.assertEqual(len(examples), 4)

    def test_explicit_python_backend_is_stable(self) -> None:
        self.assertEqual(resolve_train_backend("python"), "python")

    def test_split_examples_keeps_training_and_validation_disjoint(self) -> None:
        examples = [TrainingExample(features=[float(index)], value=0.0) for index in range(20)]
        training, validation = split_examples(examples, validation_fraction=0.25, seed=1)

        self.assertEqual(len(training), 15)
        self.assertEqual(len(validation), 5)
        self.assertEqual(len(training) + len(validation), len(examples))

    def test_shaped_target_stays_bounded_and_less_binary(self) -> None:
        state = new_game(seed=1)
        state.winner = Side.BLUE
        state.half_turns_played = 40

        blue_value = target_value(state, Side.BLUE, "shaped")
        red_value = target_value(state, Side.RED, "shaped")

        self.assertLessEqual(blue_value, 1.0)
        self.assertGreaterEqual(red_value, -1.0)
        self.assertLess(blue_value, 1.0)
        self.assertGreater(red_value, -1.0)

    def test_value_gauntlet_runs_paired_games(self) -> None:
        model = ValueNetwork(input_size=925, hidden_size=4, seed=5)
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, "model.json")
            model.save(model_path)

            result = run_value_gauntlet(
                model_path=model_path,
                games=1,
                seed=9,
                auto_neural_opponents=False,
                heuristic_search_width=1,
                heuristic_search_depth=1,
                neural_scale=80.0,
                heuristic_scale=0.5,
                neural_search_width=2,
                neural_search_depth=2,
            )

        self.assertEqual(result["games_per_side"], 1)
        self.assertEqual(result["total_games"], 4)
        self.assertEqual(len(result["opponents"]), 2)
        self.assertEqual(result["neural_scale"], 80.0)
        self.assertEqual(result["heuristic_scale"], 0.5)
        self.assertEqual(result["neural_search_width"], 2)
        self.assertEqual(result["neural_search_depth"], 2)

    def test_value_gauntlet_can_run_neural_only_head_to_head(self) -> None:
        model = ValueNetwork(input_size=925, hidden_size=4, seed=6)
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, "model.json")
            opponent_path = os.path.join(temp_dir, "opponent.json")
            model.save(model_path)
            model.save(opponent_path)

            result = run_value_gauntlet(
                model_path=model_path,
                games=1,
                seed=10,
                neural_opponent_models=[opponent_path],
                auto_neural_opponents=False,
                include_baseline_opponents=False,
                heuristic_search_width=1,
                heuristic_search_depth=1,
            )

        self.assertEqual(result["total_games"], 2)
        self.assertEqual(len(result["opponents"]), 1)
        self.assertEqual(result["opponents"][0]["kind"], "neural")

    def test_value_gauntlet_can_use_parallel_workers(self) -> None:
        model = ValueNetwork(input_size=925, hidden_size=4, seed=16)
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, "model.json")
            model.save(model_path)

            result = run_value_gauntlet(
                model_path=model_path,
                games=1,
                seed=16,
                auto_neural_opponents=False,
                include_baseline_opponents=True,
                heuristic_search_width=1,
                heuristic_search_depth=1,
                neural_search_width=1,
                neural_search_depth=1,
                workers=2,
            )

        self.assertEqual(result["workers"], 2)
        self.assertEqual(result["total_games"], 4)
        self.assertEqual(len(result["opponents"]), 2)

    def test_champion_gauntlet_reports_promotion_decision(self) -> None:
        candidate = ValueNetwork(input_size=925, hidden_size=4, seed=7)
        champion = ValueNetwork(input_size=925, hidden_size=4, seed=8)
        with tempfile.TemporaryDirectory() as temp_dir:
            candidate_path = os.path.join(temp_dir, "candidate.json")
            champion_path = os.path.join(temp_dir, "champion.json")
            candidate.save(candidate_path)
            champion.save(champion_path)

            result = run_champion_gauntlet(
                candidate_model_path=candidate_path,
                champion_model_path=champion_path,
                games=1,
                seed=11,
                include_baseline_opponents=False,
                neural_search_width=1,
                neural_search_depth=1,
                min_head_to_head_score=0.0,
                min_overall_score=0.0,
                min_head_to_head_lower_bound=0.0,
            )

        self.assertEqual(result["total_games"], 2)
        self.assertIn("champion_decision", result)
        self.assertIn("promote", result["champion_decision"])
        self.assertEqual(result["champion_decision"]["champion_model"], champion_path)


if __name__ == "__main__":
    unittest.main()
