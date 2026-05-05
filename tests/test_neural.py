from __future__ import annotations

import os
import json
import tempfile
import unittest

from grids_ai.neural import (
    TrainingExample,
    ValueNetwork,
    generate_self_play_dataset,
    load_examples,
    resolve_train_backend,
    run_value_gauntlet,
    split_examples,
    target_value,
    train_value_model,
)
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

    def test_value_network_training_reduces_single_example_loss(self) -> None:
        example = TrainingExample(features=[1.0, 0.5, -0.25], value=1.0)
        model = ValueNetwork(input_size=3, hidden_size=4, seed=2)
        first_loss = (model.predict(example.features) - example.value) ** 2

        model.fit([example], epochs=8, learning_rate=0.05, seed=2)
        final_loss = (model.predict(example.features) - example.value) ** 2

        self.assertLess(final_loss, first_loss)

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


if __name__ == "__main__":
    unittest.main()
