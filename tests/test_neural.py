from __future__ import annotations

import os
import tempfile
import unittest

from grids_ai.neural import (
    TrainingExample,
    ValueNetwork,
    generate_self_play_dataset,
    load_examples,
    train_value_model,
)


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
            )

            self.assertEqual(len(history), 1)
            self.assertTrue(os.path.exists(model_path))


if __name__ == "__main__":
    unittest.main()
