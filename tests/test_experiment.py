from __future__ import annotations

import json
import os
import tempfile
import unittest

from grids_ai.experiment import (
    ExperimentManifest,
    promotion_gate_decision,
    required_games_for_wilson_gate,
    wilson_lower_bound,
    write_experiment_manifest,
)


class ExperimentTests(unittest.TestCase):
    def test_wilson_gate_requires_strict_lower_bound(self) -> None:
        decision = promotion_gate_decision(
            head_to_head_score_rate=0.55,
            head_to_head_games=1,
            overall_score_rate=0.80,
            min_head_to_head_lower_bound=0.50,
        )

        self.assertFalse(decision["promote"])
        self.assertLess(decision["head_to_head_lower_bound"], 0.50)

    def test_required_games_for_wilson_gate_estimates_power(self) -> None:
        games = required_games_for_wilson_gate(0.5625)

        self.assertIsNotNone(games)
        self.assertGreater(games, 192)
        self.assertGreater(wilson_lower_bound(0.5625, games or 0), 0.50)

    def test_manifest_writer_creates_json(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "reports", "experiment.manifest.json")
            write_experiment_manifest(
                ExperimentManifest(
                    run_label="unit_test_run",
                    kind="test",
                    reference_model="checkpoints/reference.json",
                    datasets=("neural_data/a.jsonl",),
                    checkpoints=("checkpoints/candidate.json",),
                    reports=("reports/gate.json",),
                    parameters={"games": 96},
                ),
                path,
            )

            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)

        self.assertEqual(payload["run_label"], "unit_test_run")
        self.assertEqual(payload["datasets"], ["neural_data/a.jsonl"])
        self.assertEqual(payload["parameters"]["games"], 96)


if __name__ == "__main__":
    unittest.main()
