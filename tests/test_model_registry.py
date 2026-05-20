from __future__ import annotations

import json
import os
import tempfile
import unittest

from grids_ai.model_registry import build_model_registry, write_registry_json


class ModelRegistryTests(unittest.TestCase):
    def test_registry_ranks_models_from_champion_reports(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            reports_dir = os.path.join(temp_dir, "reports")
            checkpoints_dir = os.path.join(temp_dir, "checkpoints")
            os.makedirs(reports_dir)
            os.makedirs(checkpoints_dir)

            champion = "checkpoints/champion.json"
            candidate = "checkpoints/candidate.json"
            with open(os.path.join(checkpoints_dir, "champion.json"), "w", encoding="utf-8") as handle:
                json.dump({"input_size": 4, "hidden_size": 8, "metadata": {"role": "champion"}}, handle)
            with open(os.path.join(checkpoints_dir, "candidate.json"), "w", encoding="utf-8") as handle:
                json.dump({"input_size": 4, "hidden_size": 16, "metadata": {"role": "candidate"}}, handle)

            report = {
                "model": candidate,
                "hidden_size": 16,
                "input_size": 4,
                "total_games": 4,
                "overall": {"wins": 3, "losses": 1, "draws": 0, "score_rate": 0.75},
                "opponents": [
                    {
                        "kind": "neural",
                        "opponent": "neural:champion.json",
                        "metadata": {"model": champion},
                        "wins": 3,
                        "losses": 1,
                        "draws": 0,
                        "total_games": 4,
                        "score_rate": 0.75,
                    }
                ],
                "champion_decision": {
                    "candidate_model": candidate,
                    "champion_model": champion,
                    "promote": True,
                    "head_to_head_score_rate": 0.75,
                    "head_to_head_games": 4,
                    "reason": "candidate cleared promotion thresholds",
                },
            }
            with open(os.path.join(reports_dir, "champion_gate.json"), "w", encoding="utf-8") as handle:
                json.dump(report, handle)

            registry = build_model_registry(
                reports_dir="reports",
                checkpoints_dir="checkpoints",
                champion_model=champion,
                root=temp_dir,
            )

        rows = {row["model"]: row for row in registry["models"]}
        self.assertIn(candidate, rows)
        self.assertIn(champion, rows)
        self.assertTrue(rows[candidate]["promoted"])
        self.assertEqual(rows[candidate]["head_to_head_vs_champion"], 0.75)
        self.assertGreater(rows[candidate]["rating"], rows[champion]["rating"])

    def test_registry_json_writer_creates_parent_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = os.path.join(temp_dir, "web", "assets", "registry.json")
            write_registry_json({"models": [], "summary": {"models": 0}}, output)

            with open(output, "r", encoding="utf-8") as handle:
                payload = json.load(handle)

        self.assertEqual(payload["summary"]["models"], 0)


if __name__ == "__main__":
    unittest.main()
