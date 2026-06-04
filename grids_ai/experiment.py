from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any


DEFAULT_MIN_HEAD_TO_HEAD_SCORE = 0.55
DEFAULT_MIN_HEAD_TO_HEAD_LOWER_BOUND = 0.50
DEFAULT_MIN_OVERALL_SCORE = 0.55


def wilson_lower_bound(score_rate: float, total_games: int, *, z: float = 1.96) -> float:
    if total_games <= 0:
        return 0.0
    score_rate = max(0.0, min(1.0, score_rate))
    denominator = 1.0 + z * z / total_games
    centre = score_rate + z * z / (2 * total_games)
    margin = z * math.sqrt((score_rate * (1.0 - score_rate) + z * z / (4 * total_games)) / total_games)
    return (centre - margin) / denominator


def required_games_for_wilson_gate(
    observed_score_rate: float,
    *,
    min_lower_bound: float = DEFAULT_MIN_HEAD_TO_HEAD_LOWER_BOUND,
    z: float = 1.96,
    max_games: int = 10000,
) -> int | None:
    """Return the first game count where the observed score clears the Wilson lower-bound gate."""
    if observed_score_rate <= min_lower_bound:
        return None
    for games in range(1, max_games + 1):
        if wilson_lower_bound(observed_score_rate, games, z=z) > min_lower_bound:
            return games
    return None


def promotion_gate_decision(
    *,
    head_to_head_score_rate: float,
    head_to_head_games: int,
    overall_score_rate: float,
    min_head_to_head_score: float = DEFAULT_MIN_HEAD_TO_HEAD_SCORE,
    min_overall_score: float = DEFAULT_MIN_OVERALL_SCORE,
    min_head_to_head_lower_bound: float = DEFAULT_MIN_HEAD_TO_HEAD_LOWER_BOUND,
) -> dict[str, Any]:
    lower_bound = wilson_lower_bound(head_to_head_score_rate, head_to_head_games)
    promote = (
        head_to_head_score_rate >= min_head_to_head_score
        and overall_score_rate >= min_overall_score
        and lower_bound > min_head_to_head_lower_bound
    )
    return {
        "promote": promote,
        "head_to_head_score_rate": head_to_head_score_rate,
        "head_to_head_games": head_to_head_games,
        "head_to_head_lower_bound": lower_bound,
        "overall_score_rate": overall_score_rate,
        "thresholds": {
            "min_head_to_head_score": min_head_to_head_score,
            "min_overall_score": min_overall_score,
            "min_head_to_head_lower_bound": min_head_to_head_lower_bound,
        },
        "reason": "candidate cleared promotion thresholds" if promote else "candidate did not clear promotion thresholds",
    }


@dataclass(frozen=True)
class ExperimentManifest:
    run_label: str
    kind: str
    reference_model: str | None = None
    datasets: tuple[str, ...] = ()
    checkpoints: tuple[str, ...] = ()
    reports: tuple[str, ...] = ()
    parameters: dict[str, Any] | None = None
    notes: str = ""
    generated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["generated_at"] = self.generated_at or datetime.now(timezone.utc).isoformat(timespec="seconds")
        payload["datasets"] = list(self.datasets)
        payload["checkpoints"] = list(self.checkpoints)
        payload["reports"] = list(self.reports)
        payload["parameters"] = self.parameters or {}
        return payload


def write_experiment_manifest(manifest: ExperimentManifest, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest.to_dict(), handle, indent=2, sort_keys=True)
        handle.write("\n")
