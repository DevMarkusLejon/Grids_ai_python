from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Sequence


DEFAULT_CHAMPION_MODEL = "checkpoints/value_model_torch_128_shaped_1000_300hp.json"


@dataclass
class ModelSummary:
    model: str
    rating: float = 1000.0
    hidden_size: int | None = None
    input_size: int | None = None
    reports: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    games: int = 0
    best_overall_score: float | None = None
    latest_overall_score: float | None = None
    head_to_head_vs_champion: float | None = None
    head_to_head_games: int | None = None
    promoted: bool = False
    latest_report: str | None = None
    latest_report_mtime: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


def _as_posix(path: str | Path) -> str:
    return Path(path).as_posix()


def _relative_to_root(path: Path, root: Path) -> str:
    try:
        return _as_posix(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return _as_posix(path)


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _score_label(row: dict[str, Any]) -> tuple[int, int, int, int]:
    wins = int(row.get("wins", 0))
    losses = int(row.get("losses", 0))
    draws = int(row.get("draws", 0))
    games = int(row.get("total_games", wins + losses + draws))
    return wins, losses, draws, games


def _opponent_id(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    model = metadata.get("model")
    if isinstance(model, str) and model:
        return model
    return f"{row.get('kind', 'opponent')}:{row.get('opponent', 'unknown')}"


def _expected_score(rating: float, opponent_rating: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((opponent_rating - rating) / 400.0))


def _update_rating(ratings: dict[str, float], player: str, opponent: str, score: float, games: int) -> None:
    if games <= 0:
        return
    player_rating = ratings.setdefault(player, 1000.0)
    opponent_rating = ratings.setdefault(opponent, 1000.0)
    expected = _expected_score(player_rating, opponent_rating)
    # Aggregate Elo update: enough to rank experiments, conservative enough to avoid wild jumps.
    k = 28.0 * math.sqrt(games)
    delta = k * (score - expected)
    ratings[player] = player_rating + delta
    ratings[opponent] = opponent_rating - delta


def _load_checkpoint_metadata(checkpoints_dir: Path, root: Path) -> dict[str, dict[str, Any]]:
    metadata_by_model: dict[str, dict[str, Any]] = {}
    if not checkpoints_dir.exists():
        return metadata_by_model

    for path in checkpoints_dir.rglob("*.json"):
        payload = _read_json(path)
        if not payload or "hidden_size" not in payload or "input_size" not in payload:
            continue
        model_path = _relative_to_root(path, root)
        metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
        compact_metadata = _compact_metadata(metadata)
        kind = payload.get("kind", "value")
        compact_metadata["model_kind"] = kind if isinstance(kind, str) else "value"
        if "action_size" in payload:
            compact_metadata["action_size"] = int(payload["action_size"])
        metadata_by_model[model_path] = {
            "hidden_size": int(payload["hidden_size"]),
            "input_size": int(payload["input_size"]),
            "metadata": compact_metadata,
        }
    return metadata_by_model


def _compact_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    useful_keys = {
        "backend",
        "batch_size",
        "best_epoch",
        "best_validation_loss",
        "best_validation_policy_accuracy",
        "completed_epochs",
        "dataset",
        "datasets",
        "device",
        "epochs",
        "examples",
        "hidden_size",
        "learning_rate",
        "per_dataset_limit",
        "target",
        "training_examples",
        "policy_loss_weight",
        "validation_examples",
        "value_loss_weight",
    }
    compact = {key: value for key, value in metadata.items() if key in useful_keys}
    if isinstance(compact.get("dataset"), list):
        compact["dataset_count"] = len(compact["dataset"])
        compact["dataset"] = compact["dataset"][:4]
    if isinstance(compact.get("datasets"), list):
        compact["dataset_count"] = len(compact["datasets"])
        compact["datasets"] = compact["datasets"][:4]
    return compact


def build_model_registry(
    *,
    reports_dir: str | Path = "reports",
    checkpoints_dir: str | Path = "checkpoints",
    champion_model: str = DEFAULT_CHAMPION_MODEL,
    root: str | Path = ".",
) -> dict[str, Any]:
    root_path = Path(root)
    reports_path = root_path / reports_dir
    checkpoints_path = root_path / checkpoints_dir
    checkpoint_metadata = _load_checkpoint_metadata(checkpoints_path, root_path)

    ratings: dict[str, float] = {}
    models: dict[str, ModelSummary] = {}
    report_rows: list[dict[str, Any]] = []

    for model_path, metadata in checkpoint_metadata.items():
        summary = models.setdefault(model_path, ModelSummary(model=model_path))
        summary.hidden_size = metadata.get("hidden_size")
        summary.input_size = metadata.get("input_size")
        summary.metadata = metadata.get("metadata", {})

    if reports_path.exists():
        report_files = sorted(reports_path.glob("*.json"), key=lambda path: path.stat().st_mtime)
    else:
        report_files = []

    for report_path in report_files:
        payload = _read_json(report_path)
        if not payload or "opponents" not in payload:
            continue

        candidate = payload.get("model")
        decision = payload.get("champion_decision") if isinstance(payload.get("champion_decision"), dict) else {}
        if not isinstance(candidate, str):
            candidate = decision.get("candidate_model")
        if not isinstance(candidate, str) or not candidate:
            continue

        candidate = _as_posix(candidate)
        report_name = _relative_to_root(report_path, root_path)
        mtime = report_path.stat().st_mtime
        summary = models.setdefault(candidate, ModelSummary(model=candidate))
        summary.reports += 1
        summary.latest_report = report_name
        summary.latest_report_mtime = mtime
        summary.hidden_size = int(payload.get("hidden_size", summary.hidden_size or 0)) or summary.hidden_size
        summary.input_size = int(payload.get("input_size", summary.input_size or 0)) or summary.input_size

        overall = payload.get("overall") if isinstance(payload.get("overall"), dict) else {}
        score_rate = overall.get("score_rate")
        if isinstance(score_rate, (int, float)):
            summary.latest_overall_score = float(score_rate)
            if summary.best_overall_score is None or float(score_rate) > summary.best_overall_score:
                summary.best_overall_score = float(score_rate)
        summary.wins += int(overall.get("wins", 0))
        summary.losses += int(overall.get("losses", 0))
        summary.draws += int(overall.get("draws", 0))
        summary.games += int(payload.get("total_games", overall.get("wins", 0) + overall.get("losses", 0) + overall.get("draws", 0)))

        if decision:
            summary.promoted = summary.promoted or bool(decision.get("promote"))
            if isinstance(decision.get("head_to_head_score_rate"), (int, float)):
                summary.head_to_head_vs_champion = float(decision["head_to_head_score_rate"])
            if isinstance(decision.get("head_to_head_games"), int):
                summary.head_to_head_games = int(decision["head_to_head_games"])

        for opponent in payload.get("opponents", []):
            if not isinstance(opponent, dict):
                continue
            wins, losses, draws, games = _score_label(opponent)
            if games <= 0:
                continue
            score = (wins + 0.5 * draws) / games
            _update_rating(ratings, candidate, _opponent_id(opponent), score, games)

        report_rows.append(
            {
                "report": report_name,
                "model": candidate,
                "mtime": mtime,
                "overall_score": summary.latest_overall_score,
                "total_games": payload.get("total_games"),
                "promoted": bool(decision.get("promote")) if decision else False,
                "head_to_head_score": decision.get("head_to_head_score_rate") if decision else None,
                "reason": decision.get("reason") if decision else None,
            }
        )

    for model_path, summary in models.items():
        summary.rating = ratings.get(model_path, 1000.0)

    model_rows = []
    for rank, summary in enumerate(
        sorted(models.values(), key=lambda item: (item.rating, item.best_overall_score or 0.0, item.latest_report_mtime), reverse=True),
        start=1,
    ):
        model_rows.append(
            {
                "rank": rank,
                "model": summary.model,
                "rating": round(summary.rating, 1),
                "hidden_size": summary.hidden_size,
                "input_size": summary.input_size,
                "reports": summary.reports,
                "wins": summary.wins,
                "losses": summary.losses,
                "draws": summary.draws,
                "games": summary.games,
                "best_overall_score": summary.best_overall_score,
                "latest_overall_score": summary.latest_overall_score,
                "head_to_head_vs_champion": summary.head_to_head_vs_champion,
                "head_to_head_games": summary.head_to_head_games,
                "promoted": summary.promoted,
                "is_current_champion": _as_posix(summary.model) == _as_posix(champion_model),
                "latest_report": summary.latest_report,
                "metadata": summary.metadata,
            }
        )

    champion_row = next((row for row in model_rows if row["is_current_champion"]), None)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "champion_model": _as_posix(champion_model),
        "champion": champion_row,
        "models": model_rows,
        "reports": sorted(report_rows, key=lambda row: row["mtime"], reverse=True),
        "summary": {
            "models": len(model_rows),
            "reports": len(report_rows),
            "rated_models": sum(1 for row in model_rows if row["reports"] > 0),
        },
    }


def write_registry_json(registry: dict[str, Any], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(registry, handle, indent=2, sort_keys=True)
        handle.write("\n")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a Grids AI model registry from reports and checkpoints.")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--checkpoints-dir", default="checkpoints")
    parser.add_argument("--champion", default=DEFAULT_CHAMPION_MODEL)
    parser.add_argument("--output", default="web/assets/model-registry.json")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    registry = build_model_registry(
        reports_dir=args.reports_dir,
        checkpoints_dir=args.checkpoints_dir,
        champion_model=args.champion,
    )
    write_registry_json(registry, args.output)
    print(
        f"Wrote {args.output} with {registry['summary']['models']} models "
        f"and {registry['summary']['reports']} reports."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
