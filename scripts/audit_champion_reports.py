from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


def iter_report_paths(report_dir: Path) -> Iterable[Path]:
    yield from sorted(report_dir.glob("*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    optuna_dir = report_dir / "optuna"
    if optuna_dir.exists():
        yield from sorted(optuna_dir.glob("*.json"), key=lambda path: path.stat().st_mtime, reverse=True)


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def report_row(path: Path, reference_model: str) -> dict[str, Any] | None:
    payload = read_json(path)
    if payload is None:
        return None
    decision = payload.get("champion_decision")
    if not isinstance(decision, dict):
        return None
    if decision.get("champion_model") != reference_model:
        return None
    return {
        "report": str(path),
        "candidate": decision.get("candidate_model"),
        "champion": decision.get("champion_model"),
        "head_to_head_score_rate": float(decision.get("head_to_head_score_rate") or 0.0),
        "head_to_head_lower_bound": float(decision.get("head_to_head_lower_bound") or 0.0),
        "head_to_head_games": int(decision.get("head_to_head_games") or 0),
        "promote": bool(decision.get("promote")),
        "mtime": path.stat().st_mtime,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit champion reports against a specific reference model.")
    parser.add_argument("--reports", default="reports", help="Report directory to scan.")
    parser.add_argument(
        "--reference-model",
        default="checkpoints/policy_value_torch_192_blend_20260521-073435.json",
        help="Current strongest model that candidates must beat.",
    )
    parser.add_argument("--min-score", type=float, default=0.55)
    parser.add_argument("--min-lower-bound", type=float, default=0.50)
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report_dir = Path(args.reports)
    rows = [
        row
        for path in iter_report_paths(report_dir)
        for row in [report_row(path, args.reference_model)]
        if row is not None
    ]
    rows.sort(
        key=lambda row: (
            row["head_to_head_score_rate"],
            row["head_to_head_lower_bound"],
            row["head_to_head_games"],
        ),
        reverse=True,
    )
    passing = [
        row
        for row in rows
        if row["head_to_head_score_rate"] >= args.min_score
        and row["head_to_head_lower_bound"] > args.min_lower_bound
    ]
    summary = {
        "reference_model": args.reference_model,
        "min_score": args.min_score,
        "min_lower_bound": args.min_lower_bound,
        "reports_vs_reference": len(rows),
        "passing_reports": passing,
        "best_reports": rows[: max(0, args.top)],
    }
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(f"Reference: {args.reference_model}")
        print(f"Reports vs reference: {len(rows)}")
        print(f"Passing reports: {len(passing)}")
        for row in rows[: max(0, args.top)]:
            marker = "PASS" if row in passing else "    "
            print(
                f"{marker} score={row['head_to_head_score_rate']:.6f} "
                f"lower={row['head_to_head_lower_bound']:.6f} "
                f"games={row['head_to_head_games']} "
                f"candidate={row['candidate']} "
                f"report={row['report']}"
            )
    return 0 if passing else 1


if __name__ == "__main__":
    raise SystemExit(main())
