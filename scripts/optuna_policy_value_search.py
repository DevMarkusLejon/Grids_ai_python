from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from grids_ai.experiment import ExperimentManifest, write_experiment_manifest


def parse_int_choices(raw: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer choice")
    return values


def parse_float_choices(raw: str) -> list[float]:
    values = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one numeric choice")
    return values


def tag_float(value: float) -> str:
    return f"{value:.8g}".replace("-", "m").replace(".", "p")


def choose(trial: Any, name: str, choices: Sequence[Any]) -> Any:
    if len(choices) == 1:
        return choices[0]
    return trial.suggest_categorical(name, list(choices))


def read_model_hidden_size(path: str) -> int | None:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    value = payload.get("hidden_size")
    if value is None:
        metadata = payload.get("metadata") or {}
        value = metadata.get("hidden_size")
    return int(value) if value is not None else None


def read_model_metadata(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    metadata = payload.get("metadata")
    return metadata if isinstance(metadata, dict) else {}


def read_decision(report_path: Path) -> dict[str, Any]:
    with report_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    decision = payload.get("champion_decision")
    if not isinstance(decision, dict):
        raise ValueError(f"Report has no champion_decision: {report_path}")
    return decision


def run_logged(command: Sequence[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.strftime("%Y-%m-%dT%H:%M:%S")
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n[{started}] > {subprocess.list2cmdline(list(command))}\n")
        handle.flush()
        completed = subprocess.run(
            list(command),
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        handle.write(f"[exit] code={completed.returncode}\n")
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {completed.returncode}; see {log_path}")


def apply_promotion(
    *,
    candidate: str,
    report: str,
    decision: dict[str, Any],
    policy_scale: float,
    python_executable: str,
) -> None:
    Path("champion_model.txt").write_text(candidate + "\n", encoding="ascii")
    Path("scripts/play_strongest.cmd").write_text(
        "\n".join(
            [
                "@echo off",
                'cd /d "%~dp0\\.."',
                (
                    "python -m grids_ai.cli --blue human --red neural "
                    f"--model {candidate} --policy-scale {policy_scale:g} "
                    "--neural-search-width 3 --neural-search-depth 4 %*"
                ),
                "",
            ]
        ),
        encoding="ascii",
    )
    run_logged(
        [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            "scripts/update_model_registry.ps1",
            "-Champion",
            candidate,
        ],
        Path("logs") / "optuna_promotion.log",
    )
    message = (
        f"New Grids AI champion from Optuna: {candidate} "
        f"score={float(decision['head_to_head_score_rate']):.3f} "
        f"lower={float(decision['head_to_head_lower_bound']):.3f} report={report}"
    )
    run_logged(
        [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            "scripts/notify_important.ps1",
            "-Title",
            "Grids AI new champion",
            "-Message",
            message,
            "-Priority",
            "high",
            "-Tags",
            "trophy",
        ],
        Path("logs") / "optuna_promotion.log",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a tiered Optuna search for Grids policy/value models. "
            "Trials train a candidate, screen it cheaply against a reference model, "
            "and optionally run a full champion gate for high-scoring candidates."
        )
    )
    parser.add_argument("--data", action="append", required=True, help="Training JSONL file. Repeat to blend datasets.")
    parser.add_argument(
        "--reference-model",
        default="checkpoints/policy_value_torch_192_blend_20260521-073435.json",
        help="Baseline model for init-model and head-to-head screening.",
    )
    parser.add_argument("--study-name", default="policy_value_search")
    parser.add_argument("--storage", default="sqlite:///optuna/grids_policy_value.db")
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--timeout-hours", type=float, default=0.0)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--screen-games", type=int, default=48, help="Games per side for cheap screening.")
    parser.add_argument("--full-gate-games", type=int, default=0, help="Games per side for optional final gate. Zero disables it.")
    parser.add_argument("--full-gate-score", type=float, default=0.56, help="Run full gate when screen score reaches this.")
    parser.add_argument("--policy-scale", type=float, default=18.0)
    parser.add_argument("--neural-search-width", type=int, default=3)
    parser.add_argument("--neural-search-depth", type=int, default=4)
    parser.add_argument("--hidden-sizes", type=parse_int_choices, default=parse_int_choices("192,384,512"))
    parser.add_argument("--epoch-choices", type=parse_int_choices, default=parse_int_choices("3,4,5,6"))
    parser.add_argument("--batch-size-choices", type=parse_int_choices, default=parse_int_choices("512,1024"))
    parser.add_argument("--per-data-limit-choices", type=parse_int_choices, default=parse_int_choices("30000,60000,90000"))
    parser.add_argument("--primary-repeat-choices", type=parse_int_choices, default=parse_int_choices("1,2,3"))
    parser.add_argument(
        "--policy-loss-weight-choices",
        type=parse_float_choices,
        default=parse_float_choices("0,0.001,0.005,0.01,0.02,0.05,0.1"),
    )
    parser.add_argument("--learning-rate-min", type=float, default=1e-6)
    parser.add_argument("--learning-rate-max", type=float, default=5e-5)
    parser.add_argument("--value-loss-weight", type=float, default=1.0)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--early-stop-patience", type=int, default=2)
    parser.add_argument("--seed-base", type=int, default=20370603)
    parser.add_argument("--checkpoint-dir", default="checkpoints/optuna")
    parser.add_argument("--report-dir", default="reports/optuna")
    parser.add_argument("--log-dir", default="logs/optuna")
    parser.add_argument("--prune-on-validation", action="store_true")
    parser.add_argument("--promote-on-pass", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        import optuna
    except ImportError as exc:
        raise SystemExit(
            "Optuna is not installed. Install it with "
            '".venv-gpu\\Scripts\\python.exe -m pip install -e .[search]" '
            "or python -m pip install optuna."
        ) from exc

    if args.trials < 1:
        raise SystemExit("--trials must be at least 1.")
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1.")
    if args.screen_games < 1:
        raise SystemExit("--screen-games must be at least 1.")
    if args.learning_rate_min <= 0 or args.learning_rate_max <= args.learning_rate_min:
        raise SystemExit("--learning-rate bounds must be positive and increasing.")
    if not os.path.exists(args.reference_model):
        raise SystemExit(f"Reference model not found: {args.reference_model}")
    missing_data = [path for path in args.data if not os.path.exists(path)]
    if missing_data:
        raise SystemExit(f"Training data not found: {missing_data}")

    checkpoint_dir = Path(args.checkpoint_dir)
    report_dir = Path(args.report_dir)
    log_dir = Path(args.log_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    Path("optuna").mkdir(exist_ok=True)

    init_hidden_size = read_model_hidden_size(args.reference_model)

    pruner = optuna.pruners.MedianPruner(n_startup_trials=4) if args.prune_on_validation else optuna.pruners.NopPruner()
    study = optuna.create_study(
        direction="maximize",
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=True,
        pruner=pruner,
    )

    def objective(trial: Any) -> float:
        hidden_size = int(choose(trial, "hidden_size", args.hidden_sizes))
        epochs = int(choose(trial, "epochs", args.epoch_choices))
        batch_size = int(choose(trial, "batch_size", args.batch_size_choices))
        per_data_limit = int(choose(trial, "per_data_limit", args.per_data_limit_choices))
        primary_repeats = int(choose(trial, "primary_repeats", args.primary_repeat_choices))
        policy_loss_weight = float(choose(trial, "policy_loss_weight", args.policy_loss_weight_choices))
        learning_rate = float(trial.suggest_float("learning_rate", args.learning_rate_min, args.learning_rate_max, log=True))
        freeze_init_model = bool(trial.suggest_categorical("freeze_init_model", [True, False]))
        if init_hidden_size is not None and hidden_size <= init_hidden_size:
            freeze_init_model = False

        seed = args.seed_base + trial.number * 7919
        trial_tag = (
            f"trial{trial.number:04d}_h{hidden_size}_p{tag_float(policy_loss_weight)}_"
            f"lr{tag_float(learning_rate)}_seed{seed}"
        )
        model_path = checkpoint_dir / f"policy_value_optuna_{trial_tag}.json"
        screen_report = report_dir / f"screen_{trial_tag}.json"
        full_report = report_dir / f"full_{trial_tag}.json"
        log_path = log_dir / f"{trial_tag}.log"

        data_paths = [args.data[0]] * primary_repeats + list(args.data[1:])
        train_command = [
            args.python,
            "-m",
            "grids_ai.neural",
            "train-policy",
        ]
        for data_path in data_paths:
            train_command.extend(["--data", data_path])
        train_command.extend(
            [
                "--model",
                str(model_path),
                "--init-model",
                args.reference_model,
                "--per-data-limit",
                str(per_data_limit),
                "--hidden-size",
                str(hidden_size),
                "--batch-size",
                str(batch_size),
                "--device",
                args.device,
                "--validation-fraction",
                str(args.validation_fraction),
                "--early-stop-patience",
                str(args.early_stop_patience),
                "--learning-rate",
                str(learning_rate),
                "--value-loss-weight",
                str(args.value_loss_weight),
                "--policy-loss-weight",
                str(policy_loss_weight),
                "--seed",
                str(seed),
                "--epochs",
                str(epochs),
                "--quiet",
            ]
        )
        if freeze_init_model:
            train_command.append("--freeze-init-model")

        run_logged(train_command, log_path)
        metadata = read_model_metadata(model_path)
        validation_history = metadata.get("validation_loss_history") or []
        best_validation = metadata.get("best_validation_loss")
        if best_validation is None and validation_history:
            best_validation = min(float(value) for value in validation_history)
        if best_validation is not None:
            trial.set_user_attr("best_validation_loss", float(best_validation))
            trial.report(-float(best_validation), step=0)
            if trial.should_prune():
                raise optuna.TrialPruned()

        screen_command = [
            args.python,
            "-m",
            "grids_ai.neural",
            "champion",
            "--candidate",
            str(model_path),
            "--champion",
            args.reference_model,
            "--games",
            str(args.screen_games),
            "--seed",
            str(seed + 41017),
            "--weights",
            "trained_weights.json",
            "--neural-search-width",
            str(args.neural_search_width),
            "--neural-search-depth",
            str(args.neural_search_depth),
            "--policy-scale",
            str(args.policy_scale),
            "--workers",
            str(args.workers),
            "--only-neural-opponents",
            "--min-head-to-head-score",
            "0.55",
            "--min-overall-score",
            "0.55",
            "--min-head-to-head-lower-bound",
            "0",
            "--output",
            str(screen_report),
            "--quiet",
        ]
        run_logged(screen_command, log_path)
        screen_decision = read_decision(screen_report)
        screen_score = float(screen_decision["head_to_head_score_rate"])
        screen_lower = float(screen_decision["head_to_head_lower_bound"])
        trial.set_user_attr("candidate_model", str(model_path))
        trial.set_user_attr("screen_report", str(screen_report))
        trial.set_user_attr("screen_score", screen_score)
        trial.set_user_attr("screen_lower_bound", screen_lower)
        trial.report(screen_score, step=1)

        objective_score = screen_score
        if args.full_gate_games > 0 and screen_score >= args.full_gate_score:
            full_command = [
                args.python,
                "-m",
                "grids_ai.neural",
                "champion",
                "--candidate",
                str(model_path),
                "--champion",
                args.reference_model,
                "--games",
                str(args.full_gate_games),
                "--seed",
                str(seed + 99089),
                "--weights",
                "trained_weights.json",
                "--neural-search-width",
                str(args.neural_search_width),
                "--neural-search-depth",
                str(args.neural_search_depth),
                "--policy-scale",
                str(args.policy_scale),
                "--workers",
                str(args.workers),
                "--only-neural-opponents",
                "--min-head-to-head-score",
                "0.55",
                "--min-overall-score",
                "0.55",
                "--min-head-to-head-lower-bound",
                "0.50",
                "--output",
                str(full_report),
                "--quiet",
            ]
            run_logged(full_command, log_path)
            full_decision = read_decision(full_report)
            full_score = float(full_decision["head_to_head_score_rate"])
            full_lower = float(full_decision["head_to_head_lower_bound"])
            trial.set_user_attr("full_report", str(full_report))
            trial.set_user_attr("full_score", full_score)
            trial.set_user_attr("full_lower_bound", full_lower)
            trial.set_user_attr("promote", bool(full_decision["promote"]))
            objective_score = full_score
            if bool(full_decision["promote"]) and args.promote_on_pass:
                apply_promotion(
                    candidate=str(model_path),
                    report=str(full_report),
                    decision=full_decision,
                    policy_scale=args.policy_scale,
                    python_executable=args.python,
                )

        return objective_score

    timeout_seconds = args.timeout_hours * 3600 if args.timeout_hours and args.timeout_hours > 0 else None
    study.optimize(
        objective,
        n_trials=args.trials,
        timeout=timeout_seconds,
        gc_after_trial=True,
        catch=(RuntimeError, ValueError),
    )

    try:
        best_trial = study.best_trial
    except ValueError:
        best_trial = None

    summary = {
        "study_name": args.study_name,
        "storage": args.storage,
        "reference_model": args.reference_model,
        "data": list(args.data),
        "screen_games": args.screen_games,
        "full_gate_games": args.full_gate_games,
        "full_gate_score": args.full_gate_score,
        "policy_scale": args.policy_scale,
        "neural_search_width": args.neural_search_width,
        "neural_search_depth": args.neural_search_depth,
        "workers": args.workers,
        "timeout_hours": args.timeout_hours,
        "best_trial": best_trial.number if best_trial else None,
        "best_value": best_trial.value if best_trial else None,
        "best_params": best_trial.params if best_trial else None,
        "best_user_attrs": dict(best_trial.user_attrs) if best_trial else None,
        "trials": len(study.trials),
    }
    summary_path = report_dir / f"{args.study_name}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest_path = report_dir / f"{args.study_name}.manifest.json"
    write_experiment_manifest(
        ExperimentManifest(
            run_label=args.study_name,
            kind="optuna_policy_value_search",
            reference_model=args.reference_model,
            datasets=tuple(args.data),
            checkpoints=(
                (str(best_trial.user_attrs["candidate_model"]),)
                if best_trial and "candidate_model" in best_trial.user_attrs
                else ()
            ),
            reports=tuple(
                str(value)
                for value in (
                    best_trial.user_attrs.get("screen_report") if best_trial else None,
                    best_trial.user_attrs.get("full_report") if best_trial else None,
                    str(summary_path),
                )
                if value
            ),
            parameters={
                "trials": args.trials,
                "timeout_hours": args.timeout_hours,
                "workers": args.workers,
                "screen_games": args.screen_games,
                "full_gate_games": args.full_gate_games,
                "policy_scale": args.policy_scale,
                "neural_search_width": args.neural_search_width,
                "neural_search_depth": args.neural_search_depth,
                "device": args.device,
                "storage": args.storage,
            },
        ),
        manifest_path,
    )
    summary["manifest"] = str(manifest_path)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
