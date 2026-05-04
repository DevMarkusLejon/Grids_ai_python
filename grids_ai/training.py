from __future__ import annotations

from dataclasses import dataclass, replace
import argparse
import random
from statistics import pstdev
import sys
import time
from typing import Sequence

from .bots import DEFAULT_WEIGHTS, HeuristicBot, RandomBot, load_weights, save_weights
from .data import Side
from .engine import GameState, new_game


WIN_SCORE = 1.0
MARGIN_WEIGHT = 0.25
SPEED_WEIGHT = 0.25
COMMANDER_HEALTH_WEIGHT = 0.20
MATERIAL_WEIGHT = 0.15
BOARD_CONTROL_WEIGHT = 0.10
RESOURCE_WEIGHT = 0.05
TIMEOUT_PENALTY = 0.75
CONSISTENCY_WEIGHT = 0.20
MARGIN_SCALE = 200.0
RANDOM_OPPONENT_WEIGHT = 0.5
FIXED_TRAINING_WEIGHTS = frozenset({"win", "loss"})


@dataclass(frozen=True)
class TrainingConfig:
    generations: int = 20
    population: int = 10
    games_per_candidate: int = 4
    mutation_scale: float = 1.25
    seed: int = 7
    map_name: str = "plains"
    champion_pool_size: int = 5
    checkpoint_prefix: str | None = None
    checkpoint_every: int = 5
    resume_from_path: str | None = None
    starting_weights: dict[str, float] | None = None
    verbose_training: bool = False
    verbose_every: int = 25
    early_stop_patience: int | None = None
    early_stop_min_delta: float = 0.0
    promotion_margin: float = 0.05
    holdout_games: int = 1
    use_fixed_benchmarks: bool = True
    ai_search_width: int = 1
    ai_search_depth: int | None = None


@dataclass(frozen=True)
class TrainingResult:
    champion: dict[str, float]
    history: list[tuple[int, float]]
    benchmark_count: int
    max_score: float
    stopped_early: bool


def mutated_weights(
    rng: random.Random,
    base_weights: dict[str, float],
    mutation_scale: float,
) -> dict[str, float]:
    mutated: dict[str, float] = {}
    for key, value in base_weights.items():
        if key in FIXED_TRAINING_WEIGHTS:
            mutated[key] = value
            continue
        mutated[key] = value + rng.gauss(0.0, mutation_scale)
    return mutated


def make_heuristic_bot(weights: dict[str, float], seed: int, config: TrainingConfig) -> HeuristicBot:
    return HeuristicBot(
        weights,
        seed=seed,
        search_width=config.ai_search_width,
        search_depth=config.ai_search_depth,
    )


def archetype_benchmarks() -> list[dict[str, float]]:
    rush = dict(DEFAULT_WEIGHTS)
    rush.update(
        {
            "enemy_commander_delta": 22.0,
            "commander_distance_delta": 9.0,
            "enemy_commander_threat_delta": 4.0,
            "lethal_threat": 260.0,
            "forward_pressure_delta": 4.5,
            "draw_item": 2.5,
            "move": 4.0,
        }
    )

    material = dict(DEFAULT_WEIGHTS)
    material.update(
        {
            "enemy_unit_delta": 70.0,
            "own_unit_delta": -72.0,
            "enemy_unit_value_delta": 0.55,
            "own_unit_value_delta": -0.65,
            "deploy": 9.0,
            "hand_delta": 1.5,
        }
    )

    defensive = dict(DEFAULT_WEIGHTS)
    defensive.update(
        {
            "own_commander_delta": -28.0,
            "own_commander_threat_delta": -5.0,
            "own_lethal_risk": -340.0,
            "effective_healing": 3.0,
            "heal": 7.0,
            "forward_pressure_delta": 0.5,
        }
    )

    return [rush, material, defensive]


def benchmark_weight_pool(champion_pool: Sequence[dict[str, float]], config: TrainingConfig) -> list[dict[str, float]]:
    benchmarks = [dict(weights) for weights in champion_pool]
    if config.use_fixed_benchmarks:
        benchmarks.extend(archetype_benchmarks())
    return benchmarks


def play_match(
    blue_bot,
    red_bot,
    seed: int,
    map_name: str = "plains",
    *,
    on_step=None,
) -> GameState:
    state = new_game(seed=seed, map_name=map_name)
    turn_safety = state.config.max_half_turns * 8
    steps = 0

    while not state.is_done and steps < turn_safety:
        bot = blue_bot if state.current_side.value == "blue" else red_bot
        action = bot.choose_action(state)
        state.apply(action)
        steps += 1
        if on_step is not None:
            on_step(state, steps)

    if not state.is_done:
        state._resolve_timeout_winner()
    return state


def should_verbose_match(config: TrainingConfig, match_index: int) -> bool:
    if not config.verbose_training:
        return False
    return match_index % max(config.verbose_every, 1) == 0


def print_verbose_match_summary(
    label: str,
    state: GameState,
    *,
    score: float,
    match_index: int,
    sampled_steps: list[tuple[int, int, int]],
) -> None:
    print()
    print(f"[training] Match {match_index}: {label}")
    if sampled_steps:
        snapshots = ", ".join(
            f"step {step} B:{blue_hp} R:{red_hp}"
            for step, blue_hp, red_hp in sampled_steps
        )
        print(f"[training] Commander HP snapshots: {snapshots}")
    print(
        "[training] Result: "
        f"winner={state.winner.value if state.winner else 'none'} "
        f"reason={state.winner_reason or 'n/a'} "
        f"half_turns={state.half_turns_played} "
        f"score={score:+.3f}"
    )
    print(state.render())


def match_score(state: GameState) -> float:
    if state.winner is None:
        return 0.0
    outcome = WIN_SCORE if state.winner is Side.BLUE else -WIN_SCORE
    margin = normalized_margin(state)
    speed = speed_bonus(state)
    commander_health = commander_health_score(state)
    material = material_score(state)
    board_control = board_control_score(state)
    resources = resource_efficiency_score(state)
    timeout_penalty = TIMEOUT_PENALTY if is_timeout_game(state) else 0.0
    return (
        outcome
        + margin * MARGIN_WEIGHT
        + speed * SPEED_WEIGHT
        + commander_health * COMMANDER_HEALTH_WEIGHT
        + material * MATERIAL_WEIGHT
        + board_control * BOARD_CONTROL_WEIGHT
        + resources * RESOURCE_WEIGHT
        - timeout_penalty
    )


def normalized_margin(state: GameState) -> float:
    score_delta = state.side_score(Side.BLUE) - state.side_score(Side.RED)
    return max(-1.0, min(1.0, score_delta / MARGIN_SCALE))


def normalized_difference(blue_value: float, red_value: float) -> float:
    total = abs(blue_value) + abs(red_value)
    if total <= 0.0:
        return 0.0
    return max(-1.0, min(1.0, (blue_value - red_value) / total))


def speed_bonus(state: GameState) -> float:
    if state.winner is None:
        return 0.0
    completed_half_turns = min(state.half_turns_played, state.config.max_half_turns)
    bonus = 1.0 - (completed_half_turns / max(state.config.max_half_turns, 1))
    return bonus if state.winner is Side.BLUE else -bonus


def commander_health_score(state: GameState) -> float:
    blue_ratio = state.commander_hp(Side.BLUE) / max(state.commander_max_hp(Side.BLUE), 1)
    red_ratio = state.commander_hp(Side.RED) / max(state.commander_max_hp(Side.RED), 1)
    return max(-1.0, min(1.0, blue_ratio - red_ratio))


def material_score(state: GameState) -> float:
    blue_units = sum(1 for unit in state.units_for_side(Side.BLUE) if not unit.is_commander)
    red_units = sum(1 for unit in state.units_for_side(Side.RED) if not unit.is_commander)
    return normalized_difference(blue_units, red_units)


def board_control_score(state: GameState) -> float:
    return normalized_difference(
        state.forward_pressure(Side.BLUE),
        state.forward_pressure(Side.RED),
    )


def resource_efficiency_score(state: GameState) -> float:
    return normalized_difference(
        len(state.hands[Side.BLUE]),
        len(state.hands[Side.RED]),
    )


def is_timeout_game(state: GameState) -> bool:
    return "turn limit" in state.winner_reason


def consistency_penalty(weighted_scores: Sequence[float]) -> float:
    if len(weighted_scores) < 2:
        return 0.0
    return (pstdev(weighted_scores) / max(max_match_score(), 1.0)) * CONSISTENCY_WEIGHT * len(weighted_scores)


def evaluate_candidate(
    candidate_weights: dict[str, float],
    benchmark_pool: Sequence[dict[str, float]],
    config: TrainingConfig,
    seed_offset: int,
) -> float:
    total = 0.0
    weighted_scores: list[float] = []
    match_index = 0

    for game_index in range(config.games_per_candidate):
        seed = config.seed + seed_offset * 1000 + game_index * 100

        for benchmark_index, benchmark_weights in enumerate(benchmark_pool):
            benchmark_seed = seed + benchmark_index * 10
            match_index += 1
            verbose_match = should_verbose_match(config, match_index)
            sampled_steps: list[tuple[int, int, int]] = []

            blue_vs_bench = play_match(
                make_heuristic_bot(candidate_weights, benchmark_seed, config),
                make_heuristic_bot(benchmark_weights, benchmark_seed + 1, config),
                seed=benchmark_seed,
                map_name=config.map_name,
                on_step=(
                    lambda state, steps, snapshots=sampled_steps: snapshots.append(
                        (
                            steps,
                            state.commander_hp(Side.BLUE),
                            state.commander_hp(Side.RED),
                        )
                    )
                    if verbose_match and steps % 20 == 0
                    else None
                ),
            )
            blue_score = match_score(blue_vs_bench)
            total += blue_score
            weighted_scores.append(blue_score)
            if verbose_match:
                print_verbose_match_summary(
                    "candidate as blue vs benchmark",
                    blue_vs_bench,
                    score=blue_score,
                    match_index=match_index,
                    sampled_steps=sampled_steps,
                )

            match_index += 1
            verbose_match = should_verbose_match(config, match_index)
            sampled_steps = []
            red_vs_bench = play_match(
                make_heuristic_bot(benchmark_weights, benchmark_seed + 2, config),
                make_heuristic_bot(candidate_weights, benchmark_seed + 3, config),
                seed=benchmark_seed + 1000,
                map_name=config.map_name,
                on_step=(
                    lambda state, steps, snapshots=sampled_steps: snapshots.append(
                        (
                            steps,
                            state.commander_hp(Side.BLUE),
                            state.commander_hp(Side.RED),
                        )
                    )
                    if verbose_match and steps % 20 == 0
                    else None
                ),
            )
            red_score = -match_score(red_vs_bench)
            total += red_score
            weighted_scores.append(red_score)
            if verbose_match:
                print_verbose_match_summary(
                    "candidate as red vs benchmark",
                    red_vs_bench,
                    score=red_score,
                    match_index=match_index,
                    sampled_steps=sampled_steps,
                )

        random_seed = seed + len(benchmark_pool) * 10
        match_index += 1
        verbose_match = should_verbose_match(config, match_index)
        sampled_steps = []
        blue_vs_random = play_match(
            make_heuristic_bot(candidate_weights, random_seed, config),
            RandomBot(seed=random_seed + 1),
            seed=random_seed + 2000,
            map_name=config.map_name,
            on_step=(
                lambda state, steps, snapshots=sampled_steps: snapshots.append(
                    (
                        steps,
                        state.commander_hp(Side.BLUE),
                        state.commander_hp(Side.RED),
                    )
                )
                if verbose_match and steps % 20 == 0
                else None
            ),
        )
        blue_random_score = match_score(blue_vs_random) * RANDOM_OPPONENT_WEIGHT
        total += blue_random_score
        weighted_scores.append(blue_random_score)
        if verbose_match:
            print_verbose_match_summary(
                "candidate as blue vs random",
                blue_vs_random,
                score=blue_random_score,
                match_index=match_index,
                sampled_steps=sampled_steps,
            )

        match_index += 1
        verbose_match = should_verbose_match(config, match_index)
        sampled_steps = []
        red_vs_random = play_match(
            RandomBot(seed=random_seed + 2),
            make_heuristic_bot(candidate_weights, random_seed + 3, config),
            seed=random_seed + 3000,
            map_name=config.map_name,
            on_step=(
                lambda state, steps, snapshots=sampled_steps: snapshots.append(
                    (
                        steps,
                        state.commander_hp(Side.BLUE),
                        state.commander_hp(Side.RED),
                    )
                )
                if verbose_match and steps % 20 == 0
                else None
            ),
        )
        red_random_score = -match_score(red_vs_random) * RANDOM_OPPONENT_WEIGHT
        total += red_random_score
        weighted_scores.append(red_random_score)
        if verbose_match:
            print_verbose_match_summary(
                "candidate as red vs random",
                red_vs_random,
                score=red_random_score,
                match_index=match_index,
                sampled_steps=sampled_steps,
            )

    return total - consistency_penalty(weighted_scores)


def max_match_score() -> float:
    return (
        WIN_SCORE
        + MARGIN_WEIGHT
        + SPEED_WEIGHT
        + COMMANDER_HEALTH_WEIGHT
        + MATERIAL_WEIGHT
        + BOARD_CONTROL_WEIGHT
        + RESOURCE_WEIGHT
    )


def max_candidate_score(config: TrainingConfig, benchmark_count: int) -> float:
    return config.games_per_candidate * max_match_score() * (benchmark_count * 2 + RANDOM_OPPONENT_WEIGHT * 2)


def score_component_metadata() -> dict[str, float]:
    return {
        "win_score": WIN_SCORE,
        "margin_weight": MARGIN_WEIGHT,
        "speed_weight": SPEED_WEIGHT,
        "commander_health_weight": COMMANDER_HEALTH_WEIGHT,
        "material_weight": MATERIAL_WEIGHT,
        "board_control_weight": BOARD_CONTROL_WEIGHT,
        "resource_weight": RESOURCE_WEIGHT,
        "timeout_penalty": TIMEOUT_PENALTY,
        "consistency_weight": CONSISTENCY_WEIGHT,
        "margin_scale": MARGIN_SCALE,
        "random_opponent_weight": RANDOM_OPPONENT_WEIGHT,
    }


def build_training_metadata(
    config: TrainingConfig,
    *,
    history: list[tuple[int, float]],
    benchmark_count: int,
    max_score: float,
    stopped_early: bool,
    checkpoint_kind: str | None = None,
) -> dict[str, object]:
    return {
        "generations": config.generations,
        "population": config.population,
        "games_per_candidate": config.games_per_candidate,
        "mutation_scale": config.mutation_scale,
        "seed": config.seed,
        "map": config.map_name,
        "champion_pool_size": config.champion_pool_size,
        "verbose_training": config.verbose_training,
        "verbose_every": config.verbose_every,
        "early_stop_patience": config.early_stop_patience,
        "early_stop_min_delta": config.early_stop_min_delta,
        "promotion_margin": config.promotion_margin,
        "holdout_games": config.holdout_games,
        "use_fixed_benchmarks": config.use_fixed_benchmarks,
        "ai_search_width": config.ai_search_width,
        "ai_search_depth": config.ai_search_depth,
        "resume_from": config.resume_from_path,
        "benchmark_count": benchmark_count,
        "max_score": max_score,
        "score_components": score_component_metadata(),
        "history": history,
        "completed_generations": len(history),
        "stopped_early": stopped_early,
        "checkpoint_kind": checkpoint_kind,
    }


def save_checkpoint(
    config: TrainingConfig,
    champion: dict[str, float],
    *,
    history: list[tuple[int, float]],
    benchmark_count: int,
    max_score: float,
    stopped_early: bool,
) -> None:
    if not config.checkpoint_prefix:
        return

    latest_path = f"{config.checkpoint_prefix}.latest.json"
    save_weights(
        latest_path,
        champion,
        metadata=build_training_metadata(
            config,
            history=history,
            benchmark_count=benchmark_count,
            max_score=max_score,
            stopped_early=stopped_early,
            checkpoint_kind="latest",
        ),
    )

    if len(history) % config.checkpoint_every != 0:
        return

    snapshot_path = f"{config.checkpoint_prefix}.gen_{len(history):03d}.json"
    save_weights(
        snapshot_path,
        champion,
        metadata=build_training_metadata(
            config,
            history=history,
            benchmark_count=benchmark_count,
            max_score=max_score,
            stopped_early=stopped_early,
            checkpoint_kind="snapshot",
        ),
    )


def render_progress(
    current: int,
    total: int,
    *,
    generation: int,
    generations: int,
    width: int = 28,
) -> None:
    total = max(total, 1)
    ratio = min(max(current / total, 0.0), 1.0)
    filled = int(width * ratio)
    bar = "#" * filled + "." * (width - filled)
    message = (
        f"\rTraining gen {generation}/{generations} "
        f"[{bar}] {current}/{total}"
    )
    sys.stdout.write(message)
    sys.stdout.flush()
    if current == total:
        sys.stdout.write("\n")


def format_generation_summary(
    generation: int,
    generations: int,
    *,
    champion_score: float,
    previous_score: float,
    score_ceiling: float,
    champion_improved: bool,
    champion_pool_size: int,
    elapsed_seconds: float,
) -> str:
    score_delta = champion_score - previous_score
    ceiling_pct = 0.0 if score_ceiling <= 0.0 else (champion_score / score_ceiling) * 100.0
    status = "new champion" if champion_improved else "held steady"
    return (
        f"Generation {generation:>2}/{generations}: champion score {champion_score:.3f} "
        f"({score_delta:+.3f}), {ceiling_pct:.1f}% of ceiling, "
        f"pool {champion_pool_size}, {status}, {elapsed_seconds:.2f}s"
    )


def train(config: TrainingConfig) -> TrainingResult:
    rng = random.Random(config.seed)
    champion = dict(DEFAULT_WEIGHTS)
    if config.starting_weights is not None:
        champion.update(config.starting_weights)
    champion_pool: list[dict[str, float]] = [dict(champion)]
    history: list[tuple[int, float]] = []
    evaluation_pool = benchmark_weight_pool(champion_pool, config)
    champion_score = evaluate_candidate(champion, evaluation_pool, config, seed_offset=0)
    stopped_early = False
    generations_without_improvement = 0
    score_ceiling = max_candidate_score(config, len(evaluation_pool))

    for generation in range(1, config.generations + 1):
        generation_started = time.perf_counter()
        evaluation_pool = benchmark_weight_pool(champion_pool, config)
        eval_seed_offset = generation * 1000
        previous_score = evaluate_candidate(champion, evaluation_pool, config, eval_seed_offset)
        best_candidate = champion
        best_score = previous_score

        render_progress(0, config.population, generation=generation, generations=config.generations)
        for candidate_index in range(config.population):
            candidate = mutated_weights(rng, champion, config.mutation_scale)
            score = evaluate_candidate(candidate, evaluation_pool, config, eval_seed_offset)
            if score > best_score + config.promotion_margin:
                best_candidate = candidate
                best_score = score
            render_progress(
                candidate_index + 1,
                config.population,
                generation=generation,
                generations=config.generations,
            )

        champion_improved = best_candidate is not champion
        if champion_improved and config.holdout_games > 0:
            holdout_config = replace(config, games_per_candidate=config.holdout_games)
            holdout_pool = benchmark_weight_pool(champion_pool, holdout_config)
            holdout_seed_offset = generation * 1000 + 777
            champion_holdout = evaluate_candidate(champion, holdout_pool, holdout_config, holdout_seed_offset)
            candidate_holdout = evaluate_candidate(best_candidate, holdout_pool, holdout_config, holdout_seed_offset)
            if candidate_holdout <= champion_holdout + config.promotion_margin:
                best_candidate = champion
                best_score = previous_score
                champion_improved = False
        score_improvement = best_score - previous_score
        champion = best_candidate
        champion_score = best_score
        if champion_improved:
            champion_pool.append(dict(champion))
            champion_pool = champion_pool[-config.champion_pool_size :]
        if score_improvement >= config.early_stop_min_delta:
            generations_without_improvement = 0
        else:
            generations_without_improvement += 1
        evaluation_pool = benchmark_weight_pool(champion_pool, config)
        score_ceiling = max_candidate_score(config, len(evaluation_pool))
        history.append((generation, champion_score))
        print(
            format_generation_summary(
                generation,
                config.generations,
                champion_score=champion_score,
                previous_score=previous_score,
                score_ceiling=score_ceiling,
                champion_improved=champion_improved,
                champion_pool_size=len(champion_pool),
                elapsed_seconds=time.perf_counter() - generation_started,
            )
        )
        if champion_score >= score_ceiling:
            stopped_early = True
        patience_reached = (
            config.early_stop_patience is not None
            and generations_without_improvement >= config.early_stop_patience
        )
        if patience_reached:
            stopped_early = True
        save_checkpoint(
            config,
            champion,
            history=history,
            benchmark_count=len(evaluation_pool),
            max_score=score_ceiling,
            stopped_early=stopped_early,
        )
        if stopped_early:
            if champion_score >= score_ceiling:
                print(
                    f"Early stopping at generation {generation}: "
                    f"reached max score {score_ceiling:.3f}."
                )
            else:
                print(
                    f"Early stopping at generation {generation}: "
                    f"no improvement of at least {config.early_stop_min_delta:.3f} "
                    f"for {generations_without_improvement} generations."
                )
            break

    return TrainingResult(
        champion=champion,
        history=history,
        benchmark_count=len(evaluation_pool),
        max_score=score_ceiling,
        stopped_early=stopped_early,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Grids heuristic bot with evolutionary self-play.")
    parser.add_argument("--generations", type=int, default=20)
    parser.add_argument("--population", type=int, default=10)
    parser.add_argument("--games", type=int, default=4, help="Games per candidate evaluation.")
    parser.add_argument("--mutation-scale", type=float, default=1.25)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--map", default="plains", choices=["plains", "desert"])
    parser.add_argument("--champion-pool-size", type=int, default=5)
    parser.add_argument("--checkpoint-prefix", help="Optional prefix for periodic checkpoint files.")
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=5,
        help="Save numbered checkpoint snapshots every N generations when checkpointing is enabled.",
    )
    parser.add_argument(
        "--verbose-training",
        action="store_true",
        help="Print sampled match summaries and final boards during training.",
    )
    parser.add_argument(
        "--verbose-every",
        type=int,
        default=25,
        help="When verbose mode is enabled, print every Nth evaluated match.",
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        help="Stop after this many generations without meaningful improvement.",
    )
    parser.add_argument(
        "--early-stop-min-delta",
        type=float,
        default=0.01,
        help="Minimum champion score improvement required to reset patience.",
    )
    parser.add_argument(
        "--promotion-margin",
        type=float,
        default=0.05,
        help="Minimum paired-seed score gain required before a candidate can replace the champion.",
    )
    parser.add_argument(
        "--holdout-games",
        type=int,
        default=1,
        help="Extra held-out games used to confirm a candidate promotion. Use 0 to disable.",
    )
    parser.add_argument(
        "--no-fixed-benchmarks",
        action="store_true",
        help="Evaluate only against the rolling champion pool and random bot.",
    )
    parser.add_argument(
        "--ai-search-width",
        type=int,
        default=1,
        help="Beam width used by heuristic bots during training. 1 keeps greedy training speed.",
    )
    parser.add_argument(
        "--ai-search-depth",
        type=int,
        help="Maximum number of same-turn actions explored by training bots.",
    )
    parser.add_argument("--resume-from", help="Optional JSON weights file to continue training from.")
    parser.add_argument("--output", default="trained_weights.json")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.champion_pool_size < 1:
        raise SystemExit("--champion-pool-size must be at least 1.")
    if args.checkpoint_every < 1:
        raise SystemExit("--checkpoint-every must be at least 1.")
    if args.verbose_every < 1:
        raise SystemExit("--verbose-every must be at least 1.")
    if args.early_stop_patience is not None and args.early_stop_patience < 1:
        raise SystemExit("--early-stop-patience must be at least 1.")
    if args.early_stop_min_delta < 0.0:
        raise SystemExit("--early-stop-min-delta must be non-negative.")
    if args.promotion_margin < 0.0:
        raise SystemExit("--promotion-margin must be non-negative.")
    if args.holdout_games < 0:
        raise SystemExit("--holdout-games must be non-negative.")
    if args.ai_search_width < 1:
        raise SystemExit("--ai-search-width must be at least 1.")
    if args.ai_search_depth is not None and args.ai_search_depth < 1:
        raise SystemExit("--ai-search-depth must be at least 1.")
    starting_weights = load_weights(args.resume_from) if args.resume_from else None
    config = TrainingConfig(
        generations=args.generations,
        population=args.population,
        games_per_candidate=args.games,
        mutation_scale=args.mutation_scale,
        seed=args.seed,
        map_name=args.map,
        champion_pool_size=args.champion_pool_size,
        checkpoint_prefix=args.checkpoint_prefix,
        checkpoint_every=args.checkpoint_every,
        resume_from_path=args.resume_from,
        starting_weights=starting_weights,
        verbose_training=args.verbose_training,
        verbose_every=args.verbose_every,
        early_stop_patience=args.early_stop_patience,
        early_stop_min_delta=args.early_stop_min_delta,
        promotion_margin=args.promotion_margin,
        holdout_games=args.holdout_games,
        use_fixed_benchmarks=not args.no_fixed_benchmarks,
        ai_search_width=args.ai_search_width,
        ai_search_depth=args.ai_search_depth,
    )
    print("Training started.")
    result = train(config)
    save_weights(
        args.output,
        result.champion,
        metadata=build_training_metadata(
            config,
            history=result.history,
            benchmark_count=result.benchmark_count,
            max_score=result.max_score,
            stopped_early=result.stopped_early,
            checkpoint_kind="final",
        ),
    )
    print(f"Saved trained weights to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
