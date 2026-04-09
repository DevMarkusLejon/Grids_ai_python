from __future__ import annotations

from dataclasses import dataclass
import argparse
import random
import sys
from typing import Sequence

from .bots import DEFAULT_WEIGHTS, HeuristicBot, RandomBot, save_weights
from .engine import GameState, new_game


@dataclass(frozen=True)
class TrainingConfig:
    generations: int = 20
    population: int = 10
    games_per_candidate: int = 4
    mutation_scale: float = 1.25
    seed: int = 7
    map_name: str = "plains"


def mutated_weights(
    rng: random.Random,
    base_weights: dict[str, float],
    mutation_scale: float,
) -> dict[str, float]:
    mutated: dict[str, float] = {}
    for key, value in base_weights.items():
        mutated[key] = value + rng.gauss(0.0, mutation_scale)
    return mutated


def play_match(
    blue_bot,
    red_bot,
    seed: int,
    map_name: str = "plains",
) -> GameState:
    state = new_game(seed=seed, map_name=map_name)
    turn_safety = state.config.max_half_turns * 8
    steps = 0

    while not state.is_done and steps < turn_safety:
        bot = blue_bot if state.current_side.value == "blue" else red_bot
        action = bot.choose_action(state)
        state.apply(action)
        steps += 1

    if not state.is_done:
        state._resolve_timeout_winner()
    return state


def match_score(state: GameState) -> float:
    if state.winner is None:
        return 0.0
    return 1.0 if state.winner.value == "blue" else -1.0


def evaluate_candidate(
    candidate_weights: dict[str, float],
    benchmark_weights: dict[str, float],
    config: TrainingConfig,
    seed_offset: int,
) -> float:
    total = 0.0
    candidate_bot = HeuristicBot(candidate_weights, seed=seed_offset)
    benchmark_bot = HeuristicBot(benchmark_weights, seed=seed_offset + 1)
    random_bot = RandomBot(seed=seed_offset + 2)

    for game_index in range(config.games_per_candidate):
        seed = config.seed + seed_offset * 100 + game_index

        blue_vs_bench = play_match(candidate_bot, benchmark_bot, seed=seed, map_name=config.map_name)
        total += match_score(blue_vs_bench)

        red_vs_bench = play_match(benchmark_bot, candidate_bot, seed=seed + 1000, map_name=config.map_name)
        total -= match_score(red_vs_bench)

        blue_vs_random = play_match(candidate_bot, random_bot, seed=seed + 2000, map_name=config.map_name)
        total += match_score(blue_vs_random) * 0.5

        red_vs_random = play_match(random_bot, candidate_bot, seed=seed + 3000, map_name=config.map_name)
        total -= match_score(red_vs_random) * 0.5

    return total


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


def train(config: TrainingConfig) -> tuple[dict[str, float], list[tuple[int, float]]]:
    rng = random.Random(config.seed)
    champion = dict(DEFAULT_WEIGHTS)
    history: list[tuple[int, float]] = []
    champion_score = evaluate_candidate(champion, dict(DEFAULT_WEIGHTS), config, seed_offset=0)

    for generation in range(1, config.generations + 1):
        best_candidate = champion
        best_score = champion_score

        render_progress(0, config.population, generation=generation, generations=config.generations)
        for candidate_index in range(config.population):
            seed_offset = generation * 1000 + candidate_index
            candidate = mutated_weights(rng, champion, config.mutation_scale)
            score = evaluate_candidate(candidate, champion, config, seed_offset)
            if score > best_score:
                best_candidate = candidate
                best_score = score
            render_progress(
                candidate_index + 1,
                config.population,
                generation=generation,
                generations=config.generations,
            )

        champion = best_candidate
        champion_score = best_score
        history.append((generation, champion_score))
        print(f"Generation {generation:>2}: champion score {champion_score:.3f}")

    return champion, history


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Grids heuristic bot with evolutionary self-play.")
    parser.add_argument("--generations", type=int, default=20)
    parser.add_argument("--population", type=int, default=10)
    parser.add_argument("--games", type=int, default=4, help="Games per candidate evaluation.")
    parser.add_argument("--mutation-scale", type=float, default=1.25)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--map", default="plains", choices=["plains", "desert"])
    parser.add_argument("--output", default="trained_weights.json")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = TrainingConfig(
        generations=args.generations,
        population=args.population,
        games_per_candidate=args.games,
        mutation_scale=args.mutation_scale,
        seed=args.seed,
        map_name=args.map,
    )
    champion, history = train(config)
    save_weights(
        args.output,
        champion,
        metadata={
            "generations": config.generations,
            "population": config.population,
            "games_per_candidate": config.games_per_candidate,
            "mutation_scale": config.mutation_scale,
            "seed": config.seed,
            "map": config.map_name,
            "history": history,
        },
    )
    print(f"Saved trained weights to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
