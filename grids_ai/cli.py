from __future__ import annotations

import argparse
from collections.abc import Sequence
import sys
import time

from .bots import DEFAULT_WEIGHTS, HeuristicBot, RandomBot, load_weights
from .engine import Action, GameState, new_game


def build_bot(name: str, weights_path: str | None = None):
    if name == "random":
        return RandomBot()
    if name == "heuristic":
        weights = load_weights(weights_path) if weights_path else dict(DEFAULT_WEIGHTS)
        return HeuristicBot(weights)
    return None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play the simplified Python version of Grids.")
    parser.add_argument("--blue", default="human", choices=["human", "heuristic", "random"])
    parser.add_argument("--red", default="heuristic", choices=["human", "heuristic", "random"])
    parser.add_argument("--map", default="plains", choices=["plains", "desert"])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--delay", type=float, default=0.0, help="Pause in seconds between bot actions.")
    parser.add_argument("--weights", help="Optional JSON file containing heuristic weights.")
    return parser.parse_args(argv)


def print_state(state: GameState) -> None:
    print()
    print(state.render())
    print()
    print(state.unit_summary())
    print()
    print(state.hand_summary())
    print()


def print_recent_log(state: GameState, lines: int = 8) -> None:
    print("Recent log:")
    for line in state.log[-lines:]:
        print(f"  {line}")
    print()


def clear_screen() -> None:
    sys.stdout.write("\033[2J\033[H")
    sys.stdout.flush()


def render_spectator_view(state: GameState, last_action: str | None = None) -> None:
    clear_screen()
    print(state.render())
    print()
    print(state.unit_summary())
    print()
    print_recent_log(state)
    if last_action is not None:
        print(f"Last action: {last_action}")
    else:
        print(f"{state.current_side.short} is about to act.")
    print()


def print_actions(state: GameState) -> list[Action]:
    legal = state.legal_actions()
    for index, action in enumerate(legal):
        print(f"{index:>2}: {state.describe_action(action)}")
    return legal


def play_bot_turn(state: GameState, bot, *, delay: float = 0.0, spectator_mode: bool = False) -> None:
    if spectator_mode:
        render_spectator_view(state)
    else:
        print(f"{state.current_side.short} is controlled by a bot.")

    while not state.is_done:
        action = bot.choose_action(state)
        action_text = state.describe_action(action)
        if not spectator_mode:
            print(f"  -> {action_text}")
        state.apply(action)
        if spectator_mode:
            render_spectator_view(state, last_action=action_text)
        if delay > 0:
            time.sleep(delay)
        if action.kind == "end_turn":
            break


def human_turn(state: GameState, controller_name: str) -> bool:
    print_state(state)
    print(f"{state.current_side.short} ({controller_name}) to act. Type 'help' for commands.")

    cached_actions: list[Action] | None = None
    while not state.is_done:
        command = input("> ").strip()
        if not command:
            continue

        if command in {"help", "h", "?"}:
            print("Commands:")
            print("  show         display the board, units, and current hand")
            print("  actions      list legal actions for the current player")
            print("  do <index>   apply a numbered legal action")
            print("  auto         let the heuristic bot play the rest of this turn")
            print("  log          print the recent action log")
            print("  quit         exit immediately")
            continue

        if command == "show":
            print_state(state)
            continue

        if command == "actions":
            cached_actions = print_actions(state)
            continue

        if command.startswith("do "):
            if cached_actions is None:
                cached_actions = print_actions(state)
            try:
                _, raw_index = command.split(maxsplit=1)
                index = int(raw_index)
                action = cached_actions[index]
            except (ValueError, IndexError):
                print("That action index is not valid.")
                continue
            print(f"Applying: {state.describe_action(action)}")
            state.apply(action)
            cached_actions = None
            if action.kind == "end_turn" or state.is_done:
                return True
            print_state(state)
            continue

        if command == "auto":
            play_bot_turn(state, HeuristicBot(dict(DEFAULT_WEIGHTS)))
            return True

        if command == "log":
            for line in state.log[-12:]:
                print(line)
            continue

        if command == "quit":
            return False

        print("Unknown command. Type 'help' to see the available commands.")

    return True


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.delay < 0:
        raise SystemExit("--delay must be at least 0.")
    state = new_game(seed=args.seed, map_name=args.map)
    blue_bot = build_bot(args.blue, args.weights)
    red_bot = build_bot(args.red, args.weights)
    controllers = {"blue": blue_bot, "red": red_bot}
    spectator_mode = blue_bot is not None and red_bot is not None

    print("Starting match.")
    while not state.is_done:
        controller = controllers[state.current_side.value]
        if controller is None:
            keep_playing = human_turn(state, "human")
            if not keep_playing:
                print("Exiting match.")
                return 0
        else:
            if not spectator_mode:
                print_state(state)
            play_bot_turn(state, controller, delay=args.delay, spectator_mode=spectator_mode)

    if spectator_mode:
        render_spectator_view(state)
    else:
        print_state(state)
    print("Game over.")
    print(f"Winner: {state.winner.value} ({state.winner_reason})")
    print("Recent log:")
    for line in state.log[-12:]:
        print(f"  {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
