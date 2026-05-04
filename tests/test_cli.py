from __future__ import annotations

import unittest

from grids_ai.cli import build_bot, parse_args


class CliTests(unittest.TestCase):
    def test_parse_args_accepts_delay(self) -> None:
        args = parse_args(["--blue", "heuristic", "--red", "random", "--delay", "0.25"])
        self.assertEqual(args.blue, "heuristic")
        self.assertEqual(args.red, "random")
        self.assertEqual(args.delay, 0.25)

    def test_build_bot_returns_none_for_human(self) -> None:
        self.assertIsNone(build_bot("human"))


if __name__ == "__main__":
    unittest.main()
