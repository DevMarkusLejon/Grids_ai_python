from __future__ import annotations

import unittest

from grids_ai.encoding import action_space_for_state, encode_action_index, encode_state_planes, encode_state_vector
from grids_ai.engine import new_game


class EncodingTests(unittest.TestCase):
    def test_state_vector_and_planes_have_expected_shape(self) -> None:
        state = new_game(seed=1)

        vector = encode_state_vector(state)
        planes = encode_state_planes(state)

        self.assertEqual(len(vector), 13 * state.map_def.width * state.map_def.height + 15)
        self.assertEqual(len(planes), 13)
        self.assertEqual(len(planes[0]), state.map_def.height)
        self.assertEqual(len(planes[0][0]), state.map_def.width)

    def test_legal_action_indices_are_in_action_space(self) -> None:
        state = new_game(seed=2)
        space = action_space_for_state(state)
        indices = [encode_action_index(state, action) for action in state.legal_actions()]

        self.assertEqual(len(indices), len(set(indices)))
        self.assertTrue(all(0 <= index < space.size for index in indices))


if __name__ == "__main__":
    unittest.main()
