"""Contains tests for reinfocus.environments.state_initializer."""

import unittest

from numpy import testing

from reinfocus.environments import state_initializer


class RangedInitializerTest(unittest.TestCase):
    """Test cases for reinfocus.environments.state_initializer.RangedInitializer."""

    def test_ranged_state_initializer(self):
        """Tests that RangedInitializer samples initial states from between the given
        ranges."""

        initial_states = state_initializer.RangedInitializer(
            [[(0.0, 0.2), (0.8, 1.0)], [(0.2, 0.4), (0.6, 0.8)], [(0.4, 0.6)]]
        ).initialize(2)

        self.assertEqual(initial_states.shape, (2, 3))

        testing.assert_raises(AssertionError, testing.assert_allclose, *initial_states)

        for initial_state in initial_states:
            self.assertTrue(
                0.0 <= initial_state[0] <= 0.2 or 0.8 <= initial_state[0] <= 1.0
            )
            self.assertTrue(
                0.2 <= initial_state[1] <= 0.4 or 0.6 <= initial_state[1] <= 0.8
            )
            self.assertTrue(0.4 <= initial_state[2] <= 0.6)


if __name__ == "__main__":
    unittest.main()
