"""Contains tests for reinfocus.environment.state_initializer."""

import unittest

import numpy

from reinfocus.environment import state_initializer
from tests import test_utils


class StateInitializerTest(unittest.TestCase):
    """TestCases for reinfocus.environment.state_initializer."""

    def test_make_uniform_initializer(self):
        """Tests that make_uniform_initializer creates a state initializer that samples
        initial states from between given limits."""

        n_tests = 100

        initial_state = state_initializer.uniform(-1.0, 1.0, n_tests)()

        test_utils.all_close(
            (-1.0 <= initial_state) & (initial_state <= 1.0),
            numpy.full(n_tests, True),
        )

    def test_make_ranged_initializer(self):
        """Tests that make_ranged_initializer creates a state initializer that samples
        initial states from between the listed ranges."""

        initial_states = state_initializer.ranged(
            [[(0.0, 0.2), (0.8, 1.0)], [(0.2, 0.4), (0.6, 0.8)], [(0.4, 0.6)]]
        )()

        self.assertTrue(
            0.0 <= initial_states[0] <= 0.2 or 0.8 <= initial_states[0] <= 1.0
        )
        self.assertTrue(
            0.2 <= initial_states[1] <= 0.4 or 0.6 <= initial_states[1] <= 0.8
        )
        self.assertTrue(0.4 <= initial_states[2] <= 0.6)


if __name__ == "__main__":
    unittest.main()
