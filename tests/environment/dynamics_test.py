"""Contains tests for reinfocus.environment.dynamics."""

import unittest

import numpy

from reinfocus.environment import dynamics
from tests import test_utils


class DynamicsTest(unittest.TestCase):
    """TestCases for reinfocus.environment.dynamics."""

    def test_continuous_update(self):
        """Tests that continuous_update makes a function that returns the expected
        change given some speed."""

        fast_update = dynamics.continuous_update(2.0)
        test_utils.all_close(fast_update(2.0), [0.0, 2.0])
        test_utils.all_close(fast_update(-2.0), [0.0, -2.0])
        test_utils.all_close(fast_update(1.0), [0.0, 2.0])
        test_utils.all_close(fast_update(-1.0), [0.0, -2.0])
        test_utils.all_close(fast_update(0.1), [0.0, 0.2])
        test_utils.all_close(fast_update(-0.1), [0.0, -0.2])
        test_utils.all_close(fast_update(0.0), [0.0, 0.0])

        slow_update = dynamics.continuous_update(0.1)
        test_utils.all_close(slow_update(2.0), [0.0, 0.1])
        test_utils.all_close(slow_update(-2.0), [0.0, -0.1])
        test_utils.all_close(slow_update(1.0), [0.0, 0.1])
        test_utils.all_close(slow_update(-1.0), [0.0, -0.1])
        test_utils.all_close(slow_update(0.1), [0.0, 0.01])
        test_utils.all_close(slow_update(-0.1), [0.0, -0.01])
        test_utils.all_close(slow_update(0.0), [0.0, 0.0])

    def test_continuous_dynamics(self):
        """Tests that continuous makes a system of dynamics that moves the state with the
        expected speed."""

        fast_dynamics = dynamics.continuous((0, 1), 1)

        state = numpy.array([1, 0.5])
        test_utils.all_close(fast_dynamics(state, 1.0), [1, 1])
        test_utils.all_close(fast_dynamics(state, 0.75), [1, 1])
        test_utils.all_close(fast_dynamics(state, 0.5), [1, 1])
        test_utils.all_close(fast_dynamics(state, 0.25), [1, 0.75])
        test_utils.all_close(fast_dynamics(state, 0.1), [1, 0.6])
        test_utils.all_close(fast_dynamics(state, 0), [1, 0.5])
        test_utils.all_close(fast_dynamics(state, -0.1), [1, 0.4])
        test_utils.all_close(fast_dynamics(state, -0.25), [1, 0.25])
        test_utils.all_close(fast_dynamics(state, -0.5), [1, 0])
        test_utils.all_close(fast_dynamics(state, -0.75), [1, 0])
        test_utils.all_close(fast_dynamics(state, -1), [1, 0])

        state = numpy.array([1, 1])
        test_utils.all_close(fast_dynamics(state, 1), [1, 1])
        test_utils.all_close(fast_dynamics(state, 0), [1, 1])
        test_utils.all_close(fast_dynamics(state, -0.5), [1, 0.5])
        test_utils.all_close(fast_dynamics(state, -1), [1, 0])

        state = numpy.array([1, 0])
        test_utils.all_close(fast_dynamics(state, 1), [1, 1])
        test_utils.all_close(fast_dynamics(state, 0.5), [1, 0.5])
        test_utils.all_close(fast_dynamics(state, 0), [1, 0])
        test_utils.all_close(fast_dynamics(state, -1), [1, 0])

        slow_dynamics = dynamics.continuous((0, 1), 0.1)

        state = numpy.array([1, 0.5])
        test_utils.all_close(slow_dynamics(state, 2), [1, 0.6])
        test_utils.all_close(slow_dynamics(state, 1), [1, 0.6])
        test_utils.all_close(slow_dynamics(state, 0.1), [1, 0.51])
        test_utils.all_close(slow_dynamics(state, 0), [1, 0.5])
        test_utils.all_close(slow_dynamics(state, -0.1), [1, 0.49])
        test_utils.all_close(slow_dynamics(state, -1), [1, 0.4])
        test_utils.all_close(slow_dynamics(state, -2), [1, 0.4])

    def test_discrete_update(self):
        """Tests that discrete_update makes a function that returns the expected change
        given some set of actions."""

        moves = [0.0, 1.0, -1.0, 10.0, -10.0]
        update = dynamics.discrete_update(moves)

        for i, move in enumerate(moves):
            test_utils.all_close(update(i), [0.0, move])

    def test_discrete_dynamics(self):
        """Tests that discrete makes a system of dynamics that moves the state with the
        expected steps."""

        discrete_dynamics = dynamics.discrete((0, 1), [-1.0, -0.5, -0.1, 0, 0.1, 0.5, 1])

        state = numpy.array([1, 0.5])
        test_utils.all_close(discrete_dynamics(state, 0), [1, 0.0])
        test_utils.all_close(discrete_dynamics(state, 1), [1, 0.0])
        test_utils.all_close(discrete_dynamics(state, 2), [1, 0.4])
        test_utils.all_close(discrete_dynamics(state, 3), [1, 0.5])
        test_utils.all_close(discrete_dynamics(state, 4), [1, 0.6])
        test_utils.all_close(discrete_dynamics(state, 5), [1, 1.0])
        test_utils.all_close(discrete_dynamics(state, 6), [1, 1.0])

        state = numpy.array([1, 0.0])
        test_utils.all_close(discrete_dynamics(state, 0), [1, 0.0])
        test_utils.all_close(discrete_dynamics(state, 1), [1, 0.0])
        test_utils.all_close(discrete_dynamics(state, 2), [1, 0.0])
        test_utils.all_close(discrete_dynamics(state, 3), [1, 0.0])
        test_utils.all_close(discrete_dynamics(state, 4), [1, 0.1])
        test_utils.all_close(discrete_dynamics(state, 5), [1, 0.5])
        test_utils.all_close(discrete_dynamics(state, 6), [1, 1.0])

        state = numpy.array([1, 1.0])
        test_utils.all_close(discrete_dynamics(state, 0), [1, 0.0])
        test_utils.all_close(discrete_dynamics(state, 1), [1, 0.5])
        test_utils.all_close(discrete_dynamics(state, 2), [1, 0.9])
        test_utils.all_close(discrete_dynamics(state, 3), [1, 1.0])
        test_utils.all_close(discrete_dynamics(state, 4), [1, 1.0])
        test_utils.all_close(discrete_dynamics(state, 5), [1, 1.0])
        test_utils.all_close(discrete_dynamics(state, 6), [1, 1.0])


if __name__ == "__main__":
    unittest.main()
