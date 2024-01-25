"""Contains tests for reinfocus.learning.dynamics."""

import unittest

import numpy

from reinfocus.environment import dynamics
from tests import test_utils


class DynamicsTest(unittest.TestCase):
    """TestCases for reinfocus.learning.dynamics."""

    def test_continuous_dynamics(self):
        """Tests that ContinuousDynamics mvoes the state with the expected speed."""

        fast_dynamics = dynamics.make_continuous_dynamics((0, 1), 1)

        state = numpy.array([1, 0.5])
        test_utils.arrays_close(self, fast_dynamics(state, numpy.float32(1.0)), [1, 1])
        test_utils.arrays_close(self, fast_dynamics(state, numpy.float32(0.75)), [1, 1])
        test_utils.arrays_close(self, fast_dynamics(state, numpy.float32(0.5)), [1, 1])
        test_utils.arrays_close(
            self, fast_dynamics(state, numpy.float32(0.25)), [1, 0.75]
        )
        test_utils.arrays_close(self, fast_dynamics(state, numpy.float32(0.1)), [1, 0.6])
        test_utils.arrays_close(self, fast_dynamics(state, numpy.float32(0)), [1, 0.5])
        test_utils.arrays_close(self, fast_dynamics(state, numpy.float32(-0.1)), [1, 0.4])
        test_utils.arrays_close(
            self, fast_dynamics(state, numpy.float32(-0.25)), [1, 0.25]
        )
        test_utils.arrays_close(self, fast_dynamics(state, numpy.float32(-0.5)), [1, 0])
        test_utils.arrays_close(self, fast_dynamics(state, numpy.float32(-0.75)), [1, 0])
        test_utils.arrays_close(self, fast_dynamics(state, numpy.float32(-1)), [1, 0])

        state = numpy.array([1, 1])
        test_utils.arrays_close(self, fast_dynamics(state, numpy.float32(1)), [1, 1])
        test_utils.arrays_close(self, fast_dynamics(state, numpy.float32(0)), [1, 1])
        test_utils.arrays_close(self, fast_dynamics(state, numpy.float32(-0.5)), [1, 0.5])
        test_utils.arrays_close(self, fast_dynamics(state, numpy.float32(-1)), [1, 0])

        state = numpy.array([1, 0])
        test_utils.arrays_close(self, fast_dynamics(state, numpy.float32(1)), [1, 1])
        test_utils.arrays_close(self, fast_dynamics(state, numpy.float32(0.5)), [1, 0.5])
        test_utils.arrays_close(self, fast_dynamics(state, numpy.float32(0)), [1, 0])
        test_utils.arrays_close(self, fast_dynamics(state, numpy.float32(-1)), [1, 0])

        slow_dynamics = dynamics.make_continuous_dynamics((0, 1), 0.1)

        state = numpy.array([1, 0.5])
        test_utils.arrays_close(self, slow_dynamics(state, numpy.float32(2)), [1, 0.6])
        test_utils.arrays_close(self, slow_dynamics(state, numpy.float32(1)), [1, 0.6])
        test_utils.arrays_close(self, slow_dynamics(state, numpy.float32(0.1)), [1, 0.51])
        test_utils.arrays_close(self, slow_dynamics(state, numpy.float32(0)), [1, 0.5])
        test_utils.arrays_close(
            self, slow_dynamics(state, numpy.float32(-0.1)), [1, 0.49]
        )
        test_utils.arrays_close(self, slow_dynamics(state, numpy.float32(-1)), [1, 0.4])
        test_utils.arrays_close(self, slow_dynamics(state, numpy.float32(-2)), [1, 0.4])

    def test_discrete_dynamics(self):
        """Tests that ContinuousDynamics mvoes the state with the expected actions."""

        discrete_dynamics = dynamics.make_discrete_dynamics(
            (0, 1), [-1.0, -0.5, -0.1, 0, 0.1, 0.5, 1]
        )

        state = numpy.array([1, 0.5])
        test_utils.arrays_close(self, discrete_dynamics(state, numpy.int32(0)), [1, 0.0])
        test_utils.arrays_close(self, discrete_dynamics(state, numpy.int32(1)), [1, 0.0])
        test_utils.arrays_close(self, discrete_dynamics(state, numpy.int32(2)), [1, 0.4])
        test_utils.arrays_close(self, discrete_dynamics(state, numpy.int32(3)), [1, 0.5])
        test_utils.arrays_close(self, discrete_dynamics(state, numpy.int32(4)), [1, 0.6])
        test_utils.arrays_close(self, discrete_dynamics(state, numpy.int32(5)), [1, 1.0])
        test_utils.arrays_close(self, discrete_dynamics(state, numpy.int32(6)), [1, 1.0])

        state = numpy.array([1, 0.0])
        test_utils.arrays_close(self, discrete_dynamics(state, numpy.int32(0)), [1, 0.0])
        test_utils.arrays_close(self, discrete_dynamics(state, numpy.int32(1)), [1, 0.0])
        test_utils.arrays_close(self, discrete_dynamics(state, numpy.int32(2)), [1, 0.0])
        test_utils.arrays_close(self, discrete_dynamics(state, numpy.int32(3)), [1, 0.0])
        test_utils.arrays_close(self, discrete_dynamics(state, numpy.int32(4)), [1, 0.1])
        test_utils.arrays_close(self, discrete_dynamics(state, numpy.int32(5)), [1, 0.5])
        test_utils.arrays_close(self, discrete_dynamics(state, numpy.int32(6)), [1, 1.0])

        state = numpy.array([1, 1.0])
        test_utils.arrays_close(self, discrete_dynamics(state, numpy.int32(0)), [1, 0.0])
        test_utils.arrays_close(self, discrete_dynamics(state, numpy.int32(1)), [1, 0.5])
        test_utils.arrays_close(self, discrete_dynamics(state, numpy.int32(2)), [1, 0.9])
        test_utils.arrays_close(self, discrete_dynamics(state, numpy.int32(3)), [1, 1.0])
        test_utils.arrays_close(self, discrete_dynamics(state, numpy.int32(4)), [1, 1.0])
        test_utils.arrays_close(self, discrete_dynamics(state, numpy.int32(5)), [1, 1.0])
        test_utils.arrays_close(self, discrete_dynamics(state, numpy.int32(6)), [1, 1.0])


if __name__ == "__main__":
    unittest.main()
