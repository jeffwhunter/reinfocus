"""Contains tests for reinfocus.graphics.random."""

import numpy

from numba import cuda
from numba.cuda import testing
from numba.cuda.testing import unittest

from reinfocus.graphics import cutil
from reinfocus.graphics import random


class MakeRandomStatesTest(testing.CUDATestCase):
    """Test cases for reinfocus.graphics.random.make_random_states."""

    def test_length(self):
        """Tests that make_random_states makes the correct number of states."""

        n_states = 10

        states = random.make_random_states(n_states, 0).copy_to_host()

        self.assertEqual(len(states), n_states)


class UniformFloatTest(testing.CUDATestCase):
    """Test cases for reinfocus.graphics.random.uniform_float."""

    def test_range(self):
        """Tests that uniform_float samples from [0.0, 1.0)."""

        @cuda.jit
        def sample(target, random_states):
            i = cutil.line_index()
            if cutil.outside_shape(i, target.shape):
                return

            target[i] = random.uniform_float(random_states, i)

        tests = 100

        cpu_array = numpy.zeros(tests, dtype=numpy.float32)

        cutil.launcher(sample, tests)(cpu_array, random.make_random_states(tests, 0))

        self.assertTrue(numpy.all(numpy.logical_and(0.0 <= cpu_array, cpu_array < 1.0)))


if __name__ == "__main__":
    unittest.main()
