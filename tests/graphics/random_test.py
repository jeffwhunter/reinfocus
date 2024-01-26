"""Contains tests for reinfocus.graphics.random."""

import numpy

from numba import cuda
from numba.cuda import testing
from numba.cuda.testing import unittest

from reinfocus.graphics import cutil
from reinfocus.graphics import random
from tests.graphics import numba_test_utils


class RandomTest(testing.CUDATestCase):
    """TestCases for reinfocus.graphics.random."""

    def test_make_random_states(self):
        """Tests that make_random_states makes the correct number of states."""

        n_states = 10

        states = random.make_random_states(n_states, 0).copy_to_host()

        self.assertEqual(len(states), n_states)

    def test_uniform_float(self):
        """Tests that uniform_float samples from [0.0, 1.0)."""

        @cuda.jit
        def sample(target, random_states):
            i = cutil.line_index()
            if i < target.size:
                target[i] = random.uniform_float(random_states, i)

        tests = 100

        cpu_array = numba_test_utils.cpu_target(ndim=1, nrow=tests)

        cutil.launcher(sample, (tests, 1))(cpu_array, random.make_random_states(tests, 0))

        self.assertTrue(numpy.all(numpy.logical_and(0.0 <= cpu_array, cpu_array < 1.0)))


if __name__ == "__main__":
    unittest.main()
