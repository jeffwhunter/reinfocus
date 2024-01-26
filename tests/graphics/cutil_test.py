"""Contains tests for reinfocus.graphics.cutil."""

import numpy

from numba import cuda
from numba.cuda import testing
from numba.cuda.testing import unittest

from reinfocus.graphics import cutil
from tests import test_utils
from tests.graphics import numba_test_utils


class CutilTest(testing.CUDATestCase):
    """TestCases for reinfocus.graphics.cutil."""

    def test_launcher(self):
        """Tests that launcher creates a launchable cuda function."""

        @cuda.jit
        def return_ones(target):
            i = cutil.line_index()
            if i < target.size:
                target[i] = 1.0

        tests = 100

        cpu_array = numba_test_utils.cpu_target(ndim=1, nrow=tests)

        cutil.launcher(return_ones, (tests, 1))(cpu_array)

        test_utils.arrays_close(self, cpu_array[:, 0], numpy.ones(tests))

    def test_line_index(self):
        """Tests that line_index returns the proper thread index."""

        @cuda.jit
        def return_index(target):
            i = cutil.line_index()
            if i < target.size:
                target[i] = i

        tests = 100

        cpu_array = numba_test_utils.cpu_target(ndim=1, nrow=tests)

        cutil.launcher(return_index, (tests, 1))(cpu_array)

        test_utils.arrays_close(self, cpu_array[:, 0], range(tests))


if __name__ == "__main__":
    unittest.main()
