"""Contains tests for reinfocus.graphics.cutil."""

import numpy

from numba import cuda
from numba.cuda import testing
from numba.cuda.testing import unittest

from reinfocus.graphics import cutil
from tests import test_utils


class CutilTest(testing.CUDATestCase):
    """TestCases for reinfocus.graphics.cutil."""

    def test_enough_blocks(self):
        """Tests that enough_blocks properly divides a space."""

        line_blocks = cutil.enough_blocks(100, 8)
        self.assertIsInstance(line_blocks, int)
        self.assertEqual(line_blocks, 13)

        grid_blocks = cutil.enough_blocks((10, 20), (2, 8))
        if not isinstance(grid_blocks, tuple):
            self.fail()
        self.assertTupleEqual(grid_blocks, (5, 3))

    def test_constant_like(self):
        """Tests that constant_like produces constant filled copies of it's input."""

        self.assertEqual(cutil.constant_like(4, 0), 4)

        double_constant = cutil.constant_like(20, (0, 0))
        if not isinstance(double_constant, tuple):
            self.fail()
        self.assertTupleEqual(double_constant, (20, 20))

    def test_launcher_line(self):
        """Tests that launcher creates a 1-D launchable cuda function."""

        @cuda.jit
        def return_ones_line(target):
            i = cutil.line_index()
            if cutil.outside_shape(i, target.shape):
                return

            target[i] = 1.0

        tests = 100

        cpu_array = numpy.zeros(tests, dtype=numpy.float32)

        cutil.launcher(return_ones_line, tests)(cpu_array)

        test_utils.arrays_close(self, cpu_array, numpy.ones(tests))

    def test_launcher_grid(self):
        """Tests that launcher creates a 2-D launchable cuda function."""

        @cuda.jit
        def return_ones_grid(target):
            i = cutil.grid_index()
            if cutil.outside_shape(i, target.shape):
                return

            target[*i] = 1.0

        tests = (32, 64)

        cpu_array = numpy.zeros(tests, dtype=numpy.float32)

        cutil.launcher(return_ones_grid, tests)(cpu_array)

        test_utils.arrays_close(self, cpu_array, numpy.ones(tests))

    def test_outside_line_shape(self):
        """Tests that outside_shape returns if an index is outside one range."""

        @cuda.jit
        def toss_most_line_indices(target):
            i = cutil.line_index()
            if cutil.outside_shape(i, (3,)):
                return

            target[i] = 1.0

        tests = 100

        cpu_array = numpy.zeros(tests, dtype=numpy.float32)

        cutil.launcher(toss_most_line_indices, tests)(cpu_array)

        target = numpy.zeros(tests)
        target[0:3] = numpy.ones(3)

        test_utils.arrays_close(self, cpu_array, target)

    def test_outside_grid_shape(self):
        """Tests that outside_shape returns if an index is outside two ranges."""

        @cuda.jit
        def toss_most_grid_indices(target):
            i = cutil.grid_index()
            if cutil.outside_shape(i, (2, 3)):
                return

            target[i] = 1.0

        tests = (32, 64)

        cpu_array = numpy.zeros(tests, dtype=numpy.float32)

        cutil.launcher(toss_most_grid_indices, tests)(cpu_array)

        target = numpy.zeros(tests)
        target[0:2, 0:3] = numpy.ones((2, 3))

        test_utils.arrays_close(self, cpu_array, target)

    def test_line_index(self):
        """Tests that line_index returns the proper thread index."""

        @cuda.jit
        def return_line_index(target):
            i = cutil.line_index()
            if cutil.outside_shape(i, target.shape):
                return

            target[i] = i

        tests = 100

        cpu_array = numpy.zeros(tests, dtype=numpy.float32)

        cutil.launcher(return_line_index, tests)(cpu_array)

        test_utils.arrays_close(self, cpu_array, range(tests))

    def test_grid_index(self):
        """Tests that grid_index returns the proper thread index."""

        @cuda.jit
        def return_grid_index(target):
            i = cutil.grid_index()
            if cutil.outside_shape(i, target.shape):
                return

            target[i] = i

        tests = (32, 64)

        cpu_array = numpy.zeros(tests + (2,), dtype=numpy.float32)

        cutil.launcher(return_grid_index, tests)(cpu_array)

        test_utils.arrays_close(
            self, cpu_array, numpy.moveaxis(numpy.indices(tests), 0, -1)
        )


if __name__ == "__main__":
    unittest.main()
