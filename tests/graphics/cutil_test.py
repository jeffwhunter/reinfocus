"""Contains tests for reinfocus.graphics.cutil."""

import numpy

from numba import cuda
from numba.cuda import testing as cuda_testing
from numba.cuda.testing import unittest
from numpy import testing as numpy_testing

from reinfocus.graphics import cutil


class EnoughBlocksTest(unittest.TestCase):
    """Test cases for reinfocus.graphics.cutil.enough_blocks."""

    def test_enough_blocks(self):
        """Tests that enough_blocks properly divides a space."""

        line_blocks = cutil.enough_blocks(100, 8)
        self.assertIsInstance(line_blocks, int)
        self.assertEqual(line_blocks, 13)

        grid_blocks = cutil.enough_blocks((10, 20), (2, 8))
        if not isinstance(grid_blocks, tuple):
            self.fail()
        self.assertTupleEqual(grid_blocks, (5, 3))

        cube_blocks = cutil.enough_blocks((9, 27, 54), (2, 5, 10))
        if not isinstance(cube_blocks, tuple):
            self.fail()
        self.assertTupleEqual(cube_blocks, (5, 6, 6))


class ConstantLikeTest(unittest.TestCase):
    """Test cases for reinfocus.graphis.cutil.constant_like."""

    def test_shape(self):
        """Tests that constant_like produces constant filled copies of it's input."""

        self.assertEqual(cutil.constant_like(0, 1), 0)
        self.assertEqual(cutil.constant_like(0, (1, 1)), (0, 0))
        self.assertEqual(cutil.constant_like(0, (1, 1, 1)), (0, 0, 0))

    def test_limit(self):
        """Tests that constant_like never returns values larger than the given shape."""

        self.assertEqual(cutil.constant_like(4, 20), 4)
        self.assertEqual(cutil.constant_like(20, (1, 10)), (1, 10))
        self.assertEqual(cutil.constant_like(9, (1, 3, 9)), (1, 3, 9))


class LimitBlockSizeTest(unittest.TestCase):
    """Test cases for reinfocus.graphics.cutil.limit_block_size."""

    def test_limit_block_size(self):
        """Tests that limit_block_size always produces block sizes whose product is
        smaller than the maximum allocable."""

        self.assertLessEqual(cutil.limit_block_size(10000), cutil.CUDA_MAX_BLOCK_SIZE)
        self.assertLessEqual(
            numpy.prod(cutil.limit_block_size((64, 64))), cutil.CUDA_MAX_BLOCK_SIZE
        )
        self.assertLessEqual(
            numpy.prod(cutil.limit_block_size((16, 16, 16))), cutil.CUDA_MAX_BLOCK_SIZE
        )


class LauncherTest(cuda_testing.CUDATestCase):
    """Test cases for reinfocus.graphics.cutil.launcher."""

    def test_line(self):
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

        numpy_testing.assert_allclose(cpu_array, numpy.ones(tests))

    def test_grid(self):
        """Tests that launcher creates a 2-D launchable cuda function."""

        @cuda.jit
        def return_ones_grid(target):
            i = cutil.grid_index()
            if cutil.outside_shape(i, target.shape):
                return

            target[i] = 1.0

        tests = (32, 64)

        cpu_array = numpy.zeros(tests, dtype=numpy.float32)

        cutil.launcher(return_ones_grid, tests)(cpu_array)

        numpy_testing.assert_allclose(cpu_array, numpy.ones(tests))

    def test_cube(self):
        """Tests that launcher creates a 3-D launchable cuda function."""

        @cuda.jit
        def return_ones_cube(target):
            i = cutil.cube_index()
            if cutil.outside_shape(i, target.shape):
                return

            target[i] = 1.0

        tests = (8, 16, 32)

        cpu_array = numpy.zeros(tests, dtype=numpy.float32)

        cutil.launcher(return_ones_cube, tests)(cpu_array)

        numpy_testing.assert_allclose(cpu_array, numpy.ones(tests))


class OutsideShapeTest(cuda_testing.CUDATestCase):
    """Test cases for reinfocus.graphics.cutil.outside_shape."""

    def test_outside_line(self):
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

        numpy_testing.assert_allclose(cpu_array, target)

    def test_outside_grid(self):
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

        numpy_testing.assert_allclose(cpu_array, target)

    def test_outside_cube(self):
        """Tests that outside_shape returns if an index is outside three ranges."""

        @cuda.jit
        def toss_most_cube_indices(target):
            i = cutil.cube_index()
            if cutil.outside_shape(i, (2, 3, 4)):
                return

            target[i] = 1.0

        tests = (8, 16, 32)

        cpu_array = numpy.zeros(tests, dtype=numpy.float32)

        cutil.launcher(toss_most_cube_indices, tests)(cpu_array)

        target = numpy.zeros(tests)
        target[:2, :3, :4] = numpy.ones((2, 3, 4))

        numpy_testing.assert_allclose(cpu_array, target)


class IndexTest(cuda_testing.CUDATestCase):
    """Test cases for reinfocus.graphics.cutil.[line/grid/cube]_index."""

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

        numpy_testing.assert_allclose(cpu_array, range(tests))

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

        numpy_testing.assert_allclose(
            cpu_array, numpy.moveaxis(numpy.indices(tests), 0, -1)
        )

    def test_cube_index(self):
        """Tests that cube_index returns the proper thread index."""

        @cuda.jit
        def return_cube_index(target):
            i = cutil.cube_index()
            if cutil.outside_shape(i, target.shape):
                return

            target[i] = i

        tests = (8, 16, 32)

        cpu_array = numpy.zeros(tests + (3,), dtype=numpy.float32)

        cutil.launcher(return_cube_index, tests)(cpu_array)

        numpy_testing.assert_allclose(
            cpu_array, numpy.moveaxis(numpy.indices(tests), 0, -1)
        )


if __name__ == "__main__":
    unittest.main()
