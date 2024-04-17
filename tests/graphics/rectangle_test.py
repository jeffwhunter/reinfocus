"""Contains tests for reinfocus.graphics.rectangle."""

import numpy

from numba import cuda
from numba.cuda import testing as cuda_testing
from numba.cuda.testing import unittest
from numpy import testing as numpy_testing

from reinfocus.graphics import cutil
from reinfocus.graphics import ray
from reinfocus.graphics import shape
from reinfocus.graphics import rectangle
from reinfocus.graphics import vector
from reinfocus.graphics import world
from tests.graphics import numba_test_utils


class RectangleTest(unittest.TestCase):
    """Test cases for reinfocus.graphics.rectangle.rectangle."""

    def test_rectangle(self):
        """Tests that rectangle makes a rectangle with the expected elements."""
        numpy_testing.assert_allclose(
            rectangle.rectangle(
                vector.v2f(0, 1), vector.v2f(2, 3), 4, vector.v2f(5, 6)
            ).parameters,
            [0, 1, 2, 3, 4, 5, 6],
        )


class HitTest(cuda_testing.CUDATestCase):
    """Test cases for reinfocus.graphics.rectangle.hit."""

    def test_hit(self):
        """Tests if hit returns an appropriate hit_record for a ray hit."""

        @cuda.jit
        def hit_rectangle(target, rectangle_parameters, origin, direction):
            i = cutil.line_index()
            if cutil.outside_shape(i, target.shape):
                return

            target[i] = numba_test_utils.flatten_hit_result(
                rectangle.hit(
                    rectangle_parameters,
                    ray.ray(origin, direction),
                    numpy.float32(0),
                    numpy.float32(100),
                )
            )

        cpu_array = numpy.zeros((1, 13), dtype=numpy.float32)

        cutil.launcher(hit_rectangle, 1)(
            cpu_array,
            rectangle.rectangle(
                vector.v2f(-1, 1), vector.v2f(-1, 1), 1, vector.v2f(4, 8)
            ).parameters,
            vector.v3f(0, 0, 0),
            vector.v3f(0, 0, 1),
        )

        numpy_testing.assert_allclose(
            cpu_array[0], (1, 0, 0, 1, 0, 0, 1, 1, 0.5, 0.5, 4, 8, shape.RECTANGLE)
        )


class FastHitTest(cuda_testing.CUDATestCase):
    """Test cases for reinfocus.graphics.rectangle.fast_hit."""

    def test_fast_hit(self):
        """Tests if fast_hit returns an appropriate hit_record for a ray hit."""

        @cuda.jit
        def fast_hit_rectangle(target, fast_rectangle_parameters, origin, direction):
            i = cutil.line_index()
            if cutil.outside_shape(i, target.shape):
                return

            target[i] = numba_test_utils.flatten_hit_result(
                rectangle.fast_hit(
                    fast_rectangle_parameters[i],
                    ray.ray(origin, direction),
                    numpy.float32(0),
                    numpy.float32(100),
                )
            )

        cpu_array = numpy.zeros((1, 13), dtype=numpy.float32)

        world_data = world.FocusWorlds(1)
        world_data.update_targets([1])

        cutil.launcher(fast_hit_rectangle, 1)(
            cpu_array,
            world_data.device_data(),
            vector.v3f(0, 0, 0),
            vector.v3f(0, 0, -1),
        )


class UVTest(cuda_testing.CUDATestCase):
    """Test cases for reinfocus.graphics.rectangle.uv."""

    def test_uv(self):
        """Tests if uv returns an appropriate texture coordinate for a point on some (the
        unit?) rectangle."""

        @cuda.jit
        def get_texture_coord(target, points):
            i = cutil.line_index()
            if cutil.outside_shape(i, target.shape):
                return

            target[i] = rectangle.uv(points[i], -1, 1, -1, 1)

        tests = numpy.array(
            [
                vector.v2f(-1, -1),
                vector.v2f(-1, 1),
                vector.v2f(1, -1),
                vector.v2f(1, 1),
                vector.v2f(0, 0),
            ]
        )

        cpu_array = numpy.zeros((len(tests), 2))

        cutil.launcher(get_texture_coord, len(tests))(cpu_array, tests)

        numpy_testing.assert_allclose(
            cpu_array, [[0, 0], [0, 1], [1, 0], [1, 1], [0.5, 0.5]]
        )


if __name__ == "__main__":
    unittest.main()
