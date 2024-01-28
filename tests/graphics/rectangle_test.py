"""Contains tests for reinfocus.graphics.rectangle."""

import numpy

from numba import cuda
from numba.cuda import testing
from numba.cuda.testing import unittest

from reinfocus.graphics import cutil
from reinfocus.graphics import ray
from reinfocus.graphics import shape
from reinfocus.graphics import rectangle
from reinfocus.graphics import vector
from tests import test_utils
from tests.graphics import numba_test_utils


class RectangleTest(testing.CUDATestCase):
    """TestCases for reinfocus.graphics.rectangle."""

    def test_cpu_rectangle(self):
        """Tests that cpu_rectangle makes a CPU rectangle with the expected elements."""
        test_utils.all_close(
            rectangle.cpu_rectangle(
                vector.c2f(0, 1), vector.c2f(2, 3), 4, vector.c2f(5, 6)
            ).parameters,
            [0, 1, 2, 3, 4, 5, 6],
        )

    def test_gpu_hit_rectangle(self):
        """Tests if gpu_hit_rectangle returns an appropriate hit_record for a ray hit."""

        @cuda.jit
        def hit_rectangle(target, rectangle_parameters, origin, direction):
            i = cutil.line_index()
            if cutil.outside_shape(i, target.shape):
                return

            target[i] = numba_test_utils.flatten_hit_result(
                rectangle.gpu_hit_rectangle(
                    rectangle_parameters,
                    ray.cpu_to_gpu_ray(origin, direction),
                    0,
                    100,
                )
            )

        cpu_array = numpy.zeros((1, 13), dtype=numpy.float32)

        cutil.launcher(hit_rectangle, 1)(
            cpu_array,
            rectangle.cpu_rectangle(
                vector.c2f(-1, 1), vector.c2f(-1, 1), 1, vector.c2f(4, 8)
            ).parameters,
            vector.c3f(0, 0, 0),
            vector.c3f(0, 0, 1),
        )

        test_utils.all_close(
            cpu_array[0], (1, 0, 0, 1, 0, 0, 1, 1, 0.5, 0.5, 4, 8, shape.RECTANGLE)
        )

    def test_gpu_rectangle_uv(self):
        """Tests if gpu_rectangle_uv returns an appropriate texture coordinate for a point
        on some (the unit?) rectangle."""

        @cuda.jit
        def get_texture_coord(target, points):
            i = cutil.line_index()
            if cutil.outside_shape(i, target.shape):
                return

            target[i] = vector.g2f_to_c2f(
                rectangle.gpu_rectangle_uv(vector.c2f_to_g2f(points[i]), -1, 1, -1, 1)
            )

        tests = numpy.array(
            [
                vector.c2f(-1, -1),
                vector.c2f(-1, 1),
                vector.c2f(1, -1),
                vector.c2f(1, 1),
                vector.c2f(0, 0),
            ]
        )

        cpu_array = numpy.zeros((len(tests), 2))

        cutil.launcher(get_texture_coord, len(tests))(cpu_array, tests)

        test_utils.all_close(cpu_array, [[0, 0], [0, 1], [1, 0], [1, 1], [0.5, 0.5]])


if __name__ == "__main__":
    unittest.main()
