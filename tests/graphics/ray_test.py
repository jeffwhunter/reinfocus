"""Contains tests for reinfocus.graphics.ray."""

import numpy

from numba import cuda
from numba.cuda import testing
from numba.cuda.testing import unittest

from reinfocus.graphics import cutil
from reinfocus.graphics import ray
from reinfocus.graphics import vector
from tests import test_utils
from tests.graphics import numba_test_utils


class RayTest(testing.CUDATestCase):
    """TestCases for reinfocus.graphics.ray."""

    def test_gpu_ray(self):
        """Tests that gpu_ray constructs a GPU ray with the expected origin and
        direction."""

        @cuda.jit
        def copy_gpu_ray(target, origin, direction):
            i = cutil.line_index()
            if cutil.outside_shape(i, target.shape):
                return

            target[i] = numba_test_utils.flatten_ray(
                ray.cpu_to_gpu_ray(origin, direction)
            )

        cpu_array = numpy.zeros((1, 6), dtype=numpy.float32)

        cutil.launcher(copy_gpu_ray, 1)(
            cpu_array, vector.c3f(1, 2, 3), vector.c3f(4, 5, 6)
        )

        test_utils.arrays_close(
            self, cpu_array[0], vector.c3f(1, 2, 3) + vector.c3f(4, 5, 6)
        )

    def test_gpu_point_at_parameter(self):
        """Tests that gpu_point_at_parameter correctly finds the point t distance along
        ray."""

        @cuda.jit
        def find_gpu_point_at_parameter(target, origin, direction, t):
            i = cutil.line_index()
            if cutil.outside_shape(i, target.shape):
                return

            target[i] = vector.g3f_to_c3f(
                ray.gpu_point_at_parameter(ray.cpu_to_gpu_ray(origin, direction), t)
            )

        cpu_array = numpy.zeros((1, 3), dtype=numpy.float32)

        cutil.launcher(find_gpu_point_at_parameter, 1)(
            cpu_array, vector.c3f(1, 2, 3), vector.c3f(4, 5, 6), 2
        )

        test_utils.arrays_close(self, cpu_array[0], vector.c3f(9, 12, 15))


if __name__ == "__main__":
    unittest.main()
