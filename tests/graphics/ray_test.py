"""Contains tests for reinfocus.graphics.ray."""

import numpy

from numba import cuda
from numba.cuda import testing as cuda_testing
from numba.cuda.testing import unittest
from numpy import testing as numpy_testing

from reinfocus.graphics import cutil
from reinfocus.graphics import ray
from reinfocus.graphics import vector
from tests.graphics import numba_test_utils


class RayTest(cuda_testing.CUDATestCase):
    """Test cases for reinfocus.graphics.ray.ray."""

    def test_ray(self):
        """Tests that ray constructs a ray with the expected origin and direction."""

        @cuda.jit
        def copy_ray(target, origin, direction):
            i = cutil.line_index()
            if cutil.outside_shape(i, target.shape):
                return

            target[i] = numba_test_utils.flatten_ray(ray.ray(origin, direction))

        cpu_array = numpy.zeros((1, 6), dtype=numpy.float32)

        cutil.launcher(copy_ray, 1)(cpu_array, vector.v3f(1, 2, 3), vector.v3f(4, 5, 6))

        numpy_testing.assert_allclose(
            cpu_array[0], vector.v3f(1, 2, 3) + vector.v3f(4, 5, 6)
        )


class PointAtParameterTest(cuda_testing.CUDATestCase):
    """Test cases for reinfocus.graphics.ray.point_at_parameter."""

    def test_point_at_parameter(self):
        """Tests that point_at_parameter correctly finds the point t distance along
        ray."""

        @cuda.jit
        def find_point_at_parameter(target, origin, direction, t):
            i = cutil.line_index()
            if cutil.outside_shape(i, target.shape):
                return

            target[i] = ray.point_at_parameter(ray.ray(origin, direction), t)

        cpu_array = numpy.zeros((1, 3), dtype=numpy.float32)

        cutil.launcher(find_point_at_parameter, 1)(
            cpu_array, vector.v3f(1, 2, 3), vector.v3f(4, 5, 6), 2
        )

        numpy_testing.assert_allclose(cpu_array[0], vector.v3f(9, 12, 15))


if __name__ == "__main__":
    unittest.main()
