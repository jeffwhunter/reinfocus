"""Contains tests for reinfocus.graphics.sphere."""

import numpy

from numba import cuda
from numba.cuda import testing
from numba.cuda.testing import unittest

from reinfocus.graphics import cutil
from reinfocus.graphics import ray
from reinfocus.graphics import shape
from reinfocus.graphics import sphere
from reinfocus.graphics import vector
from tests import test_utils
from tests.graphics import numba_test_utils


class SphereTest(testing.CUDATestCase):
    """TestCases for reinfocus.graphics.sphere."""

    def test_cpu_sphere(self):
        """Tests that cpu_sphere makes a CPU sphere with the expected elements."""

        test_utils.arrays_close(
            self,
            sphere.cpu_sphere(vector.c3f(1, 2, 3), 4, vector.c2f(5, 6)).parameters,
            [1, 2, 3, 4, 5, 6],
        )

    def test_gpu_hit_sphere(self):
        """Tests if gpu_hit_sphere returns an appropriate hit_record for a ray hit."""

        @cuda.jit
        def hit_sphere(target, sphere_parameters, origin, direction):
            i = cutil.line_index()
            if cutil.outside_shape(i, target.shape):
                return

            target[i] = numba_test_utils.flatten_hit_result(
                sphere.gpu_hit_sphere(
                    sphere_parameters, ray.cpu_to_gpu_ray(origin, direction), 0, 100
                )
            )

        cpu_array = numpy.zeros((1, 13), dtype=numpy.float32)

        cutil.launcher(hit_sphere, 1)(
            cpu_array,
            sphere.cpu_sphere(vector.c3f(0, 0, 0), 1, vector.c2f(4, 8)).parameters,
            vector.c3f(10, 0, 0),
            vector.c3f(-1, 0, 0),
        )

        test_utils.arrays_close(
            self, cpu_array[0], (1, 1, 0, 0, 1, 0, 0, 9, 1, 0.5, 4, 8, shape.SPHERE)
        )

    def test_gpu_sphere_uv(self):
        """Tests if gpu_sphere_uv returns an appropriate texture coordinate for a point
        on the unit sphere."""

        @cuda.jit
        def get_texture_coord(target, point):
            i = cutil.line_index()
            if cutil.outside_shape(i, target.shape):
                return

            target[i] = vector.g2f_to_c2f(sphere.gpu_sphere_uv(vector.c3f_to_g3f(point)))

        cpu_array = numpy.zeros((1, 2), dtype=numpy.float32)

        cutil.launcher(get_texture_coord, 1)(cpu_array, vector.c3f(-1, 0, 0))

        test_utils.arrays_close(self, cpu_array[0], (0.0, 0.5))


if __name__ == "__main__":
    unittest.main()
