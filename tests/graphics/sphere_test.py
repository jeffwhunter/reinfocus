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

    def test_sphere(self):
        """Tests that sphere makes a sphere with the expected elements."""

        test_utils.all_close(
            sphere.sphere(vector.v3f(1, 2, 3), 4, vector.v2f(5, 6)).parameters,
            [1, 2, 3, 4, 5, 6],
        )

    def test_hit(self):
        """Tests if hit returns an appropriate hit_record for a ray hit."""

        @cuda.jit
        def hit_sphere(target, sphere_parameters, origin, direction):
            i = cutil.line_index()
            if cutil.outside_shape(i, target.shape):
                return

            target[i] = numba_test_utils.flatten_hit_result(
                sphere.hit(
                    sphere_parameters,
                    ray.ray(origin, direction),
                    numpy.float32(0),
                    numpy.float32(100),
                )
            )

        cpu_array = numpy.zeros((1, 13), dtype=numpy.float32)

        cutil.launcher(hit_sphere, 1)(
            cpu_array,
            sphere.sphere(vector.v3f(0, 0, 0), 1, vector.v2f(4, 8)).parameters,
            vector.v3f(10, 0, 0),
            vector.v3f(-1, 0, 0),
        )

        test_utils.all_close(
            cpu_array[0], (1, 1, 0, 0, 1, 0, 0, 9, 1, 0.5, 4, 8, shape.SPHERE)
        )

    def test_uv(self):
        """Tests if uv returns an appropriate texture coordinate for a point on the unit
        sphere."""

        @cuda.jit
        def get_texture_coord(target, point):
            i = cutil.line_index()
            if cutil.outside_shape(i, target.shape):
                return

            target[i] = sphere.uv(point)

        cpu_array = numpy.zeros((1, 2), dtype=numpy.float32)

        cutil.launcher(get_texture_coord, 1)(cpu_array, vector.v3f(-1, 0, 0))

        test_utils.all_close(cpu_array[0], (0.0, 0.5))


if __name__ == "__main__":
    unittest.main()
