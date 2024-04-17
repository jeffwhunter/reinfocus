"""Contains tests for reinfocus.graphics.world."""

import numpy

from numba import cuda
from numba.cuda import testing as cuda_testing
from numba.cuda.testing import unittest
from numpy import testing as numpy_testing

from reinfocus.graphics import cutil
from reinfocus.graphics import ray
from reinfocus.graphics import rectangle
from reinfocus.graphics import shape
from reinfocus.graphics import sphere
from reinfocus.graphics import vector
from reinfocus.graphics import world
from tests.graphics import numba_test_utils


class WorldTest(cuda_testing.CUDATestCase):
    """Test cases for reinfocus.graphics.world.World."""

    def test_environment_sizes(self):
        """Tests that device_data contains the proper environment sizes."""

        testee = world.Worlds(
            [
                sphere.sphere(vector.v3f(1, 2, 3), 4, vector.v2f(5, 6)),
                rectangle.rectangle(
                    vector.v2f(-1, 1), vector.v2f(-1, 1), 1, vector.v2f(4, 8)
                ),
            ],
            [
                rectangle.rectangle(
                    vector.v2f(-0.5, 0.5),
                    vector.v2f(-0.5, 0.5),
                    0.5,
                    vector.v2f(8, 4),
                )
            ],
        )

        numpy_testing.assert_allclose(testee.device_data()[world.MW_ENV_SIZES], [2, 1])

    def test_parameters(self):
        """Tests that device_data contains the proper shape parameters."""

        testee = world.Worlds(
            [
                sphere.sphere(vector.v3f(1, 2, 3), 4, vector.v2f(5, 6)),
                rectangle.rectangle(
                    vector.v2f(-1, 1), vector.v2f(-1, 1), 1, vector.v2f(4, 8)
                ),
            ],
            [
                rectangle.rectangle(
                    vector.v2f(-0.5, 0.5),
                    vector.v2f(-0.5, 0.5),
                    0.5,
                    vector.v2f(8, 4),
                )
            ],
        )

        numpy_testing.assert_allclose(
            testee.device_data()[world.MW_PARAMETERS],
            [
                [
                    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0],
                    [-1.0, 1.0, -1.0, 1.0, 1.0, 4.0, 8.0],
                ],
                [
                    [-0.5, 0.5, -0.5, 0.5, 0.5, 8.0, 4.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ],
            ],
        )

    def test_types(self):
        """Tests that device_data contains the proper shape types."""

        testee = world.Worlds(
            [
                sphere.sphere(vector.v3f(1, 2, 3), 4, vector.v2f(5, 6)),
                rectangle.rectangle(
                    vector.v2f(-1, 1), vector.v2f(-1, 1), 1, vector.v2f(4, 8)
                ),
            ],
            [
                rectangle.rectangle(
                    vector.v2f(-0.5, 0.5),
                    vector.v2f(-0.5, 0.5),
                    0.5,
                    vector.v2f(8, 4),
                )
            ],
        )

        numpy_testing.assert_allclose(
            testee.device_data()[world.MW_TYPES],
            [[shape.SPHERE, shape.RECTANGLE], [shape.RECTANGLE, shape.SPHERE]],
        )


class FocusWorldTest(cuda_testing.CUDATestCase):
    """Test cases for reinfocus.graphics.world.FocusWorld."""

    def test_world_parameters(self):
        """Tests that device_data contains the proper rectangle parameters."""

        testee = world.FocusWorlds(3)

        testee.update_targets([1, 2, 3])

        sizes = testee.device_data()[:, 0]

        numpy_testing.assert_allclose(testee.device_data()[:, 1], [-1, -2, -3])
        numpy_testing.assert_array_less(0, sizes)

        testee.update_targets([3, 2, 1])

        reversed_sizes = testee.device_data()[:, 0]

        numpy_testing.assert_allclose(testee.device_data()[:, 1], [-3, -2, -1])
        numpy_testing.assert_array_less(0, reversed_sizes)

        numpy_testing.assert_allclose(sizes, list(reversed(reversed_sizes)))


class HitTest(cuda_testing.CUDATestCase):
    """Test cases for reinfocus.graphics.world.hit."""

    def test_sphere_world(self):
        """Tests if hit returns an appropriate hit_record for spheres."""

        @cuda.jit
        def hit_sphere_world(target, device_worlds, origin, direction):
            i = cutil.line_index()
            if cutil.outside_shape(i, target.shape):
                return

            target[i] = numba_test_utils.flatten_hit_result(
                world.hit(
                    device_worlds[world.MW_PARAMETERS][0],
                    device_worlds[world.MW_TYPES][0],
                    ray.ray(origin, direction),
                    numpy.float32(0),
                    numpy.float32(100),
                )
            )

        cpu_array = numpy.zeros((1, 13), dtype=numpy.float32)

        testee = world.Worlds([sphere.sphere(vector.v3f(0, 0, 0), 1, vector.v2f(4, 8))])

        cutil.launcher(hit_sphere_world, 1)(
            cpu_array,
            testee.device_data(),
            vector.v3f(10, 0, 0),
            vector.v3f(-1, 0, 0),
        )

        numpy_testing.assert_allclose(
            cpu_array[0], (1, 1, 0, 0, 1, 0, 0, 9, 1, 0.5, 4, 8, shape.SPHERE)
        )

    def test_rectangle_world(self):
        """Tests if hit_world returns an appropriate hit_record for rectangles."""

        @cuda.jit
        def hit_rectangle_world(target, device_worlds, origin, direction):
            i = cutil.line_index()
            if cutil.outside_shape(i, target.shape):
                return

            target[i] = numba_test_utils.flatten_hit_result(
                world.hit(
                    device_worlds[world.MW_PARAMETERS][0],
                    device_worlds[world.MW_TYPES][0],
                    ray.ray(origin, direction),
                    numpy.float32(0),
                    numpy.float32(100),
                )
            )

        cpu_array = numpy.zeros((1, 13), dtype=numpy.float32)

        testee = world.Worlds(
            [
                rectangle.rectangle(
                    vector.v2f(-1, 1), vector.v2f(-1, 1), 1, vector.v2f(4, 8)
                )
            ]
        )

        cutil.launcher(hit_rectangle_world, 1)(
            cpu_array,
            testee.device_data(),
            vector.v3f(0, 0, 0),
            vector.v3f(0, 0, 1),
        )

        numpy_testing.assert_allclose(
            cpu_array[0],
            numpy.asarray(
                (1, 0, 0, 1, 0, 0, 1, 1, 0.5, 0.5, 4, 8, numpy.float32(shape.RECTANGLE))
            ),
        )


if __name__ == "__main__":
    unittest.main()
