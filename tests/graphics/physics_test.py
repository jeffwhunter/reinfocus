"""Contains tests for reinfocus.graphics.physics."""

import numpy

from numba import cuda
from numba.cuda import testing
from numba.cuda.testing import unittest
from numpy import linalg

from reinfocus.graphics import cutil
from reinfocus.graphics import hit_record
from reinfocus.graphics import physics
from reinfocus.graphics import random
from reinfocus.graphics import ray
from reinfocus.graphics import rectangle
from reinfocus.graphics import shape
from reinfocus.graphics import sphere
from reinfocus.graphics import vector
from reinfocus.graphics import world
from tests import test_utils
from tests.graphics import numba_test_utils


class PhysicsTest(testing.CUDATestCase):
    """TestCases for reinfocus.graphics.physics."""

    def test_random_in_unit_sphere(self):
        """Tests that random_in_unit_sphere makes 3D vectors in the unit sphere."""

        @cuda.jit
        def sample_from_sphere(target, random_states):
            i = cutil.line_index()
            if cutil.outside_shape(i, target.shape):
                return

            target[i] = physics.random_in_unit_sphere(random_states, i)

        tests = 100

        cpu_array = numpy.zeros((tests, 3), dtype=numpy.float32)

        cutil.launcher(sample_from_sphere, tests)(
            cpu_array, random.make_random_states(tests, 0)
        )

        self.assertTrue(numpy.all(linalg.norm(cpu_array, axis=-1) < 1.0))

    def test_colour_checkerboard(self):
        """Tests that colour_checkerboard produces the expected colours for different
        frequencies and positions."""

        @cuda.jit
        def checkerboard_colour_points(target, f, p):
            i = cutil.line_index()
            if cutil.outside_shape(i, target.shape):
                return

            target[i] = physics.colour_checkerboard(f[i], p[i])

        tests = cuda.to_device(
            numpy.array(
                [
                    [vector.v2f(1, 1), vector.v2f(0.25, 0.25)],
                    [vector.v2f(1, 1), vector.v2f(0.25, 0.75)],
                    [vector.v2f(1, 1), vector.v2f(0.75, 0.25)],
                    [vector.v2f(1, 1), vector.v2f(0.75, 0.75)],
                    [vector.v2f(2, 2), vector.v2f(0.25, 0.25)],
                    [vector.v2f(2, 2), vector.v2f(0.25, 0.75)],
                    [vector.v2f(2, 2), vector.v2f(0.75, 0.25)],
                    [vector.v2f(2, 2), vector.v2f(0.75, 0.75)],
                ]
            )
        )

        cpu_array = numpy.zeros((len(tests), 3), dtype=numpy.float32)

        cutil.launcher(checkerboard_colour_points, len(tests))(
            cpu_array, tests[:, 0], tests[:, 1]
        )

        test_utils.all_close(
            cpu_array,
            [
                [1, 0, 0],
                [1, 0, 0],
                [1, 0, 0],
                [1, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [0, 1, 0],
                [1, 0, 0],
            ],
        )

    def test_scatter_with_rectangles(self):
        """Tests that scatter scatters a rectangle hit in the expected way."""

        @cuda.jit
        def scatter_with_rectangle(target, random_states):
            i = cutil.line_index()
            if cutil.outside_shape(i, target.shape):
                return

            target[i] = numba_test_utils.flatten_coloured_ray(
                physics.scatter(
                    hit_record.hit_record(
                        vector.d_v3f(0, 0, 0),
                        vector.d_v3f(0, 0, 1),
                        numpy.float32(1.0),
                        vector.d_v2f(2**-4, 2**-4),
                        vector.d_v2f(1.0, 1.0),
                        numpy.float32(shape.RECTANGLE),
                    ),
                    random_states,
                    i,
                )
            )

        cpu_array = numpy.zeros((1, 9), dtype=numpy.float32)

        cutil.launcher(scatter_with_rectangle, 1)(
            cpu_array, random.make_random_states(1, 0)
        )

        test_utils.all_close(cpu_array[0, 0:3], [0, 0, 0])

        self.assertTrue(
            numpy.less(linalg.norm(cpu_array[0, 3:6] - numpy.array([0, 0, 1])), 1.0)
        )
        test_utils.all_close(cpu_array[0, 6:9], [1, 0, 0])

    def test_scatter_with_spheres(self):
        """Tests that scatter scatters a sphere hit in the expected way."""

        @cuda.jit
        def scatter_with_sphere(target, random_states):
            i = cutil.line_index()
            if cutil.outside_shape(i, target.shape):
                return

            target[i] = numba_test_utils.flatten_coloured_ray(
                physics.scatter(
                    hit_record.hit_record(
                        vector.d_v3f(0, 0, 1),
                        vector.d_v3f(0, 0, 1),
                        numpy.float32(1.0),
                        vector.d_v2f(2**-7, 2**-6),
                        vector.d_v2f(1.0, 1.0),
                        numpy.float32(shape.SPHERE),
                    ),
                    random_states,
                    i,
                )
            )

        cpu_array = numpy.zeros((1, 9), dtype=numpy.float32)

        cutil.launcher(scatter_with_sphere, 1)(cpu_array, random.make_random_states(1, 0))

        test_utils.all_close(cpu_array[0, 0:3], [0, 0, 1])
        self.assertTrue(
            numpy.less(linalg.norm(cpu_array[0, 3:6] - numpy.array([0, 0, 1])), 1.0)
        )
        test_utils.all_close(cpu_array[0, 6:9], [1, 0, 0])

    def test_find_colour_with_rectangles(self):
        """Tests that find_colour finds the expected colour when we fire a ray at a
        rectangle."""

        @cuda.jit
        def find_rectangle_colour(target, random_states, shapes_parameters, shapes_types):
            i = cutil.line_index()
            if cutil.outside_shape(i, target.shape):
                return

            target[i] = physics.find_colour(
                shapes_parameters,
                shapes_types,
                ray.ray(
                    vector.d_v3f(-(2**-4), -(2**-4), 0),
                    vector.d_v3f(-(2**-4), -(2**-4), 1),
                ),
                random_states,
                i,
            )

        cpu_array = numpy.zeros((1, 3), dtype=numpy.float32)

        world_data = world.World(
            rectangle.rectangle(vector.v2f(-1, 1), vector.v2f(-1, 1), 1)
        )

        cutil.launcher(find_rectangle_colour, 1)(
            cpu_array,
            random.make_random_states(1, 0),
            world_data.device_shape_parameters(),
            world_data.device_shape_types(),
        )

        self.assertTrue(0 < cpu_array[0, 0] <= 1.0)
        test_utils.all_close(cpu_array[0, 1:3], [0, 0])

    def test_find_colour_with_spheres(self):
        """Tests that find_colour finds the expected colour when we fire a ray at a
        sphere."""

        @cuda.jit
        def find_sphere_colour(target, random_states, shapes_parameters, shapes_types):
            i = cutil.line_index()
            if cutil.outside_shape(i, target.shape):
                return

            target[i] = physics.find_colour(
                shapes_parameters,
                shapes_types,
                ray.ray(
                    vector.d_v3f(-(2**-7), -(2**-6), 0),
                    vector.d_v3f(-(2**-7), -(2**-6), 1),
                ),
                random_states,
                i,
            )

        cpu_array = numpy.zeros((1, 3), dtype=numpy.float32)

        world_data = world.World(sphere.sphere(vector.v3f(0, 0, 10), 1))

        cutil.launcher(find_sphere_colour, 1)(
            cpu_array,
            random.make_random_states(1, 0),
            world_data.device_shape_parameters(),
            world_data.device_shape_types(),
        )

        self.assertTrue(0 < cpu_array[0, 1] <= 1.0)
        test_utils.all_close(cpu_array[0, ::2], [0, 0])


if __name__ == "__main__":
    unittest.main()
