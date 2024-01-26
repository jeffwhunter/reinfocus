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
        """Tests that random_in_unit_sphere makes 3D GPU vectors in the unit sphere."""

        @cuda.jit
        def sample_from_sphere(target, random_states):
            i = cutil.line_index()
            if i < target.size:
                target[i] = vector.g3f_to_c3f(
                    physics.random_in_unit_sphere(random_states, i)
                )

        tests = 100

        cpu_array = numba_test_utils.cpu_target(nrow=tests)

        cutil.launcher(sample_from_sphere, (tests, 1))(
            cpu_array, random.make_random_states(tests, 0)
        )

        self.assertTrue(numpy.all(linalg.norm(cpu_array, axis=-1) < 1.0))

    def test_colour_checkerboard(self):
        """Tests that colour_checkerboard produces the expected colours for different
        frequencies and positions."""

        @cuda.jit
        def checkerboard_colour_points(target, f, p):
            i = cutil.line_index()
            if i < target.size:
                target[i] = vector.g3f_to_c3f(
                    physics.colour_checkerboard(
                        vector.c2f_to_g2f(f[i]), vector.c2f_to_g2f(p[i])
                    )
                )

        tests = cuda.to_device(
            numpy.array(
                [
                    [vector.c2f(1, 1), vector.c2f(0.25, 0.25)],
                    [vector.c2f(1, 1), vector.c2f(0.25, 0.75)],
                    [vector.c2f(1, 1), vector.c2f(0.75, 0.25)],
                    [vector.c2f(1, 1), vector.c2f(0.75, 0.75)],
                    [vector.c2f(2, 2), vector.c2f(0.25, 0.25)],
                    [vector.c2f(2, 2), vector.c2f(0.25, 0.75)],
                    [vector.c2f(2, 2), vector.c2f(0.75, 0.25)],
                    [vector.c2f(2, 2), vector.c2f(0.75, 0.75)],
                ]
            )
        )

        cpu_array = numba_test_utils.cpu_target(nrow=len(tests))

        cutil.launcher(checkerboard_colour_points, (len(tests), 1))(
            cpu_array, tests[:, 0], tests[:, 1]
        )

        test_utils.arrays_close(
            self,
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
            if i < target.size:
                target[i] = numba_test_utils.flatten_coloured_ray(
                    physics.scatter(
                        hit_record.gpu_hit_record(
                            vector.g3f(0, 0, 0),
                            vector.g3f(0, 0, 1),
                            1.0,
                            vector.g2f(2**-4, 2**-4),
                            vector.g2f(1.0, 1.0),
                            shape.RECTANGLE,
                        ),
                        random_states,
                        i,
                    )
                )

        cpu_array = numba_test_utils.cpu_target(ndim=9)

        cutil.launcher(scatter_with_rectangle, (1, 1))(
            cpu_array, random.make_random_states(1, 0)
        )

        test_utils.arrays_close(self, cpu_array[0, 0:3], [0, 0, 0])

        self.assertTrue(
            numpy.less(linalg.norm(cpu_array[0, 3:6] - numpy.array([0, 0, 1])), 1.0)
        )
        test_utils.arrays_close(self, cpu_array[0, 6:9], [1, 0, 0])

    def test_scatter_with_spheres(self):
        """Tests that scatter scatters a sphere hit in the expected way."""

        @cuda.jit
        def scatter_with_sphere(target, random_states):
            i = cutil.line_index()
            if i < target.size:
                target[i] = numba_test_utils.flatten_coloured_ray(
                    physics.scatter(
                        hit_record.gpu_hit_record(
                            vector.g3f(0, 0, 1),
                            vector.g3f(0, 0, 1),
                            1.0,
                            vector.g2f(2**-7, 2**-6),
                            vector.g2f(1.0, 1.0),
                            shape.SPHERE,
                        ),
                        random_states,
                        i,
                    )
                )

        cpu_array = numba_test_utils.cpu_target(ndim=9)

        cutil.launcher(scatter_with_sphere, (1, 1))(
            cpu_array, random.make_random_states(1, 0)
        )

        test_utils.arrays_close(self, cpu_array[0, 0:3], [0, 0, 1])
        self.assertTrue(
            numpy.less(linalg.norm(cpu_array[0, 3:6] - numpy.array([0, 0, 1])), 1.0)
        )
        test_utils.arrays_close(self, cpu_array[0, 6:9], [1, 0, 0])

    def test_find_colour_with_rectangles(self):
        """Tests that find_colour finds the expected colour when we fire a ray at a
        rectangle."""

        @cuda.jit
        def find_rectangle_colour(target, random_states, shapes_parameters, shapes_types):
            i = cutil.line_index()
            if i < target.size:
                target[i] = vector.g3f_to_c3f(
                    physics.find_colour(
                        shapes_parameters,
                        shapes_types,
                        ray.gpu_ray(
                            vector.g3f(-(2**-4), -(2**-4), 0),
                            vector.g3f(-(2**-4), -(2**-4), 1),
                        ),
                        random_states,
                        i,
                    )
                )

        cpu_array = numba_test_utils.cpu_target()

        cpu_world = world.World(rectangle.cpu_rectangle(-1, 1, -1, 1, 1))

        cutil.launcher(find_rectangle_colour, (1, 1))(
            cpu_array,
            random.make_random_states(1, 0),
            cpu_world.device_shape_parameters(),
            cpu_world.device_shape_types(),
        )

        self.assertTrue(0 < cpu_array[0, 0] <= 1.0)
        test_utils.arrays_close(self, cpu_array[0, 1:3], [0, 0])

    def test_find_colour_with_spheres(self):
        """Tests that find_colour finds the expected colour when we fire a ray at a
        sphere."""

        @cuda.jit
        def find_sphere_colour(target, random_states, shapes_parameters, shapes_types):
            i = cutil.line_index()
            if i < target.size:
                target[i] = vector.g3f_to_c3f(
                    physics.find_colour(
                        shapes_parameters,
                        shapes_types,
                        ray.gpu_ray(
                            vector.g3f(-(2**-7), -(2**-6), 0),
                            vector.g3f(-(2**-7), -(2**-6), 1),
                        ),
                        random_states,
                        i,
                    )
                )

        cpu_array = numba_test_utils.cpu_target()

        cpu_world = world.World(sphere.cpu_sphere(vector.c3f(0, 0, 10), 1))

        cutil.launcher(find_sphere_colour, (1, 1))(
            cpu_array,
            random.make_random_states(1, 0),
            cpu_world.device_shape_parameters(),
            cpu_world.device_shape_types(),
        )

        self.assertTrue(0 < cpu_array[0, 1] <= 1.0)
        test_utils.arrays_close(self, cpu_array[0, ::2], [0, 0])


if __name__ == "__main__":
    unittest.main()
