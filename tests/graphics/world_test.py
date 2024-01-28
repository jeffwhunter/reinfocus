"""Contains tests for reinfocus.graphics.world."""

import numpy

from numba import cuda
from numba.cuda import testing
from numba.cuda.testing import unittest

from reinfocus.graphics import cutil
from reinfocus.graphics import ray
from reinfocus.graphics import rectangle
from reinfocus.graphics import shape
from reinfocus.graphics import sphere
from reinfocus.graphics import vector
from reinfocus.graphics import world
from tests import test_utils
from tests.graphics import numba_test_utils


class WorldTest(testing.CUDATestCase):
    """TestCases for reinfocus.graphics.world."""

    def test_different_world_parameters(self):
        """Tests that World.device_shape_parameters can handle spheres and rectangles."""

        w = world.World(
            sphere.cpu_sphere(vector.c3f(1, 2, 3), 4, vector.c2f(5, 6)),
            rectangle.cpu_rectangle(
                vector.c2f(-1, 1), vector.c2f(-1, 1), 1, vector.c2f(4, 8)
            ),
        )

        test_utils.all_close(
            w.device_shape_parameters(),
            numpy.array([[1, 2, 3, 4, 5, 6, 0], [-1, 1, -1, 1, 1, 4, 8]]),
        )

    def test_sphere_world_parameters(self):
        """Tests that World.device_shape_parameters can handle spheres."""

        w = world.World(
            sphere.cpu_sphere(vector.c3f(1, 2, 3), 4, vector.c2f(5, 6)),
            sphere.cpu_sphere(vector.c3f(7, 8, 9), 10, vector.c2f(11, 12)),
        )

        test_utils.all_close(
            w.device_shape_parameters(),
            numpy.array([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]]),
        )

    def test_world_shape_types(self):
        """Tests that World.device_shape_types can handle spheres and rectangles."""

        w = world.World(
            sphere.cpu_sphere(vector.c3f(1, 2, 3), 4, vector.c2f(5, 6)),
            rectangle.cpu_rectangle(
                vector.c2f(-1, 1), vector.c2f(-1, 1), 1, vector.c2f(4, 8)
            ),
        )

        test_utils.all_close(
            w.device_shape_types(), numpy.array([shape.SPHERE, shape.RECTANGLE])
        )

    def test_gpu_hit_sphere_world(self):
        """Tests if gpu_hit_world returns an appropriate hit_record for spheres."""

        @cuda.jit
        def hit_sphere_world(target, shapes_parameters, shapes_types, origin, direction):
            i = cutil.line_index()
            if cutil.outside_shape(i, target.shape):
                return

            target[i] = numba_test_utils.flatten_hit_result(
                world.gpu_hit_world(
                    shapes_parameters,
                    shapes_types,
                    ray.cpu_to_gpu_ray(origin, direction),
                    0,
                    100,
                )
            )

        cpu_array = numpy.zeros((1, 13), dtype=numpy.float32)

        cpu_world = world.World(
            sphere.cpu_sphere(vector.c3f(0, 0, 0), 1, vector.c2f(4, 8))
        )

        cutil.launcher(hit_sphere_world, 1)(
            cpu_array,
            cpu_world.device_shape_parameters(),
            cpu_world.device_shape_types(),
            vector.c3f(10, 0, 0),
            vector.c3f(-1, 0, 0),
        )

        test_utils.all_close(
            cpu_array[0], (1, 1, 0, 0, 1, 0, 0, 9, 1, 0.5, 4, 8, shape.SPHERE)
        )

    def test_gpu_hit_rectangle_world(self):
        """Tests if gpu_hit_world returns an appropriate hit_record for rectangles."""

        @cuda.jit
        def hit_rectangle_world(
            target, shapes_parameters, shapes_types, origin, direction
        ):
            i = cutil.line_index()
            if cutil.outside_shape(i, target.shape):
                return

            target[i] = numba_test_utils.flatten_hit_result(
                world.gpu_hit_world(
                    shapes_parameters,
                    shapes_types,
                    ray.cpu_to_gpu_ray(origin, direction),
                    0,
                    100,
                )
            )

        cpu_array = numpy.zeros((1, 13), dtype=numpy.float32)

        cpu_world = world.World(
            rectangle.cpu_rectangle(
                vector.c2f(-1, 1), vector.c2f(-1, 1), 1, vector.c2f(4, 8)
            )
        )

        cutil.launcher(hit_rectangle_world, 1)(
            cpu_array,
            cpu_world.device_shape_parameters(),
            cpu_world.device_shape_types(),
            vector.c3f(0, 0, 0),
            vector.c3f(0, 0, 1),
        )

        test_utils.all_close(
            cpu_array[0], (1, 0, 0, 1, 0, 0, 1, 1, 0.5, 0.5, 4, 8, shape.RECTANGLE)
        )

    def test_one_sphere_world(self):
        """Tests that one_sphere_world creates a world with one sphere."""

        test_utils.all_close(
            world.one_sphere_world().device_shape_types(), [shape.SPHERE]
        )

    def test_two_sphere_world(self):
        """Tests that two_sphere_world creates a world with two spheres."""

        test_utils.all_close(
            world.two_sphere_world().device_shape_types(), [shape.SPHERE] * 2
        )

    def test_one_rect_world(self):
        """Tests that one_rect_world creates a world with one rectangle."""

        test_utils.all_close(
            world.one_rect_world().device_shape_types(), [shape.RECTANGLE]
        )

    def test_two_rect_world(self):
        """Tests that two_rect_world creates a world with two rectangles."""

        test_utils.all_close(
            world.two_rect_world().device_shape_types(), [shape.RECTANGLE] * 2
        )

    def test_mixed_world(self):
        """Tests that mixed_world creates a world with two rectangles."""

        types = world.mixed_world().device_shape_types()
        self.assertLessEqual(set(types), set([shape.RECTANGLE, shape.SPHERE]))
        self.assertIs(len(types), 2)


if __name__ == "__main__":
    unittest.main()
