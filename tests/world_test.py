"""Contains tests for reinfocus.world."""

import numpy as np
from numba import cuda
from numba.cuda.testing import unittest
from reinfocus import ray
from reinfocus import rectangle as rec
from reinfocus import shape as sha
from reinfocus import sphere as sph
from reinfocus import vector as vec
from reinfocus import world as wor
from tests import numba_test_case as ntc
from tests import numba_test_utils as ntu

class WorldTest(ntc.NumbaTestCase):
    """TestCases for reinfocus.world."""
    # pylint: disable=no-value-for-parameter

    def test_different_world_parameters(self):
        """Tests that World.device_shape_parameters can handle spheres and rectangles."""
        w = wor.World(sph.cpu_sphere(vec.c3f(1, 2, 3), 4), rec.cpu_rectangle(-1, 1, -1, 1, 1))
        self.arrays_close(
            w.device_shape_parameters(),
            np.array([[1, 2, 3, 4, 0], [-1, 1, -1, 1, 1]]))

    def test_sphere_world_parameters(self):
        """Tests that World.device_shape_parameters can handle spheres."""
        w = wor.World(sph.cpu_sphere(vec.c3f(1, 2, 3), 4), sph.cpu_sphere(vec.c3f(5, 6, 7), 8))
        self.arrays_close(
            w.device_shape_parameters(),
            np.array([[1, 2, 3, 4], [5, 6, 7, 8]]))

    def test_world_shape_types(self):
        """Tests that World.device_shape_types can handle spheres and rectangles."""
        w = wor.World(sph.cpu_sphere(vec.c3f(1, 2, 3), 4), rec.cpu_rectangle(-1, 1, -1, 1, 1))
        self.arrays_close(
            w.device_shape_types(),
            np.array([sha.SPHERE, sha.RECTANGLE]))

    def test_gpu_hit_sphere_world(self):
        """Tests if gpu_hit_world returns an appropriate hit_record for spheres."""
        @cuda.jit
        def hit_sphere_world(target, shapes_parameters, shapes_types, origin, direction):
            i = cuda.grid(1) # type: ignore
            if i < target.size:
                target[i] = ntu.flatten_hit_result(
                    wor.gpu_hit_world(
                        shapes_parameters,
                        shapes_types,
                        ray.cpu_to_gpu_ray(origin, direction),
                        0,
                        100))

        cpu_array = ntu.cpu_target(ndim=11)

        world = wor.World(sph.cpu_sphere(vec.c3f(0, 0, 0), 1))

        hit_sphere_world[1, 1]( # type: ignore
            cpu_array,
            world.device_shape_parameters(),
            world.device_shape_types(),
            vec.c3f(10, 0, 0),
            vec.c3f(-1, 0, 0))

        self.arrays_close(cpu_array[0], (1, 1, 0, 0, 1, 0, 0, 9, .5, .5, sha.SPHERE))

    def test_gpu_hit_rectangle_world(self):
        """Tests if gpu_hit_world returns an appropriate hit_record for rectangles."""
        @cuda.jit
        def hit_rectangle_world(target, shapes_parameters, shapes_types, origin, direction):
            i = cuda.grid(1) # type: ignore
            if i < target.size:
                target[i] = ntu.flatten_hit_result(
                    wor.gpu_hit_world(
                        shapes_parameters,
                        shapes_types,
                        ray.cpu_to_gpu_ray(origin, direction),
                        0,
                        100))

        cpu_array = ntu.cpu_target(ndim=11)

        world = wor.World(rec.cpu_rectangle(-1, 1, -1, 1, 1))

        hit_rectangle_world[1, 1]( # type: ignore
            cpu_array,
            world.device_shape_parameters(),
            world.device_shape_types(),
            vec.c3f(0, 0, 0),
            vec.c3f(0, 0, 1))

        self.arrays_close(cpu_array[0], (1, 0, 0, 1, 0, 0, 1, 1, .5, .5, sha.RECTANGLE))

    def test_one_sphere_world(self):
        """Tests that one_sphere_world creates a world with one sphere."""
        self.arrays_close(wor.one_sphere_world().device_shape_types(), [sha.SPHERE])

    def test_two_sphere_world(self):
        """Tests that two_sphere_world creates a world with two spheres."""
        self.arrays_close(wor.two_sphere_world().device_shape_types(), [sha.SPHERE] * 2)

    def test_one_rect_world(self):
        """Tests that one_rect_world creates a world with one rectangle."""
        self.arrays_close(wor.one_rect_world().device_shape_types(), [sha.RECTANGLE])

    def test_two_rect_world(self):
        """Tests that two_rect_world creates a world with two rectangles."""
        self.arrays_close(wor.two_rect_world().device_shape_types(), [sha.RECTANGLE] * 2)

    def test_mixed_world(self):
        """Tests that mixed_world creates a world with two rectangles."""
        types = wor.mixed_world().device_shape_types()
        self.assertLessEqual(set(types), set([sha.RECTANGLE, sha.SPHERE]))
        self.assertIs(len(types), 2)

if __name__ == '__main__':
    unittest.main()
