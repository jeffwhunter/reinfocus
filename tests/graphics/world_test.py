"""Contains tests for reinfocus.graphics.world."""

import numpy as np
from numba import cuda
from numba.cuda.testing import CUDATestCase, unittest

import tests.test_utils as tu
from reinfocus.graphics import ray
from reinfocus.graphics import rectangle as rec
from reinfocus.graphics import shape as sha
from reinfocus.graphics import sphere as sph
from reinfocus.graphics import vector as vec
from reinfocus.graphics import world as wor
from tests.graphics import numba_test_utils as ntu


class WorldTest(CUDATestCase):
    # pylint: disable=no-value-for-parameter
    """TestCases for reinfocus.graphics.world."""

    def test_different_world_parameters(self):
        """Tests that World.device_shape_parameters can handle spheres and rectangles."""

        w = wor.World(
            sph.cpu_sphere(vec.c3f(1, 2, 3), 4, vec.c2f(5, 6)),
            rec.cpu_rectangle(-1, 1, -1, 1, 1, vec.c2f(4, 8)),
        )

        tu.arrays_close(
            self,
            w.device_shape_parameters(),
            np.array([[1, 2, 3, 4, 5, 6, 0], [-1, 1, -1, 1, 1, 4, 8]]),
        )

    def test_sphere_world_parameters(self):
        """Tests that World.device_shape_parameters can handle spheres."""

        w = wor.World(
            sph.cpu_sphere(vec.c3f(1, 2, 3), 4, vec.c2f(5, 6)),
            sph.cpu_sphere(vec.c3f(7, 8, 9), 10, vec.c2f(11, 12)),
        )

        tu.arrays_close(
            self,
            w.device_shape_parameters(),
            np.array([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]]),
        )

    def test_world_shape_types(self):
        """Tests that World.device_shape_types can handle spheres and rectangles."""

        w = wor.World(
            sph.cpu_sphere(vec.c3f(1, 2, 3), 4, vec.c2f(5, 6)),
            rec.cpu_rectangle(-1, 1, -1, 1, 1, vec.c2f(4, 8)),
        )

        tu.arrays_close(
            self, w.device_shape_types(), np.array([sha.SPHERE, sha.RECTANGLE])
        )

    def test_gpu_hit_sphere_world(self):
        """Tests if gpu_hit_world returns an appropriate hit_record for spheres."""

        @cuda.jit
        def hit_sphere_world(target, shapes_parameters, shapes_types, origin, direction):
            i = cuda.grid(1)  # type: ignore
            if i < target.size:
                target[i] = ntu.flatten_hit_result(
                    wor.gpu_hit_world(
                        shapes_parameters,
                        shapes_types,
                        ray.cpu_to_gpu_ray(origin, direction),
                        0,
                        100,
                    )
                )

        cpu_array = ntu.cpu_target(ndim=13)

        world = wor.World(sph.cpu_sphere(vec.c3f(0, 0, 0), 1, vec.c2f(4, 8)))

        hit_sphere_world[1, 1](  # type: ignore
            cpu_array,
            world.device_shape_parameters(),
            world.device_shape_types(),
            vec.c3f(10, 0, 0),
            vec.c3f(-1, 0, 0),
        )

        tu.arrays_close(
            self, cpu_array[0], (1, 1, 0, 0, 1, 0, 0, 9, 1, 0.5, 4, 8, sha.SPHERE)
        )

    def test_gpu_hit_rectangle_world(self):
        """Tests if gpu_hit_world returns an appropriate hit_record for rectangles."""

        @cuda.jit
        def hit_rectangle_world(
            target, shapes_parameters, shapes_types, origin, direction
        ):
            i = cuda.grid(1)  # type: ignore
            if i < target.size:
                target[i] = ntu.flatten_hit_result(
                    wor.gpu_hit_world(
                        shapes_parameters,
                        shapes_types,
                        ray.cpu_to_gpu_ray(origin, direction),
                        0,
                        100,
                    )
                )

        cpu_array = ntu.cpu_target(ndim=13)

        world = wor.World(rec.cpu_rectangle(-1, 1, -1, 1, 1, vec.c2f(4, 8)))

        hit_rectangle_world[1, 1](  # type: ignore
            cpu_array,
            world.device_shape_parameters(),
            world.device_shape_types(),
            vec.c3f(0, 0, 0),
            vec.c3f(0, 0, 1),
        )

        tu.arrays_close(
            self, cpu_array[0], (1, 0, 0, 1, 0, 0, 1, 1, 0.5, 0.5, 4, 8, sha.RECTANGLE)
        )

    def test_one_sphere_world(self):
        """Tests that one_sphere_world creates a world with one sphere."""

        tu.arrays_close(self, wor.one_sphere_world().device_shape_types(), [sha.SPHERE])

    def test_two_sphere_world(self):
        """Tests that two_sphere_world creates a world with two spheres."""

        tu.arrays_close(
            self, wor.two_sphere_world().device_shape_types(), [sha.SPHERE] * 2
        )

    def test_one_rect_world(self):
        """Tests that one_rect_world creates a world with one rectangle."""

        tu.arrays_close(self, wor.one_rect_world().device_shape_types(), [sha.RECTANGLE])

    def test_two_rect_world(self):
        """Tests that two_rect_world creates a world with two rectangles."""

        tu.arrays_close(
            self, wor.two_rect_world().device_shape_types(), [sha.RECTANGLE] * 2
        )

    def test_mixed_world(self):
        """Tests that mixed_world creates a world with two rectangles."""

        types = wor.mixed_world().device_shape_types()
        self.assertLessEqual(set(types), set([sha.RECTANGLE, sha.SPHERE]))
        self.assertIs(len(types), 2)


if __name__ == "__main__":
    unittest.main()
