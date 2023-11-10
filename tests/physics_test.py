"""Contains tests for reinfocus.physics."""

import numpy as np
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states
from numba.cuda.testing import unittest
from reinfocus import hit_record as hit
from reinfocus import physics as phy
from reinfocus import ray
from reinfocus import rectangle as rec
from reinfocus import shape as sha
from reinfocus import sphere as sph
from reinfocus import vector as vec
from reinfocus import world as wor
from tests import numba_test_case as ntc
from tests import numba_test_utils as ntu

class PhysicsTest(ntc.NumbaTestCase):
    """TestCases for reinfocus.physics."""
    # pylint: disable=no-value-for-parameter

    def test_random_in_unit_sphere(self):
        """Tests that random_in_unit_sphere makes 3D GPU vectors in the unit sphere."""
        @cuda.jit
        def sample_from_sphere(target, random_states):
            i = cuda.grid(1) # type: ignore
            if i < target.size:
                target[i] = vec.g3f_to_c3f(phy.random_in_unit_sphere(random_states, i))

        tests = 100

        cpu_array = ntu.cpu_target(nrow=tests)

        sample_from_sphere[tests, 1]( # type: ignore
            cpu_array,
            create_xoroshiro128p_states(tests, seed=0))

        self.arrays_close(
            np.sum(np.abs(cpu_array) ** 2, axis=-1) ** .5 < 1.0, np.ones(tests))

    def test_colour_checkerboard(self):
        """Tests that colour_checkerboard produces the expected colours for different
            frequencies and positions."""
        @cuda.jit
        def checkerboard_colour_points(target, f, p):
            i = cuda.grid(1) # type: ignore
            if i < target.size:
                target[i] = vec.g3f_to_c3f(
                    phy.colour_checkerboard(vec.c2f_to_g2f(f[i]), vec.c2f_to_g2f(p[i])))

        tests = cuda.to_device(
            np.array([
                [vec.c2f(1, 1), vec.c2f(.25, .25)],
                [vec.c2f(1, 1), vec.c2f(.25, .75)],
                [vec.c2f(1, 1), vec.c2f(.75, .25)],
                [vec.c2f(1, 1), vec.c2f(.75, .75)],
                [vec.c2f(2, 2), vec.c2f(.25, .25)],
                [vec.c2f(2, 2), vec.c2f(.25, .75)],
                [vec.c2f(2, 2), vec.c2f(.75, .25)],
                [vec.c2f(2, 2), vec.c2f(.75, .75)]]))

        cpu_array = ntu.cpu_target(nrow=len(tests))

        checkerboard_colour_points[len(tests), 1]( # type: ignore
            cpu_array,
            tests[:, 0],
            tests[:, 1])

        self.arrays_close(
            cpu_array,
            [
                [1, 0, 0],
                [1, 0, 0],
                [1, 0, 0],
                [1, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [0, 1, 0],
                [1, 0, 0]])

    def test_rect_scatter(self):
        """Tests that rect_scatter scatters a ray hit in the expected way."""
        @cuda.jit
        def scatter_from_rectangle(target, random_states):
            i = cuda.grid(1) # type: ignore
            if i < target.size:
                target[i] = ntu.flatten_coloured_ray(
                    phy.rect_scatter(
                        hit.gpu_hit_record(
                            vec.g3f(0, 0, 0),
                            vec.g3f(0, 0, 1),
                            1.,
                            vec.g2f(2 ** -4, 2 ** -4),
                            sha.RECTANGLE),
                        random_states,
                        i))

        cpu_array = ntu.cpu_target(ndim=9)

        scatter_from_rectangle[1, 1]( # type: ignore
            cpu_array,
            create_xoroshiro128p_states(1, seed=0))

        self.arrays_close(cpu_array[0, 0 : 3], [0, 0, 0])
        self.assertTrue(
            np.sum(np.abs(cpu_array[0, 3 : 6] - np.array([0, 0, 1])) ** 2) ** .5 < 1.)
        self.arrays_close(cpu_array[0, 6 : 9], [1, 0, 0])

    def test_sphere_scatter(self):
        """Tests that sphere_scatter scatters a sphere hit in the expected way."""
        @cuda.jit
        def scatter_from_sphere(target, random_states):
            i = cuda.grid(1) # type: ignore
            if i < target.size:
                target[i] = ntu.flatten_coloured_ray(
                    phy.sphere_scatter(
                        hit.gpu_hit_record(
                            vec.g3f(0, 0, 1),
                            vec.g3f(0, 0, 1),
                            1.,
                            vec.g2f(2 ** -7, 2 ** -6),
                            sha.SPHERE),
                        random_states,
                        i))

        cpu_array = ntu.cpu_target(ndim=9)

        scatter_from_sphere[1, 1]( # type: ignore
            cpu_array,
            create_xoroshiro128p_states(1, seed=0))

        self.arrays_close(cpu_array[0, 0 : 3], [0, 0, 1])
        self.assertTrue(
            np.sum(np.abs(cpu_array[0, 3 : 6] - np.array([0, 0, 1])) ** 2) ** .5 < 1.)
        self.arrays_close(cpu_array[0, 6 : 9], [1, 0, 0])

    def test_scatter_with_rectangles(self):
        """Tests that scatter scatters a rectangle hit in the expected way."""
        @cuda.jit
        def scatter_with_rectangle(target, random_states):
            i = cuda.grid(1) # type: ignore
            if i < target.size:
                target[i] = ntu.flatten_coloured_ray(
                    phy.scatter(
                        hit.gpu_hit_record(
                            vec.g3f(0, 0, 0),
                            vec.g3f(0, 0, 1),
                            1.,
                            vec.g2f(2 ** -4, 2 ** -4),
                            sha.RECTANGLE),
                        random_states,
                        i))

        cpu_array = ntu.cpu_target(ndim=9)

        scatter_with_rectangle[1, 1]( # type: ignore
            cpu_array,
            create_xoroshiro128p_states(1, seed=0))

        self.arrays_close(cpu_array[0, 0 : 3], [0, 0, 0])
        self.assertTrue(
            np.sum(np.abs(cpu_array[0, 3 : 6] - np.array([0, 0, 1])) ** 2) ** .5 < 1.)
        self.arrays_close(cpu_array[0, 6 : 9], [1, 0, 0])

    def test_scatter_with_spheres(self):
        """Tests that scatter scatters a sphere hit in the expected way."""
        @cuda.jit
        def scatter_with_sphere(target, random_states):
            i = cuda.grid(1) # type: ignore
            if i < target.size:
                target[i] = ntu.flatten_coloured_ray(
                    phy.scatter(
                        hit.gpu_hit_record(
                            vec.g3f(0, 0, 1),
                            vec.g3f(0, 0, 1),
                            1.,
                            vec.g2f(2 ** -7, 2 ** -6),
                            sha.SPHERE),
                        random_states,
                        i))

        cpu_array = ntu.cpu_target(ndim=9)

        scatter_with_sphere[1, 1]( # type: ignore
            cpu_array,
            create_xoroshiro128p_states(1, seed=0))

        self.arrays_close(cpu_array[0, 0 : 3], [0, 0, 1])
        self.assertTrue(
            np.sum(np.abs(cpu_array[0, 3 : 6] - np.array([0, 0, 1])) ** 2) ** .5 < 1.)
        self.arrays_close(cpu_array[0, 6 : 9], [1, 0, 0])

    def test_find_colour_with_rectangles(self):
        """Tests that find_colour finds the expected colour when we fire a ray at a rectangle."""
        @cuda.jit
        def find_rectangle_colour(target, random_states, shapes_parameters, shapes_types):
            i = cuda.grid(1) # type: ignore
            if i < target.size:
                target[i] = vec.g3f_to_c3f(
                    phy.find_colour(
                        shapes_parameters,
                        shapes_types,
                        ray.gpu_ray(
                            vec.g3f(-2 ** -4, -2 ** -4, 0),
                            vec.g3f(-2 ** -4, -2 ** -4, 1)),
                        random_states,
                        i))

        cpu_array = ntu.cpu_target()

        world = wor.World(rec.cpu_rectangle(-1, 1, -1, 1, 1))

        find_rectangle_colour[1, 1]( # type: ignore
            cpu_array,
            create_xoroshiro128p_states(1, seed=0),
            world.device_shape_parameters(),
            world.device_shape_types())

        self.assertTrue(0 < cpu_array[0, 0] <= 1.)
        self.arrays_close(cpu_array[0, 1 : 3], [0, 0])

    def test_find_colour_with_spheres(self):
        """Tests that find_colour finds the expected colour when we fire a ray at a sphere."""
        @cuda.jit
        def find_sphere_colour(target, random_states, shapes_parameters, shapes_types):
            i = cuda.grid(1) # type: ignore
            if i < target.size:
                target[i] = vec.g3f_to_c3f(
                    phy.find_colour(
                        shapes_parameters,
                        shapes_types,
                        ray.gpu_ray(
                            vec.g3f(-2 ** -7, -2 ** -6, 0),
                            vec.g3f(-2 ** -7, -2 ** -6, 1)),
                        random_states,
                        i))

        cpu_array = ntu.cpu_target()

        world = wor.World(sph.cpu_sphere(vec.c3f(0, 0, 10), 1))

        find_sphere_colour[1, 1]( # type: ignore
            cpu_array,
            create_xoroshiro128p_states(1, seed=0),
            world.device_shape_parameters(),
            world.device_shape_types())

        self.assertTrue(0 < cpu_array[0, 0] <= 1.)
        self.arrays_close(cpu_array[0, 1 : 3], [0, 0])

if __name__ == '__main__':
    unittest.main()
