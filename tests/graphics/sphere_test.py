"""Contains tests for reinfocus.sphere."""

from numba import cuda
from numba.cuda.testing import unittest
from reinfocus.graphics import ray
from reinfocus.graphics import shape as sha
from reinfocus.graphics import sphere as sph
from reinfocus.graphics import vector as vec
from tests.graphics import numba_test_case as ntc
from tests.graphics import numba_test_utils as ntu

class SphereTest(ntc.NumbaTestCase):
    """TestCases for reinfocus.sphere."""
    # pylint: disable=no-value-for-parameter

    def test_cpu_sphere(self):
        """Tests that cpu_sphere makes a CPU sphere with the expected elements."""
        self.arrays_close(sph.cpu_sphere(vec.c3f(1, 2, 3), 4).parameters, [1, 2, 3, 4])

    def test_gpu_hit_sphere(self):
        """Tests if gpu_hit_sphere returns an appropriate hit_record for a ray hit."""
        @cuda.jit
        def hit_sphere(target, sphere_parameters, origin, direction):
            i = cuda.grid(1) # type: ignore
            if i < target.size:
                target[i] = ntu.flatten_hit_result(
                    sph.gpu_hit_sphere(
                        sphere_parameters,
                        ray.cpu_to_gpu_ray(origin, direction),
                        0,
                        100))

        cpu_array = ntu.cpu_target(ndim=11)

        hit_sphere[1, 1]( # type: ignore
            cpu_array,
            sph.cpu_sphere(vec.c3f(0, 0, 0), 1).parameters,
            vec.c3f(10, 0, 0),
            vec.c3f(-1, 0, 0))

        self.arrays_close(cpu_array[0], (1, 1, 0, 0, 1, 0, 0, 9, .5, .5, sha.SPHERE))

    def test_gpu_sphere_uv(self):
        """Tests if gpu_sphere_uv returns an appropriate texture coordinate for a point
            on the unit sphere."""
        @cuda.jit
        def get_texture_coord(target, point):
            i = cuda.grid(1) # type: ignore
            if i < target.size:
                target[i] = vec.g2f_to_c2f(sph.gpu_sphere_uv(vec.c3f_to_g3f(point)))

        cpu_array = ntu.cpu_target(ndim=2)

        get_texture_coord[1, 1]( # type: ignore
            cpu_array,
            vec.c3f(-1, 0, 0))

        self.arrays_close(cpu_array[0], (0., .5))

if __name__ == '__main__':
    unittest.main()
