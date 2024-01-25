"""Contains tests for reinfocus.graphics.ray."""

from numba import cuda
from numba.cuda.testing import CUDATestCase, unittest

import tests.test_utils as tu
from reinfocus.graphics import ray
from reinfocus.graphics import vector as vec
from tests.graphics import numba_test_utils as ntu


class RayTest(CUDATestCase):
    # pylint: disable=no-value-for-parameter
    """TestCases for reinfocus.graphics.ray."""

    def test_gpu_ray(self):
        """Tests that gpu_ray constructs a GPU ray with the expected origin and
        direction."""

        @cuda.jit
        def copy_gpu_ray(target, origin, direction):
            i = cuda.grid(1)  # type: ignore
            if i < target.size:
                target[i] = ntu.flatten_ray(ray.cpu_to_gpu_ray(origin, direction))

        cpu_array = ntu.cpu_target(ndim=6)

        copy_gpu_ray[1, 1](cpu_array, vec.c3f(1, 2, 3), vec.c3f(4, 5, 6))  # type: ignore

        tu.arrays_close(self, cpu_array[0], vec.c3f(1, 2, 3) + vec.c3f(4, 5, 6))

    def test_gpu_point_at_parameter(self):
        """Tests that gpu_point_at_parameter correctly finds the point t distance along
        ray."""

        @cuda.jit
        def find_gpu_point_at_parameter(target, origin, direction, t):
            i = cuda.grid(1)  # type: ignore
            if i < target.size:
                target[i] = vec.g3f_to_c3f(
                    ray.gpu_point_at_parameter(ray.cpu_to_gpu_ray(origin, direction), t)
                )

        cpu_array = ntu.cpu_target()

        find_gpu_point_at_parameter[1, 1](  # type: ignore
            cpu_array, vec.c3f(1, 2, 3), vec.c3f(4, 5, 6), 2
        )

        tu.arrays_close(self, cpu_array[0], vec.c3f(9, 12, 15))


if __name__ == "__main__":
    unittest.main()
