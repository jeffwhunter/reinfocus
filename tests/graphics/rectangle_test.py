"""Contains tests for reinfocus.graphics.rectangle."""

import numpy as np
from numba import cuda
from numba.cuda.testing import CUDATestCase, unittest

import tests.test_utils as tu
from reinfocus.graphics import ray
from reinfocus.graphics import shape as sha
from reinfocus.graphics import rectangle as rec
from reinfocus.graphics import vector as vec
from tests.graphics import numba_test_utils as ntu


class RectangleTest(CUDATestCase):
    # pylint: disable=no-value-for-parameter
    """TestCases for reinfocus.graphics.rectangle."""

    def test_cpu_rectangle(self):
        """Tests that cpu_rectangle makes a CPU rectangle with the expected elements."""
        tu.arrays_close(
            self,
            rec.cpu_rectangle(0, 1, 2, 3, 4, vec.c2f(5, 6)).parameters,
            [0, 1, 2, 3, 4, 5, 6],
        )

    def test_gpu_hit_rectangle(self):
        """Tests if gpu_hit_rectangle returns an appropriate hit_record for a ray hit."""

        @cuda.jit
        def hit_rectangle(target, rectangle_parameters, origin, direction):
            i = cuda.grid(1)  # type: ignore
            if i < target.size:
                target[i] = ntu.flatten_hit_result(
                    rec.gpu_hit_rectangle(
                        rectangle_parameters,
                        ray.cpu_to_gpu_ray(origin, direction),
                        0,
                        100,
                    )
                )

        cpu_array = ntu.cpu_target(ndim=13)

        hit_rectangle[1, 1](  # type: ignore
            cpu_array,
            rec.cpu_rectangle(-1, 1, -1, 1, 1, vec.c2f(4, 8)).parameters,
            vec.c3f(0, 0, 0),
            vec.c3f(0, 0, 1),
        )

        tu.arrays_close(
            self, cpu_array[0], (1, 0, 0, 1, 0, 0, 1, 1, 0.5, 0.5, 4, 8, sha.RECTANGLE)
        )

    def test_gpu_rectangle_uv(self):
        """Tests if gpu_rectangle_uv returns an appropriate texture coordinate for a point
        on some (the unit?) rectangle."""

        @cuda.jit
        def get_texture_coord(target, points):
            i = cuda.grid(1)  # type: ignore
            if i < target.size:
                target[i] = vec.g2f_to_c2f(
                    rec.gpu_rectangle_uv(vec.c2f_to_g2f(points[i]), -1, 1, -1, 1)
                )

        tests = np.array(
            [
                vec.c2f(-1, -1),
                vec.c2f(-1, 1),
                vec.c2f(1, -1),
                vec.c2f(1, 1),
                vec.c2f(0, 0),
            ]
        )

        cpu_array = ntu.cpu_target(ndim=2, nrow=len(tests))

        get_texture_coord[len(tests), 1](cpu_array, tests)  # type: ignore

        tu.arrays_close(self, cpu_array, [[0, 0], [0, 1], [1, 0], [1, 1], [0.5, 0.5]])


if __name__ == "__main__":
    unittest.main()
