"""Contains tests for reinfocus.rectangle."""

import numpy as np
from numba import cuda
from numba.cuda.testing import unittest
from reinfocus import ray
from reinfocus import shape as sha
from reinfocus import rectangle as rec
from reinfocus import vector as vec
from tests import numba_test_case as ntc
from tests import numba_test_utils as ntu

class RectangleTest(ntc.NumbaTestCase):
    """TestCases for reinfocus.rectangle."""
    # pylint: disable=no-value-for-parameter

    def test_cpu_rectangle(self):
        """Tests that cpu_rectangle makes a CPU rectangle with the expected elements."""
        self.arrays_close(rec.cpu_rectangle(0, 1, 2, 3, 4).parameters, [0, 1, 2, 3, 4])

    def test_gpu_hit_rectangle(self):
        """Tests if gpu_hit_rectangle returns an appropriate hit_record for a ray hit."""
        @cuda.jit
        def hit_rectangle(target, rectangle_parameters, origin, direction):
            i = cuda.grid(1) # type: ignore
            if i < target.size:
                target[i] = ntu.flatten_hit_result(
                    rec.gpu_hit_rectangle(
                        rectangle_parameters,
                        ray.cpu_to_gpu_ray(origin, direction),
                        0,
                        100))

        cpu_array = ntu.cpu_target(ndim=11)

        hit_rectangle[1, 1]( # type: ignore
            cpu_array,
            rec.cpu_rectangle(-1, 1, -1, 1, 1).parameters,
            vec.c3f(0, 0, 0),
            vec.c3f(0, 0, 1))

        self.arrays_close(cpu_array[0], (1, 0, 0, 1, 0, 0, 1, 1, .5, .5, sha.RECTANGLE))

    def test_gpu_rectangle_uv(self):
        """Tests if gpu_rectangle_uv returns an appropriate texture coordinate for a point
            on some (the unit?) rectangle."""
        @cuda.jit
        def get_texture_coord(target, points):
            i = cuda.grid(1) # type: ignore
            if i < target.size:
                target[i] = vec.g2f_to_c2f(
                    rec.gpu_rectangle_uv(vec.c2f_to_g2f(points[i]), -1, 1, -1, 1))

        tests = np.array(
            [vec.c2f(-1, -1), vec.c2f(-1, 1), vec.c2f(1, -1), vec.c2f(1, 1), vec.c2f(0, 0)])

        cpu_array = ntu.cpu_target(ndim=2, nrow=len(tests))

        get_texture_coord[len(tests), 1]( # type: ignore
            cpu_array,
            tests)

        self.arrays_close(cpu_array, [[0, 0], [0, 1], [1, 0], [1, 1], [.5, .5]])

if __name__ == '__main__':
    unittest.main()
