# pylint: disable=no-member

"""Contains tests for reinfocus.graphics.hit_record."""

import numpy as np

from numba import cuda
from numba.cuda.testing import CUDATestCase, unittest

import tests.test_utils as tu
from reinfocus.graphics import hit_record as hit
from reinfocus.graphics import vector as vec
from tests.graphics import numba_test_utils as ntu

class HitRecordTest(CUDATestCase):
    """TestCases for reinfocus.graphics.hit_record."""
    # pylint: disable=no-value-for-parameter

    def test_empty_record(self):
        """Tests that empty_record makes an empty hit record on the GPU."""
        @cuda.jit
        def make_empty_record(target):
            i = cuda.grid(1) # type: ignore
            if i < target.size:
                rec = hit.gpu_empty_hit_record()
                target[i] = ntu.flatten_hit_record(rec)

        cpu_array = ntu.cpu_target(ndim=10)

        make_empty_record[1, 1]( # type: ignore
            cpu_array)

        tu.arrays_close(self, cpu_array[0], (0, 0, 0, 0, 0, 0, 0, 0, 0, 0))

    def test_hit_record(self):
        """Tests that hit_record makes an appropriate hit record on the GPU."""
        @cuda.jit
        def make_hit_record(target, args):
            i = cuda.grid(1) # type: ignore
            if i < target.size:
                target[i] = ntu.flatten_hit_record(
                    hit.gpu_hit_record(
                        vec.g3f(args[hit.P][0], args[hit.P][1], args[hit.P][2]),
                        vec.g3f(args[hit.N][0], args[hit.N][1], args[hit.N][2]),
                        args[hit.T],
                        vec.g2f(args[hit.UV][0], args[hit.UV][1]),
                        args[hit.M]))

        cpu_array = ntu.cpu_target(ndim=10)

        make_hit_record[1, 1]( # type: ignore
            cpu_array,
            (np.array([0, 1, 2]), np.array([3, 4, 5]), 6, np.array([7, 8]), 9))

        tu.arrays_close(self, cpu_array[0], (0, 1, 2, 3, 4, 5, 6, 7, 8, 9))

if __name__ == '__main__':
    unittest.main()
