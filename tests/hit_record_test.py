# pylint: disable=no-member
# type: ignore

"""Contains tests for reinfocus.hit_record."""

import numpy as np

import numba as nb

from numba import cuda
from numba.cuda.testing import unittest

from reinfocus import hit_record as hr
from tests import numba_test_case as ntc
from tests import numba_test_utils as ntu

@cuda.jit()
def flatten_hit_record(hit_record):
    """Flattens a hit_record into a tuple."""
    p = hit_record[hr.P]
    n = hit_record[hr.N]
    uv = hit_record[hr.UV]
    return (p.x, p.y, p.z, n.x, n.y, n.z, hit_record[hr.T], uv.x, uv.y, hit_record[hr.M])

class HitRecordTest(ntc.NumbaTestCase):
    """TestCases for reinfocus.hit_record."""
    # pylint: disable=no-value-for-parameter

    def test_empty_record(self):
        """Tests that empty_record makes an empty hit record on the GPU."""
        @cuda.jit()
        def make_empty_record(target):
            i = cuda.grid(1) # type: ignore
            if i < target.size:
                rec = hr.gpu_empty_hit_record()
                target[i] = flatten_hit_record(rec)

        cpu_array = ntu.cpu_target(ndim=10)

        make_empty_record[1, 1]( # type: ignore
            cpu_array)

        self.arrays_close(cpu_array[0], (0, 0, 0, 0, 0, 0, 0, 0, 0, 0))

    def test_hit_record(self):
        """Tests that hit_record makes an appropriate hit record on the GPU."""
        @cuda.jit()
        def make_hit_record(target, args):
            i = cuda.grid(1) # type: ignore
            if i < target.size:
                target[i] = flatten_hit_record(
                    hr.gpu_hit_record(
                        cuda.float32x3(args[hr.P][0], args[hr.P][1], args[hr.P][2]),
                        cuda.float32x3(args[hr.N][0], args[hr.N][1], args[hr.N][2]),
                        nb.float32(args[hr.T]),
                        cuda.float32x2(args[hr.UV][0], args[hr.UV][1]),
                        nb.float32(args[hr.M])))

        cpu_array = ntu.cpu_target(ndim=10)

        make_hit_record[1, 1]( # type: ignore
            cpu_array,
            (np.array([0, 1, 2]), np.array([3, 4, 5]), 6, np.array([7, 8]), 9))

        self.arrays_close(cpu_array[0], (0, 1, 2, 3, 4, 5, 6, 7, 8, 9))

if __name__ == '__main__':
    unittest.main()
