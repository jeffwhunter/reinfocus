"""Contains tests for reinfocus.graphics.hit_record."""

import numpy

from numba import cuda
from numba.cuda import testing
from numba.cuda.testing import unittest

from reinfocus.graphics import cutil
from reinfocus.graphics import hit_record
from reinfocus.graphics import vector
from tests import test_utils
from tests.graphics import numba_test_utils


class HitRecordTest(testing.CUDATestCase):
    """TestCases for reinfocus.graphics.hit_record."""

    def test_empty_record(self):
        """Tests that empty_record makes an empty hit record on the GPU."""

        @cuda.jit
        def make_empty_record(target):
            i = cutil.line_index()
            if cutil.outside_shape(i, target.shape):
                return

            rec = hit_record.gpu_empty_hit_record()
            target[i] = numba_test_utils.flatten_hit_record(rec)

        cpu_array = numpy.ones((1, 12), dtype=numpy.float32)

        cutil.launcher(make_empty_record, 1)(cpu_array)

        test_utils.all_close(cpu_array[0], numpy.zeros(12))

    def test_hit_record(self):
        """Tests that hit_record makes an appropriate hit record on the GPU."""

        @cuda.jit
        def make_hit_record(target, args):
            i = cutil.line_index()
            if cutil.outside_shape(i, target.shape):
                return

            target[i] = numba_test_utils.flatten_hit_record(
                hit_record.gpu_hit_record(
                    vector.g3f(
                        args[hit_record.P][0],
                        args[hit_record.P][1],
                        args[hit_record.P][2],
                    ),
                    vector.g3f(
                        args[hit_record.N][0],
                        args[hit_record.N][1],
                        args[hit_record.N][2],
                    ),
                    args[hit_record.T],
                    vector.g2f(args[hit_record.UV][0], args[hit_record.UV][1]),
                    vector.g2f(args[hit_record.UF][0], args[hit_record.UF][1]),
                    args[hit_record.M],
                )
            )

        cpu_array = numpy.zeros((1, 12), dtype=numpy.float32)

        cutil.launcher(make_hit_record, (1, 1))(
            cpu_array,
            (
                numpy.array([0, 1, 2]),
                numpy.array([3, 4, 5]),
                6,
                numpy.array([7, 8]),
                numpy.array([9, 10]),
                11,
            ),
        )

        test_utils.all_close(cpu_array[0], range(12))


if __name__ == "__main__":
    unittest.main()
