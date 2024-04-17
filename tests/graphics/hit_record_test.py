"""Contains tests for reinfocus.graphics.hit_record."""

import numpy

from numba import cuda
from numba.cuda import testing as cuda_testing
from numba.cuda.testing import unittest
from numpy import testing as numpy_testing

from reinfocus.graphics import cutil
from reinfocus.graphics import hit_record
from reinfocus.graphics import vector
from tests.graphics import numba_test_utils


class EmptyHitRecordTest(cuda_testing.CUDATestCase):
    """Test cases for reinfocus.graphics.hit_record.empty_hit_record."""

    def test_empty(self):
        """Tests that empty_record makes an empty hit record."""

        @cuda.jit
        def make_empty_record(target):
            i = cutil.line_index()
            if cutil.outside_shape(i, target.shape):
                return

            rec = hit_record.empty_hit_record()
            target[i] = numba_test_utils.flatten_hit_record(rec)

        cpu_array = numpy.ones((1, 12), dtype=numpy.float32)

        cutil.launcher(make_empty_record, 1)(cpu_array)

        numpy_testing.assert_allclose(cpu_array[0], numpy.zeros(12))


class HitRecordTest(cuda_testing.CUDATestCase):
    """Test cases for reinfocus.graphics.hit_record.hit_record."""

    def test_expected(self):
        """Tests that hit_record makes an appropriate hit record."""

        @cuda.jit
        def make_hit_record(target, args):
            i = cutil.line_index()
            if cutil.outside_shape(i, target.shape):
                return

            target[i] = numba_test_utils.flatten_hit_record(
                hit_record.hit_record(
                    vector.d_v3f(
                        args[hit_record.P][0],
                        args[hit_record.P][1],
                        args[hit_record.P][2],
                    ),
                    vector.d_v3f(
                        args[hit_record.N][0],
                        args[hit_record.N][1],
                        args[hit_record.N][2],
                    ),
                    args[hit_record.T],
                    vector.d_v2f(args[hit_record.UV][0], args[hit_record.UV][1]),
                    vector.d_v2f(args[hit_record.UF][0], args[hit_record.UF][1]),
                    args[hit_record.M],
                )
            )

        cpu_array = numpy.zeros((1, 12), dtype=numpy.float32)

        cutil.launcher(make_hit_record, (1, 1))(
            cpu_array,
            (
                numpy.array([0, 1, 2]),
                numpy.array([3, 4, 5]),
                numpy.float32(6),
                numpy.array([7, 8]),
                numpy.array([9, 10]),
                numpy.float32(11),
            ),
        )

        numpy_testing.assert_allclose(cpu_array[0], range(12))


if __name__ == "__main__":
    unittest.main()
