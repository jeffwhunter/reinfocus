"""Contains tests for reinfocus.graphics.vector."""

import math

import numpy

from numba import cuda
from numba.cuda import testing
from numba.cuda.testing import unittest

from reinfocus.graphics import cutil
from reinfocus.graphics import vector
from tests import test_utils


class VectorTest(testing.CUDATestCase):
    # pylint: disable=too-many-public-methods
    """TestCases for reinfocus.graphics.vector."""

    def test_v2f(self):
        """Tests that v2f makes a vector with the expected elements."""

        test_utils.all_close(vector.v2f(1, 2), vector.v2f(1, 2))
        test_utils.differ(vector.v2f(1, 2), vector.v2f(1, 3))

    def test_v3f(self):
        """Tests that v3f makes a vector with the expected elements."""

        test_utils.all_close(vector.v3f(1, 2, 3), vector.v3f(1, 2, 3))
        test_utils.differ(vector.v3f(1, 2, 3), vector.v3f(1, 2, 4))

    def test_add_v3f(self):
        """Tests that add_v3f properly adds three vectors."""

        test_utils.all_close(
            vector.add_v3f(
                (vector.v3f(1, 2, 3), vector.v3f(4, 5, 6), vector.v3f(7, 8, 9))
            ),
            vector.v3f(12, 15, 18),
        )

    def test_sub_v3f(self):
        """Tests that sub_v3f properly subtracts one vector from another."""

        test_utils.all_close(
            vector.sub_v3f(vector.v3f(4, 5, 6), vector.v3f(3, 2, 1)),
            vector.v3f(1, 3, 5),
        )

    def test_smul_v3f(self):
        """Tests that smul_v3f properly multiplies a vector by a scalar."""

        test_utils.all_close(vector.smul_v3f(vector.v3f(1, 2, 3), 3), vector.v3f(3, 6, 9))

    def test_cross_v3f(self):
        """Tests that cross_v3f properly produces the cross product of two vectors."""

        test_utils.all_close(
            vector.cross_v3f(vector.v3f(1, 2, 3), vector.v3f(4, 5, 6)),
            vector.v3f(-3, 6, -3),
        )

    def test_length_v3f(self):
        """Tests that length_v3f properly produces the length of a vector."""

        self.assertAlmostEqual(vector.length_v3f(vector.v3f(2, 3, 6)), 7)

    def test_norm_v3f(self):
        """Tests that norm_v3f properly normalizes a vector."""

        test_utils.all_close(
            vector.norm_v3f(vector.v3f(1, -1, 2)),
            vector.v3f(1 / math.sqrt(6), -1 / math.sqrt(6), math.sqrt(2 / 3)),
        )

    def test_d_v2f(self):
        """Tests that d_v2f makes a 2D vector with the expected elements."""

        @cuda.jit
        def make_v2f(target, a, b):
            i = cutil.line_index()
            if cutil.outside_shape(i, target.shape):
                return

            target[i] = vector.d_v2f(a, b)

        cpu_array = numpy.zeros((1, 2), dtype=numpy.float32)

        cutil.launcher(make_v2f, 1)(cpu_array, 2, 3)

        test_utils.all_close(cpu_array[0], vector.v2f(2, 3))

    def test_d_v3f(self):
        """Tests that d_v3f makes a vector with the expected elements."""

        @cuda.jit
        def make_v3f(target, a, b, c):
            i = cutil.line_index()
            if cutil.outside_shape(i, target.shape):
                return

            target[i] = vector.d_v3f(a, b, c)

        cpu_array = numpy.zeros((1, 3), dtype=numpy.float32)

        cutil.launcher(make_v3f, 1)(cpu_array, 2, 3, 5)

        test_utils.all_close(cpu_array[0], vector.v3f(2, 3, 5))

    def test_d_v3f_to_v3ui(self):
        """tests that d_v3f_to_v3ui properly casts a vector."""

        @cuda.jit
        def cast_to_unsigned(target, a, b, c):
            i = cutil.line_index()
            if cutil.outside_shape(i, target.shape):
                return

            target[i] = vector.d_v3f_to_v3ui(vector.d_v3f(a, b, c))

        cpu_array = numpy.zeros((1, 3), dtype=numpy.uint8)

        cutil.launcher(cast_to_unsigned, 1)(cpu_array, 2, 3, 5)

        test_utils.all_close(cpu_array[0], (2, 3, 5))

    def test_d_add_v3f(self):
        """Tests that d_add_v3f properly adds vectors."""

        @cuda.jit
        def add_v3f(target, a, b, c):
            i = cutil.line_index()
            if cutil.outside_shape(i, target.shape):
                return

            target[i] = vector.d_add_v3f((a, b, c))

        cpu_array = numpy.zeros((1, 3), dtype=numpy.float32)

        cutil.launcher(add_v3f, 1)(
            cpu_array, vector.v3f(1, 2, 3), vector.v3f(4, 5, 6), vector.v3f(7, 8, 9)
        )

        test_utils.all_close(cpu_array[0], vector.v3f(12, 15, 18))

    def test_d_sub_v2f(self):
        """Tests that d_sub_v2f properly subtracts one vector from another."""

        @cuda.jit
        def subtract_v2f(target, a, b):
            i = cutil.line_index()
            if cutil.outside_shape(i, target.shape):
                return

            target[i] = vector.d_sub_v2f(a, b)

        cpu_array = numpy.zeros((1, 2), dtype=numpy.float32)

        cutil.launcher(subtract_v2f, 1)(cpu_array, vector.v2f(3, 4), vector.v2f(2, 1))

        test_utils.all_close(cpu_array[0], vector.v2f(1, 3))

    def test_d_sub_v3f(self):
        """Tests that d_sub_v3f properly subtracts one vector from another."""

        @cuda.jit
        def subtract_v3f(target, a, b):
            i = cutil.line_index()
            if cutil.outside_shape(i, target.shape):
                return

            target[i] = vector.d_sub_v3f(a, b)

        cpu_array = numpy.zeros((1, 3), dtype=numpy.float32)

        cutil.launcher(subtract_v3f, 1)(
            cpu_array, vector.v3f(4, 5, 6), vector.v3f(3, 2, 1)
        )

        test_utils.all_close(cpu_array[0], vector.v3f(1, 3, 5))

    def test_d_smul_v2f(self):
        """Tests that d_smul_v2f properly multiplies a vector by a scalar."""

        @cuda.jit
        def scale_v2f(target, a, b):
            i = cutil.line_index()
            if cutil.outside_shape(i, target.shape):
                return

            target[i] = vector.d_smul_v2f(a, b)

        cpu_array = numpy.zeros((1, 2), dtype=numpy.float32)

        cutil.launcher(scale_v2f, 1)(cpu_array, vector.v2f(1, 2), 3)

        test_utils.all_close(cpu_array[0], vector.v2f(3, 6))

    def test_d_smul_v3f(self):
        """Tests that d_smul_v3f properly multiplies a vector by a scalar."""

        @cuda.jit
        def scale_v3f(target, a, b):
            i = cutil.line_index()
            if cutil.outside_shape(i, target.shape):
                return

            target[i] = vector.d_smul_v3f(a, b)

        cpu_array = numpy.zeros((1, 3), dtype=numpy.float32)

        cutil.launcher(scale_v3f, 1)(cpu_array, vector.v3f(1, 2, 3), 3)

        test_utils.all_close(cpu_array[0], vector.v3f(3, 6, 9))

    def test_d_vmul_v3f(self):
        """Tests that d_vmul_v3f properly produces the Hadamard product of two vectors."""

        @cuda.jit
        def elementwise_multiply_g3f(target, a, b):
            i = cutil.line_index()
            if cutil.outside_shape(i, target.shape):
                return

            target[i] = vector.d_vmul_v3f(a, b)

        cpu_array = numpy.zeros((1, 3), dtype=numpy.float32)

        cutil.launcher(elementwise_multiply_g3f, 1)(
            cpu_array, vector.v3f(1, 2, 3), vector.v3f(1, 2, 3)
        )

        test_utils.all_close(cpu_array[0], vector.v3f(1, 4, 9))

    def test_d_dot_v2f(self):
        """Tests that d_dot_v2f properly produces the dot product of two vectors."""

        @cuda.jit
        def dot_v2f(target, a, b):
            i = cutil.line_index()
            if cutil.outside_shape(i, target.shape):
                return

            target[i] = vector.d_dot_v2f(a, b)

        cpu_array = numpy.zeros(1, dtype=numpy.float32)

        cutil.launcher(dot_v2f, 1)(cpu_array, vector.v2f(2, 3), vector.v2f(4, 5))

        self.assertAlmostEqual(cpu_array[0], 23)

    def test_d_dot_v3f(self):
        """Tests that d_dot_v3f properly produces the dot product of two vectors."""

        @cuda.jit
        def dot_v3f(target, a, b):
            i = cutil.line_index()
            if cutil.outside_shape(i, target.shape):
                return

            target[i] = vector.d_dot_v3f(a, b)

        cpu_array = numpy.zeros(1, dtype=numpy.float32)

        cutil.launcher(dot_v3f, 1)(cpu_array, vector.v3f(1, 2, 3), vector.v3f(4, 5, 6))

        self.assertAlmostEqual(cpu_array[0], 32)

    def test_d_cross_v3f(self):
        """Tests that d_cross_v3f properly produces the cross product of two vectors."""

        @cuda.jit
        def cross_v3f(target, a, b):
            i = cutil.line_index()
            if cutil.outside_shape(i, target.shape):
                return

            target[i] = vector.d_cross_v3f(a, b)

        cpu_array = numpy.zeros((1, 3), dtype=numpy.float32)

        cutil.launcher(cross_v3f, 1)(cpu_array, vector.v3f(1, 2, 3), vector.v3f(4, 5, 6))

        test_utils.all_close(cpu_array[0], vector.v3f(-3, 6, -3))

    def test_d_squared_lengtv3f(self):
        """Tests that d_squared_length_v3f properly produces the squared length of a
        vector."""

        @cuda.jit
        def v3f_squared_length(target, a):
            i = cutil.line_index()
            if cutil.outside_shape(i, target.shape):
                return

            target[i] = vector.d_squared_length_v3f(a)

        cpu_array = numpy.zeros(1, dtype=numpy.float32)

        cutil.launcher(v3f_squared_length, 1)(cpu_array, vector.v3f(1, 2, 3))

        self.assertAlmostEqual(cpu_array[0], 14)

    def test_d_length_v3f(self):
        """Tests that d_length_v3f properly produces the length of a vector."""

        @cuda.jit
        def v3f_length(target, a):
            i = cutil.line_index()
            if cutil.outside_shape(i, target.shape):
                return

            target[i] = vector.d_length_v3f(a)

        cpu_array = numpy.zeros(1, dtype=numpy.float32)

        cutil.launcher(v3f_length, 1)(cpu_array, vector.v3f(2, 3, 6))

        self.assertAlmostEqual(cpu_array[0], 7)

    def test_d_norm_v3f(self):
        """Tests that d_norm_v3f properly normalizes a vector."""

        @cuda.jit
        def v3f_norm(target, a):
            i = cutil.line_index()
            if cutil.outside_shape(i, target.shape):
                return

            target[i] = vector.d_norm_v3f(a)

        cpu_array = numpy.zeros((1, 3), dtype=numpy.float32)

        cutil.launcher(v3f_norm, 1)(cpu_array, vector.v3f(1, -1, 2))

        test_utils.all_close(
            cpu_array[0],
            vector.v3f(1 / math.sqrt(6), -1 / math.sqrt(6), math.sqrt(2 / 3)),
        )


if __name__ == "__main__":
    unittest.main()
