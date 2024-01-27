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

    def test_c3f(self):
        """Tests that c3f makes a CPU vector with the expected elements."""

        test_utils.arrays_close(self, vector.c3f(1, 2, 3), vector.c3f(1, 2, 3))
        test_utils.arrays_not_close(self, vector.c3f(1, 2, 3), vector.c3f(2, 2, 2))

    def test_add3_c3f(self):
        """Tests that add3_c3f properly adds three CPU vectors."""

        test_utils.arrays_close(
            self,
            vector.add3_c3f(
                vector.c3f(1, 2, 3), vector.c3f(4, 5, 6), vector.c3f(7, 8, 9)
            ),
            vector.c3f(12, 15, 18),
        )

    def test_sub_c3f(self):
        """Tests that sub_c3f properly subtracts one CPU vector from another."""

        test_utils.arrays_close(
            self,
            vector.sub_c3f(vector.c3f(4, 5, 6), vector.c3f(3, 2, 1)),
            vector.c3f(1, 3, 5),
        )

    def test_smul_c3f(self):
        """Tests that smul_c3f properly multiplies a CPU vector by a scalar."""

        test_utils.arrays_close(
            self, vector.smul_c3f(vector.c3f(1, 2, 3), 3), vector.c3f(3, 6, 9)
        )

    def test_div_c3f(self):
        """Tests that div_c3f properly divides a CPU vector by a scalar."""

        test_utils.arrays_close(
            self, vector.div_c3f(vector.c3f(3, 6, 9), 3), vector.c3f(1, 2, 3)
        )

    def test_cross_c3f(self):
        """Tests that cross_c3f properly produces the cross product of two CPU
        vectors."""

        test_utils.arrays_close(
            self,
            vector.cross_c3f(vector.c3f(1, 2, 3), vector.c3f(4, 5, 6)),
            vector.c3f(-3, 6, -3),
        )

    def test_length_c3f(self):
        """Tests that length_c3f properly produces the length of a CPU vector."""

        self.assertAlmostEqual(vector.length_c3f(vector.c3f(2, 3, 6)), 7)

    def test_norm_c3f(self):
        """Tests that norm_c3f properly normalizes a CPU vector."""

        test_utils.arrays_close(
            self,
            vector.norm_c3f(vector.c3f(1, -1, 2)),
            vector.c3f(1 / math.sqrt(6), -1 / math.sqrt(6), math.sqrt(2 / 3)),
        )

    def test_empty_g2f(self):
        """Tests that g2f makes a 2D GPU vector with the expected elements."""

        @cuda.jit
        def copy_empty_g2f(target):
            i = cutil.line_index()
            if cutil.outside_shape(i, target.shape):
                return

            target[i] = vector.g2f_to_c2f(vector.empty_g2f())

        cpu_array = numpy.zeros((1, 2), dtype=numpy.float32)

        cutil.launcher(copy_empty_g2f, 1)(cpu_array)

        test_utils.arrays_close(self, cpu_array[0], vector.c2f(0, 0))

    def test_g2f(self):
        """Tests that g2f makes a 2D GPU vector with the expected elements."""

        @cuda.jit
        def copy_g2f(target, source):
            i = cutil.line_index()
            if cutil.outside_shape(i, target.shape):
                return

            target[i] = vector.g2f_to_c2f(vector.g2f(*source))

        cpu_array = numpy.zeros((1, 2), dtype=numpy.float32)

        cutil.launcher(copy_g2f, 1)(cpu_array, vector.c2f(1, 2))

        test_utils.arrays_close(self, cpu_array[0], vector.c2f(1, 2))

    def test_empty_g3f(self):
        """Tests that g3f makes an empty 3D GPU vector."""

        @cuda.jit
        def copy_empty_g3f(target):
            i = cutil.line_index()
            if cutil.outside_shape(i, target.shape):
                return

            target[i] = vector.g3f_to_c3f(vector.empty_g3f())

        cpu_array = numpy.zeros((1, 3), dtype=numpy.float32)

        cutil.launcher(copy_empty_g3f, 1)(cpu_array)

        test_utils.arrays_close(self, cpu_array[0], vector.c3f(0, 0, 0))

    def test_g3f(self):
        """Tests that g3f makes a GPU vector with the expected elements."""

        @cuda.jit
        def copy_g3f(target, source):
            i = cutil.line_index()
            if cutil.outside_shape(i, target.shape):
                return

            target[i] = vector.g3f_to_c3f(vector.g3f(*source))

        cpu_array = numpy.zeros((1, 3), dtype=numpy.float32)

        cutil.launcher(copy_g3f, 1)(cpu_array, vector.c3f(1, 2, 3))

        test_utils.arrays_close(self, cpu_array[0], vector.c3f(1, 2, 3))

    def test_c3f_to_g3f(self):
        """Tests that c3f_to_g3f makes a GPU vector with elements from the CPU vector."""

        @cuda.jit
        def copy_g3f_from_c3f(target, source):
            i = cutil.line_index()
            if cutil.outside_shape(i, target.shape):
                return

            target[i] = vector.g3f_to_c3f(vector.c3f_to_g3f(source))

        cpu_array = numpy.zeros((1, 3), dtype=numpy.float32)

        cutil.launcher(copy_g3f_from_c3f, 1)(cpu_array, vector.c3f(1, 2, 3))

        test_utils.arrays_close(self, cpu_array[0], vector.c3f(1, 2, 3))

    def test_add_g3f(self):
        """Tests that add_g3f properly adds two GPU vectors."""

        @cuda.jit
        def add_2_g3f(target, a, b):
            i = cutil.line_index()
            if cutil.outside_shape(i, target.shape):
                return

            target[i] = vector.g3f_to_c3f(vector.add_g3f(vector.g3f(*a), vector.g3f(*b)))

        cpu_array = numpy.zeros((1, 3), dtype=numpy.float32)

        cutil.launcher(add_2_g3f, 1)(cpu_array, vector.c3f(1, 2, 3), vector.c3f(4, 5, 6))

        test_utils.arrays_close(self, cpu_array[0], vector.c3f(5, 7, 9))

    def test_add3_g3f(self):
        """Tests that add3_g3f properly adds three GPU vectors."""

        @cuda.jit
        def add_3_g3f(target, a, b, c):
            i = cutil.line_index()
            if cutil.outside_shape(i, target.shape):
                return

            target[i] = vector.g3f_to_c3f(
                vector.add3_g3f(vector.g3f(*a), vector.g3f(*b), vector.g3f(*c))
            )

        cpu_array = numpy.zeros((1, 3), dtype=numpy.float32)

        cutil.launcher(add_3_g3f, 1)(
            cpu_array, vector.c3f(1, 2, 3), vector.c3f(4, 5, 6), vector.c3f(7, 8, 9)
        )

        test_utils.arrays_close(self, cpu_array[0], vector.c3f(12, 15, 18))

    def test_neg_g3f(self):
        """Tests that neg_g3f properly negates a GPU vector."""

        @cuda.jit
        def negate_g3f(target, a):
            i = cutil.line_index()
            if cutil.outside_shape(i, target.shape):
                return

            target[i] = vector.g3f_to_c3f(vector.neg_g3f(vector.g3f(*a)))

        cpu_array = numpy.zeros((1, 3), dtype=numpy.float32)

        cutil.launcher(negate_g3f, 1)(cpu_array, vector.c3f(1, -2, 3))

        test_utils.arrays_close(self, cpu_array[0], vector.c3f(-1, 2, -3))

    def test_sub_g3f(self):
        """Tests that sub_g3f properly subtracts one GPU vector from another."""

        @cuda.jit
        def subtract_g3f(target, a, b):
            i = cutil.line_index()
            if cutil.outside_shape(i, target.shape):
                return

            target[i] = vector.g3f_to_c3f(vector.sub_g3f(vector.g3f(*a), vector.g3f(*b)))

        cpu_array = numpy.zeros((1, 3), dtype=numpy.float32)

        cutil.launcher(subtract_g3f, 1)(
            cpu_array, vector.c3f(4, 5, 6), vector.c3f(3, 2, 1)
        )

        test_utils.arrays_close(self, cpu_array[0], vector.c3f(1, 3, 5))

    def test_smul_g3f(self):
        """Tests that smul_g3f properly multiplies a GPU vector by a scalar."""

        @cuda.jit
        def scale_g3f(target, a, b):
            i = cutil.line_index()
            if cutil.outside_shape(i, target.shape):
                return

            target[i] = vector.g3f_to_c3f(vector.smul_g3f(vector.g3f(*a), b))

        cpu_array = numpy.zeros((1, 3), dtype=numpy.float32)

        cutil.launcher(scale_g3f, 1)(cpu_array, vector.c3f(1, 2, 3), 3)

        test_utils.arrays_close(self, cpu_array[0], vector.c3f(3, 6, 9))

    def test_vmul_g3f(self):
        """Tests that vmul_g3f properly produces the Hadamard product of two GPU
        vectors."""

        @cuda.jit
        def elementwise_multiply_g3f(target, a, b):
            i = cutil.line_index()
            if cutil.outside_shape(i, target.shape):
                return

            target[i] = vector.g3f_to_c3f(vector.vmul_g3f(vector.g3f(*a), vector.g3f(*b)))

        cpu_array = numpy.zeros((1, 3), dtype=numpy.float32)

        cutil.launcher(elementwise_multiply_g3f, 1)(
            cpu_array, vector.c3f(1, 2, 3), vector.c3f(1, 2, 3)
        )

        test_utils.arrays_close(self, cpu_array[0], vector.c3f(1, 4, 9))

    def test_div_g3f(self):
        """Tests that div_g3f properly divides a GPU vector by a scalar."""

        @cuda.jit
        def divide_g3f(target, a, b):
            i = cutil.line_index()
            if cutil.outside_shape(i, target.shape):
                return

            target[i] = vector.g3f_to_c3f(vector.div_g3f(vector.g3f(*a), b))

        cpu_array = numpy.zeros((1, 3), dtype=numpy.float32)

        cutil.launcher(divide_g3f, 1)(cpu_array, vector.c3f(3, 6, 9), 3)

        test_utils.arrays_close(self, cpu_array[0], vector.c3f(1, 2, 3))

    def test_dot_g3f(self):
        """Tests that dot_g3f properly produces the dot product of two GPU vectors."""

        @cuda.jit
        def dot_multiply_g3f(target, a, b):
            i = cutil.line_index()
            if cutil.outside_shape(i, target.shape):
                return

            target[i] = vector.dot_g3f(vector.g3f(*a), vector.g3f(*b))

        cpu_array = numpy.zeros(1, dtype=numpy.float32)

        cutil.launcher(dot_multiply_g3f, 1)(
            cpu_array, vector.c3f(1, 2, 3), vector.c3f(4, 5, 6)
        )

        self.assertAlmostEqual(cpu_array[0], 32)

    def test_cross_g3f(self):
        """Tests that cross_g3f properly produces the cross product of two GPU vectors."""

        @cuda.jit
        def cross_multiply_g3f(target, a, b):
            i = cutil.line_index()
            if cutil.outside_shape(i, target.shape):
                return

            target[i] = vector.g3f_to_c3f(
                vector.cross_g3f(vector.g3f(*a), vector.g3f(*b))
            )

        cpu_array = numpy.zeros((1, 3), dtype=numpy.float32)

        cutil.launcher(cross_multiply_g3f, 1)(
            cpu_array, vector.c3f(1, 2, 3), vector.c3f(4, 5, 6)
        )

        test_utils.arrays_close(self, cpu_array[0], vector.c3f(-3, 6, -3))

    def test_squared_length_g3f(self):
        """Tests that squared_length_g3f properly produces the squared length of a GPU
        vector."""

        @cuda.jit
        def find_g3f_vector_squared_length(target, a):
            i = cutil.line_index()
            if cutil.outside_shape(i, target.shape):
                return

            target[i] = vector.squared_length_g3f(vector.g3f(*a))

        cpu_array = numpy.zeros(1, dtype=numpy.float32)

        cutil.launcher(find_g3f_vector_squared_length, 1)(cpu_array, vector.c3f(1, 2, 3))

        self.assertAlmostEqual(cpu_array[0], 14)

    def test_length_g3f(self):
        """Tests that length_g3f properly produces the length of a GPU vector."""

        @cuda.jit
        def find_g3f_vector_length(target, a):
            i = cutil.line_index()
            if cutil.outside_shape(i, target.shape):
                return

            target[i] = vector.length_g3f(vector.g3f(*a))

        cpu_array = numpy.zeros(1, dtype=numpy.float32)

        cutil.launcher(find_g3f_vector_length, 1)(cpu_array, vector.c3f(2, 3, 6))

        self.assertAlmostEqual(cpu_array[0], 7)

    def test_norm_g3f(self):
        """Tests that norm_g3f properly normalizes a GPU vector."""

        @cuda.jit
        def normalize_g3f(target, a):
            i = cutil.line_index()
            if cutil.outside_shape(i, target.shape):
                return

            target[i] = vector.g3f_to_c3f(vector.norm_g3f(vector.g3f(*a)))

        cpu_array = numpy.zeros((1, 3), dtype=numpy.float32)

        cutil.launcher(normalize_g3f, 1)(cpu_array, vector.c3f(1, -1, 2))

        test_utils.arrays_close(
            self,
            cpu_array[0],
            vector.c3f(1 / math.sqrt(6), -1 / math.sqrt(6), math.sqrt(2 / 3)),
        )


if __name__ == "__main__":
    unittest.main()
