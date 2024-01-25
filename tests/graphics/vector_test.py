"""Contains tests for reinfocus.graphics.vector."""

from math import sqrt

from numba import cuda
from numba.cuda.testing import CUDATestCase, unittest

import tests.test_utils as tu
from reinfocus.graphics import vector as vec
from tests.graphics import numba_test_utils as ntu


class VectorTest(CUDATestCase):
    # pylint: disable=no-value-for-parameter,too-many-public-methods
    """TestCases for reinfocus.graphics.vector."""

    def test_c3f(self):
        """Tests that c3f makes a CPU vector with the expected elements."""

        tu.arrays_close(self, vec.c3f(1, 2, 3), vec.c3f(1, 2, 3))
        tu.arrays_not_close(self, vec.c3f(1, 2, 3), vec.c3f(2, 2, 2))

    def test_add3_c3f(self):
        """Tests that add3_c3f properly adds three CPU vectors."""

        tu.arrays_close(
            self,
            vec.add3_c3f(vec.c3f(1, 2, 3), vec.c3f(4, 5, 6), vec.c3f(7, 8, 9)),
            vec.c3f(12, 15, 18),
        )

    def test_sub_c3f(self):
        """Tests that sub_c3f properly subtracts one CPU vector from another."""

        tu.arrays_close(
            self, vec.sub_c3f(vec.c3f(4, 5, 6), vec.c3f(3, 2, 1)), vec.c3f(1, 3, 5)
        )

    def test_smul_c3f(self):
        """Tests that smul_c3f properly multiplies a CPU vector by a scalar."""

        tu.arrays_close(self, vec.smul_c3f(vec.c3f(1, 2, 3), 3), vec.c3f(3, 6, 9))

    def test_div_c3f(self):
        """Tests that div_c3f properly divides a CPU vector by a scalar."""

        tu.arrays_close(self, vec.div_c3f(vec.c3f(3, 6, 9), 3), vec.c3f(1, 2, 3))

    def test_cross_c3f(self):
        """Tests that cross_c3f properly produces the cross product of two CPU
        vectors."""

        tu.arrays_close(
            self, vec.cross_c3f(vec.c3f(1, 2, 3), vec.c3f(4, 5, 6)), vec.c3f(-3, 6, -3)
        )

    def test_length_c3f(self):
        """Tests that length_c3f properly produces the length of a CPU vector."""

        self.assertAlmostEqual(vec.length_c3f(vec.c3f(2, 3, 6)), 7)

    def test_norm_c3f(self):
        """Tests that norm_c3f properly normalizes a CPU vector."""

        tu.arrays_close(
            self,
            vec.norm_c3f(vec.c3f(1, -1, 2)),
            vec.c3f(1 / sqrt(6), -1 / sqrt(6), sqrt(2 / 3)),
        )

    def test_empty_g2f(self):
        """Tests that g2f makes a 2D GPU vector with the expected elements."""

        @cuda.jit
        def copy_empty_g2f(target):
            i = cuda.grid(1)  # type: ignore
            if i < target.size:
                target[i] = vec.g2f_to_c2f(vec.empty_g2f())

        cpu_array = ntu.cpu_target(ndim=2)

        copy_empty_g2f[1, 1](cpu_array)  # type: ignore

        tu.arrays_close(self, cpu_array[0], vec.c2f(0, 0))

    def test_g2f(self):
        """Tests that g2f makes a 2D GPU vector with the expected elements."""

        @cuda.jit
        def copy_g2f(target, source):
            i = cuda.grid(1)  # type: ignore
            if i < target.size:
                target[i] = vec.g2f_to_c2f(vec.g2f(*source))

        cpu_array = ntu.cpu_target(ndim=2)

        copy_g2f[1, 1](cpu_array, vec.c2f(1, 2))  # type: ignore

        tu.arrays_close(self, cpu_array[0], vec.c2f(1, 2))

    def test_empty_g3f(self):
        """Tests that g3f makes an empty 3D GPU vector."""

        @cuda.jit
        def copy_empty_g3f(target):
            i = cuda.grid(1)  # type: ignore
            if i < target.size:
                target[i] = vec.g3f_to_c3f(vec.empty_g3f())

        cpu_array = ntu.cpu_target()

        copy_empty_g3f[1, 1](cpu_array)  # type: ignore

        tu.arrays_close(self, cpu_array[0], vec.c3f(0, 0, 0))

    def test_g3f(self):
        """Tests that g3f makes a GPU vector with the expected elements."""

        @cuda.jit
        def copy_g3f(target, source):
            i = cuda.grid(1)  # type: ignore
            if i < target.size:
                target[i] = vec.g3f_to_c3f(vec.g3f(*source))

        cpu_array = ntu.cpu_target()

        copy_g3f[1, 1](cpu_array, vec.c3f(1, 2, 3))  # type: ignore

        tu.arrays_close(self, cpu_array[0], vec.c3f(1, 2, 3))

    def test_c3f_to_g3f(self):
        """Tests that c3f_to_g3f makes a GPU vector with elements from the CPU vector."""

        @cuda.jit
        def copy_g3f_from_c3f(target, source):
            i = cuda.grid(1)  # type: ignore
            if i < target.size:
                target[i] = vec.g3f_to_c3f(vec.c3f_to_g3f(source))

        cpu_array = ntu.cpu_target()

        copy_g3f_from_c3f[1, 1](cpu_array, vec.c3f(1, 2, 3))  # type: ignore

        tu.arrays_close(self, cpu_array[0], vec.c3f(1, 2, 3))

    def test_add_g3f(self):
        """Tests that add_g3f properly adds two GPU vectors."""

        @cuda.jit
        def add_2_g3f(target, a, b):
            i = cuda.grid(1)  # type: ignore
            if i < target.size:
                target[i] = vec.g3f_to_c3f(vec.add_g3f(vec.g3f(*a), vec.g3f(*b)))

        cpu_array = ntu.cpu_target()

        add_2_g3f[1, 1](cpu_array, vec.c3f(1, 2, 3), vec.c3f(4, 5, 6))  # type: ignore

        tu.arrays_close(self, cpu_array[0], vec.c3f(5, 7, 9))

    def test_add3_g3f(self):
        """Tests that add3_g3f properly adds three GPU vectors."""

        @cuda.jit
        def add_3_g3f(target, a, b, c):
            i = cuda.grid(1)  # type: ignore
            if i < target.size:
                target[i] = vec.g3f_to_c3f(
                    vec.add3_g3f(vec.g3f(*a), vec.g3f(*b), vec.g3f(*c))
                )

        cpu_array = ntu.cpu_target()

        add_3_g3f[1, 1](  # type: ignore
            cpu_array, vec.c3f(1, 2, 3), vec.c3f(4, 5, 6), vec.c3f(7, 8, 9)
        )

        tu.arrays_close(self, cpu_array[0], vec.c3f(12, 15, 18))

    def test_neg_g3f(self):
        """Tests that neg_g3f properly negates a GPU vector."""

        @cuda.jit
        def negate_g3f(target, a):
            i = cuda.grid(1)  # type: ignore
            if i < target.size:
                target[i] = vec.g3f_to_c3f(vec.neg_g3f(vec.g3f(*a)))

        cpu_array = ntu.cpu_target()

        negate_g3f[1, 1](cpu_array, vec.c3f(1, -2, 3))  # type: ignore

        tu.arrays_close(self, cpu_array[0], vec.c3f(-1, 2, -3))

    def test_sub_g3f(self):
        """Tests that sub_g3f properly subtracts one GPU vector from another."""

        @cuda.jit
        def subtract_g3f(target, a, b):
            i = cuda.grid(1)  # type: ignore
            if i < target.size:
                target[i] = vec.g3f_to_c3f(vec.sub_g3f(vec.g3f(*a), vec.g3f(*b)))

        cpu_array = ntu.cpu_target()

        subtract_g3f[1, 1](cpu_array, vec.c3f(4, 5, 6), vec.c3f(3, 2, 1))  # type: ignore

        tu.arrays_close(self, cpu_array[0], vec.c3f(1, 3, 5))

    def test_smul_g3f(self):
        """Tests that smul_g3f properly multiplies a GPU vector by a scalar."""

        @cuda.jit
        def scale_g3f(target, a, b):
            i = cuda.grid(1)  # type: ignore
            if i < target.size:
                target[i] = vec.g3f_to_c3f(vec.smul_g3f(vec.g3f(*a), b))

        cpu_array = ntu.cpu_target()

        scale_g3f[1, 1](cpu_array, vec.c3f(1, 2, 3), 3)  # type: ignore

        tu.arrays_close(self, cpu_array[0], vec.c3f(3, 6, 9))

    def test_vmul_g3f(self):
        """Tests that vmul_g3f properly produces the Hadamard product of two GPU
        vectors."""

        @cuda.jit
        def elementwise_multiply_g3f(target, a, b):
            i = cuda.grid(1)  # type: ignore
            if i < target.size:
                target[i] = vec.g3f_to_c3f(vec.vmul_g3f(vec.g3f(*a), vec.g3f(*b)))

        cpu_array = ntu.cpu_target()

        elementwise_multiply_g3f[1, 1](  # type: ignore
            cpu_array, vec.c3f(1, 2, 3), vec.c3f(1, 2, 3)
        )

        tu.arrays_close(self, cpu_array[0], vec.c3f(1, 4, 9))

    def test_div_g3f(self):
        """Tests that div_g3f properly divides a GPU vector by a scalar."""

        @cuda.jit
        def divide_g3f(target, a, b):
            i = cuda.grid(1)  # type: ignore
            if i < target.size:
                target[i] = vec.g3f_to_c3f(vec.div_g3f(vec.g3f(*a), b))

        cpu_array = ntu.cpu_target()

        divide_g3f[1, 1](cpu_array, vec.c3f(3, 6, 9), 3)  # type: ignore

        tu.arrays_close(self, cpu_array[0], vec.c3f(1, 2, 3))

    def test_dot_g3f(self):
        """Tests that dot_g3f properly produces the dot product of two GPU vectors."""

        @cuda.jit
        def dot_multiply_g3f(target, a, b):
            i = cuda.grid(1)  # type: ignore
            if i < target.size:
                target[i] = vec.dot_g3f(vec.g3f(*a), vec.g3f(*b))

        cpu_array = ntu.cpu_target(ndim=1)

        dot_multiply_g3f[1, 1](  # type: ignore
            cpu_array, vec.c3f(1, 2, 3), vec.c3f(4, 5, 6)
        )

        self.assertAlmostEqual(cpu_array[0][0], 32)

    def test_cross_g3f(self):
        """Tests that cross_g3f properly produces the cross product of two GPU vectors."""

        @cuda.jit
        def cross_multiply_g3f(target, a, b):
            i = cuda.grid(1)  # type: ignore
            if i < target.size:
                target[i] = vec.g3f_to_c3f(vec.cross_g3f(vec.g3f(*a), vec.g3f(*b)))

        cpu_array = ntu.cpu_target()

        cross_multiply_g3f[1, 1](  # type: ignore
            cpu_array, vec.c3f(1, 2, 3), vec.c3f(4, 5, 6)
        )

        tu.arrays_close(self, cpu_array[0], vec.c3f(-3, 6, -3))

    def test_squared_length_g3f(self):
        """Tests that squared_length_g3f properly produces the squared length of a GPU
        vector."""

        @cuda.jit
        def find_g3f_vector_squared_length(target, a):
            i = cuda.grid(1)  # type: ignore
            if i < target.size:
                target[i] = vec.squared_length_g3f(vec.g3f(*a))

        cpu_array = ntu.cpu_target(ndim=1)

        find_g3f_vector_squared_length[1, 1](cpu_array, vec.c3f(1, 2, 3))  # type: ignore

        self.assertAlmostEqual(cpu_array[0][0], 14)

    def test_length_g3f(self):
        """Tests that length_g3f properly produces the length of a GPU vector."""

        @cuda.jit
        def find_g3f_vector_length(target, a):
            i = cuda.grid(1)  # type: ignore
            if i < target.size:
                target[i] = vec.length_g3f(vec.g3f(*a))

        cpu_array = ntu.cpu_target(ndim=1)

        find_g3f_vector_length[1, 1](cpu_array, vec.c3f(2, 3, 6))  # type: ignore

        self.assertAlmostEqual(cpu_array[0][0], 7)

    def test_norm_g3f(self):
        """Tests that norm_g3f properly normalizes a GPU vector."""

        @cuda.jit
        def normalize_g3f(target, a):
            i = cuda.grid(1)  # type: ignore
            if i < target.size:
                target[i] = vec.g3f_to_c3f(vec.norm_g3f(vec.g3f(*a)))

        cpu_array = ntu.cpu_target()

        normalize_g3f[1, 1](cpu_array, vec.c3f(1, -1, 2))  # type: ignore

        tu.arrays_close(
            self, cpu_array[0], vec.c3f(1 / sqrt(6), -1 / sqrt(6), sqrt(2 / 3))
        )


if __name__ == "__main__":
    unittest.main()
