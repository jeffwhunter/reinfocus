"""Contains tests for reinfocus.vector."""

from math import sqrt
from numba import cuda
from numba.cuda.testing import unittest

from reinfocus import vector as vec
from tests import numba_test_case as ntc
from tests import numba_test_utils as ntu

class CPUVectorTest(ntc.NumbaTestCase):
    """TestCases for reinfocus.vector."""

    def test_cpu_vector(self):
        """Tests that cpu_vector makes a CPU vector with the expected elements."""
        self.arrays_close(vec.cpu_vector(1, 2, 3), vec.cpu_vector(1, 2, 3))
        self.arrays_not_close(vec.cpu_vector(1, 2, 3), vec.cpu_vector(2, 2, 2))

    def test_cpu_add3(self):
        """Tests that cpu_add3 properly adds three CPU vectors."""
        self.arrays_close(
            vec.cpu_add3(vec.cpu_vector(1, 2, 3), vec.cpu_vector(4, 5, 6), vec.cpu_vector(7, 8, 9)),
            vec.cpu_vector(12, 15, 18))

    def test_cpu_sub(self):
        """Tests that cpu_sub properly subtracts one CPU vector from another."""
        self.arrays_close(
            vec.cpu_sub(vec.cpu_vector(4, 5, 6), vec.cpu_vector(3, 2, 1)),
            vec.cpu_vector(1, 3, 5))

    def test_cpu_smul(self):
        """Tests that cpu_smul properly multiplies a CPU vector by a scalar."""
        self.arrays_close(vec.cpu_smul(vec.cpu_vector(1, 2, 3), 3), vec.cpu_vector(3, 6, 9))

    def test_cpu_div(self):
        """Tests that cpu_div properly divides a CPU vector by a scalar."""
        self.arrays_close(vec.cpu_div(vec.cpu_vector(3, 6, 9), 3), vec.cpu_vector(1, 2, 3))

    def test_cpu_cross(self):
        """Tests that cpu_cross properly produces the cross product of two CPU vectors."""
        self.arrays_close(
            vec.cpu_cross(vec.cpu_vector(1, 2, 3), vec.cpu_vector(4, 5, 6)),
            vec.cpu_vector(-3, 6, -3))

    def test_cpu_length(self):
        """Tests that cpu_length properly produces the length of a CPU vector."""
        self.assertAlmostEqual(vec.cpu_length(vec.cpu_vector(2, 3, 6)), 7)

    def test_cpu_norm_vector(self):
        """Tests that cpu_norm_vector properly normalizes a CPU vector."""
        self.arrays_close(
            vec.cpu_norm_vector(vec.cpu_vector(1, -1, 2)),
            vec.cpu_vector(1 / sqrt(6), -1 / sqrt(6), sqrt(2 / 3)))

class GPUVectorTest(ntc.NumbaTestCase):
    """TestCases for reinfocus.vector."""
    # pylint: disable=no-value-for-parameter

    def test_gpu_vector(self):
        """Tests that gpu_vector makes a GPU vector with the expected elements."""
        @cuda.jit()
        def copy_gpu_vector(target, source):
            i = cuda.grid(1) # type: ignore
            if i < target.size:
                target[i] = vec.to_cpu_vector(vec.gpu_vector(*source))

        cpu_array = ntu.cpu_target()

        copy_gpu_vector[1, 1](cpu_array, vec.cpu_vector(1, 2, 3)) # type: ignore

        self.arrays_close(cpu_array[0], vec.cpu_vector(1, 2, 3))

    def test_to_gpu_vector(self):
        """Tests that to_gpu_vector makes a GPU vector with elements from the CPU vector."""
        @cuda.jit()
        def copy_gpu_from_cpu_vector(target, source):
            i = cuda.grid(1) # type: ignore
            if i < target.size:
                target[i] = vec.to_cpu_vector(vec.to_gpu_vector(source))

        cpu_array = ntu.cpu_target()

        copy_gpu_from_cpu_vector[1, 1](cpu_array, vec.cpu_vector(1, 2, 3)) # type: ignore

        self.arrays_close(cpu_array[0], vec.cpu_vector(1, 2, 3))

    def test_gpu_add(self):
        """Tests that gpu_add properly adds two GPU vectors."""
        @cuda.jit()
        def add_2_gpu_vectors(target, a, b):
            i = cuda.grid(1) # type: ignore
            if i < target.size:
                target[i] = vec.to_cpu_vector(vec.gpu_add(vec.gpu_vector(*a), vec.gpu_vector(*b)))

        cpu_array = ntu.cpu_target()

        add_2_gpu_vectors[1, 1]( # type: ignore
            cpu_array,
            vec.cpu_vector(1, 2, 3),
            vec.cpu_vector(4, 5, 6))

        self.arrays_close(cpu_array[0], vec.cpu_vector(5, 7, 9))

    def test_gpu_add3(self):
        """Tests that gpu_add3 properly adds three GPU vectors."""
        @cuda.jit()
        def add_3_gpu_vectors(target, a, b, c):
            i = cuda.grid(1) # type: ignore
            if i < target.size:
                target[i] = vec.to_cpu_vector(
                    vec.gpu_add3(vec.gpu_vector(*a), vec.gpu_vector(*b), vec.gpu_vector(*c)))

        cpu_array = ntu.cpu_target()

        add_3_gpu_vectors[1, 1]( # type: ignore
            cpu_array,
            vec.cpu_vector(1, 2, 3),
            vec.cpu_vector(4, 5, 6),
            vec.cpu_vector(7, 8, 9))

        self.arrays_close(cpu_array[0], vec.cpu_vector(12, 15, 18))

    def test_gpu_neg(self):
        """Tests that gpu_neg properly negates a GPU vector."""
        @cuda.jit()
        def negate_gpu_vector(target, a):
            i = cuda.grid(1) # type: ignore
            if i < target.size:
                target[i] = vec.to_cpu_vector(vec.gpu_neg(vec.gpu_vector(*a)))

        cpu_array = ntu.cpu_target()

        negate_gpu_vector[1, 1](cpu_array, vec.cpu_vector(1, -2, 3)) # type: ignore

        self.arrays_close(cpu_array[0], vec.cpu_vector(-1, 2, -3))

    def test_gpu_sub(self):
        """Tests that gpu_sub properly subtracts one GPU vector from another."""
        @cuda.jit()
        def subtract_gpu_vectors(target, a, b):
            i = cuda.grid(1) # type: ignore
            if i < target.size:
                target[i] = vec.to_cpu_vector(vec.gpu_sub(vec.gpu_vector(*a), vec.gpu_vector(*b)))

        cpu_array = ntu.cpu_target()

        subtract_gpu_vectors[1, 1]( # type: ignore
            cpu_array,
            vec.cpu_vector(4, 5, 6),
            vec.cpu_vector(3, 2, 1))

        self.arrays_close(cpu_array[0], vec.cpu_vector(1, 3, 5))

    def test_gpu_smul(self):
        """Tests that gpu_smul properly multiplies a GPU vector by a scalar."""
        @cuda.jit()
        def scale_gpu_vector(target, a, b):
            i = cuda.grid(1) # type: ignore
            if i < target.size:
                target[i] = vec.to_cpu_vector(vec.gpu_smul(vec.gpu_vector(*a), b))

        cpu_array = ntu.cpu_target()

        scale_gpu_vector[1, 1](cpu_array, vec.cpu_vector(1, 2, 3), 3) # type: ignore

        self.arrays_close(cpu_array[0], vec.cpu_vector(3, 6, 9))

    def test_gpu_vmul(self):
        """Tests that gpu_vmul properly produces the Hadamard product of two GPU vectors."""
        @cuda.jit()
        def multiply_vectors_elementwise(target, a, b):
            i = cuda.grid(1) # type: ignore
            if i < target.size:
                target[i] = vec.to_cpu_vector(vec.gpu_vmul(vec.gpu_vector(*a), vec.gpu_vector(*b)))

        cpu_array = ntu.cpu_target()

        multiply_vectors_elementwise[1, 1]( # type: ignore
            cpu_array,
            vec.cpu_vector(1, 2, 3),
            vec.cpu_vector(1, 2, 3))

        self.arrays_close(cpu_array[0], vec.cpu_vector(1, 4, 9))

    def test_gpu_div(self):
        """Tests that gpu_div properly divides a GPU vector by a scalar."""
        @cuda.jit()
        def divide_gpu_vector(target, a, b):
            i = cuda.grid(1) # type: ignore
            if i < target.size:
                target[i] = vec.to_cpu_vector(vec.gpu_div(vec.gpu_vector(*a), b))

        cpu_array = ntu.cpu_target()

        divide_gpu_vector[1, 1](cpu_array, vec.cpu_vector(3, 6, 9), 3) # type: ignore

        self.arrays_close(cpu_array[0], vec.cpu_vector(1, 2, 3))

    def test_gpu_dot(self):
        """Tests that gpu_dot properly produces the dot product of two GPU vectors."""
        @cuda.jit()
        def dot_multiply_gpu_vectors(target, a, b):
            i = cuda.grid(1) # type: ignore
            if i < target.size:
                target[i] = vec.gpu_dot(vec.gpu_vector(*a), vec.gpu_vector(*b))

        cpu_array = ntu.cpu_target(ndim=1)

        dot_multiply_gpu_vectors[1, 1]( # type: ignore
            cpu_array,
            vec.cpu_vector(1, 2, 3),
            vec.cpu_vector(4, 5, 6))

        self.assertAlmostEqual(cpu_array[0][0], 32)

    def test_gpu_cross(self):
        """Tests that gpu_cross properly produces the cross product of two GPU vectors."""
        @cuda.jit()
        def cross_multiply_gpu_vectors(target, a, b):
            i = cuda.grid(1) # type: ignore
            if i < target.size:
                target[i] = vec.to_cpu_vector(vec.gpu_cross(vec.gpu_vector(*a), vec.gpu_vector(*b)))

        cpu_array = ntu.cpu_target()

        cross_multiply_gpu_vectors[1, 1]( # type: ignore
            cpu_array,
            vec.cpu_vector(1, 2, 3),
            vec.cpu_vector(4, 5, 6))

        self.arrays_close(cpu_array[0], vec.cpu_vector(-3, 6, -3))

    def test_gpu_squared_length(self):
        """Tests that gpu_squared_length properly produces the squared length of a GPU vector."""
        @cuda.jit()
        def find_gpu_vector_squared_length(target, a):
            i = cuda.grid(1) # type: ignore
            if i < target.size:
                target[i] = vec.gpu_squared_length(vec.gpu_vector(*a))

        cpu_array = ntu.cpu_target(ndim=1)

        find_gpu_vector_squared_length[1, 1](cpu_array, vec.cpu_vector(1, 2, 3)) # type: ignore

        self.assertAlmostEqual(cpu_array[0][0], 14)

    def test_gpu_length(self):
        """Tests that gpu_length properly produces the length of a GPU vector."""
        @cuda.jit()
        def find_gpu_vector_length(target, a):
            i = cuda.grid(1) # type: ignore
            if i < target.size:
                target[i] = vec.gpu_length(vec.gpu_vector(*a))

        cpu_array = ntu.cpu_target(ndim=1)

        find_gpu_vector_length[1, 1](cpu_array, vec.cpu_vector(2, 3, 6)) # type: ignore

        self.assertAlmostEqual(cpu_array[0][0], 7)

    def test_gpu_norm_vector(self):
        """Tests that gpu_norm_vector properly normalizes a GPU vector."""
        @cuda.jit()
        def normalize_gpu_vector(target, a):
            i = cuda.grid(1) # type: ignore
            if i < target.size:
                target[i] = vec.to_cpu_vector(vec.gpu_norm_vector(vec.gpu_vector(*a)))

        cpu_array = ntu.cpu_target()

        normalize_gpu_vector[1, 1](cpu_array, vec.cpu_vector(1, -1, 2)) # type: ignore

        self.arrays_close(cpu_array[0], vec.cpu_vector(1 / sqrt(6), -1 / sqrt(6), sqrt(2 / 3)))

if __name__ == '__main__':
    unittest.main()
