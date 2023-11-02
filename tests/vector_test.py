"""Contains tests for reinfocus.vector."""
from math import sqrt
import typing
import numpy as np

from numba import cuda
from numba.cuda.testing import unittest, CUDATestCase
from numpy import testing as npt
from reinfocus import vector as vc

def cpu_target(ndim=3):
    """Makes a single vector target array for GPU test output."""
    return np.array([(0.0,) * ndim])

class VectorTestCase(CUDATestCase):
    """Base class for CPU and GPU vector unit tests."""

    def arrays_close(self, a, b):
        """Asserts that two arrays are fairly close."""
        self.assertIsNone(npt.assert_allclose(a, b))

    def arrays_not_close(self, a, b):
        """Asserts that two arrays are not fairly close."""
        self.assertIsNone(npt.assert_raises(AssertionError, npt.assert_allclose, a, b))

class CPUVectorTest(VectorTestCase):
    """TestCases for reinfocus.vector."""

    def test_cpu_vector(self):
        """Tests that cpu_vector constructs a CPU vector with the expected elements."""
        self.arrays_close(vc.cpu_vector(1, 2, 3), vc.cpu_vector(1, 2, 3))
        self.arrays_not_close(vc.cpu_vector(1, 2, 3), vc.cpu_vector(2, 2, 2))

    def test_cpu_add3(self):
        """Tests that cpu_add3 properly adds three CPU vectors."""
        self.arrays_close(
            vc.cpu_add3(vc.cpu_vector(1, 2, 3), vc.cpu_vector(4, 5, 6), vc.cpu_vector(7, 8, 9)),
            vc.cpu_vector(12, 15, 18))

    def test_cpu_sub(self):
        """Tests that cpu_sub properly subtracts one CPU vector from another."""
        self.arrays_close(
            vc.cpu_sub(vc.cpu_vector(4, 5, 6), vc.cpu_vector(3, 2, 1)),
            vc.cpu_vector(1, 3, 5))

    def test_cpu_smul(self):
        """Tests that cpu_smul properly multiplies a CPU vector by a scalar."""
        self.arrays_close(vc.cpu_smul(vc.cpu_vector(1, 2, 3), 3), vc.cpu_vector(3, 6, 9))

    def test_cpu_div(self):
        """Tests that cpu_div properly divides a CPU vector by a scalar."""
        self.arrays_close(vc.cpu_div(vc.cpu_vector(3, 6, 9), 3), vc.cpu_vector(1, 2, 3))

    def test_cpu_cross(self):
        """Tests that cpu_cross properly produces the cross product of two CPU vectors."""
        self.arrays_close(
            vc.cpu_cross(vc.cpu_vector(1, 2, 3), vc.cpu_vector(4, 5, 6)),
            vc.cpu_vector(-3, 6, -3))

    def test_cpu_length(self):
        """Tests that cpu_length properly produces the length of a CPU vector."""
        self.assertAlmostEqual(vc.cpu_length(vc.cpu_vector(2, 3, 6)), 7)

    def test_cpu_norm_vector(self):
        """Tests that cpu_norm_vector properly normalizes a CPU vector."""
        self.arrays_close(
            vc.cpu_norm_vector(vc.cpu_vector(1, -1, 2)),
            vc.cpu_vector(1 / sqrt(6), -1 / sqrt(6), sqrt(2 / 3)))

class GPUVectorTest(VectorTestCase):
    """TestCases for reinfocus.vector."""
    # pylint: disable=no-value-for-parameter

    def test_gpu_vector(self):
        """Tests that gpu_vector constructs a GPU vector with the expected elements."""
        @cuda.jit()
        @typing.no_type_check
        def copy_gpu_vector(target, source):
            i = cuda.grid(1) # type: ignore
            if i < target.size:
                target[i] = vc.gpu_vector(*source)

        cpu_array = cpu_target()

        copy_gpu_vector[1, 1](cpu_array, vc.cpu_vector(1, 2, 3)) # type: ignore

        self.arrays_close(cpu_array[0], vc.cpu_vector(1, 2, 3))

    def test_gpu_add(self):
        """Tests that gpu_add properly adds two GPU vectors."""
        @cuda.jit()
        def add_2_gpu_vectors(target, a, b):
            i = cuda.grid(1) # type: ignore
            if i < target.size:
                target[i] = vc.gpu_add(vc.gpu_vector(*a), vc.gpu_vector(*b))

        cpu_array = cpu_target()

        add_2_gpu_vectors[1, 1]( # type: ignore
            cpu_array,
            vc.cpu_vector(1, 2, 3),
            vc.cpu_vector(4, 5, 6))

        self.arrays_close(cpu_array[0], vc.cpu_vector(5, 7, 9))

    def test_gpu_add3(self):
        """Tests that gpu_add3 properly adds three GPU vectors."""
        @cuda.jit()
        def add_3_gpu_vectors(target, a, b, c):
            i = cuda.grid(1) # type: ignore
            if i < target.size:
                target[i] = vc.gpu_add3(vc.gpu_vector(*a), vc.gpu_vector(*b), vc.gpu_vector(*c))

        cpu_array = cpu_target()

        add_3_gpu_vectors[1, 1]( # type: ignore
            cpu_array,
            vc.cpu_vector(1, 2, 3),
            vc.cpu_vector(4, 5, 6),
            vc.cpu_vector(7, 8, 9))

        self.arrays_close(cpu_array[0], vc.cpu_vector(12, 15, 18))

    def test_gpu_neg(self):
        """Tests that gpu_neg properly negates a GPU vector."""
        @cuda.jit()
        def negate_gpu_vector(target, a):
            i = cuda.grid(1) # type: ignore
            if i < target.size:
                target[i] = vc.gpu_neg(vc.gpu_vector(*a))

        cpu_array = cpu_target()

        negate_gpu_vector[1, 1](cpu_array, vc.cpu_vector(1, -2, 3)) # type: ignore

        self.arrays_close(cpu_array[0], vc.cpu_vector(-1, 2, -3))

    def test_gpu_sub(self):
        """Tests that gpu_sub properly subtracts one GPU vector from another."""
        @cuda.jit()
        def subtract_gpu_vectors(target, a, b):
            i = cuda.grid(1) # type: ignore
            if i < target.size:
                target[i] = vc.gpu_sub(vc.gpu_vector(*a), vc.gpu_vector(*b))

        cpu_array = cpu_target()

        subtract_gpu_vectors[1, 1]( # type: ignore
            cpu_array,
            vc.cpu_vector(4, 5, 6),
            vc.cpu_vector(3, 2, 1))

        self.arrays_close(cpu_array[0], vc.cpu_vector(1, 3, 5))

    def test_gpu_smul(self):
        """Tests that gpu_smul properly multiplies a GPU vector by a scalar."""
        @cuda.jit()
        def scale_gpu_vector(target, a, b):
            i = cuda.grid(1) # type: ignore
            if i < target.size:
                target[i] = vc.gpu_smul(vc.gpu_vector(*a), b)

        cpu_array = cpu_target()

        scale_gpu_vector[1, 1](cpu_array, vc.cpu_vector(1, 2, 3), 3) # type: ignore

        self.arrays_close(cpu_array[0], vc.cpu_vector(3, 6, 9))

    def test_gpu_vmul(self):
        """Tests that gpu_vmul properly produces the Hadamard product of two GPU vectors."""
        @cuda.jit()
        def multiply_vectors_elementwise(target, a, b):
            i = cuda.grid(1) # type: ignore
            if i < target.size:
                target[i] = vc.gpu_vmul(vc.gpu_vector(*a), vc.gpu_vector(*b))

        cpu_array = cpu_target()

        multiply_vectors_elementwise[1, 1]( # type: ignore
            cpu_array,
            vc.cpu_vector(1, 2, 3),
            vc.cpu_vector(1, 2, 3))

        self.arrays_close(cpu_array[0], vc.cpu_vector(1, 4, 9))

    def test_gpu_div(self):
        """Tests that gpu_div properly divides a GPU vector by a scalar."""
        @cuda.jit()
        def divide_gpu_vector(target, a, b):
            i = cuda.grid(1) # type: ignore
            if i < target.size:
                target[i] = vc.gpu_div(vc.gpu_vector(*a), b)

        cpu_array = cpu_target()

        divide_gpu_vector[1, 1](cpu_array, vc.cpu_vector(3, 6, 9), 3) # type: ignore

        self.arrays_close(cpu_array[0], vc.cpu_vector(1, 2, 3))

    def test_gpu_dot(self):
        """Tests that gpu_dot properly produces the dot product of two GPU vectors."""
        @cuda.jit()
        def dot_multiply_gpu_vectors(target, a, b):
            i = cuda.grid(1) # type: ignore
            if i < target.size:
                target[i] = vc.gpu_dot(vc.gpu_vector(*a), vc.gpu_vector(*b))

        cpu_array = cpu_target(ndim=1)

        dot_multiply_gpu_vectors[1, 1]( # type: ignore
            cpu_array,
            vc.cpu_vector(1, 2, 3),
            vc.cpu_vector(4, 5, 6))

        self.assertAlmostEqual(cpu_array[0][0], 32)

    def test_gpu_cross(self):
        """Tests that gpu_cross properly produces the cross product of two GPU vectors."""
        @cuda.jit()
        def cross_multiply_gpu_vectors(target, a, b):
            i = cuda.grid(1) # type: ignore
            if i < target.size:
                target[i] = vc.gpu_cross(vc.gpu_vector(*a), vc.gpu_vector(*b))

        cpu_array = cpu_target()

        cross_multiply_gpu_vectors[1, 1]( # type: ignore
            cpu_array,
            vc.cpu_vector(1, 2, 3),
            vc.cpu_vector(4, 5, 6))

        self.arrays_close(cpu_array[0], vc.cpu_vector(-3, 6, -3))

    def test_gpu_squared_length(self):
        """Tests that gpu_squared_length properly produces the squared length of a GPU vector."""
        @cuda.jit()
        def find_gpu_vector_squared_length(target, a):
            i = cuda.grid(1) # type: ignore
            if i < target.size:
                target[i] = vc.gpu_squared_length(vc.gpu_vector(*a))

        cpu_array = cpu_target(ndim=1)

        find_gpu_vector_squared_length[1, 1](cpu_array, vc.cpu_vector(1, 2, 3)) # type: ignore

        self.assertAlmostEqual(cpu_array[0][0], 14)

    def test_gpu_length(self):
        """Tests that gpu_length properly produces the length of a GPU vector."""
        @cuda.jit()
        def find_gpu_vector_length(target, a):
            i = cuda.grid(1) # type: ignore
            if i < target.size:
                target[i] = vc.gpu_length(vc.gpu_vector(*a))

        cpu_array = cpu_target(ndim=1)

        find_gpu_vector_length[1, 1](cpu_array, vc.cpu_vector(2, 3, 6)) # type: ignore

        self.assertAlmostEqual(cpu_array[0][0], 7)

    def test_gpu_norm_vector(self):
        """Tests that gpu_norm_vector properly normalizes a GPU vector."""
        @cuda.jit()
        def normalize_gpu_vector(target, a):
            i = cuda.grid(1) # type: ignore
            if i < target.size:
                target[i] = vc.gpu_norm_vector(vc.gpu_vector(*a))

        cpu_array = cpu_target()

        normalize_gpu_vector[1, 1](cpu_array, vc.cpu_vector(1, -1, 2)) # type: ignore

        self.arrays_close(cpu_array[0], vc.cpu_vector(1 / sqrt(6), -1 / sqrt(6), sqrt(2 / 3)))

if __name__ == '__main__':
    unittest.main()
