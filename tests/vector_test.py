"""Contains tests for reinfocus.vector."""
import numpy as np

from numba import cuda
from numba.cuda.testing import unittest, CUDATestCase
from reinfocus import vector

class VectorTest(CUDATestCase):
    """TestCases for reinfocus.vector."""

    def test_cpu_vector(self):
        """Checks that cpu_vector constructs a CPU vector with the expected elements."""
        v = vector.cpu_vector(1, 2, 3)
        self.assertIsNone(np.testing.assert_allclose(v, [1, 2, 3]))

    def test_gpu_vector(self):
        """Checks that gpu_vector constructs a GPU vector with the expected elements."""
        @cuda.jit()
        def get_vector(v, x, y, z):
            i = cuda.grid(1)
            if i < v.size:
                v[i] = vector.gpu_vector(x, y, z)

        v = np.array([vector.cpu_vector(0, 0, 0)])

        get_vector[1, 1](v, 1, 2, 3)
        self.assertIsNone(np.testing.assert_allclose(v[0], [1, 2, 3]))

if __name__ == '__main__':
    unittest.main()
