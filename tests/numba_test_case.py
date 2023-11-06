"""Contains utilities for numba based unit tests."""

from numba.cuda.testing import CUDATestCase
from numpy import testing as npt

class NumbaTestCase(CUDATestCase):
    """Base class for CPU and GPU vector unit tests."""

    def arrays_close(self, a, b):
        """Asserts that two arrays are fairly close."""
        self.assertIsNone(npt.assert_allclose(a, b))

    def arrays_not_close(self, a, b):
        """Asserts that two arrays are not fairly close."""
        self.assertIsNone(npt.assert_raises(AssertionError, npt.assert_allclose, a, b))
