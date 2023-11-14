"""Contains utilities for array based unit tests."""

import unittest

from numpy import testing as npt

def arrays_close(test_case: unittest.TestCase, a, b):
    """Asserts that two arrays are fairly close."""
    test_case.assertIsNone(npt.assert_allclose(a, b, atol=1e-7))

def arrays_not_close(test_case: unittest.TestCase, a, b):
    """Asserts that two arrays are not fairly close."""
    test_case.assertIsNone(npt.assert_raises(AssertionError, npt.assert_allclose, a, b))
