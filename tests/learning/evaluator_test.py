"""Contains tests for reinfocus.learning.evaluator."""

import unittest

import reinfocus.learning.evaluator as eva
import tests.test_utils as tu

class EvaluatorTest(unittest.TestCase):
    """TestCases for reinfocus.learning.evaluator."""

    def test_iqm_interval(self):
        """Tests that iqm_interval does something, at least."""
        tu.arrays_close(self, eva.iqm_interval([1, 2, 3, 4, 5]), [3, 2, 4])

if __name__ == '__main__':
    unittest.main()
