"""Contains tests for reinfocus.learning.focus_environment."""

import unittest

import reinfocus.learning.focus_environment as env

class FocusEnvironmentTest(unittest.TestCase):
    """TestCases for reinfocus.learning.focus_environment."""

    def test_make_observation_normer(self):
        """Tests that make_observation_normer creates a normer that norms as expected."""
        normer = env.make_observation_normer(5, 5)
        self.assertEqual(normer(0), -1)
        self.assertEqual(normer(5), 0)
        self.assertEqual(normer(10), 1)

if __name__ == '__main__':
    unittest.main()
