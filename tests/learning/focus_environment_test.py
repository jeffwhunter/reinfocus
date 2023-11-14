"""Contains tests for reinfocus.learning.focus_environment."""

import unittest

import numpy as np

import reinfocus.learning.focus_environment as env
import tests.test_utils as tu

class FocusEnvironmentTest(unittest.TestCase):
    """TestCases for reinfocus.learning.focus_environment."""

    def test_make_observation_normer(self):
        """Tests that make_observation_normer creates a normer that norms as expected."""
        normer = env.make_observation_normer(5, 5)
        self.assertEqual(normer(0), -1)
        self.assertEqual(normer(5), 0)
        self.assertEqual(normer(10), 1)
        array_normer = env.make_observation_normer(np.array([1]), np.array([2]))
        tu.arrays_close(self, array_normer(np.array([0])), np.array([-.5]))
        tu.arrays_close(self, array_normer(np.array([1])), np.array([0]))
        tu.arrays_close(self, array_normer(np.array([2])), np.array([.5]))

    def test_make_lens_distance_penalty(self):
        """Tests that make_lens_distance_penalty creates a rewarder that gives
            the proper penalties."""
        penalty = env.make_lens_distance_penalty(1.)
        self.assertEqual(penalty([0., 0.]), 0)
        self.assertEqual(penalty([0., 1.]), -1)
        self.assertEqual(penalty([1., 0.]), -1)
        self.assertEqual(penalty([1., 1.]), 0)

    def test_make_lens_on_target_reward(self):
        """Tests that make_lens_on_target_reward creates a rewarder that gives a reward
            of one when the lens is within the given distance."""
        reward = env.make_lens_on_target_reward(.1)
        self.assertEqual(reward([.5, .65]), 0)
        self.assertEqual(reward([.5, .55]), 1)
        self.assertEqual(reward([.5, .45]), 1)
        self.assertEqual(reward([.5, .35]), 0)

    def test_make_focus_reward(self):
        """Tests that make_focus_reward creates a rewarder that gives a reward equal to
            the focus."""
        reward = env.make_focus_reward()
        self.assertEqual(reward([1, 2, 3]), 3)
        self.assertEqual(reward([4, 5, 6]), 6)
        self.assertEqual(reward([3, 2, 1]), 1)
        self.assertEqual(reward([6, 5, 4]), 4)

if __name__ == '__main__':
    unittest.main()
