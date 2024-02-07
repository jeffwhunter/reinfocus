"""Contains tests for reinfocus.environment.observation_rewarder."""

import unittest

import numpy

from reinfocus.environment import observation_rewarder


class ObservationRewarderTest(unittest.TestCase):
    """TestCases for reinfocus.environment.observation_rewarder."""

    def test_make_lens_distance_penalty(self):
        """Tests that make_lens_distance_penalty creates a rewarder that gives
        the proper penalties."""

        penalty = observation_rewarder.distance_penalty(1.0)
        self.assertEqual(penalty(numpy.array([0.0, 0.0])), 0)
        self.assertEqual(penalty(numpy.array([0.0, 1.0])), -1)
        self.assertEqual(penalty(numpy.array([1.0, 0.0])), -1)
        self.assertEqual(penalty(numpy.array([1.0, 1.0])), 0)

    def test_make_lens_on_target_reward(self):
        """Tests that make_lens_on_target_reward creates a rewarder that gives a reward
        of one when the lens is within the given distance."""

        reward = observation_rewarder.on_target_reward(0.1)
        self.assertEqual(reward(numpy.array([0.5, 0.65])), 0)
        self.assertEqual(reward(numpy.array([0.5, 0.55])), 1)
        self.assertEqual(reward(numpy.array([0.5, 0.45])), 1)
        self.assertEqual(reward(numpy.array([0.5, 0.35])), 0)

    def test_make_focus_reward(self):
        """Tests that make_focus_reward creates a rewarder that gives a reward equal to
        the focus."""

        reward = observation_rewarder.focus_reward()
        self.assertEqual(reward(numpy.array([1, 2, 3])), 3)
        self.assertEqual(reward(numpy.array([4, 5, 6])), 6)
        self.assertEqual(reward(numpy.array([3, 2, 1])), 1)
        self.assertEqual(reward(numpy.array([6, 5, 4])), 4)


if __name__ == "__main__":
    unittest.main()
