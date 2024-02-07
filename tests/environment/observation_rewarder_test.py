"""Contains tests for reinfocus.environment.observation_rewarder."""

import unittest

import numpy

from reinfocus.environment import observation_rewarder


class ObservationRewarderTest(unittest.TestCase):
    """TestCases for reinfocus.environment.observation_rewarder."""

    def test_distance(self):
        """Tests that distance creates a rewarder that gives rewards that linearly scale
        between the given low and high rewards."""

        reward = observation_rewarder.distance(1, -3, 7)
        self.assertEqual(reward(numpy.array([0, 0]), numpy.array([])), 7)
        self.assertEqual(reward(numpy.array([0, 1]), numpy.array([])), -3)
        self.assertEqual(reward(numpy.array([1, 0]), numpy.array([])), -3)
        self.assertEqual(reward(numpy.array([1, 1]), numpy.array([])), 7)
        self.assertEqual(reward(numpy.array([1, 0.5]), numpy.array([])), 2)
        self.assertEqual(reward(numpy.array([0.5, 1]), numpy.array([])), 2)

    def test_on_target(self):
        """Tests that on_target creates a rewarder that gives the proper reward when the
        lens is on and off target."""

        reward = observation_rewarder.on_target(0.1, -3, 7)
        self.assertEqual(reward(numpy.array([0.5, 0.65]), numpy.array([])), -3)
        self.assertEqual(reward(numpy.array([0.5, 0.55]), numpy.array([])), 7)
        self.assertEqual(reward(numpy.array([0.5, 0.45]), numpy.array([])), 7)
        self.assertEqual(reward(numpy.array([0.5, 0.35]), numpy.array([])), -3)

    def test_focus(self):
        """Tests that focus creates a rewarder that gives a reward equal to the focus."""

        reward = observation_rewarder.focus()
        self.assertEqual(reward(numpy.array([]), numpy.array([1, 2, 3])), 3)
        self.assertEqual(reward(numpy.array([]), numpy.array([4, 5, 6])), 6)
        self.assertEqual(reward(numpy.array([]), numpy.array([3, 2, 1])), 1)
        self.assertEqual(reward(numpy.array([]), numpy.array([6, 5, 4])), 4)


if __name__ == "__main__":
    unittest.main()
