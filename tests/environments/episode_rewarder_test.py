"""Contains tests for reinfocus.environments.episode_rewarder."""

import unittest

from unittest import mock

import numpy

from numpy import testing
from numpy.typing import NDArray

from reinfocus.environments import episode_rewarder


class DeltaRewarderTest(unittest.TestCase):
    """Test cases for reinfocus.environments.episode_rewarder.DeltaRewarder."""

    def test_reward(self):
        """Tests that DeltaRewarder only gives out rewards proportional to the change in
        state."""

        testee = episode_rewarder.DeltaRewarder(1, 2)

        testing.assert_allclose(
            testee.reward(
                numpy.array([]),
                numpy.array([[4, 1], [3, 2], [2, 3], [1, 4]]),
                numpy.array([]),
            ),
            [0, 0, 0, 0],
        )

        testing.assert_allclose(
            testee.reward(
                numpy.array([]),
                numpy.array([[4, 1], [3, 4], [2, 1], [1, 5]]),
                numpy.array([]),
            ),
            [0, -1, -1, -0.5],
        )

        testing.assert_allclose(
            testee.reward(
                numpy.array([]),
                numpy.array([[4, 0.6], [3, 1], [2, 4], [1, 3.5]]),
                numpy.array([]),
            ),
            [-0.2, -1.5, -1.5, -0.75],
        )


class DistanceRewarderTest(unittest.TestCase):
    """Test cases for reinfocus.environments.episode_rewarder.DistanceRewarder."""

    def test_reward(self):
        """Tests that DistanceRewarder creates a rewarder that gives some reward
        proportional to the distance between two given state elements."""

        testing.assert_allclose(
            episode_rewarder.DistanceRewarder((0, 1), 1, -3, 7).reward(
                numpy.array([]),
                numpy.array([[0, 0], [0, 1], [1, 0], [1, 1], [1, 0.5], [0.5, 0]]),
                numpy.array([]),
            ),
            [7, -3, -3, 7, 2, 2],
        )


class ObservationRewarderTest(unittest.TestCase):
    """Test cases for reinfocus.environments.episode_rewarder.ObservationRewarder."""

    def test_reward(self):
        """Tests that ObservationRewarder copies the reward from some element of the
        state."""

        observations = numpy.array([[4, 1], [3, 2], [2, 3], [1, 4]])

        testing.assert_allclose(
            episode_rewarder.ObservationRewarder(0).reward(
                numpy.array([]),
                numpy.array([]),
                observations,
            ),
            [4, 3, 2, 1],
        )

        testing.assert_allclose(
            episode_rewarder.ObservationRewarder(1).reward(
                numpy.array([]),
                numpy.array([]),
                observations,
            ),
            [1, 2, 3, 4],
        )


class OnTargetRewarderTest(unittest.TestCase):
    """Test cases for reinfocus.environments.episode_rewarder.OnTargetRewarder."""

    def test_reward(self):
        """Tests that OnTargetRewarder gives the proper reward when the two given state
        elements are and aren't close enough."""

        testing.assert_allclose(
            episode_rewarder.OnTargetRewarder((0, 1), 0.1, -3, 7).reward(
                numpy.array([]),
                numpy.array([[0.5, 0.65], [0.5, 0.55], [0.5, 0.45], [0.5, 0.35]]),
                numpy.array([]),
            ),
            [-3, 7, 7, -3],
        )


class SumRewarderTest(unittest.TestCase):
    """Test cases for reinfocus.environments.episode_rewarder.SumRewarder."""

    def test_reward(self):
        """Tests that SumRewarder gives the a reward that is the sum of rewards from other
        rewarders."""

        def make_rewarder(reward: NDArray):
            rewarder = mock.Mock()
            rewarder.reward.return_value = reward
            return rewarder

        testing.assert_allclose(
            episode_rewarder.SumRewarder(
                make_rewarder(numpy.array([0, 1, 2, 3])),
                make_rewarder(numpy.array([-1, 0, 1, 0])),
                make_rewarder(numpy.array([2, 4, 6, 8])),
            ).reward(numpy.array([]), numpy.array([]), numpy.array([])),
            [1, 5, 9, 11],
        )


if __name__ == "__main__":
    unittest.main()
