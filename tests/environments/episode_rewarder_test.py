"""Contains tests for reinfocus.environments.episode_rewarder."""

import unittest

import numpy

from numpy import testing

from reinfocus.environments import episode_rewarder


class DistanceRewarderTest(unittest.TestCase):
    """TestCase for reinfocus.environments.episode_rewarder.DistanceRewarder."""

    def test_distance_rewarder(self):
        """Tests that DistanceRewarder creates a rewarder that gives some reward
        proportional to the distance between two given state elements."""

        testing.assert_allclose(
            episode_rewarder.DistanceRewarder((0, 1), 1, -3, 7).reward(
                numpy.array([[0, 0], [0, 1], [1, 0], [1, 1], [1, 0.5], [0.5, 0]]),
                numpy.array([]),
                numpy.array([]),
            ),
            [7, -3, -3, 7, 2, 2],
        )


class ObservationElementRewarderTest(unittest.TestCase):
    """TestCase for reinfocus.environments.episode_rewarder.ObservationElementRewarder."""

    def test_observation_element_rewarder(self):
        """Tests that ObservationElementRewarder creates a rewarder that gives the proper
        rewarder copied from some element of the state."""

        observations = numpy.array([[4, 1], [3, 2], [2, 3], [1, 4]])

        testing.assert_allclose(
            episode_rewarder.ObservationElementRewarder(0).reward(
                numpy.array([]),
                observations,
                numpy.array([]),
            ),
            [4, 3, 2, 1],
        )

        testing.assert_allclose(
            episode_rewarder.ObservationElementRewarder(1).reward(
                numpy.array([]),
                observations,
                numpy.array([]),
            ),
            [1, 2, 3, 4],
        )


class OnTargetRewarderTest(unittest.TestCase):
    """TestCase for reinfocus.environments.episode_rewarder.OnTargetRewarder."""

    def test_on_target_rewarder(self):
        """Tests that OnTargetRewarder creates a rewarder that gives the proper reward
        when the two given state elements are and aren't close enough."""

        testing.assert_allclose(
            episode_rewarder.OnTargetRewarder((0, 1), 0.1, -3, 7).reward(
                numpy.array([[0.5, 0.65], [0.5, 0.55], [0.5, 0.45], [0.5, 0.35]]),
                numpy.array([]),
                numpy.array([]),
            ),
            [-3, 7, 7, -3],
        )


if __name__ == "__main__":
    unittest.main()
