"""Contains tests for reinfocus.environments.episode_rewarder."""

import unittest

from unittest import mock

import numpy

from numpy import testing
from numpy.typing import NDArray

from reinfocus.environments import episode_rewarder


class BaseRewarderTest(unittest.TestCase):
    """Test cases for reinfocus.environments.episode_rewarder.BaseRewarder."""

    class _BaseRewarder(episode_rewarder.BaseRewarder):
        # pylint: disable=too-few-public-methods
        """A minimal implementation of BaseRewarder to allow testing of it."""

        def __init__(self, reward: NDArray[numpy.float32]):
            """Creates a _BaseRewarder.

            Args:
                reward: The reward signal to emit."""

            super().__init__()

            self._reward = reward

        def reward(
            self, states: NDArray[numpy.float32], observations: NDArray[numpy.float32]
        ) -> NDArray[numpy.float32]:
            """Emits the preconfigured reward signal.

            Args:
                states: The states, which will be ignored.
                observations: The observations, which will be ignored.

            Returns:
                The reward signal configured in the initializer."""

            return self._reward

    def test_plus(self):
        """Tests that BaseRewarders can be combined with +."""

        testee = BaseRewarderTest._BaseRewarder(
            numpy.array([1, 2])
        ) + BaseRewarderTest._BaseRewarder(numpy.array([3, 4]))

        testing.assert_allclose(
            testee.reward(numpy.array([]), numpy.array([])), numpy.array([4, 6])
        )

    def test_times(self):
        """Tests that BaseRewarders can be combined with *."""

        testee = BaseRewarderTest._BaseRewarder(
            numpy.array([1, 2])
        ) * BaseRewarderTest._BaseRewarder(numpy.array([3, 4]))

        testing.assert_allclose(
            testee.reward(numpy.array([]), numpy.array([])), numpy.array([3, 8])
        )


class DeltaRewarderTest(unittest.TestCase):
    """Test cases for reinfocus.environments.episode_rewarder.DeltaRewarder."""

    def test_reward(self):
        """Tests that DeltaRewarder only gives out rewards proportional to the change in
        state."""

        testee = episode_rewarder.DeltaRewarder(1, 2)

        testee.reset(numpy.array([[4, 1], [3, 2], [2, 3], [1, 4]]), numpy.array([]))

        testing.assert_allclose(
            testee.reward(numpy.array([[4, 1], [3, 4], [2, 1], [1, 5]]), numpy.array([])),
            [0, -1, -1, -0.5],
        )

        testing.assert_allclose(
            testee.reward(
                numpy.array([[4, 0.6], [3, 1], [2, 4], [1, 3.5]]), numpy.array([])
            ),
            [-0.2, -1.5, -1.5, -0.75],
        )

    def test_reset(self):
        """Tests that reset partially resets some environments of the rewarder."""

        testee = episode_rewarder.DeltaRewarder(1, 2)

        testee.reset(numpy.array([[4, 1], [3, 2], [2, 3], [1, 4]]), numpy.array([]))

        testing.assert_allclose(
            testee.reward(numpy.array([[4, 1], [3, 4], [2, 1], [1, 5]]), numpy.array([])),
            [0, -1, -1, -0.5],
        )

        testee.reset(
            numpy.array([[3, 2], [1, 4]]),
            numpy.array([]),
            numpy.array([False, True, False, True]),
        )

        testing.assert_allclose(
            testee.reward(
                numpy.array([[4, 0.6], [3, 1], [2, 4], [1, 3.5]]), numpy.array([])
            ),
            [-0.2, -0.5, -1.5, -0.25],
        )


class DistanceRewarderTest(unittest.TestCase):
    """Test cases for reinfocus.environments.episode_rewarder.DistanceRewarder."""

    def test_reward(self):
        """Tests that DistanceRewarder creates a rewarder that gives some reward
        proportional to the distance between two given state elements."""

        testing.assert_allclose(
            episode_rewarder.DistanceRewarder((0, 1), 1, -3, 7).reward(
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
            episode_rewarder.ObservationRewarder(0).reward(numpy.array([]), observations),
            [4, 3, 2, 1],
        )

        testing.assert_allclose(
            episode_rewarder.ObservationRewarder(1).reward(numpy.array([]), observations),
            [1, 2, 3, 4],
        )


class OnTargetRewarderTest(unittest.TestCase):
    """Test cases for reinfocus.environments.episode_rewarder.OnTargetRewarder."""

    def test_reward(self):
        """Tests that OnTargetRewarder gives the proper reward when the two given state
        elements are and aren't close enough."""

        testing.assert_allclose(
            episode_rewarder.OnTargetRewarder((0, 1), 0.1, -3, 7).reward(
                numpy.array([[0.5, 0.65], [0.5, 0.55], [0.5, 0.45], [0.5, 0.35]]),
                numpy.array([]),
            ),
            [-3, 7, 7, -3],
        )


class OpRewarderTest(unittest.TestCase):
    """Test cases for reinfocus.environments.episode_rewarder.OpRewarder."""

    def test_reset(self):
        """Tests that OpRewarder passes the reset signal to it's child rewarders."""

        l_rewarder = mock.Mock()
        r_rewarder = mock.Mock()

        states = numpy.ones((3, 2), dtype=numpy.float32)
        observations = numpy.zeros((3, 1), dtype=numpy.float32)
        dones = numpy.arange(3) % 2 == 0

        episode_rewarder.OpRewarder(l_rewarder, r_rewarder, lambda l, r: l + r).reset(
            states, observations, dones
        )

        l_rewarder.reset.assert_called_once_with(states, observations, dones)
        r_rewarder.reset.assert_called_once_with(states, observations, dones)

    def test_reward(self):
        """Tests that OpRewarder properly combines the rewards from it's child
        rewarders."""

        l_rewarder = mock.Mock()
        l_rewarder.reward.return_value = numpy.array([1, 2, 3, 4], dtype=numpy.float32)

        r_rewarder = mock.Mock()
        r_rewarder.reward.return_value = numpy.array([1, 3, 5, 7], dtype=numpy.float32)

        testing.assert_allclose(
            episode_rewarder.OpRewarder(
                l_rewarder, r_rewarder, lambda l, r: l - r
            ).reward(numpy.array([]), numpy.array([])),
            [0, -1, -2, -3],
        )


class StoppedRewarderTest(unittest.TestCase):
    """Test cases for reinfocus.environments.episode_rewarder.StoppedRewarder."""

    def test_reward(self):
        """Tests that StoppedRewarder only gives out rewards when some state element is
        stopped."""

        testee = episode_rewarder.StoppedRewarder(1, 1.5, reward=3)

        testee.reset(numpy.array([[4, 1], [3, 2], [2, 3], [1, 4]]), numpy.array([]))

        testing.assert_allclose(
            testee.reward(numpy.array([[4, 1], [3, 4], [2, 1], [1, 5]]), numpy.array([])),
            [3, 0, 0, 3],
        )

        testing.assert_allclose(
            testee.reward(numpy.array([[4, 0], [3, 3], [2, 3], [1, 3]]), numpy.array([])),
            [3, 3, 0, 0],
        )

    def test_reset(self):
        """Tests that reset partially resets some environments of the rewarder."""

        testee = episode_rewarder.StoppedRewarder(1, 1.5)

        testee.reset(numpy.array([[4, 1], [3, 2], [2, 3], [1, 4]]), numpy.array([]))

        testing.assert_allclose(
            testee.reward(numpy.array([[4, 1], [3, 4], [2, 1], [1, 5]]), numpy.array([])),
            [1, 0, 0, 1],
        )

        testee.reset(
            numpy.array([[4, 2], [2, 2]]),
            numpy.array([]),
            numpy.array([True, False, True, False]),
        )

        testing.assert_allclose(
            testee.reward(numpy.array([[4, 0], [3, 3], [2, 3], [1, 3]]), numpy.array([])),
            [0, 1, 1, 0],
        )


if __name__ == "__main__":
    unittest.main()
