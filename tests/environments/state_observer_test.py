"""Contains tests for reinfocus.environments.state_observer."""

import unittest

from collections.abc import Callable
from unittest import mock

import numpy

from gymnasium import spaces
from gymnasium.vector import utils
from numba.cuda import testing as cuda_testing
from numpy import testing
from numpy.typing import NDArray

from reinfocus.environments import state_observer
from reinfocus.graphics import world


def make_observer(
    num_envs: int = 0,
    space: spaces.Space = spaces.Box(0, 1),
    observe: Callable[
        [NDArray[numpy.float32]], NDArray[numpy.float32]
    ] = lambda state: state,
) -> state_observer.ScalarObserver[numpy.float32, NDArray[numpy.float32]]:
    """Creates a mock scalar observer.

    Args:
        num_envs: The number of environments this scalar observer will observe.
        space: The observation space from a single environment that will be observed.
        observe: A function that observes a batch of states and returns observations.

    Returns:
        A mocked scalar observer."""

    observer = mock.Mock()
    observer.single_observation_space = space
    observer.observation_space = utils.batch_space(space, num_envs)
    observer.observe.side_effect = observe
    return observer


class ScalarObserverTest(unittest.TestCase):
    """Test cases for reinfocus.environments.state_observer.ScalarObserver."""

    def test_spaces(self):
        """Tests that scalar observers have the correct observation spaces."""

        class _ScalarObserver(state_observer.ScalarObserver):
            # pylint: disable=too-few-public-methods
            def observe(self, state: NDArray[numpy.float32]) -> NDArray[numpy.float32]:
                """A dummy function needed to implement ScalarObserver."""

                raise NotImplementedError

        num_envs = 5
        low = 0
        high = 10

        testee = _ScalarObserver(num_envs, low, high)

        self.assertEqual(testee.single_observation_space.shape, (1,))
        self.assertEqual(testee.single_observation_space.low, low)
        self.assertEqual(testee.single_observation_space.high, high)

        self.assertEqual(testee.observation_space.shape, (num_envs,))
        testing.assert_allclose(testee.observation_space.low, numpy.full(num_envs, low))
        testing.assert_allclose(testee.observation_space.high, numpy.full(num_envs, high))


class NormalizedObserverTest(unittest.TestCase):
    """Test cases for reinfocus.environments.state_observer.NormalizedObserver."""

    def test_checks_num_envs(self):
        """Tests that NormalizedObserver checks that it's children observe the same number
        of environments."""

        self.assertRaises(
            AssertionError,
            lambda: state_observer.NormalizedObserver(
                [make_observer(3), make_observer(2)]
            ),
        )

    def test_spaces(self):
        """Tests that NormalizedObserver has the correct observation spaces."""

        num_envs = 3

        observers = [
            make_observer(num_envs, spaces.Box(1, 2)),
            make_observer(num_envs, spaces.Box(3, 5)),
        ]

        n_observers = len(observers)

        testee = state_observer.NormalizedObserver(observers)

        self.assertEqual(testee.observation_space.shape, (num_envs, n_observers))
        self.assertEqual(testee.single_observation_space.shape, (n_observers,))

    def test_observation(self):
        """Tests that NormalizedObserver correctly normalizes observations."""

        testee = state_observer.NormalizedObserver(
            [
                make_observer(space=spaces.Box(0, 2)),
                make_observer(space=spaces.Box(1, 4)),
            ]
        )

        testing.assert_allclose(
            testee.observe(numpy.arange(5, dtype=numpy.float32)),
            [[-1.0, -1.0], [0.0, -1.0], [1.0, -(1 / 3)], [1.0, (1 / 3)], [1.0, 1.0]],
        )


class IndexedElementObserverTest(unittest.TestCase):
    """Test cases for reinfocus.environments.state_observer.IndexedElementObserver."""

    def test_observation(self):
        """Tests that IndexedElementObserver correctly observes the given state
        element."""

        state = numpy.array([[0, 1], [1, 3], [2, 5], [3, 7], [4, 9]])

        testing.assert_allclose(
            state_observer.IndexedElementObserver(5, 0, 0, 4).observe(state),
            [0, 1, 2, 3, 4],
        )

        testing.assert_allclose(
            state_observer.IndexedElementObserver(5, 1, 0, 9).observe(state),
            [1, 3, 5, 7, 9],
        )


class FocusObserverTest(cuda_testing.CUDATestCase):
    """Test cases for reinfocus.environments.state_observer.FocusObserver."""

    def test_spaces(self):
        """Tests that the observation space has a minimum lower than it's maximum."""

        observation_space = state_observer.FocusObserver(
            0, 0, 1, (0, 1), mock.Mock()
        ).observation_space

        testing.assert_array_less(observation_space.low, observation_space.high)

    def test_observation(self):
        """Tests that FocusObserver correctly returns observations that increase as the
        focus plane moves towards the target."""

        num_envs = 5
        ends = (5, 10)

        worlds = world.FocusWorlds(num_envs)

        testees = [
            state_observer.FocusObserver(3, 0, 1, ends, worlds),
            state_observer.FocusObserver(3, 1, 0, ends, worlds),
        ]

        for mid_point in numpy.linspace(*ends, num_envs):
            state = numpy.vstack(
                [
                    numpy.linspace(ends[0], mid_point, num_envs),
                    numpy.linspace(ends[1], mid_point, num_envs),
                ]
            ).T

            observations = numpy.vstack([testee.observe(state) for testee in testees])

            self.assertTrue(numpy.all(observations[:, 1:] > observations[:, :-1]))


if __name__ == "__main__":
    unittest.main()
