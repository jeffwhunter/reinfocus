"""Contains tests for reinfocus.environments.state_observer."""

import unittest

from collections.abc import Callable, Sequence
from unittest import mock

import numpy

from gymnasium import spaces
from gymnasium.vector import utils
from numba.cuda import testing as cuda_testing
from numpy import testing
from numpy.typing import NDArray

from reinfocus.environments import state_observer
from reinfocus.graphics import render


def make_observer(
    num_envs: int = 1,
    space: spaces.Space = spaces.Box(0, 1),
    observe: Callable[
        [NDArray[numpy.float32], NDArray[numpy.bool_] | None], NDArray[numpy.float32]
    ] = lambda state, _: state,
) -> state_observer.BaseObserver:
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
    observer.reset.side_effect = lambda state, dones: observe(state, dones)
    return observer


class BaseObserverTest(unittest.TestCase):
    """Test cases for reinfocus.environments.state_observer.BaseObserver."""

    def test_scalar_spaces(self):
        """Tests that base observers have the correct observation spaces."""

        class _ScalarObserver(state_observer.BaseObserver):
            # pylint: disable=too-few-public-methods
            def observe(
                self,
                states: NDArray[numpy.float32],
                dones: NDArray[numpy.bool_] | None = None,
            ) -> NDArray[numpy.float32]:
                """A dummy function needed to implement BaseObserver."""

                raise NotImplementedError

        num_envs = 5
        low = 0
        high = 10

        testee = _ScalarObserver(num_envs, low, high)

        self.assertEqual(testee.single_observation_space.shape, (1,))
        self.assertEqual(testee.single_observation_space.low, low)
        self.assertEqual(testee.single_observation_space.high, high)

        target_shape = (num_envs, 1)

        self.assertEqual(testee.observation_space.shape, target_shape)
        testing.assert_allclose(
            testee.observation_space.low, numpy.full(target_shape, low)
        )
        testing.assert_allclose(
            testee.observation_space.high, numpy.full(target_shape, high)
        )

    def test_array_spaces(self):
        """Tests that base observers have the correct multidimensional observation
        spaces."""

        class _ArrayObserver(state_observer.BaseObserver):
            # pylint: disable=too-few-public-methods
            def observe(
                self,
                states: NDArray[numpy.float32],
                dones: NDArray[numpy.bool_] | None = None,
            ) -> NDArray[numpy.float32]:
                """A dummy function needed to implement BaseObserver."""

                raise NotImplementedError

        num_envs = 5
        num_obs = 3
        low = numpy.array([1, 2, 3])
        high = numpy.array([2, 4, 6])

        testee = _ArrayObserver(num_envs, low, high)

        self.assertTupleEqual(testee.single_observation_space.shape, (num_obs,))
        testing.assert_allclose(testee.single_observation_space.low, low)
        testing.assert_allclose(testee.single_observation_space.high, high)

        target_shape = (num_envs, num_obs)

        self.assertTupleEqual(testee.observation_space.shape, target_shape)
        testing.assert_allclose(
            testee.observation_space.low, numpy.full(target_shape, low)
        )
        testing.assert_allclose(
            testee.observation_space.high, numpy.full(target_shape, high)
        )


class WrapperObserverTest(unittest.TestCase):
    """Test cases for reinfocus.environments.state_observer.WrapperObserver."""

    class _WrapperObserver(state_observer.WrapperObserver):
        # pylint: disable=too-few-public-methods
        def observe(
            self,
            states: NDArray[numpy.float32],
            dones: NDArray[numpy.bool_] | None = None,
        ) -> NDArray[numpy.float32]:
            """A dummy function needed to implement WrapperObserver."""

            raise NotImplementedError

    def test_spaces(self):
        """Tests that wrapper observers have the correct observation spaces."""

        num_envs = 5

        observation_sizes = [2, 3]

        n_observations = sum(observation_sizes)

        testee = WrapperObserverTest._WrapperObserver(
            [
                make_observer(
                    num_envs=num_envs, space=spaces.Box(0, 1, (observation_size,))
                )
                for observation_size in observation_sizes
            ],
            numpy.zeros(n_observations),
            numpy.ones(n_observations),
        )

        self.assertEqual(testee.observation_space.shape, (num_envs, n_observations))

    def test_wrapped_observations(self):
        """Tests that wrapper observers correctly append observations from the wrapped
        observers."""

        num_envs = 3

        observer_coefficients = [[-1, 1], [-1, 1, -1]]

        n_observations = sum(len(coefficients) for coefficients in observer_coefficients)

        def _multiplier(coefficients: Sequence[float]):
            return lambda state, _: numpy.hstack(
                [state * coefficient for coefficient in coefficients]
            )

        testee = WrapperObserverTest._WrapperObserver(
            [
                make_observer(
                    num_envs=num_envs,
                    space=spaces.Box(0, 1, (len(coefficients),)),
                    observe=_multiplier(coefficients),
                )
                for coefficients in observer_coefficients
            ],
            numpy.zeros(n_observations),
            numpy.ones(n_observations),
        )

        state = numpy.reshape([1, 2, 3], (num_envs, 1))
        target = [[-1, 1, -1, 1, -1], [-2, 2, -2, 2, -2], [-3, 3, -3, 3, -3]]

        testing.assert_allclose(testee.wrapped_observations(state), target)
        testing.assert_allclose(testee.wrapped_observations(state[::-1]), target[::-1])

    def test_reset(self):
        """Tests that reset calls reset on the wrapped observers."""

        observers = [make_observer() for _ in range(3)]

        WrapperObserverTest._WrapperObserver(observers, 0, 1).reset(
            numpy.zeros((1, 1), dtype=numpy.float32)
        )

        for observer in observers:
            observer.reset.assert_called_once()


class DeltaObserverTest(unittest.TestCase):
    """Test cases for reinfocus.environments.state_observer.DeltaObserver."""

    def test_spaces(self):
        """Tests that the observation space has an appropriately sized low and high."""

        num_envs = 4

        original_low = 2
        original_high = 5

        target_diff = original_high - original_low

        observation_space = state_observer.DeltaObserver(
            make_observer(num_envs, spaces.Box(original_low, original_high))
        ).observation_space

        target_shape = (num_envs, 1)

        testing.assert_allclose(
            observation_space.low, numpy.full(target_shape, -target_diff)
        )

        testing.assert_allclose(
            observation_space.high, numpy.full(target_shape, target_diff)
        )

    def test_spaces_with_max_change(self):
        """Tests that the observation space has an appropriately sized low and high when
        max_change is set."""

        num_envs = 4

        target_diff = 1

        observation_space = state_observer.DeltaObserver(
            make_observer(num_envs, spaces.Box(2, 5)), max_change=target_diff
        ).observation_space

        target_shape = (num_envs, 1)

        testing.assert_allclose(
            observation_space.low, numpy.full(target_shape, -target_diff)
        )

        testing.assert_allclose(
            observation_space.high, numpy.full(target_shape, target_diff)
        )

    def test_spaces_with_multidimensional_max_change(self):
        """Tests that the observation space has an appropriately sized low and high when
        max_change is set to some multidimensional value."""

        num_envs = 4

        observation_space = state_observer.DeltaObserver(
            [
                make_observer(num_envs, spaces.Box(2, 5)),
                make_observer(
                    num_envs, spaces.Box(numpy.array([3, 6]), numpy.array([7, 9]))
                ),
            ],
            max_change=numpy.array([1, 2, numpy.nan]),
        ).observation_space

        target_shape = (num_envs, 3)

        testing.assert_allclose(
            observation_space.low, numpy.full(target_shape, [-1, -2, -3])
        )

        testing.assert_allclose(
            observation_space.high, numpy.full(target_shape, [1, 2, 3])
        )

    def test_spaces_with_original(self):
        """Tests that the observation space has an appropriately sized low and high when
        also including the original observations."""

        num_envs = 4

        original_low = numpy.array([1, 2, 3])
        original_high = numpy.array([2, 4, 6])

        target_diff = original_high - original_low

        observation_space = state_observer.DeltaObserver(
            make_observer(num_envs, spaces.Box(original_low, original_high)),
            include_original=True,
        ).observation_space

        target_shape = (num_envs, original_low.shape[0] * 2)

        testing.assert_allclose(
            observation_space.low,
            numpy.full(target_shape, numpy.append(original_low, -target_diff)),
        )

        testing.assert_allclose(
            observation_space.high,
            numpy.full(target_shape, numpy.append(original_high, target_diff)),
        )

    def test_observation(self):
        """Tests that DeltaObserver correctly returns observations that are the change in
        observations of some wrapped observer."""

        num_envs = 3
        state_shape = (num_envs, 1)

        observer = make_observer(num_envs=num_envs)

        testee = state_observer.DeltaObserver(observer)

        testing.assert_allclose(
            testee.reset(numpy.reshape([0, 1, 2], state_shape)), [[0], [0], [0]]
        )

        testing.assert_allclose(
            testee.observe(numpy.reshape([0, 2, 4], state_shape)), [[0], [1], [2]]
        )

    def test_partial_observation(self):
        """Tests that partial observations correctly observe only some environments."""

        num_envs = 4

        observer = make_observer(num_envs=num_envs)

        testee = state_observer.DeltaObserver(observer)

        testing.assert_allclose(
            testee.reset(numpy.reshape([0, 1, 2, 3], (num_envs, 1))), [[0], [0], [0], [0]]
        )

        testing.assert_allclose(
            testee.observe(
                numpy.reshape([3, 6], (2, 1)), numpy.array([False, True, False, True])
            ),
            [[2], [3]],
        )

        testing.assert_allclose(
            testee.observe(
                numpy.reshape([-2, 1], (2, 1)), numpy.array([True, True, False, False])
            ),
            [[-2], [-2]],
        )

    def test_observation_with_original(self):
        """Tests that DeltaObserver correctly returns observations from some wrapped
        observer appened to changes in those observations over time."""

        num_envs = 3
        state_shape = (num_envs, 1)

        observer = make_observer(
            num_envs=num_envs,
            space=spaces.Box(0, 1, (2,)),
            observe=lambda state, _: numpy.hstack([-state, state]),
        )

        testee = state_observer.DeltaObserver(observer, include_original=True)

        testing.assert_allclose(
            testee.reset(numpy.reshape([0, 1, 2], state_shape)),
            [[0, 0, 0, 0], [-1, 1, 0, 0], [-2, 2, 0, 0]],
        )

        testing.assert_allclose(
            testee.observe(numpy.reshape([0, 3, 6], state_shape)),
            [[0, 0, 0, 0], [-3, 3, -2, 2], [-6, 6, -4, 4]],
        )

    def test_observation_with_reset(self):
        """Tests that DeltaObserver correctly observes no change immediately after
        being reset."""

        num_envs = 2
        state_shape = (num_envs, 1)

        observer = make_observer(num_envs=num_envs)

        testee = state_observer.DeltaObserver(observer)

        testing.assert_allclose(
            testee.reset(numpy.reshape([0, 1], state_shape)), [[0], [0]]
        )

        testing.assert_allclose(
            testee.reset(numpy.array([[3]]), numpy.array([False, True])), [[0]]
        )

        testing.assert_allclose(
            testee.observe(numpy.reshape([2, 4], state_shape)), [[2], [1]]
        )

    def test_multidimensional_observation(self):
        """Tests that DeltaObserver correctly observes the changes in multiple
        observers."""

        num_envs = 4
        state_shape = (num_envs, 1)

        observers = [
            make_observer(num_envs=num_envs),
            make_observer(num_envs=num_envs, observe=lambda state, _: -state),
        ]

        testee = state_observer.DeltaObserver(observers)

        testing.assert_allclose(
            testee.reset(numpy.reshape([0, 1, 2, 3], state_shape)),
            [[0, 0], [0, 0], [0, 0], [0, 0]],
        )

        testing.assert_allclose(
            testee.observe(numpy.reshape([0, 2, 4, 6], state_shape)),
            [[0, 0], [1, -1], [2, -2], [3, -3]],
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

        renderer = render.FastRenderer()

        testees = [
            state_observer.FocusObserver(num_envs, 0, 1, ends, renderer),
            state_observer.FocusObserver(num_envs, 1, 0, ends, renderer),
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

    def test_partial_observation(self):
        """Tests that partial observations correctly observe only some environments."""

        num_envs = 5
        ends = (5, 10)

        renderer = render.FastRenderer()

        testee = state_observer.FocusObserver(num_envs, 0, 1, ends, renderer)

        dones = numpy.array([True, False, True, False, True])

        observations = testee.observe(
            numpy.array([[5, 10], [7.5, 10], [10, 10]]),
            dones,
        )

        self.assertTrue(numpy.all(observations[1:] > observations[:-1]))
        self.assertEqual(len(observations), dones.sum())


class IndexedElementObserverTest(unittest.TestCase):
    """Test cases for reinfocus.environments.state_observer.IndexedElementObserver."""

    def test_observation(self):
        """Tests that IndexedElementObserver correctly observes the given state
        element."""

        num_envs = 5
        target_shape = (num_envs, 1)

        state = numpy.array([[0, 1], [1, 3], [2, 5], [3, 7], [4, 9]])

        testing.assert_allclose(
            state_observer.IndexedElementObserver(num_envs, 0, 0, 4).observe(state),
            numpy.reshape([0, 1, 2, 3, 4], target_shape),
        )

        testing.assert_allclose(
            state_observer.IndexedElementObserver(num_envs, 1, 0, 9).observe(state),
            numpy.reshape([1, 3, 5, 7, 9], target_shape),
        )

    def test_partial_observation(self):
        """Tests that partial observations correctly observe only some environments."""

        num_envs = 5

        state = numpy.array([[0, 1], [1, 3], [2, 5], [3, 7], [4, 9]])
        even_i = numpy.arange(num_envs) % 2 == 0
        odd_i = numpy.arange(num_envs) % 2 == 1

        testing.assert_allclose(
            state_observer.IndexedElementObserver(num_envs, 0, 0, 4).observe(
                state[even_i], even_i
            ),
            numpy.reshape([0, 2, 4], (3, 1)),
        )

        testing.assert_allclose(
            state_observer.IndexedElementObserver(num_envs, 1, 0, 9).observe(
                state[odd_i], odd_i
            ),
            numpy.reshape([3, 7], (2, 1)),
        )


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

    def test_multidimensional_spaces(self):
        """Tests that NormalizedObserver has the correct multidimensional observation
        spaces."""

        num_envs = 3

        observers = [
            make_observer(num_envs, spaces.Box(numpy.array([1, 2]), numpy.array([3, 4]))),
            make_observer(num_envs, spaces.Box(numpy.array([2, 4]), numpy.array([6, 8]))),
        ]

        n_observers = len(observers)

        testee = state_observer.NormalizedObserver(observers)

        self.assertEqual(testee.observation_space.shape, (num_envs, 2 * n_observers))
        self.assertEqual(testee.single_observation_space.shape, (2 * n_observers,))

    def test_observation(self):
        """Tests that NormalizedObserver correctly normalizes observations."""

        testee = state_observer.NormalizedObserver(
            [
                make_observer(space=spaces.Box(0, 2)),
                make_observer(space=spaces.Box(1, 4)),
            ]
        )

        testing.assert_allclose(
            testee.observe(numpy.array([[0], [1], [2], [3], [4]])),
            [[-1.0, -1.0], [0.0, -1.0], [1.0, -(1 / 3)], [1.0, (1 / 3)], [1.0, 1.0]],
        )

    def test_partial_observation(self):
        """Tests that partial observations correctly observe only some environments."""

        dones = numpy.arange(5) % 2 == 0

        testee = state_observer.NormalizedObserver(
            [
                make_observer(space=spaces.Box(0, 2)),
                make_observer(space=spaces.Box(1, 4)),
            ]
        )

        testing.assert_allclose(
            testee.observe(numpy.array([[0], [2], [4]]), dones),
            [[-1.0, -1.0], [1.0, -(1 / 3)], [1.0, 1.0]],
        )

    def test_reset(self):
        """Tests that NormalizedObserver correctly normalizes observations from reset."""

        testee = state_observer.NormalizedObserver(
            [
                make_observer(space=spaces.Box(0, 2)),
                make_observer(space=spaces.Box(1, 4)),
            ]
        )

        testing.assert_allclose(
            testee.reset(numpy.array([[0], [1], [2], [3], [4]])),
            [[-1.0, -1.0], [0.0, -1.0], [1.0, -(1 / 3)], [1.0, (1 / 3)], [1.0, 1.0]],
        )

    def test_multidimensional_observation(self):
        """Tests that NormalizedObserver correctly normalizes multidimensional
        observations."""

        num_envs = 9

        testee = state_observer.NormalizedObserver(
            [
                make_observer(
                    num_envs=num_envs,
                    space=spaces.Box(numpy.array([-2, -1]), numpy.array([-1, 0])),
                    observe=lambda state, _: numpy.hstack([state, -state]),
                ),
                make_observer(
                    num_envs=num_envs,
                    space=spaces.Box(numpy.array([0, 1]), numpy.array([1, 2])),
                    observe=lambda state, _: numpy.hstack([-state, state]),
                ),
            ]
        )

        testing.assert_allclose(
            testee.observe(numpy.linspace(-2.0, 2.0, num_envs).reshape(9, 1)),
            [
                [-1.0, 1.0, 1.0, -1.0],
                [0.0, 1.0, 1.0, -1.0],
                [1.0, 1.0, 1.0, -1.0],
                [1.0, 1.0, 0.0, -1.0],
                [1.0, 1.0, -1.0, -1.0],
                [1.0, 0.0, -1.0, -1.0],
                [1.0, -1.0, -1.0, -1.0],
                [1.0, -1.0, -1.0, 0.0],
                [1.0, -1.0, -1.0, 1.0],
            ],
        )


if __name__ == "__main__":
    unittest.main()
