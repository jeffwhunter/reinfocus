"""Contains tests for reinfocus.environments.vector_environment."""

import unittest

from collections.abc import Callable
from unittest import mock

import numpy

from numpy import testing
from numpy.typing import NDArray

from reinfocus.environments import vector_environment
from reinfocus.environments.types import ActionT, IState, StateT


def make_ender(
    is_terminated: NDArray[numpy.bool_] = numpy.full(1, False),
    is_truncated: NDArray[numpy.bool_] = numpy.full(1, False),
) -> mock.Mock:
    """Creates a mock episode ender.

    Args:
        is_terminated: The return value from is_terminated called on this episode ender.
        is_trunated: The return value from is_truncated called on this episode ender.

    Returns:
        A mocked episode ender."""

    ender = mock.Mock()
    ender.is_terminated.return_value = is_terminated
    ender.is_truncated.return_value = is_truncated
    return ender


def make_initializer(*initial_states: IState) -> mock.Mock:
    """Creates a mocked state initializer.

    Args:
        initial_states: An array of states objects that will be iterated through when
            initialize is called on this state initializer.

    Returns:
        A mocked state initializer."""
    
    assert len(initial_states) > 0

    initializer = mock.Mock()
    if len(initial_states) > 1:
        initializer.initialize.side_effect = initial_states
    else:
        initializer.initialize.return_value = initial_states[0]
    return initializer


def make_observer(
    observation_space: mock.Mock = mock.Mock(),
    single_observation_space: mock.Mock = mock.Mock(),
) -> mock.Mock:
    """Creates a mocked state observer.

    Args:
        observation_space: The batch observation space this observer will observe.
        single_observation_space: An individual observation space from the batch space.

    Returns:
        A mocked state observer."""

    observer = mock.Mock()
    observer.observe.side_effect = lambda state: state
    observer.reset.side_effect = lambda state, _: state
    observer.observation_space = observation_space
    observer.single_observation_space = single_observation_space
    return observer


def make_transformer(
    transform: Callable[[StateT, NDArray[ActionT]], StateT] = lambda s, _: s,
    action_space: mock.Mock = mock.Mock(),
    single_action_space: mock.Mock = mock.Mock(),
) -> mock.Mock:
    """Creates a mocked state transformer.

    Args:
        transform: A function that transforms one batch of states into another, given a
            batch of actions.
        action_space: The batch action space this transformer will transform with.
        single_action_space: An individual action space from the batch space.

    Returns:
        A mocked state transformer."""

    transformer = mock.Mock()
    transformer.transform.side_effect = transform
    transformer.action_space = action_space
    transformer.single_action_space = single_action_space
    return transformer


def make_testee(
    ender: mock.Mock = make_ender(),
    initializer: mock.Mock = mock.Mock(),
    observer: mock.Mock = make_observer(),
    rewarder: mock.Mock = mock.Mock(),
    transformer: mock.Mock = make_transformer(),
    visualizer: mock.Mock = mock.Mock(),
    num_envs: int = 1,
    render_mode: str | None = None,
) -> vector_environment.VectorEnvironment:
    # pylint: disable=too-many-arguments
    """A convenience function for creating VectorEnvironments.

    Args:
        ender: An episode ender.
        initializer: A state initializer.
        observer: A state observer.
        rewarder: An episode rewarder.
        transformer: A state transformer.
        visualizer: An environment visualizer.
        num_envs: The number of individual environments to simulate.
        render_mode: The render mode of the created environments."""

    return vector_environment.VectorEnvironment(
        ender=ender,
        initializer=initializer,
        observer=observer,
        rewarder=rewarder,
        transformer=transformer,
        visualizer=visualizer,
        num_envs=num_envs,
        render_mode=render_mode,
    )


class VectorEnvironmentTest(unittest.TestCase):
    """Test cases for reinfocus.environments.vector_environment.VectorEnvironment."""

    def test_spaces(self):
        """Tests that the environment reports appropriate spaces retreived from its state
        observer and transformer."""

        target_action_space = mock.Mock()
        target_observation_space = mock.Mock()
        target_single_action_space = mock.Mock()
        target_single_observation_space = mock.Mock()

        testee = make_testee(
            observer=make_observer(
                observation_space=target_observation_space,
                single_observation_space=target_single_observation_space,
            ),
            transformer=make_transformer(
                action_space=target_action_space,
                single_action_space=target_single_action_space,
            ),
        )

        self.assertEqual(testee.action_space, target_action_space)
        self.assertEqual(testee.single_action_space, target_single_action_space)
        self.assertEqual(testee.observation_space, target_observation_space)
        self.assertEqual(testee.single_observation_space, target_single_observation_space)

    def test_initialization(self):
        """Tests that the environment correctly initializes states with its
        initializer."""

        target = numpy.array([[-4], [8]])

        testing.assert_allclose(
            make_testee(initializer=make_initializer(target)).reset()[0], target
        )

    def test_transforms(self):
        """Tests that the environment correctly transforms states with it's
        transformer."""

        target = numpy.array([[-4], [8]])

        testee = make_testee(
            initializer=make_initializer(target),
            transformer=make_transformer(lambda state, _: numpy.flip(state)),
        )

        testee.reset()

        testing.assert_allclose(testee.step(numpy.empty(0))[0], numpy.flip(target))

    def test_reward(self):
        """Tests that the environment correctly rewards episodes with it's rewarder."""

        rewarder = mock.Mock()
        rewarder.reward.side_effect = lambda s, o, a: s[:, 0] + o[:, 0] + a[:, 0]

        testee = make_testee(
            initializer=make_initializer(numpy.array([[-4], [8]])), rewarder=rewarder
        )

        testee.reset()

        testing.assert_allclose(testee.step(numpy.array([[3], [5]]))[1], [-5, 21])

    def test_terminated_and_truncated(self):
        """Tests that the environment correctly reports termination and truncation from
        its episode ender."""

        testee = make_testee(
            ender=make_ender(
                is_terminated=numpy.array([True, False]),
                is_truncated=numpy.array([False, True]),
            ),
            initializer=make_initializer(numpy.array([[-4], [8]])),
        )

        testee.reset()

        is_terminated, is_truncated = testee.step(numpy.empty(0))[2:4]

        testing.assert_allclose(is_terminated, (True, False))
        testing.assert_allclose(is_truncated, (False, True))

    def test_done_are_reset(self):
        """Tests that the environment reinitializes individual episodes that the episode
        ender has reported are finished."""

        ender = make_ender(is_terminated=numpy.array([True, False]))

        initializer = make_initializer(numpy.array([[-4], [8]]), numpy.array([[5]]))

        testee = make_testee(ender=ender, initializer=initializer)

        testee.reset()

        testing.assert_allclose(testee.step(numpy.empty(0))[0], numpy.array([[5], [8]]))

    def test_reset_ender(self):
        """Tests that reset resets the episode ender."""

        ender = make_ender()

        testee = make_testee(ender=ender)

        testee.reset()

        ender.reset.assert_called_once()

    def test_reset_individual_ender(self):
        """Tests that individual environments of the episode ender are reset during step
        after the episode ender reports the end of an episode."""

        state_targets = numpy.array([[5]])
        done_targets = numpy.array([True, False])

        ender = make_ender(is_terminated=done_targets)

        testee = make_testee(
            ender=ender,
            initializer=make_initializer(numpy.array([[-4], [8]]), state_targets),
            num_envs=2,
        )

        testee.reset()

        ender.reset.assert_called_once()

        testee.step(numpy.empty(0))

        testing.assert_allclose(ender.reset.call_args.args[0], state_targets)
        testing.assert_allclose(ender.reset.call_args.args[1], done_targets)

    def test_reset_observer(self):
        """Tests that reset resets the state observer."""

        observer = make_observer()

        testee = make_testee(observer=observer)

        testee.reset()

        observer.reset.assert_called_once()

    def test_reset_individual_observer(self):
        """Tests that individual environments of the state observer are reset during step
        after the episode ender reports the end of an episode."""

        state_targets = numpy.array([[5]])
        done_targets = numpy.array([True, False])

        observer = make_observer()

        testee = make_testee(
            ender=make_ender(is_terminated=done_targets),
            initializer=make_initializer(numpy.array([[-4], [8]]), state_targets),
            observer=observer,
            num_envs=2,
        )

        testee.reset()

        observer.reset.assert_called_once()

        testee.step(numpy.empty(0))

        testing.assert_allclose(observer.reset.call_args.args[0], state_targets)
        testing.assert_allclose(observer.reset.call_args.args[1], done_targets)

    def test_no_render(self):
        """Tests that the environment doesn't render anything if it's render_mode isn't
        set."""

        testee = make_testee()

        testee.reset()

        self.assertIsNone(testee.render())

    def test_rgb_array_render(self):
        """Tests that the environment renders something produced by the visualzer when its
        render mode is set."""

        target = mock.Mock()

        visualizer = mock.Mock()
        visualizer.visualize.return_value = target

        testee = make_testee(visualizer=visualizer, render_mode="rgb_array")

        testee.reset()

        self.assertEqual(testee.render(), target)


if __name__ == "__main__":
    unittest.main()
