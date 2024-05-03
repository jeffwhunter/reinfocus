"""Contains tests for reinfocus.environments.environment."""

import unittest

from unittest import mock

import numpy

from numpy import testing

from reinfocus.environments import environment
from reinfocus.environments.types import IState


def make_ender(is_terminated: bool = False, is_truncated: bool = False) -> mock.Mock:
    """Creates a mock episode ender.

    Args:
        is_terminated: The return value from is_terminated called on this episode ender.
        is_trunated: The return value from is_truncated called on this episode ender.

    Returns:
        A mocked episode ender."""

    episode_ender = mock.Mock()
    episode_ender.is_terminated.return_value = [is_terminated]
    episode_ender.is_truncated.return_value = [is_truncated]
    return episode_ender


def make_initializer(initial_state: IState = numpy.zeros((1, 1))) -> mock.Mock:
    """Creates a mocked state initializer.

    Args:
        initial_state: An state objected that will be returned when initialize is called
        on this initializer.

    Returns:
        A mocked state initializer."""

    state_initializer = mock.Mock()
    state_initializer.initialize.return_value = initial_state
    return state_initializer


def make_observer() -> mock.Mock:
    """Creates a mocked state observer that simply returns the state packed in an array,
    so that a single environment can unpack the obserations into a single observation.

    Returns:
        A mocked state observer."""

    state_observer = mock.Mock()
    state_observer.observe.side_effect = lambda state: [state]
    return state_observer


def make_transformer() -> mock.Mock:
    """Creates a mocked state transformer that always returns the old state as the new
    state.

    Returns:
        A mocked state transformer."""

    state_transformer = mock.Mock()
    state_transformer.transform.side_effect = lambda states, _: states
    return state_transformer


def make_rewarder() -> mock.Mock:
    """Creates a mocked episode rewarder that always returns an array of zeros as the
    reward.

    Returns:
        A mocked episode rewarder."""

    rewarder = mock.Mock()
    rewarder.reward.return_value = [0]
    return rewarder


def make_testee(
    ender: mock.Mock = make_ender(),
    initializer: mock.Mock = make_initializer(),
    observer: mock.Mock = make_observer(),
    rewarder: mock.Mock = make_rewarder(),
    transformer: mock.Mock = make_transformer(),
    visualizer: mock.Mock = mock.Mock(),
    render_mode: str | None = None,
) -> environment.Environment:
    # pylint: disable=too-many-arguments
    """A convenience function for creating Environments.

    Args:
        ender: An episode ender.
        initializer: A state initializer.
        observer: A state observer.
        rewarder: An episode rewarder.
        transformer: A state transformer.
        visualizer: An environment visualizer.
        render_mode: The render mode of the created environments."""

    return environment.Environment(
        ender=ender,
        initializer=initializer,
        observer=observer,
        rewarder=rewarder,
        transformer=transformer,
        visualizer=visualizer,
        render_mode=render_mode,
    )


class EnvironmentTest(unittest.TestCase):
    """Test cases for reinfocus.environments.environment.Environment."""

    def test_spaces(self):
        """Tests that the environment reports appropriate spaces retreived from its state
        observer and transformer."""

        target_action_space = mock.Mock()
        target_observation_space = mock.Mock()

        observer = mock.Mock()
        observer.single_observation_space = target_observation_space

        transformer = mock.Mock()
        transformer.single_action_space = target_action_space

        testee = make_testee(observer=observer, transformer=transformer)

        self.assertEqual(testee.action_space, target_action_space)
        self.assertEqual(testee.observation_space, target_observation_space)

    def test_initialization(self):
        """Tests that the environment correctly initializes states with its
        initializer."""

        target = numpy.array([-4, 8])

        testing.assert_allclose(
            make_testee(initializer=make_initializer(target)).reset()[0], target
        )

    def test_transforms(self):
        """Tests that the environment correctly transforms states with it's
        transformer."""

        target = numpy.array([-4, 8])

        testee = make_testee(initializer=make_initializer(target))

        testee.reset()

        testing.assert_allclose(testee.step(0)[0], target)

    def test_reward(self):
        """Tests that the environment correctly rewards episodes with it's rewarder."""

        rewarder = mock.Mock()
        rewarder.reward.side_effect = lambda a, s, o: s[0] + o[0][1] + a

        testee = make_testee(
            initializer=make_initializer(numpy.array([-4, 8])), rewarder=rewarder
        )

        testee.reset()

        self.assertEqual(testee.step(3)[1], 7)

    def test_terminated_and_truncated(self):
        """Tests that the environment correctly reports termination and truncation from
        its episode ender."""

        testee = make_testee(ender=make_ender(True, False))

        testee.reset()

        testing.assert_allclose(testee.step(0)[2:4], [True, False])

        testee = make_testee(ender=make_ender(False, True))

        testee.reset()

        testing.assert_allclose(testee.step(0)[2:4], [False, True])

    def test_reset_ender(self):
        """Tests that reset resets the episode ender."""

        ender = make_ender()

        testee = make_testee(ender=ender)

        testee.reset()

        ender.reset.assert_called_once()

    def test_reset_observer(self):
        """Tests that reset resets the state observer."""

        observer = make_observer()

        testee = make_testee(observer=observer)

        testee.reset()

        observer.reset.assert_called_once()

    def test_no_render(self):
        """Tests that render returns None when render_mode is None."""

        testee = make_testee()

        testee.reset()

        self.assertIsNone(testee.render())

    def test_rgb_array_render(self):
        """Tests that render returns a value from it's visualizer when render_mode is
        'rgb_array'."""

        target = mock.Mock()

        visualizer = mock.Mock()
        visualizer.visualize.return_value = target

        testee = make_testee(visualizer=visualizer, render_mode="rgb_array")

        testee.reset()

        self.assertEqual(testee.render(), target)


if __name__ == "__main__":
    unittest.main()
