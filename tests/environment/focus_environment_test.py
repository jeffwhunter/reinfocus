"""Contains tests for reinfocus.environment.focus_environment."""

import unittest

from unittest import mock

from reinfocus.environment import focus_environment

from reinfocus.environment.types import ActionT


def make_ender() -> mock.Mock:
    """Returns a Mock that pretends to be an EpisodeEnder."""

    ender = mock.Mock()
    ender.is_early_end.return_value = (False, 0.0)
    return ender


def make_obs_producer() -> mock.Mock:
    """Returns a Mock that pretends to be an FocusObservationProducer"""

    obs_producer = mock.Mock()
    obs_producer.produce_observation.side_effect = lambda state, _: (*state, 0.0)
    return obs_producer


def make_testee(
    dynamics_function: mock.Mock | None = None,
    initializer: mock.Mock | None = None,
    obs_filter: mock.Mock | None = None,
    visualizer: mock.Mock | None = None,
    render_mode: str | None = None,
) -> focus_environment.FocusEnvironment[ActionT]:
    """Produces a FocusEnvironment with all the given dependencies; missing ones will
    be mocked."""

    return focus_environment.FocusEnvironment[ActionT](
        focus_environment.FocusEnvironmentDependencies(
            dynamics_function=dynamics_function or mock.Mock(),
            ender=make_ender(),
            initializer=initializer or mock.Mock(return_value=(0.0, 0.0)),
            obs_filter=obs_filter or mock.Mock(side_effect=lambda obs: obs),
            obs_producer=make_obs_producer(),
            rewarder=mock.Mock(side_effect=lambda _: 0.0),
            visualizer=visualizer or mock.Mock(),
        ),
        render_mode=render_mode,
    )


class FocusEnvironmentTest(unittest.TestCase):
    """TestCases for reinfocus.environment.focus_environment."""

    def test_environment_dynamics(self):
        """Tests that an environment reports state changes produced by it's dynamics
        function."""

        target = (-4.0, 8.0)

        testee = make_testee(dynamics_function=mock.Mock(return_value=target))

        testee.reset()

        self.assertEqual(testee.step(0.0)[0][0:2], target)

    def test_environment_initialization(self):
        """Tests that an environment initializes it's state with the return value from
        it's state initializer."""

        target = (-4.0, 8.0)

        self.assertEqual(
            make_testee(initializer=mock.Mock(return_value=target)).reset()[0][0:2],
            target,
        )

    def test_environment_observability(self):
        """Tests that observation_space is the right shape for the given observation
        filter."""

        target = mock.Mock()

        obs_filter = mock.Mock()
        obs_filter.observation_space.return_value = target

        self.assertEqual(make_testee(obs_filter=obs_filter).observation_space, target)

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

    def test_close_does_nothing(self):
        """Test that close does nothing! Wheeee!!!"""

        self.assertIsNone(make_testee().close())


if __name__ == "__main__":
    unittest.main()
