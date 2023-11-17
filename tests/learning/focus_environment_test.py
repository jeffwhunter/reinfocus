"""Contains tests for reinfocus.learning.focus_environment."""

import unittest

import numpy as np

import reinfocus.graphics.world as wor
import reinfocus.learning.focus_environment as env
import reinfocus.vision as vis
import tests.test_utils as tu

class FocusEnvironmentTest(unittest.TestCase):
    """TestCases for reinfocus.learning.focus_environment."""

    def test_make_relative_dynamics(self):
        """Tests that make_relative_dynamics creates a function that moves the second
            element of the state between the limits, according to the action."""
        dynamics = env.make_relative_dynamics(0, 1)

        state = np.array([1, .5])
        tu.arrays_close(self, dynamics(state, 1.), [1, 1])
        tu.arrays_close(self, dynamics(state, .75), [1, 1])
        tu.arrays_close(self, dynamics(state, .5), [1, 1])
        tu.arrays_close(self, dynamics(state, .25), [1, .75])
        tu.arrays_close(self, dynamics(state, .1), [1, .6])
        tu.arrays_close(self, dynamics(state, 0), [1, .5])
        tu.arrays_close(self, dynamics(state, -.1), [1, .4])
        tu.arrays_close(self, dynamics(state, -.25), [1, .25])
        tu.arrays_close(self, dynamics(state, -.5), [1, 0])
        tu.arrays_close(self, dynamics(state, -.75), [1, 0])
        tu.arrays_close(self, dynamics(state, -1), [1, 0])

        state = np.array([1, 1])
        tu.arrays_close(self, dynamics(state, 1), [1, 1])
        tu.arrays_close(self, dynamics(state, 0), [1, 1])
        tu.arrays_close(self, dynamics(state, -.5), [1, .5])
        tu.arrays_close(self, dynamics(state, -1), [1, 0])

        state = np.array([1, 0])
        tu.arrays_close(self, dynamics(state, 1), [1, 1])
        tu.arrays_close(self, dynamics(state, .5), [1, .5])
        tu.arrays_close(self, dynamics(state, 0), [1, 0])
        tu.arrays_close(self, dynamics(state, -1), [1, 0])

    def test_make_uniform_initializer(self):
        """Tests that make_uniform_initializer creates a state initializer that samples
            initial states from between given limits."""
        n_tests = 100

        initial_state = env.make_uniform_initializer(-1., 1., n_tests)()

        tu.arrays_close(
            self,
            (-1. <= initial_state) & (initial_state <= 1.),
            np.full(n_tests, True))

    def test_make_ranged_initializer(self):
        """Tests that make_ranged_initializer creates a state initializer that samples
            initial states from between the listed ranges."""
        initial_states = env.make_ranged_initializer([
            [(0., .2), (.8, 1.)],
            [(.2, .4), (.6, .8)],
            [(.4, .6)]])()

        self.assertTrue(
            (0. <= initial_states[0] and initial_states[0] <= .2) or
            (.8 <= initial_states[0] and initial_states[0] <= 1.))

        self.assertTrue(
            (.2 <= initial_states[1] and initial_states[1] <= .4) or
            (.6 <= initial_states[1] and initial_states[1] <= .8))

        self.assertTrue(.4 <= initial_states[2] and initial_states[2] <= .6)

    def test_make_observation_normer(self):
        """Tests that make_observation_normer creates a normer that norms as expected."""
        normer = env.make_observation_normer(5, 5)
        self.assertEqual(normer(0), -1)
        self.assertEqual(normer(5), 0)
        self.assertEqual(normer(10), 1)
        array_normer = env.make_observation_normer(np.array([1]), np.array([2]))
        tu.arrays_close(self, array_normer(np.array([0])), np.array([-.5]))
        tu.arrays_close(self, array_normer(np.array([1])), np.array([0]))
        tu.arrays_close(self, array_normer(np.array([2])), np.array([.5]))

    def test_make_observation_filter(self):
        """Tests that make_observation_filter creates a filter that filters as expected."""
        original = env.make_observation_filter()
        no_target = env.make_observation_filter([1, 2])
        only_focus = env.make_observation_filter([2])

        tu.arrays_close(self, original([1, 2, 3]), [1, 2, 3])
        tu.arrays_close(self, no_target([1, 2, 3]), [2, 3])
        tu.arrays_close(self, only_focus([1, 2, 3]), [3])


    def test_make_lens_distance_penalty(self):
        """Tests that make_lens_distance_penalty creates a rewarder that gives
            the proper penalties."""
        penalty = env.make_lens_distance_penalty(1.)
        self.assertEqual(penalty([0., 0.]), 0)
        self.assertEqual(penalty([0., 1.]), -1)
        self.assertEqual(penalty([1., 0.]), -1)
        self.assertEqual(penalty([1., 1.]), 0)

    def test_make_lens_on_target_reward(self):
        """Tests that make_lens_on_target_reward creates a rewarder that gives a reward
            of one when the lens is within the given distance."""
        reward = env.make_lens_on_target_reward(.1)
        self.assertEqual(reward([.5, .65]), 0)
        self.assertEqual(reward([.5, .55]), 1)
        self.assertEqual(reward([.5, .45]), 1)
        self.assertEqual(reward([.5, .35]), 0)

    def test_make_focus_reward(self):
        """Tests that make_focus_reward creates a rewarder that gives a reward equal to
            the focus."""
        reward = env.make_focus_reward()
        self.assertEqual(reward([1, 2, 3]), 3)
        self.assertEqual(reward([4, 5, 6]), 6)
        self.assertEqual(reward([3, 2, 1]), 1)
        self.assertEqual(reward([6, 5, 4]), 4)

    def test_render_and_measure(self):
        """Tests that render and measure produces measured focus_values that increase as
            the lens moves towards the target."""
        world = wor.one_rect_world()

        self.assertGreaterEqual(
            env.render_and_measure(world, 10),
            env.render_and_measure(world, 5))

    def test_pretty_render(self):
        """Tests that pretty_render produces images that focus as the lens moves towards
            the target."""
        world = wor.one_rect_world()

        self.assertGreaterEqual(
            vis.focus_value(env.pretty_render(world, 10)),
            vis.focus_value(env.pretty_render(world, 5)))

    def test_find_focus_value_limits_returns_min_lower_than_max(self):
        """Tests that find_focus_value_limits produces a min lower than it's max."""
        limits = env.find_focus_value_limits()

        self.assertGreaterEqual(limits[1], limits[0])

    def test_envionment_initialization(self):
        """Tests that FocusEnvironment.reset initializes the state with the given limits
            for different initialization modes."""
        u_env = env.FocusEnvironment()
        d_env = env.FocusEnvironment(
            modes=env.FocusEnvironment.Modes(initializer_type=env.InitializerType.DEVIATED))

        u_obs = u_env.reset()[0]
        d_obs = d_env.reset()[0]

        self.assertTrue(-1 <= u_obs[env.TARGET] and u_obs[env.TARGET] <= 1)

        self.assertTrue(
            (-.8 <= d_obs[env.TARGET] and d_obs[env.TARGET] <= -.5) or
            (.5 <= d_obs[env.TARGET] and d_obs[env.TARGET] <= .8))

    def test_environment_observability(self):
        """Tests that FocusEnvironment.observation_space is the right shape under
            different observability modes."""
        self.assertEqual(
            env.FocusEnvironment(
                modes=env.FocusEnvironment.Modes(observable_type=env.ObservableType.FULL)
            )
                .observation_space
                .shape,
            (3,))
        self.assertEqual(
            env.FocusEnvironment(
                modes=env.FocusEnvironment.Modes(observable_type=env.ObservableType.NO_TARGET)
            )
                .observation_space
                .shape,
            (2,))
        self.assertEqual(
            env.FocusEnvironment(
                modes=env.FocusEnvironment.Modes(observable_type=env.ObservableType.ONLY_FOCUS)
            )
                .observation_space
                .shape,
            (1,))

    def test_environment_produces_observation_inside_observation_space(self):
        """Tests that FocusEnvironment.reset() produces an observation inside
            FocusEnvironment.observation_space."""
        environment = env.FocusEnvironment()

        self.assertTrue(environment.observation_space.contains(environment.reset()[0]))

    def test_environment_step_towards_target_increases_focus_value(self):
        """Tests that FocusEnvironment.step() with an action that moves the lens towards
            the target will return an observation with a focus_value higher than the
            prior one."""
        environment = env.FocusEnvironment()

        observation = environment.reset()[0]

        action = (observation[env.TARGET] - observation[env.LENS]) / 4.

        self.assertGreaterEqual(
            environment.step(action)[0][env.FOCUS],
            observation[env.FOCUS])

    def test_no_render(self):
        """Tests that FocusEnvironment.render returns None when render_mode is None."""
        environment = env.FocusEnvironment()

        environment.reset()

        self.assertIsNone(environment.render())

    def test_environment_render(self):
        """Tests that FocusEnvironment.render produces images that focus as the lens
            approaches the target."""
        environment = env.FocusEnvironment(render_mode="rgb_array")

        observation = environment.reset()[0]

        old_image = environment.render()

        environment.step((observation[env.TARGET] - observation[env.LENS]) / 4.)

        new_image = environment.render()

        assert old_image is not None
        assert new_image is not None

        self.assertGreaterEqual(vis.focus_value(new_image), vis.focus_value(old_image))

    def test_close_does_nothing(self):
        """Test that close does nothing! Wheeee!!!"""
        self.assertIsNone(env.FocusEnvironment().close())

if __name__ == '__main__':
    unittest.main()
