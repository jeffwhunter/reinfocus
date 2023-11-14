"""Contains tests for reinfocus.learning.focus_environment."""

import unittest

import numpy as np

import reinfocus.graphics.world as wor
import reinfocus.learning.focus_environment as env
import reinfocus.vision as vis
import tests.test_utils as tu

class FocusEnvironmentTest(unittest.TestCase):
    """TestCases for reinfocus.learning.focus_environment."""

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
            environment.step(action)[env.FOCUS],
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
