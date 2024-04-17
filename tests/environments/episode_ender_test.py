"""Contains tests for reinfocus.environments.episode_ender."""

import unittest

import numpy

from numpy import testing

from reinfocus.environments import episode_ender


class EndlessEpisodeEnderTest(unittest.TestCase):
    """Test cases for reinfocus.environments.episode_ender.EndlessEpisodeEnder."""

    def test_never_ends(self):
        """Tests that is_terminated and is_truncated are always False."""

        num_envs = 5

        testee = episode_ender.EndlessEpisodeEnder(num_envs)

        testee.step(numpy.zeros(num_envs, dtype=numpy.float32))

        testing.assert_allclose(testee.is_terminated(), [False] * num_envs)
        testing.assert_allclose(testee.is_truncated(), [False] * num_envs)


class OnTargetEpisodeEnderTest(unittest.TestCase):
    """Test cases for reinfocus.environments.episode_ender.OnTarget."""

    def test_is_terminated(self):
        """Tests that is_terminated is always False (ie: the MDP has no end state)."""

        num_envs = 3

        testee = episode_ender.OnTargetEpisodeEnder(num_envs, (0, 1), 1, 1, 1)

        testee.step(numpy.array([[-1, -1], [0, 0], [1, 1]]))

        testing.assert_allclose(testee.is_terminated(), [False] * num_envs)

    def test_is_truncated_on_target(self):
        """Tests that is_truncated correctly ends the episode when on target."""

        num_envs = 3

        testee = episode_ender.OnTargetEpisodeEnder(num_envs, (0, 1), 2, 2)

        testee.step(numpy.array([[0, 2], [0, 1], [0, 1]]))

        testing.assert_allclose(testee.is_truncated(), [False] * num_envs)

        testee.step(numpy.array([[0, 2], [0, 2], [0, 1]]))

        testing.assert_allclose(testee.is_truncated(), [False, False, True])

    def test_is_truncated_time_limit(self):
        """Tests that is_truncated correctly ends the episode when time is up."""

        num_envs = 3

        testee = episode_ender.OnTargetEpisodeEnder(
            num_envs, (0, 1), 2, max_episode_steps=2
        )

        testee.step(numpy.array([[0, 2], [0, 1], [0, 1]]))

        testing.assert_allclose(testee.is_truncated(), [False] * num_envs)

        testee.step(numpy.array([[0, 2], [0, 2], [0, 1]]))

        testing.assert_allclose(testee.is_truncated(), [True] * num_envs)

    def test_check_indices(self):
        """Tests that truncation does not depend on the check indices."""

        testee = episode_ender.OnTargetEpisodeEnder(2, (3, 7), 2, 1)

        testee.step(numpy.array([[0, 0, 0, 1, 0, 0, 0, 3], [0, 0, 0, 1, 0, 0, 0, 2]]))

        testing.assert_allclose(testee.is_truncated(), [False, True])

    def test_truncation_partial_reset_on_target(self):
        """Thats that a partial reset properly resets is_truncated when on target."""

        num_envs = 2

        testee = episode_ender.OnTargetEpisodeEnder(num_envs, (0, 1), 2, 2)

        testee.step(numpy.array([[0, 1], [0, 1]]))

        testing.assert_allclose(testee.is_truncated(), [False] * num_envs)

        testee.reset(numpy.array([True, False]))

        testee.step(numpy.array([[0, 1], [0, 1]]))

        testing.assert_allclose(testee.is_truncated(), [False, True])

    def test_truncation_partial_reset_time_limit(self):
        """Thats that a partial reset properly resets is_truncated when time is up."""

        num_envs = 2

        testee = episode_ender.OnTargetEpisodeEnder(
            num_envs, (0, 1), 2, max_episode_steps=2
        )

        testee.step(numpy.array([[0, 2], [0, 2]]))

        testing.assert_allclose(testee.is_truncated(), [False] * num_envs)

        testee.reset(numpy.array([True, False]))

        testee.step(numpy.array([[0, 2], [0, 2]]))

        testing.assert_allclose(testee.is_truncated(), [False, True])

    def test_truncation_reset_on_target(self):
        """Thats that a full reset properly resets is_truncated when on target."""

        num_envs = 2

        testee = episode_ender.OnTargetEpisodeEnder(num_envs, (0, 1), 2, 2)

        testee.step(numpy.array([[0, 1], [0, 1]]))

        testing.assert_allclose(testee.is_truncated(), [False] * num_envs)

        testee.reset()

        testee.step(numpy.array([[0, 1], [0, 1]]))

        testing.assert_allclose(testee.is_truncated(), [False, False])

        testee.step(numpy.array([[0, 1], [0, 2]]))

        testing.assert_allclose(testee.is_truncated(), [True, False])

    def test_truncation_reset_time_limit(self):
        """Thats that a full reset properly resets is_truncated when time is up."""

        num_envs = 2

        testee = episode_ender.OnTargetEpisodeEnder(
            num_envs, (0, 1), 2, max_episode_steps=2
        )

        testee.step(numpy.array([[0, 2], [0, 2]]))

        testing.assert_allclose(testee.is_truncated(), [False] * num_envs)

        testee.reset()

        testee.step(numpy.array([[0, 2], [0, 2]]))

        testing.assert_allclose(testee.is_truncated(), [False] * num_envs)

        testee.step(numpy.array([[0, 2], [0, 2]]))

        testing.assert_allclose(testee.is_truncated(), [True] * num_envs)

    def test_status_properly_reports(self):
        """Tests status returns a string that shows how close the episode is to ending."""

        num_envs = 3

        testee = episode_ender.OnTargetEpisodeEnder(num_envs, (0, 1), 2, 2)

        self.assertEqual([testee.status(i) for i in range(num_envs)], [""] * num_envs)

        testee.step(numpy.array([[0, 2], [0, 1], [0, 1]]))

        self.assertEqual(
            [testee.status(i) for i in range(num_envs)],
            ["", "on target 1 / 2", "on target 1 / 2"],
        )

        testee.step(numpy.array([[0, 2], [0, 2], [0, 1]]))

        self.assertEqual(
            [testee.status(i) for i in range(num_envs)], ["", "", "on target 2 / 2"]
        )


if __name__ == "__main__":
    unittest.main()
