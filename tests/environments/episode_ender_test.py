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
    """Test cases for reinfocus.environments.episode_ender.OnTargetEpisodeEnder."""

    def test_is_terminated(self):
        """Tests that is_terminated is always False (ie: the MDP has no end state)."""

        num_envs = 3

        testee = episode_ender.OnTargetEpisodeEnder(num_envs, (0, 1), 1, 1)

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


class StoppedEpisodeEnderTest(unittest.TestCase):
    """Test cases for reinfocus.environments.episode_ender.StoppedEpisodeEnder."""

    def test_is_terminated(self):
        """Tests that is_terminated is always False (ie: the MDP has no end state)."""

        num_envs = 3

        testee = episode_ender.StoppedEpisodeEnder(num_envs, 1, 0.5, 0)

        testee.step(numpy.array([[-1, -1], [0, 0], [1, 1]]))

        testing.assert_allclose(testee.is_terminated(), [False] * num_envs)

    def test_is_truncated_stopped(self):
        """Tests that is_truncated correctly ends the episode when stopped."""

        num_envs = 4

        testee = episode_ender.StoppedEpisodeEnder(num_envs, 0, 0.5, 1)

        testee.step(numpy.array([[1, 0], [2, 0], [3, 0], [4, 0]]))

        testing.assert_allclose(testee.is_truncated(), [False] * num_envs)

        testee.step(numpy.array([[0.6, 0], [1.4, 0], [3.6, 0], [4.4, 0]]))

        testing.assert_allclose(testee.is_truncated(), [True, False, False, True])

    def test_check_index(self):
        """Tests that truncation does not depend on the check index."""

        num_envs = 4

        testee = episode_ender.StoppedEpisodeEnder(num_envs, 1, 0.5, 1)

        testee.step(numpy.array([[0, 1], [0, 2], [0, 3], [0, 4]]))

        testing.assert_allclose(testee.is_truncated(), [False] * num_envs)

        testee.step(numpy.array([[0, 0.4], [0, 1.6], [0, 3.4], [0, 4.6]]))

        testing.assert_allclose(testee.is_truncated(), [False, True, True, False])

    def test_early_end_steps(self):
        """Tests that truncation responds appropriately to early_end_steps."""

        num_envs = 4

        testee = episode_ender.StoppedEpisodeEnder(num_envs, 0, 0.5, 2)

        testee.step(numpy.array([[1, 0], [2, 0], [3, 0], [4, 0]]))

        testing.assert_allclose(testee.is_truncated(), [False] * num_envs)

        testee.step(numpy.array([[0.6, 0], [1.4, 0], [3.6, 0], [4.4, 0]]))

        testing.assert_allclose(testee.is_truncated(), [False] * num_envs)

        testee.step(numpy.array([[0.6, 0], [1.4, 0], [3.6, 0], [4.4, 0]]))

        testing.assert_allclose(testee.is_truncated(), [True, False, False, True])

    def test_slow_move(self):
        """Tests that states which move farther than the threshold over multiple steps
        don't end."""

        num_envs = 4

        testee = episode_ender.StoppedEpisodeEnder(num_envs, 0, 0.5, 2)

        testee.step(numpy.array([[1, 0], [2, 0], [3, 0], [4, 0]]))

        testing.assert_allclose(testee.is_truncated(), [False] * num_envs)

        testee.step(numpy.array([[0.7, 0], [2.3, 0], [2.8, 0], [4.2, 0]]))

        testing.assert_allclose(testee.is_truncated(), [False] * num_envs)

        testee.step(numpy.array([[0.4, 0], [2.6, 0], [2.6, 0], [4.4, 0]]))

        testing.assert_allclose(testee.is_truncated(), [False, False, True, True])

    def test_reset(self):
        """Tests that truncation responds appropriately after a reset."""

        num_envs = 4

        testee = episode_ender.StoppedEpisodeEnder(num_envs, 0, 0.5, 2)

        testee.step(numpy.array([[1, 0], [2, 0], [3, 0], [4, 0]]))

        testing.assert_allclose(testee.is_truncated(), [False] * num_envs)

        testee.step(numpy.array([[0.6, 0], [1.4, 0], [3.6, 0], [4.4, 0]]))

        testing.assert_allclose(testee.is_truncated(), [False] * num_envs)

        testee.reset(numpy.array([True, False, True, False]))

        testee.step(numpy.array([[0.6, 0], [1.4, 0], [3.6, 0], [4.4, 0]]))

        testing.assert_allclose(testee.is_truncated(), [False, False, False, True])

    def test_status_properly_reports(self):
        """Tests status returns a string that shows how close the episode is to ending."""

        num_envs = 4

        testee = episode_ender.StoppedEpisodeEnder(num_envs, 0, 0.5, 2)

        self.assertEqual([testee.status(i) for i in range(num_envs)], [""] * num_envs)

        testee.step(numpy.array([[1, 0], [2, 0], [3, 0], [4, 0]]))

        self.assertEqual([testee.status(i) for i in range(num_envs)], [""] * num_envs)

        testee.step(numpy.array([[0.7, 0], [1.8, 0], [3.2, 0], [4.3, 0]]))

        self.assertEqual(
            [testee.status(i) for i in range(num_envs)],
            ["stopped 1 / 2"] * num_envs,
        )

        testee.step(numpy.array([[0.4, 0], [1.6, 0], [3.4, 0], [4.9, 0]]))

        self.assertEqual(
            [testee.status(i) for i in range(num_envs)],
            ["stopped 1 / 2", "stopped 2 / 2", "stopped 2 / 2", ""],
        )


if __name__ == "__main__":
    unittest.main()
