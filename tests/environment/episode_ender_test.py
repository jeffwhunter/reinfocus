"""Contains tests for reinfocus.environment.dynamics."""

import unittest

import numpy

from reinfocus.environment import episode_ender


class OnTargetEpisodeEnderTest(unittest.TestCase):
    """Test cases for reinfocus.environment.episode_ender.OnTargetEpisodeEnder."""

    def test_is_early_end_ended(self):
        """Tests that is_early_end correctly decides if an episode is over early."""

        testee = episode_ender.OnTargetEpisodeEnder(0.1, 2)

        self.assertFalse(testee.is_early_end(numpy.array([0.0, 0.1])))
        self.assertFalse(testee.is_early_end(numpy.array([0.0, 0.1])))

        testee.reset()

        self.assertFalse(testee.is_early_end(numpy.array([0.0, 0.05])))
        self.assertFalse(testee.is_early_end(numpy.array([0.0, 0.1])))

        testee.reset()

        self.assertFalse(testee.is_early_end(numpy.array([0.0, 0.05])))
        self.assertTrue(testee.is_early_end(numpy.array([0.0, 0.05])))

    def test_reset_resets_the_end_counter(self):
        """Thats that reset properly resets an episode."""

        testee = episode_ender.OnTargetEpisodeEnder(0.1, 2)

        self.assertFalse(testee.is_early_end(numpy.array([0.0, 0.05])))

        testee.reset()

        self.assertFalse(testee.is_early_end(numpy.array([0.0, 0.05])))

    def test_status_properly_reports(self):
        """Tests status returns a string that shows how close the episode is to ending."""

        testee = episode_ender.OnTargetEpisodeEnder(0.1, 2)

        self.assertEqual(testee.status(), "")

        testee.is_early_end(numpy.array([0.0, 0.1]))

        self.assertEqual(testee.status(), "")

        testee.is_early_end(numpy.array([0.0, 0.05]))

        self.assertEqual(testee.status(), "on target 1 / 2")


if __name__ == "__main__":
    unittest.main()