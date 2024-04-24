"""Contains tests for reinfocus.histories."""

import unittest

import numpy

from numpy import testing

from reinfocus import histories


class HistoriesTest(unittest.TestCase):
    """Test cases for reinfocus.histories.Histories."""

    def test_new_histories_are_empty(self):
        """Tests that newly initialized histories are empty."""

        num_histories = 3

        testee = histories.Histories(num_histories, 5)

        for i in range(num_histories):
            testing.assert_allclose(testee.get_history(i), [])

    def test_append_events(self):
        """Tests that events appended to histories can be retrieved."""

        max_n = 3
        num_histories = 2

        testee = histories.Histories(num_histories, max_n)

        testee.append_events([1, 4])
        testee.append_events([2, 3])
        testee.append_events([3, 2])
        testee.append_events([4, 1])

        testing.assert_allclose(testee.get_history(0), [2, 3, 4])
        testing.assert_allclose(testee.get_history(1), [3, 2, 1])

    def test_most_recent_events(self):
        """Tests that the most recent events can be retrieved."""

        max_n = 2
        num_histories = 4

        testee = histories.Histories(num_histories, max_n)

        testee.append_events([1, 2, 3, 4])
        testee.reset([False, False, True, False])

        testing.assert_allclose(testee.most_recent_events(), [1, 2, numpy.nan, 4])

    def test_reset(self):
        """Tests that histories can be individually reset."""

        max_n = 2
        num_histories = 3

        testee = histories.Histories(num_histories, max_n)

        testee.append_events([1, 3, 5])
        testee.reset([True, False, False])

        testee.append_events([2, 4, 6])
        testee.reset([False, False, True])

        testing.assert_allclose(testee.get_history(0), [2])
        testing.assert_allclose(testee.get_history(1), [3, 4])
        testing.assert_allclose(testee.get_history(2), [])

        testee.append_events([1, 5, 1])
        testee.reset([True, False, True])

        testing.assert_allclose(testee.get_history(1), [4, 5])


if __name__ == "__main__":
    unittest.main()
