"""Contains tests for reinfocus.environments.visualization."""

import unittest

import numpy

from matplotlib import colors
from numpy import testing

from reinfocus.environments import visualization


def _simple_colormap():
    """Produces a colourmap that changes smoothly from (0, 0, 0, 0) to (1, 1, 1, 1)."""

    return colors.LinearSegmentedColormap.from_list("", ["black", "white"])


class FadingColoursTest(unittest.TestCase):
    """Test cases for reinfocus.environments.visualization.fading_colours."""

    def test_fade(self):
        """Tests that fading_colours makes a cool fade!"""

        colour_dimension = 4

        testing.assert_allclose(
            visualization.fading_colours(_simple_colormap(), 5, 3, p=1),
            [
                (0.6,) * colour_dimension,
                (0.8,) * colour_dimension,
                (1.0,) * colour_dimension,
            ],
        )

    def test_high_power_fades_fast(self):
        """Tests that higher values of p produces a faster fade with the same maximum."""

        black_to_white = _simple_colormap()

        lower = visualization.fading_colours(black_to_white, 5, 5, p=2)
        higher = visualization.fading_colours(black_to_white, 5, 5, p=1)

        testing.assert_allclose(lower[-1], higher[-1])
        testing.assert_array_less(lower[:-1], higher[:-1])

    def test_high_power_increasingly_fades(self):
        """Tests that when p is larger than one, the fade increases by more each step."""

        colour_differences = numpy.diff(
            visualization.fading_colours(_simple_colormap(), 5, 5, p=3), axis=0
        )

        self.assertTrue(numpy.all(colour_differences[1:] > colour_differences[:-1]))


class HistoriesTest(unittest.TestCase):
    """Test cases for reinfocus.environments.visualization.Histories."""

    def test_new_histories_are_empty(self):
        """Tests that newly initialized histories are empty."""

        num_histories = 3

        testee = visualization.Histories(num_histories, 5)

        for i in range(num_histories):
            testing.assert_allclose(testee.get_history(i), [])

    def test_append_events(self):
        """Tests that events appended to histories can be retrieved."""

        max_n = 3
        num_histories = 2

        testee = visualization.Histories(num_histories, max_n)

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

        testee = visualization.Histories(num_histories, max_n)

        testee.append_events([1, 2, 3, 4])
        testee.reset([False, False, True, False])

        testing.assert_allclose(testee.most_recent_events(), [1, 2, numpy.nan, 4])

    def test_reset(self):
        """Tests that histories can be individually reset."""

        max_n = 2
        num_histories = 3

        testee = visualization.Histories(num_histories, max_n)

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
