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


if __name__ == "__main__":
    unittest.main()
