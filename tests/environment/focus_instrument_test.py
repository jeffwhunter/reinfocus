"""Contains tests for reinfocus.environment.focus_instrument."""

import unittest

from reinfocus.environment import focus_instrument
from reinfocus.graphics import world


class FocusInstrumentTest(unittest.TestCase):
    """TestCases for reinfocus.environment.focus_instrument."""

    def test_render_and_measure(self):
        """Tests that render and measure produces measured focus_values that increase as
        the lens moves towards the target."""

        rect_world = world.one_rect_world()

        self.assertGreaterEqual(
            focus_instrument.render_and_measure(rect_world, 10),
            focus_instrument.render_and_measure(rect_world, 5),
        )

    def test_focus_extrema_returns_min_lower_than_max(self):
        """Tests that focus_extrema produces a min lower than it's max."""

        limits = focus_instrument.focus_extrema((5.0, 10.0))

        self.assertGreaterEqual(limits[1], limits[0])


if __name__ == "__main__":
    unittest.main()
