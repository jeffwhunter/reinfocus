"""Contains tests for reinfocus.vision."""

import unittest

import numpy

from reinfocus import vision
from reinfocus.graphics import render
from reinfocus.graphics import world


class FocusValueTest(unittest.TestCase):
    """Test cases for reinfocus.vision.focus_value."""

    def test_empty_image(self):
        """Tests that focus_value properly measures the focus in an empty image."""

        self.assertAlmostEqual(
            vision.focus_value(numpy.zeros((10, 10, 3), dtype=numpy.uint8)), 0
        )

    def test_full_image(self):
        """Tests that focus_value properly measures the focus in a full image."""

        self.assertAlmostEqual(
            vision.focus_value(numpy.ones((10, 10, 3), dtype=numpy.uint8)), 0
        )

    def test_checkerboard(self):
        """Tests that focus_value returns a large value for a checkerboard image."""

        frame = numpy.zeros((10, 10, 3), dtype=numpy.uint8)
        frame[0:10:2, :, :] = 255
        frame[:, 0:10:2, :] = 255 - frame[:, 0:10:2, :]
        self.assertGreater(vision.focus_value(frame), 1)


class FocusValuesTest(unittest.TestCase):
    """Test cases for reinfocus.vision.focus_values."""

    def test_ray_traced_images(self):
        """Tests that in focus images have higher focus_values than out of focus
        images."""

        num_envs = 5

        worlds = world.FocusWorlds(num_envs)
        worlds.update_targets([10] * num_envs)

        focus_values = vision.focus_values(render.fast_render(worlds, [40, 20, 10, 5, 1]))

        self.assertGreater(focus_values[2], focus_values[3])
        self.assertGreater(focus_values[3], focus_values[4])

        self.assertGreater(focus_values[2], focus_values[1])
        self.assertGreater(focus_values[1], focus_values[0])


if __name__ == "__main__":
    unittest.main()
