"""Contains tests for reinfocus.vision."""

import unittest

import numpy

from reinfocus import vision
from reinfocus.graphics import render
from reinfocus.graphics import world


class VisionTest(unittest.TestCase):
    """TestCases for reinfocus.vision."""

    def test_focus_value_on_empty_image(self):
        """Tests that focus_value properly measures the focus in an empty image."""

        self.assertAlmostEqual(
            vision.focus_value(numpy.zeros((10, 10, 3), dtype=numpy.uint8)), 0
        )

    def test_focus_value_on_full_image(self):
        """Tests that focus_value properly measures the focus in a full image."""

        self.assertAlmostEqual(
            vision.focus_value(numpy.ones((10, 10, 3), dtype=numpy.uint8)), 0
        )

    def test_focus_value_on_checkerboard(self):
        """Tests that focus_value returns a large value for a checkerboard image."""

        frame = numpy.zeros((10, 10, 3), dtype=numpy.uint8)
        frame[0:10:2, :, :] = 255
        frame[:, 0:10:2, :] = 255 - frame[:, 0:10:2, :]
        self.assertGreater(vision.focus_value(frame), 1)

    def test_focus_value_on_ray_traced_images(self):
        """Tests that in focus images have higher focus_values than out of focus
        images."""

        world_data = world.one_rect_world(world.ShapeParameters(distance=10))

        distant_focus = vision.focus_value(
            render.render(world_data=world_data, focus_distance=40)
        )
        far_focus = vision.focus_value(
            render.render(world_data=world_data, focus_distance=20)
        )
        in_focus = vision.focus_value(
            render.render(world_data=world_data, focus_distance=10)
        )
        near_focus = vision.focus_value(
            render.render(world_data=world_data, focus_distance=5)
        )
        close_focus = vision.focus_value(
            render.render(world_data=world_data, focus_distance=1)
        )

        self.assertGreater(in_focus, near_focus)
        self.assertGreater(near_focus, close_focus)

        self.assertGreater(in_focus, far_focus)
        self.assertGreater(far_focus, distant_focus)


if __name__ == "__main__":
    unittest.main()
