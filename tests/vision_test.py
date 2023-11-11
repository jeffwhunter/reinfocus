"""Contains tests for reinfocus.vision."""

import unittest

import numpy as np
import numpy.typing as npt
from reinfocus import vision as vis
from reinfocus.graphics import render as ren
from reinfocus.graphics import world as wor

def make_frame(x: int = 10, y: int = 10, d: float = 0) -> npt.NDArray:
    """Returns an x by y image full of d."""
    return np.array([[(np.float32(d),) * 3] * y] * x)

class VisionTest(unittest.TestCase):
    """TestCases for reinfocus.vision."""

    def test_focus_value_on_empty_image(self):
        """Tests that focus_value properly measures the focus in an empty image."""
        self.assertAlmostEqual(vis.focus_value(make_frame()), 0)

    def test_focus_value_on_full_image(self):
        """Tests that focus_value properly measures the focus in a full image."""
        self.assertAlmostEqual(vis.focus_value(make_frame(d=1.)), 0)

    def test_focus_value_on_checkerboard(self):
        """Tests that focus_value returns a large value for a checkerboard image."""
        frame = make_frame()
        frame[0 : 10 : 2, :, :] = 1.
        frame[:, 0 : 10 : 2, :] = 1. - frame[:, 0 : 10 : 2, :]
        self.assertGreater(vis.focus_value(frame), 9.)

    def test_focus_value_on_ray_traced_images(self):
        """Tests that in focus images have higher focus_values than out of focus
            images."""
        world = wor.one_rect_world(distance=10)

        distant_focus = vis.focus_value(ren.render(world=world, focus_distance=40))
        far_focus = vis.focus_value(ren.render(world=world, focus_distance=20))
        in_focus = vis.focus_value(ren.render(world=world, focus_distance=10))
        near_focus = vis.focus_value(ren.render(world=world, focus_distance=5))
        close_focus = vis.focus_value(ren.render(world=world, focus_distance=1))

        self.assertGreater(in_focus, near_focus)
        self.assertGreater(near_focus, close_focus)

        self.assertGreater(in_focus, far_focus)
        self.assertGreater(far_focus, distant_focus)

if __name__ == '__main__':
    unittest.main()
