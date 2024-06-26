"""Contains tests for reinfocus.graphics.render."""

import numpy

from numba.cuda import testing
from numba.cuda.testing import unittest

from reinfocus.graphics import camera
from reinfocus.graphics import cutil
from reinfocus.graphics import random
from reinfocus.graphics import render
from reinfocus.graphics import shape_factory
from reinfocus.graphics import world


class MakeRenderTargetTest(unittest.TestCase):
    """Test cases for reinfocus.graphics.render.make_render_target."""

    def test_shape(self):
        """Tests that make_render_target produces targets of the proper shape."""

        target = (-4, 8)

        self.assertEqual(render.make_render_target(target).shape, target + (3,))


class RenderTest(testing.CUDATestCase):
    """Test cases for reinfocus.graphics.render.[device_]render."""

    def test_device_average_colour(self):
        """Tests that device_render produces a known image for a known set of
        parameters."""

        frame_shape = (1, 300, 300)

        frame = render.make_render_target(frame_shape)

        world_data = world.Worlds(
            shape_factory.one_rect(shape_factory.ShapeParameters(r_size=30))
        )

        cameras = camera.Cameras(camera.make_gpu_camera())

        cutil.launcher(render.device_render, frame_shape)(
            frame,
            cameras.device_data(),
            100,
            random.make_random_states(int(numpy.prod(frame_shape)), 0),
            world_data.device_data(),
        )

        average_colour = numpy.average(frame.copy_to_host(), axis=(0, 1, 2))

        self.assertTrue(numpy.all(average_colour >= numpy.multiply([0.25, 0.25, 0], 255)))
        self.assertTrue(numpy.all(average_colour <= numpy.multiply([0.5, 0.5, 0], 255)))

    def test_average_colour(self):
        """Tests that render produces a known image for a known set of parameters."""

        average_colour = numpy.average(
            render.render(
                world_data=world.Worlds(
                    shape_factory.one_sphere(shape_factory.ShapeParameters(r_size=30))
                ),
                cameras=camera.Cameras(camera.make_gpu_camera()),
                frame_shape=(300, 300),
            ),
            axis=(0, 1, 2),
        )

        self.assertTrue(
            numpy.all(
                average_colour >= numpy.multiply([0.4, 0.4, 0.1], 255)  # type: ignore
            )
        )
        self.assertTrue(
            numpy.all(
                average_colour <= numpy.multiply([0.6, 0.6, 0.2], 255)  # type: ignore
            )
        )


class FastRendererTest(testing.CUDATestCase):
    """Test cases for reinfocus.graphics.render.FastRenderer."""

    def test_render(self):
        """Tests that render produces a known image for a known set of parameters."""

        targets = [10]

        testee = render.FastRenderer(r_size=30)
        testee.update_targets(targets)
        testee.update_focus_planes(targets)

        average_colour = numpy.average(testee.render(300), axis=(0, 1, 2))

        self.assertTrue(numpy.all(average_colour >= numpy.multiply([0.25, 0.25, 0], 255)))
        self.assertTrue(numpy.all(average_colour <= numpy.multiply([0.5, 0.5, 0], 255)))

    def test_random_states_grow(self):
        """Tests that enough random states are created if the frame grows, and that
        they're still usable with smaller frames."""

        targets = [10]

        testee = render.FastRenderer()
        testee.update_targets(targets)
        testee.update_focus_planes(targets)

        testee.render(300)
        testee.render(600)

        result = testee.render(300)

        self.assertTrue(numpy.all(result <= 255))
        self.assertTrue(numpy.all(result >= 0))


if __name__ == "__main__":
    unittest.main()
