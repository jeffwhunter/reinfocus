"""Contains tests for reinfocus.graphics.render."""

import numpy

from numba.cuda import testing
from numba.cuda.testing import unittest

from reinfocus.graphics import camera
from reinfocus.graphics import cutil
from reinfocus.graphics import random
from reinfocus.graphics import render
from reinfocus.graphics import vector
from reinfocus.graphics import world


class RenderTest(testing.CUDATestCase):
    """TestCases for reinfocus.graphics.render."""

    def test_device_render(self):
        """Tests that device_render produces a known image for a known set of
        parameters."""

        frame_shape = (300, 300)

        frame = render.make_render_target(frame_shape)

        world_data = world.one_rect_world(world.ShapeParameters(r_size=30))

        cutil.launcher(render.device_render, frame_shape)(
            frame,
            camera.camera(
                camera.CameraOrientation(
                    vector.v3f(0, 0, -10), vector.v3f(0, 0, 0), vector.v3f(0, 1, 0)
                ),
                camera.CameraView(1, 30.0),
                camera.CameraLens(0.1, 10),
            ),
            100,
            random.make_random_states(frame_shape[0] * frame_shape[1], 0),
            (world_data.device_shape_parameters(), world_data.device_shape_types()),
        )

        average_colour = numpy.average(frame.copy_to_host(), axis=(0, 1))

        self.assertTrue(numpy.all(average_colour >= numpy.multiply([0.25, 0.25, 0], 255)))
        self.assertTrue(numpy.all(average_colour <= numpy.multiply([0.5, 0.5, 0], 255)))

    def test_make_render_target(self):
        """Tests that make_render_target produces targets of the proper shape."""

        target = (-4, 8)

        self.assertEqual(render.make_render_target(target).shape, target + (3,))

    def test_render(self):
        """Tests that render produces a known image for a known set of parameters."""

        average_colour = numpy.average(
            render.render(
                frame_shape=(300, 300),
                world_data=world.one_sphere_world(world.ShapeParameters(r_size=30)),
            ),
            axis=(0, 1),
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


if __name__ == "__main__":
    unittest.main()
