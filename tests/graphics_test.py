"""Contains tests for reinfocus.graphics."""

import numpy as np
from numba.cuda.random import create_xoroshiro128p_states
from numba.cuda.testing import unittest
from reinfocus import camera as cam
from reinfocus import graphics as gra
from reinfocus import vector as vec
from reinfocus import world as wor
from tests import numba_test_case as ntc

class GraphicsTest(ntc.NumbaTestCase):
    """TestCases for reinfocus.graphics."""
    # pylint: disable=no-value-for-parameter

    def test_device_render(self):
        """Tests that device_render produces a known image for a known set of parameters."""

        frame_shape = (300, 300)

        device_frame = gra.make_device_frame(*frame_shape)

        world = wor.one_rect_world(r_size=30)

        gra.device_render[(19, 19), (16, 16)]( # type: ignore
            device_frame,
            cam.cpu_camera(
                cam.CameraOrientation(
                    vec.c3f(0, 0, -10),
                    vec.c3f(0, 0, 0),
                    vec.c3f(0, 1, 0)),
                cam.CameraView(1, 30.0),
                cam.CameraLens(0.1, 10)),
            100,
            create_xoroshiro128p_states(frame_shape[0] * frame_shape[1], seed=0),
            (world.device_shape_parameters(), world.device_shape_types()))

        average_colour = np.average(device_frame.copy_to_host(), axis=(0, 1))

        self.assertTrue(np.all(average_colour >= [.25, .25, 0]))
        self.assertTrue(np.all(average_colour <= [.5, .5, 0]))

    def test_render(self):
        """Tests that render produces a known image for a known set of parameters."""

        average_colour = np.average(
            gra.render(frame_shape=(300, 300), world=wor.one_sphere_world(r_size=30)),
            axis=(0, 1))

        self.assertTrue(np.all(average_colour >= [.4, .4, .1]))
        self.assertTrue(np.all(average_colour <= [.6, .6, .2]))

if __name__ == '__main__':
    unittest.main()
