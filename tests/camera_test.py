"""Contains tests for reinfocus.camera."""

from numba.cuda.testing import unittest

from reinfocus import camera as cam, vector as vec
from tests import numba_test_case as ntc

class CameraTest(ntc.NumbaTestCase):
    """TestCases for reinfocus.camera."""

    def test_camera_thingy(self):
        """Tests something about a camera."""

        def flatten(l):
            return tuple(item for sublist in l for item in sublist)

        def flatten_camera(camera):
            return flatten(camera[0:7]) + (camera[7],)

        self.arrays_close(
            flatten_camera(
                cam.cpu_camera(
                    cam.CameraOrientation(
                        vec.c3f(0, 0, -1),
                        vec.c3f(0, 0, 0),
                        vec.c3f(0, 1, 0)),
                    cam.CameraView(1.0, 90.0),
                    cam.CameraLens(2.0, 10.0))
            ),
            flatten_camera(
                (
                    vec.c3f(-10, -10, -10),
                    vec.c3f(20, 0, 0),
                    vec.c3f(0, 20, 0),
                    vec.c3f(0, 0, 0),
                    vec.c3f(1, 0, 0),
                    vec.c3f(0, 1, 0),
                    vec.c3f(0, 0, 1),
                    1
                )))

if __name__ == '__main__':
    unittest.main()
