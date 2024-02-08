"""Contains tests for reinfocus.graphics.camera."""

import numpy

from numba import cuda
from numba.cuda import testing
from numba.cuda.testing import unittest
from numpy import linalg

from reinfocus.graphics import camera
from reinfocus.graphics import cutil
from reinfocus.graphics import random
from reinfocus.graphics import vector
from tests import test_utils
from tests.graphics import numba_test_utils


class CameraTest(testing.CUDATestCase):
    """TestCases for reinfocus.graphics.camera."""

    def test_camera(self):
        """Tests that camera makes a camera with the expected elements."""

        def flatten(l):
            return tuple(item for sublist in l for item in sublist)

        def flatten_camera(cam):
            return flatten(cam[0:7]) + (cam[7],)

        test_utils.all_close(
            flatten_camera(
                camera.camera(
                    camera.CameraOrientation(
                        vector.v3f(0, 0, -1),
                        vector.v3f(0, 0, 0),
                        vector.v3f(0, 1, 0),
                    ),
                    camera.CameraView(1.0, 90.0),
                    camera.CameraLens(2.0, 10.0),
                )
            ),
            flatten_camera(
                (
                    vector.v3f(-10, -10, -10),
                    vector.v3f(20, 0, 0),
                    vector.v3f(0, 20, 0),
                    vector.v3f(0, 0, 0),
                    vector.v3f(1, 0, 0),
                    vector.v3f(0, 1, 0),
                    vector.v3f(0, 0, 1),
                    1,
                )
            ),
        )

    def test_random_in_unit_disc(self):
        """Tests that random_in_unit_disc makes 2D vectors in the unit disc."""

        @cuda.jit
        def sample_from_disc(target, random_states):
            i = cutil.line_index()
            if cutil.outside_shape(i, target.shape):
                return

            target[i] = camera.random_in_unit_disc(random_states, i)

        tests = 100

        cpu_array = numpy.zeros((tests, 2), dtype=numpy.float32)

        cutil.launcher(sample_from_disc, tests)(
            cpu_array, random.make_random_states(tests, 0)
        )

        self.assertTrue(numpy.all(linalg.norm(cpu_array, axis=-1) < 1.0))

    def test_get_ray(self):
        """Tests that get_ray returns a ray through the expected pixel."""

        @cuda.jit
        def get_test_ray(target, cam, random_states):
            i = cutil.line_index()
            if cutil.outside_shape(i, target.shape):
                return

            target[i] = numba_test_utils.flatten_ray(
                camera.get_ray(
                    cam, numpy.float32(0.5), numpy.float32(0.5), random_states, i
                )
            )

        cpu_array = numpy.zeros((1, 6), dtype=numpy.float32)

        cutil.launcher(get_test_ray, 1)(
            cpu_array,
            camera.camera(
                camera.CameraOrientation(
                    vector.v3f(0, 0, -1), vector.v3f(0, 0, 0), vector.v3f(0, 1, 0)
                ),
                camera.CameraView(1.0, 90.0),
                camera.CameraLens(0.2, 10.0),
            ),
            random.make_random_states(1, 0),
        )

        self.assertTrue(numpy.all(numpy.abs(cpu_array[0, 0:5]) < 0.1))
        self.assertAlmostEqual(cpu_array[0, 5], -10)


if __name__ == "__main__":
    unittest.main()
