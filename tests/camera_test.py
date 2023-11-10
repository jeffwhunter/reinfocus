"""Contains tests for reinfocus.camera."""

import numpy as np
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states
from numba.cuda.testing import unittest

from reinfocus import camera as cam
from reinfocus import vector as vec
from tests import numba_test_case as ntc
from tests import numba_test_utils as ntu

class CameraTest(ntc.NumbaTestCase):
    """TestCases for reinfocus.camera."""
    # pylint: disable=no-value-for-parameter

    def test_cpu_camera(self):
        """Tests that cpu_camera makes a CPU camera with the expected elements."""

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

    def test_random_in_unit_disc(self):
        """Tests that random_in_unit_disc makes 2D GPU vectors in the unit disc."""
        @cuda.jit
        def sample_from_disc(target, random_states):
            i = cuda.grid(1) # type: ignore
            if i < target.size:
                target[i] = vec.g2f_to_c2f(cam.random_in_unit_disc(random_states, i))

        tests = 100

        cpu_array = ntu.cpu_target(ndim=2, nrow=tests)

        sample_from_disc[tests, 1]( # type: ignore
            cpu_array,
            create_xoroshiro128p_states(tests, seed=0))

        self.arrays_close(
            np.sum(np.abs(cpu_array) ** 2, axis=-1) ** .5 < 1.0, np.ones(tests))

    def test_get_ray(self):
        """Tests that get_ray returns a GPU ray through the expected pixel."""
        @cuda.jit
        def get_test_ray(target, camera, random_states):
            i = cuda.grid(1) # type: ignore
            if i < target.size:
                target[i] = ntu.flatten_ray(
                    cam.get_ray(cam.to_gpu_camera(camera), .5, .5, random_states, i))

        cpu_array = ntu.cpu_target(ndim=6)

        get_test_ray[1, 1]( # type: ignore
            cpu_array,
            cam.cpu_camera(
                cam.CameraOrientation(
                    vec.c3f(0, 0, -1),
                    vec.c3f(0, 0, 0),
                    vec.c3f(0, 1, 0)),
                cam.CameraView(1.0, 90.0),
                cam.CameraLens(0.2, 10.0)),
            create_xoroshiro128p_states(1, seed=0))

        self.arrays_close(np.abs(cpu_array[0, 0:5]) < .1, np.ones(5))
        self.assertAlmostEqual(cpu_array[0, 5], -10)

if __name__ == '__main__':
    unittest.main()
