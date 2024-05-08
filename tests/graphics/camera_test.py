"""Contains tests for reinfocus.graphics.camera."""

import numpy

from numba import cuda
from numba.cuda import testing as cuda_testing
from numba.cuda.testing import unittest
from numpy import linalg
from numpy import testing as numpy_testing

from reinfocus.graphics import camera
from reinfocus.graphics import cutil
from reinfocus.graphics import random
from reinfocus.graphics import vector
from tests.graphics import numba_test_utils


class CamerasTest(cuda_testing.CUDATestCase):
    """Test cases for reinfocus.graphics.camera.Cameras."""

    def test_device_data(self):
        """Tests that Cameras produces device data with the expected elements."""

        numpy_testing.assert_allclose(
            camera.Cameras(camera.make_gpu_camera()).device_data()[0],
            [-2.68, -2.68, -10, 5.36, 0, 0, 0, 5.36, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0.05],
            atol=0.01,
        )

    def test_from_cameras(self):
        """Tests that from_cameras produces the expected camera from Cameras."""

        @cuda.jit
        def from_cameras(target, cams):
            i = cutil.line_index()
            if cutil.outside_shape(i, target.shape):
                return

            cam = camera.from_cameras(cams, i)

            target[i, 0:3] = cam[camera.CAM_LOWER_LEFT]
            target[i, 3:6] = cam[camera.CAM_HORIZONTAL]
            target[i, 6:9] = cam[camera.CAM_VERTICAL]
            target[i, 9:12] = cam[camera.CAM_ORIGIN]
            target[i, 12:15] = cam[camera.CAM_U]
            target[i, 15:18] = cam[camera.CAM_V]
            target[i, 18] = cam[camera.CAM_LENS_RADIUS]

        cpu_array = numpy.zeros((1, 19), dtype=numpy.float32)

        cutil.launcher(from_cameras, 1)(
            cpu_array, camera.Cameras(camera.make_gpu_camera()).device_data()
        )

        numpy_testing.assert_allclose(
            cpu_array[0],
            [-2.68, -2.68, -10, 5.36, 0, 0, 0, 5.36, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0.05],
            atol=0.01,
        )


class FastCamerasTest(cuda_testing.CUDATestCase):
    """Test cases for reinfocus.graphics.camera.FastCameras."""

    def test_device_data(self):
        """Tests that FastCameras produces device data with the expected elements."""

        testee = camera.FastCameras()
        testee.update([10])

        numpy_testing.assert_allclose(
            testee.device_data()[camera.FCAM_DYNAMIC][0],
            [[-2.68, -2.68, -10], [5.36, 0, 0], [0, 5.36, 0]],
            atol=0.01,
        )

        numpy_testing.assert_allclose(testee.device_data()[camera.FCAM_ORIGIN], [0, 0, 0])
        numpy_testing.assert_allclose(testee.device_data()[camera.FCAM_U], [1, 0, 0])
        numpy_testing.assert_allclose(testee.device_data()[camera.FCAM_V], [0, 1, 0])
        self.assertEqual(testee.device_data()[camera.FCAM_LENS_RADIUS], 0.05)

    def test_from_fast_cameras(self):
        """Tests that from_fast_cameras produces the expected cameras."""

        @cuda.jit
        def from_fast_cameras(target, cams):
            i = cutil.line_index()
            if cutil.outside_shape(i, target.shape):
                return

            cam = camera.from_fast_cameras(cams, i)

            target[i, 0:3] = cam[camera.CAM_LOWER_LEFT]
            target[i, 3:6] = cam[camera.CAM_HORIZONTAL]
            target[i, 6:9] = cam[camera.CAM_VERTICAL]
            target[i, 9:12] = cam[camera.CAM_ORIGIN]
            target[i, 12:15] = cam[camera.CAM_U]
            target[i, 15:18] = cam[camera.CAM_V]
            target[i, 18] = cam[camera.CAM_LENS_RADIUS]

        cpu_array = numpy.zeros((1, 19), dtype=numpy.float32)

        cameras = camera.FastCameras()
        cameras.update([10])

        cutil.launcher(from_fast_cameras, 1)(cpu_array, cameras.device_data())

        numpy_testing.assert_allclose(
            cpu_array[0],
            [-2.68, -2.68, -10, 5.36, 0, 0, 0, 5.36, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0.05],
            atol=0.01,
        )


class MakeGpuCameraTest(unittest.TestCase):
    """Test cases for reinfocus.graphics.camera.make_gpu_camera."""

    def test_elements(self):
        """Tests that make_gpu_camera makes a camera with the expected elements."""

        def flatten(l):
            return tuple(item for sublist in l for item in sublist)

        def flatten_camera(cam):
            return flatten(cam[0:6]) + (cam[6],)

        numpy_testing.assert_allclose(
            flatten_camera(
                camera.make_gpu_camera(aperture=2, look_at=vector.v3f(0, 0, -1), vfov=90),
            ),
            flatten_camera(
                (
                    vector.v3f(-10, -10, -10),
                    vector.v3f(20, 0, 0),
                    vector.v3f(0, 20, 0),
                    vector.v3f(0, 0, 0),
                    vector.v3f(1, 0, 0),
                    vector.v3f(0, 1, 0),
                    1,
                )
            ),
        )


class RandomInUnitDiscTest(cuda_testing.CUDATestCase):
    """Test cases for reinfocus.graphics.camera.random_in_unit_disc."""

    def test_result_in_unit_disc(self):
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


class GetRayTest(cuda_testing.CUDATestCase):
    """Test cases for reinfocus.graphics.camera.test_get_ray."""

    def test_middle_pixel(self):
        """Tests that get_ray through the middle pixel returns a ray pointing forawrds."""

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
            camera.make_gpu_camera(look_at=vector.v3f(0, 0, -1)),
            random.make_random_states(1, 0),
        )

        self.assertTrue(numpy.all(numpy.abs(cpu_array[0, 0:5]) < 0.1))
        self.assertAlmostEqual(cpu_array[0, 5], -10)


if __name__ == "__main__":
    unittest.main()
