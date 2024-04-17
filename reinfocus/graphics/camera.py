"""Methods that relate to 3D cameras."""

import math

from collections.abc import Collection

import numpy

from numba import cuda
from numba.cuda.cudadrv.devicearray import DeviceNDArray
from reinfocus.graphics import random
from reinfocus.graphics import ray
from reinfocus.graphics import vector

GpuCamera = tuple[
    vector.V3F,
    vector.V3F,
    vector.V3F,
    vector.V3F,
    vector.V3F,
    vector.V3F,
    numpy.float32,
]

GpuCameras = DeviceNDArray

# [Gpu]Camera indices
CAM_LOWER_LEFT = 0
CAM_HORIZONTAL = 1
CAM_VERTICAL = 2
CAM_ORIGIN = 3
CAM_U = 4
CAM_V = 5
CAM_LENS_RADIUS = 6

FastGpuCameras = tuple[
    DeviceNDArray,
    vector.V3F,
    vector.V3F,
    vector.V3F,
    numpy.float32,
]

# FastGpuCamera indices
FCAM_DYNAMIC = 0
FCAM_ORIGIN = 1
FCAM_U = 2
FCAM_V = 3
FCAM_LENS_RADIUS = 4

FCAM_DYNAMIC_LOWER_LEFT = 0
FCAM_DYNAMIC_HORIZONTAL = 1
FCAM_DYNAMIC_VERTICAL = 2


class Cameras:
    # pylint: disable=too-few-public-methods
    """A collection of cameras that can be conveniently transfered to the GPU."""

    def __init__(self, *cameras: GpuCamera):
        """Creates a Cameras.

        Args:
            cameras: The collection of GpuCameras to include."""

        self._d_cameras = cuda.to_device(
            numpy.hstack(
                [
                    [cam[CAM_LOWER_LEFT] for cam in cameras],
                    [cam[CAM_HORIZONTAL] for cam in cameras],
                    [cam[CAM_VERTICAL] for cam in cameras],
                    [cam[CAM_ORIGIN] for cam in cameras],
                    [cam[CAM_U] for cam in cameras],
                    [cam[CAM_V] for cam in cameras],
                    numpy.reshape(
                        [cam[CAM_LENS_RADIUS] for cam in cameras], (len(cameras), 1)
                    ),
                ]
            )
        )

    def device_data(self) -> GpuCameras:
        """Returns a GPU array containing all the properties of these cameras.

        Returns:
            A GPU array containing all the properties of these cameras."""

        return self._d_cameras


class FastCameras:
    # pylint: disable=too-few-public-methods
    """A collection of cameras that can be conveniently transfered to the GPU. Reduces the
    amound of GPU data needed by assuming all cameras will have the same properties except
    for focus distance."""

    def __init__(
        self,
        focus_planes: Collection[float],
        aspect_ratio: float = 1,
        look_from: vector.V3F = vector.v3f(0, 0, 0),
        look_at: vector.V3F = vector.v3f(0, 0, -10),
        up: vector.V3F = vector.v3f(0, 1, 0),
        aperture: float = 0.1,
        vfov: float = 30,
    ):
        # pylint: disable=too-many-arguments
        """Creates a FastCameras.

        Args:
            focus_planes: The focus planes of the various cameras to create.
            aspect_ratio: The aspect ratio of all cameras.
            look_from: The position of all cameras.
            look_at: The direction all cameras are looking in.
            up: The up direction of all cameras.
            aperture: The aperture of all cameras.
            vfov: The vertical field of view of all cameras."""

        half_height = math.tan((vfov * math.pi / 180.0) / 2.0)
        half_width = aspect_ratio * half_height

        w = vector.norm_v3f(vector.sub_v3f(look_from, look_at))
        u = vector.norm_v3f(vector.cross_v3f(up, w))
        v = vector.cross_v3f(w, u)

        parameters = numpy.array(
            [
                [
                    vector.sub_v3f(
                        look_from,
                        vector.add_v3f(
                            (
                                vector.smul_v3f(u, half_width * focus_plane),
                                vector.smul_v3f(v, half_height * focus_plane),
                                vector.smul_v3f(w, focus_plane),
                            )
                        ),
                    ),
                    vector.smul_v3f(u, 2.0 * half_width * focus_plane),
                    vector.smul_v3f(v, 2.0 * half_height * focus_plane),
                ]
                for focus_plane in focus_planes
            ],
            dtype=numpy.float32,
        )

        self._d_fast_cameras = (
            cuda.to_device(parameters),
            look_from,
            u,
            v,
            numpy.divide(aperture, 2.0),
        )

    def device_data(self) -> FastGpuCameras:
        """Returns a tuple containing all the properties of these cameras.

        Returns:
            A tuple containing all the properties of these cameras."""

        return self._d_fast_cameras


def make_gpu_camera(
    aperture: float = 0.1,
    aspect_ratio: float = 1,
    focus_distance: float = 10,
    look_at: vector.V3F = vector.v3f(0, 0, -10),
    look_from: vector.V3F = vector.v3f(0, 0, 0),
    up: vector.V3F = vector.v3f(0, 1, 0),
    vfov: float = 30,
) -> GpuCamera:
    # pylint: disable=too-many-arguments
    """Makes a Camera with a given viewpoint and perspective.

    Args:
        orientation: The camera's orientation.
        view: The geometry of the camera's final output image.
        lens: The camera's lens.

    Returns:
        A tuple containing all the information needed to render an image from the
        specified camera."""

    half_height = math.tan((vfov * math.pi / 180.0) / 2.0)
    half_width = aspect_ratio * half_height
    w = vector.norm_v3f(vector.sub_v3f(look_from, look_at))
    u = vector.norm_v3f(vector.cross_v3f(up, w))
    v = vector.cross_v3f(w, u)

    return (
        vector.sub_v3f(
            look_from,
            vector.add_v3f(
                (
                    vector.smul_v3f(u, half_width * focus_distance),
                    vector.smul_v3f(v, half_height * focus_distance),
                    vector.smul_v3f(w, focus_distance),
                )
            ),
        ),
        vector.smul_v3f(u, 2.0 * half_width * focus_distance),
        vector.smul_v3f(v, 2.0 * half_height * focus_distance),
        look_from,
        u,
        v,
        numpy.divide(aperture, 2.0),
    )


@cuda.jit
def random_in_unit_disc(random_states: DeviceNDArray, pixel_index: int) -> vector.V2F:
    """Returns a 2D vector somewhere in the unit disc.

    Args:
        random_states: An array of RNG states.
        pixel_index: Which RNG state to use.

    Returns:
        A 2D vector somewhere in the unit disc."""

    while True:
        p = vector.d_sub_v2f(
            vector.d_smul_v2f(
                vector.d_v2f(
                    random.uniform_float(random_states, pixel_index),
                    random.uniform_float(random_states, pixel_index),
                ),
                numpy.float32(2.0),
            ),
            vector.d_v2f(1, 1),
        )
        if numpy.less(vector.d_dot_v2f(p, p), 1.0):
            return p


@cuda.jit
def from_cameras(cameras: GpuCameras, env_index: int) -> GpuCamera:
    """Retrieves a tuple representing a camera from a collection of cameras.

    Args:
        cameras: The device data from a Cameras collection.
        env_index: The index of the camera to retrieve from cameras.

    Returns:
        A tuple containing all the information needed to render an image from the
        specified camera."""

    return (
        vector.d_array_to_v3f(
            cameras[env_index, CAM_LOWER_LEFT * 3 : (CAM_LOWER_LEFT + 1) * 3]
        ),
        vector.d_array_to_v3f(
            cameras[env_index, CAM_HORIZONTAL * 3 : (CAM_HORIZONTAL + 1) * 3]
        ),
        vector.d_array_to_v3f(
            cameras[env_index, CAM_VERTICAL * 3 : (CAM_VERTICAL + 1) * 3]
        ),
        vector.d_array_to_v3f(cameras[env_index, CAM_ORIGIN * 3 : (CAM_ORIGIN + 1) * 3]),
        vector.d_array_to_v3f(cameras[env_index, CAM_U * 3 : (CAM_U + 1) * 3]),
        vector.d_array_to_v3f(cameras[env_index, CAM_V * 3 : (CAM_V + 1) * 3]),
        cameras[env_index, CAM_LENS_RADIUS * 3],
    )


@cuda.jit
def from_fast_cameras(cameras: FastGpuCameras, env_index: int) -> GpuCamera:
    """Retrieves a tuple representing a camera from a collection of cameras.

    Args:
        cameras: The device data from a FastCameras collection.
        env_index: The index of the camera to retrieve from cameras.

    Returns:
        A tuple containing all the information needed to render an image from the
        specified camera."""

    return (
        vector.d_array_to_v3f(cameras[FCAM_DYNAMIC][env_index, FCAM_DYNAMIC_LOWER_LEFT]),
        vector.d_array_to_v3f(cameras[FCAM_DYNAMIC][env_index, FCAM_DYNAMIC_HORIZONTAL]),
        vector.d_array_to_v3f(cameras[FCAM_DYNAMIC][env_index, FCAM_DYNAMIC_VERTICAL]),
        cameras[FCAM_ORIGIN],
        cameras[FCAM_U],
        cameras[FCAM_V],
        cameras[FCAM_LENS_RADIUS],
    )


@cuda.jit
def get_ray(
    cam: GpuCamera,
    s: numpy.float32,
    t: numpy.float32,
    random_states: DeviceNDArray,
    pixel_index: int,
) -> ray.Ray:
    """Returns a defocussed ray passing through camera pixel (s, t).

    Args:
        cam: The camera whose pixels the ray will pass through.
        s: The horizontal coordinate of the pixel the ray will pass through.
        t: The vertical coordinate of the pixel the ray will pass through.
        random_states: An array of RNG states.
        pixel_index: Which RNG state to use.

    Returns:
        A ray shooting out of camera through (s, t)."""

    rd = vector.d_smul_v2f(
        random_in_unit_disc(random_states, pixel_index), cam[CAM_LENS_RADIUS]
    )
    offset_origin = vector.d_add_v3f(
        (
            cam[CAM_ORIGIN],
            vector.d_smul_v3f(cam[CAM_U], rd[0]),
            vector.d_smul_v3f(cam[CAM_V], rd[1]),
        )
    )

    return ray.ray(
        offset_origin,
        vector.d_sub_v3f(
            vector.d_add_v3f(
                (
                    cam[CAM_LOWER_LEFT],
                    vector.d_smul_v3f(cam[CAM_HORIZONTAL], s),
                    vector.d_smul_v3f(cam[CAM_VERTICAL], t),
                )
            ),
            offset_origin,
        ),
    )
