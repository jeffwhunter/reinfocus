"""Methods that relate to 3D cameras."""

import dataclasses
import math

import numpy

from numba import cuda
from numba.cuda.cudadrv import devicearray
from reinfocus.graphics import random
from reinfocus.graphics import ray
from reinfocus.graphics import vector

Camera = tuple[
    vector.V3F,
    vector.V3F,
    vector.V3F,
    vector.V3F,
    vector.V3F,
    vector.V3F,
    vector.V3F,
    numpy.float32,
]

LOWER_LEFT = 0
HORIZONTAL = 1
VERTICAL = 2
ORIGIN = 3
CAM_U = 4
CAM_V = 5
CAM_W = 6
LENS_RADIUS = 7


@dataclasses.dataclass
class CameraOrientation:
    """Represents the orientation of a 3D camera in space.

    Args:
        look_at: The position the camera is looking at.
        look_from: The position of the camera.
        up: Which direction is up for the camera."""

    look_at: vector.V3F
    look_from: vector.V3F
    up: vector.V3F


@dataclasses.dataclass
class CameraView:
    """Represents the view of a camera.

    Args:
        aspect: Output image aspect ratio.
        vfov: Vertical field of view in degrees."""

    aspect: float
    vfov: float


@dataclasses.dataclass
class CameraLens:
    """Represents the lens of a camera.

    Args:
        aperture: How large the lens is.
        focus_dist: Distance from look_from of plane of perfect focus."""

    aperture: float
    focus_dist: float


def camera(orientation: CameraOrientation, view: CameraView, lens: CameraLens) -> Camera:
    """Makes a Camera with a given viewpoint and perspective.

    Args:
        orientation: The camera's orientation.
        view: The geometry of the camera's final output image.
        lens: The camera's lens.

    Returns:
        A camera."""

    half_height = math.tan((view.vfov * math.pi / 180.0) / 2.0)
    half_width = view.aspect * half_height
    w = vector.norm_v3f(vector.sub_v3f(orientation.look_from, orientation.look_at))
    u = vector.norm_v3f(vector.cross_v3f(orientation.up, w))
    v = vector.cross_v3f(w, u)

    return (
        vector.sub_v3f(
            orientation.look_from,
            vector.add_v3f(
                (
                    vector.smul_v3f(u, half_width * lens.focus_dist),
                    vector.smul_v3f(v, half_height * lens.focus_dist),
                    vector.smul_v3f(w, lens.focus_dist),
                )
            ),
        ),
        vector.smul_v3f(u, 2.0 * half_width * lens.focus_dist),
        vector.smul_v3f(v, 2.0 * half_height * lens.focus_dist),
        orientation.look_from,
        u,
        v,
        w,
        numpy.divide(lens.aperture, 2.0),
    )


@cuda.jit
def random_in_unit_disc(
    random_states: devicearray.DeviceNDArray, pixel_index: int
) -> vector.V2F:
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
def get_ray(
    cam: Camera,
    s: numpy.float32,
    t: numpy.float32,
    random_states: devicearray.DeviceNDArray,
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
        random_in_unit_disc(random_states, pixel_index), cam[LENS_RADIUS]
    )
    offset_origin = vector.d_add_v3f(
        (
            cam[ORIGIN],
            vector.d_smul_v3f(cam[CAM_U], rd[0]),
            vector.d_smul_v3f(cam[CAM_V], rd[1]),
        )
    )

    return ray.ray(
        offset_origin,
        vector.d_sub_v3f(
            vector.d_add_v3f(
                (
                    cam[LOWER_LEFT],
                    vector.d_smul_v3f(cam[HORIZONTAL], s),
                    vector.d_smul_v3f(cam[VERTICAL], t),
                )
            ),
            offset_origin,
        ),
    )
