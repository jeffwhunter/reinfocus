"""Methods that relate to 3D cameras."""

import dataclasses
import math

from numba import cuda
from numba.cuda.cudadrv import devicearray
from reinfocus.graphics import random
from reinfocus.graphics import ray
from reinfocus.graphics import vector

CpuCamera = tuple[
    vector.C3F,
    vector.C3F,
    vector.C3F,
    vector.C3F,
    vector.C3F,
    vector.C3F,
    vector.C3F,
    float,
]

GpuCamera = tuple[
    vector.G3F,
    vector.G3F,
    vector.G3F,
    vector.G3F,
    vector.G3F,
    vector.G3F,
    vector.G3F,
    float,
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

    look_at: vector.C3F
    look_from: vector.C3F
    up: vector.C3F


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


@cuda.jit
def to_gpu_camera(camera: CpuCamera) -> GpuCamera:
    """Moves a camera from the GPU to the CPU.

    Args:
        camera: The CPU representation of a camera.

    Returns:
        A GPU representation of that camera."""

    return (
        vector.c3f_to_g3f(camera[LOWER_LEFT]),
        vector.c3f_to_g3f(camera[HORIZONTAL]),
        vector.c3f_to_g3f(camera[VERTICAL]),
        vector.c3f_to_g3f(camera[ORIGIN]),
        vector.c3f_to_g3f(camera[CAM_U]),
        vector.c3f_to_g3f(camera[CAM_V]),
        vector.c3f_to_g3f(camera[CAM_W]),
        camera[LENS_RADIUS],
    )


def cpu_camera(
    orientation: CameraOrientation, view: CameraView, lens: CameraLens
) -> CpuCamera:
    """Makes a representation of a camera suitable for transfer to the GPU.

    Args:
        orientation: The camera's orientation.
        view: The geometry of the camera's final output image.
        lens: The camera's lens.

    Returns:
        A CPU retresentation of a camera."""

    half_height = math.tan((view.vfov * math.pi / 180.0) / 2.0)
    half_width = view.aspect * half_height
    w = vector.norm_c3f(vector.sub_c3f(orientation.look_from, orientation.look_at))
    u = vector.norm_c3f(vector.cross_c3f(orientation.up, w))
    v = vector.cross_c3f(w, u)

    return (
        vector.sub_c3f(
            orientation.look_from,
            vector.add3_c3f(
                vector.smul_c3f(u, half_width * lens.focus_dist),
                vector.smul_c3f(v, half_height * lens.focus_dist),
                vector.smul_c3f(w, lens.focus_dist),
            ),
        ),
        vector.smul_c3f(u, 2.0 * half_width * lens.focus_dist),
        vector.smul_c3f(v, 2.0 * half_height * lens.focus_dist),
        orientation.look_from,
        u,
        v,
        w,
        lens.aperture / 2.0,
    )


@cuda.jit
def random_in_unit_disc(
    random_states: devicearray.DeviceNDArray, pixel_index: int
) -> vector.G2F:
    """Returns a 2D GPU vector somewhere in the unit disc.

    Args:
        random_states: An array of RNG states.
        pixel_index: Which RNG state to use.

    Returns:
        A 2D GPU vector somewhere in the unit disc."""

    while True:
        p = vector.sub_g2f(
            vector.smul_g2f(
                vector.g2f(
                    random.uniform_float(random_states, pixel_index),
                    random.uniform_float(random_states, pixel_index),
                ),
                2.0,
            ),
            vector.g2f(1, 1),
        )
        if vector.dot_g2f(p, p) < 1.0:
            return p


@cuda.jit
def get_ray(
    camera: GpuCamera,
    s: float,
    t: float,
    random_states: devicearray.DeviceNDArray,
    pixel_index: int,
) -> ray.GpuRay:
    """Returns a defocussed ray passing through camera pixel (s, t).

    Args:
        camera: The camera whose pixels the ray will pass through.
        s: The horizontal coordinate of the pixel the ray will pass through.
        t: The vertical coordinate of the pixel the ray will pass through.
        random_states: An array of RNG states.
        pixel_index: Which RNG state to use.

    Returns:
        A ray shooting out of camera through (s, t)."""

    rd = vector.smul_g2f(
        random_in_unit_disc(random_states, pixel_index), camera[LENS_RADIUS]
    )
    offset_origin = vector.add3_g3f(
        camera[ORIGIN],
        vector.smul_g3f(camera[CAM_U], rd.x),
        vector.smul_g3f(camera[CAM_V], rd.y),
    )

    return ray.gpu_ray(
        offset_origin,
        vector.sub_g3f(
            vector.add3_g3f(
                camera[LOWER_LEFT],
                vector.smul_g3f(camera[HORIZONTAL], s),
                vector.smul_g3f(camera[VERTICAL], t),
            ),
            offset_origin,
        ),
    )
