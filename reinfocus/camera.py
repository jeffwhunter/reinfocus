"""Methods that relate to 3D cameras."""

import math
from dataclasses import dataclass
from typing import Tuple

from numba import cuda
from numba.cuda.cudadrv import devicearray as cda
from numba.cuda.random import xoroshiro128p_uniform_float32
from reinfocus import ray
from reinfocus import vector as vec

CpuCamera = Tuple[
    vec.C3F,
    vec.C3F,
    vec.C3F,
    vec.C3F,
    vec.C3F,
    vec.C3F,
    vec.C3F,
    float]

GpuCamera = Tuple[
    vec.G3F,
    vec.G3F,
    vec.G3F,
    vec.G3F,
    vec.G3F,
    vec.G3F,
    vec.G3F,
    float]

LOWER_LEFT = 0
HORIZONTAL = 1
VERTICAL = 2
ORIGIN = 3
CAM_U = 4
CAM_V = 5
CAM_W = 6
LENS_RADIUS = 7

@dataclass
class CameraOrientation:
    """Represents the orientation of a 3D camera in space.

    Args:
        look_at: The position the camera is looking at.
        look_from: The position of the camera.
        up: Which direction is up for the camera."""
    look_at: vec.C3F
    look_from: vec.C3F
    up: vec.C3F

@dataclass
class CameraView:
    """Represents the view of a camera.

    Args:
        aspect: Output image aspect ratio.
        vfov: Vertical field of view in degrees."""
    aspect: float
    vfov: float

@dataclass
class CameraLens:
    """Represents the lens of a camera.

    Args:
        aperture: How large the lens is.
        focus_dist: Distance from look_from of plane of perfect focus."""
    aperture: float
    focus_dist: float

@cuda.jit()
def to_gpu_camera(camera: CpuCamera) -> GpuCamera:
    """Moves a camera from the GPU to the CPU.
    
    Args:
        camera: The CPU representation of a camera.

    Returns:
        A GPU representation of that camera."""
    return (
        vec.c3f_to_g3f(camera[LOWER_LEFT]),
        vec.c3f_to_g3f(camera[HORIZONTAL]),
        vec.c3f_to_g3f(camera[VERTICAL]),
        vec.c3f_to_g3f(camera[ORIGIN]),
        vec.c3f_to_g3f(camera[CAM_U]),
        vec.c3f_to_g3f(camera[CAM_V]),
        vec.c3f_to_g3f(camera[CAM_W]),
        camera[LENS_RADIUS])

def cpu_camera(
    orientation: CameraOrientation,
    view: CameraView,
    lens: CameraLens
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
    w = vec.norm_c3f(vec.sub_c3f(orientation.look_from, orientation.look_at))
    u = vec.norm_c3f(vec.cross_c3f(orientation.up, w))
    v = vec.cross_c3f(w, u)

    return (
        vec.sub_c3f(
            orientation.look_from,
            vec.add3_c3f(
                vec.smul_c3f(u, half_width * lens.focus_dist),
                vec.smul_c3f(v, half_height * lens.focus_dist),
                vec.smul_c3f(w, lens.focus_dist))),
        vec.smul_c3f(u, 2.0 * half_width * lens.focus_dist),
        vec.smul_c3f(v, 2.0 * half_height * lens.focus_dist),
        orientation.look_from,
        u,
        v,
        w,
        lens.aperture / 2.0)

@cuda.jit
def random_in_unit_disc(
    random_states: cda.DeviceNDArray,
    pixel_index: int
) -> vec.G2F:
    """Returns a 2D GPU vector somewhere in the unit disc.

    Args:
        random_states: An array of RNG states.
        pixel_index: Which RNG state to use.

    Returns:
        A 2D GPU vector somewhere in the unit disc."""
    while True:
        p = vec.sub_g2f(
            vec.smul_g2f(
                vec.g2f(
                    xoroshiro128p_uniform_float32(random_states, pixel_index), # type: ignore
                    xoroshiro128p_uniform_float32(random_states, pixel_index)), # type: ignore
                2.0),
            vec.g2f(1, 1))
        if vec.dot_g2f(p, p) < 1.0:
            return p


@cuda.jit
def get_ray(
    camera: GpuCamera,
    s: float,
    t: float,
    random_states: cda.DeviceNDArray,
    pixel_index: int
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
    rd = vec.smul_g2f(random_in_unit_disc(random_states, pixel_index), camera[LENS_RADIUS])
    offset_origin = vec.add3_g3f(
        camera[ORIGIN],
        vec.smul_g3f(camera[CAM_U], rd.x),
        vec.smul_g3f(camera[CAM_V], rd.y))

    return ray.gpu_ray(
        offset_origin,
        vec.sub_g3f(
            vec.add3_g3f(
                camera[LOWER_LEFT],
                vec.smul_g3f(camera[HORIZONTAL], s),
                vec.smul_g3f(camera[VERTICAL], t)),
            offset_origin))
