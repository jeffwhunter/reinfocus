"""Methods relating to 3D rays."""

from typing import Tuple

from numba import cuda
from reinfocus.graphics import vector as vec

GpuRay = Tuple[vec.G3F, vec.G3F]

ORIGIN = 0
DIRECTION = 1


@cuda.jit
def gpu_ray(origin: vec.G3F, direction: vec.G3F) -> GpuRay:
    """Makes a 3D ray on the GPU originating at origin and pointing in direction.

    Args:
        origin: The origin of the ray.
        direction: The direction of the ray.

    Returns:
        A 3D GPU ray pointing from origin to direction."""

    return origin, direction


@cuda.jit
def cpu_to_gpu_ray(origin: vec.C3F, direction: vec.C3F) -> GpuRay:
    """Converts a CPU to a GPU 3D ray.

    Args:
        origin: The origin of the ray.
        direction: The direction of the ray.

    Returns:
        A 3D GPU ray pointing from origin to direction."""

    return vec.c3f_to_g3f(origin), vec.c3f_to_g3f(direction)


@cuda.jit
def gpu_point_at_parameter(ray: GpuRay, t: float) -> vec.G3F:
    """Returns the point at the end of ray scaled by t."""

    return vec.add_g3f(ray[ORIGIN], vec.smul_g3f(ray[DIRECTION], t))
