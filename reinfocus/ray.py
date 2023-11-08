"""Methods relating to 3D rays."""

from numba import cuda
from reinfocus import types as typ
from reinfocus import vector as vec

ORIGIN = 0
DIRECTION = 1

@cuda.jit
def gpu_ray(
    origin: typ.G3F,
    direction: typ.G3F
) -> typ.GpuRay:
    """Makes a 3D ray on the GPU originating at origin and pointing in direction.

    Args:
        origin: The origin of the ray.
        direction: The direction of the ray.

    Returns:
        A 3D GPU ray pointing from origin to direction."""
    return origin, direction

@cuda.jit
def cpu_to_gpu_ray(
    origin: typ.C3F,
    direction: typ.C3F
) -> typ.GpuRay:
    """Converts a CPU to a GPU 3D ray.

    Args:
        origin: The origin of the ray.
        direction: The direction of the ray.

    Returns:
        A 3D GPU ray pointing from origin to direction."""
    return vec.c3f_to_g3f(origin), vec.c3f_to_g3f(direction)

@cuda.jit
def gpu_point_at_parameter(ray: typ.GpuRay, t: float) -> typ.G3F:
    """Returns the point at the end of ray scaled by t."""
    return vec.add_g3f(ray[ORIGIN], vec.smul_g3f(ray[DIRECTION], t))
