"""Methods relating to 3D rays."""

from numba import cuda
from reinfocus import types as typ, vector as vec

ORIGIN = 0
DIRECTION = 1

@cuda.jit
def gpu_ray(
    origin: typ.G3F,
    direction: typ.G3F
) -> typ.GpuRay:
    """Makes a 3D ray on the GPU originating at origin and pointing in direction."""
    return origin, direction

@cuda.jit
def gpu_point_at_parameter(ray: typ.GpuRay, t: float) -> typ.G3F:
    """Returns the point at the end of ray scaled by t."""
    return vec.add_g3f(ray[ORIGIN], vec.smul_g3f(ray[DIRECTION], t))
