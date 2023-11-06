"""Methods relating to 3D rays."""

from numba import cuda

from reinfocus import vector as vc

ORIGIN = 0
DIRECTION = 1

@cuda.jit
def gpu_ray(origin, direction):
    """Makes a 3D ray on the GPU originating at origin and pointing in direction."""
    return (origin, direction)

@cuda.jit
def gpu_point_at_parameter(ray, t):
    """Returns the point at the end of ray scaled by t."""
    return vc.gpu_add(ray[ORIGIN], vc.gpu_smul(ray[DIRECTION], t))
