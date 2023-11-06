# pylint: disable=no-member
# type: ignore

"""Methods relating to 3D rays."""

from typing import Tuple

import numba as nb

from numba import cuda
from reinfocus import vector as vec

ORIGIN = 0
DIRECTION = 1

GpuRay = Tuple[vec.GpuVector, vec.GpuVector]

@cuda.jit
def gpu_ray(
    origin: vec.GpuVector,
    direction: vec.GpuVector
) -> GpuRay:
    """Makes a 3D ray on the GPU originating at origin and pointing in direction."""
    return (origin, direction)

@cuda.jit
def gpu_point_at_parameter(ray: GpuRay, t: nb.float32) -> vec.GpuVector:
    """Returns the point at the end of ray scaled by t."""
    return vec.gpu_add(ray[ORIGIN], vec.gpu_smul(ray[DIRECTION], t))
