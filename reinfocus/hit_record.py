# pylint: disable=no-member
# type: ignore

"""Hit records hold information about rays hitting objects."""

from typing import Tuple

import numba as nb

from numba import cuda
from reinfocus import vector as vec

P = 0
N = 1
T = 2
UV = 3
M = 4

GpuHitRecord = Tuple[vec.GpuVector, vec.GpuVector, nb.float32, cuda.float32x2, nb.float32]

@cuda.jit
def gpu_empty_hit_record() -> GpuHitRecord:
    """Makes an empty hit record on the GPU."""
    return (
        vec.empty_gpu_vector(),
        vec.empty_gpu_vector(),
        nb.float32(0),
        cuda.float32x2(0, 0),
        nb.float32(0))

@cuda.jit
def gpu_hit_record(
    p: vec.GpuVector,
    n: vec.GpuVector,
    t: nb.float32,
    uv: cuda.float32x2,
    m: nb.float32
) -> GpuHitRecord:
    """Makes a hit record on the GPU.

    Args:
        p: The position of the hit.
        n: The normal of the hit.
        t: How close the hit is to maximum ray length.
        uv: The texture coordinates of the hit.
        m: What got hit: SPHERE or RECTANGLE.
    """
    return (p, n, t, uv, m)
