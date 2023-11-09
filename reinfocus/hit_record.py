"""Hit records hold information about rays hitting objects."""

from typing import Tuple

import numba as nb
from numba import cuda
from reinfocus import vector as vec

GpuHitRecord = Tuple[vec.G3F, vec.G3F, nb.float32, vec.G2F, nb.float32] # type: ignore

P = 0
N = 1
T = 2
UV = 3
M = 4

@cuda.jit
def gpu_empty_hit_record() -> GpuHitRecord:
    """Makes an empty hit record on the GPU.

    Returns:
        An GPU representation of a ray not hitting anything."""
    return (
        vec.empty_g3f(),
        vec.empty_g3f(),
        nb.float32(0),
        vec.empty_g2f(),
        nb.float32(0))

@cuda.jit
def gpu_hit_record(
    p: vec.G3F,
    n: vec.G3F,
    t: float,
    uv: vec.G2F,
    m: float
) -> GpuHitRecord:
    """Makes a hit record on the GPU.

    Args:
        p: The position of the hit.
        n: The normal of the hit.
        t: How close the hit is to maximum ray length.
        uv: The texture coordinates of the hit.
        m: What got hit: SPHERE or RECTANGLE.

    Returns:
        A GPU representation of a ray tracer hit."""
    return (p, n, t, uv, m)
