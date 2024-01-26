"""Hit records hold information about rays hitting objects."""

from numba import cuda
from reinfocus.graphics import vector

GpuHitRecord = tuple[vector.G3F, vector.G3F, float, vector.G2F, vector.G2F, float]

P = 0
N = 1
T = 2
UV = 3
UF = 4
M = 5


@cuda.jit
def gpu_empty_hit_record() -> GpuHitRecord:
    """Makes an empty hit record on the GPU.

    Returns:
        An GPU representation of a ray not hitting anything."""

    return (
        vector.empty_g3f(),
        vector.empty_g3f(),
        0.0,
        vector.empty_g2f(),
        vector.empty_g2f(),
        0.0,
    )


@cuda.jit
def gpu_hit_record(
    p: vector.G3F, n: vector.G3F, t: float, uv: vector.G2F, uf: vector.G2F, m: float
) -> GpuHitRecord:
    """Makes a hit record on the GPU.

    Args:
        p: The position of the hit.
        n: The normal of the hit.
        t: How close the hit is to maximum ray length.
        uv: The texture coordinates of the hit.
        uf: The checkerboard frequency of the thing hit.
        m: What got hit: SPHERE or RECTANGLE.

    Returns:
        A GPU representation of a ray tracer hit."""

    return (p, n, t, uv, uf, m)
