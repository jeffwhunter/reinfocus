"""Hit records hold information about rays hitting objects."""

import numpy

from numba import cuda
from reinfocus.graphics import vector

HitRecord = tuple[
    vector.V3F, vector.V3F, numpy.float32, vector.V2F, vector.V2F, numpy.float32
]

HitResult = tuple[bool, HitRecord]

P = 0
N = 1
T = 2
UV = 3
UF = 4
M = 5


@cuda.jit
def empty_hit_record() -> HitRecord:
    """Makes an empty hit record.

    Returns:
        An representation of a ray not hitting anything."""

    return (
        vector.d_v3f(0.0, 0.0, 0.0),
        vector.d_v3f(0.0, 0.0, 0.0),
        numpy.float32(0.0),
        vector.d_v2f(0.0, 0.0),
        vector.d_v2f(0.0, 0.0),
        numpy.float32(0.0),
    )


@cuda.jit
def hit_record(
    p: vector.V3F,
    n: vector.V3F,
    t: numpy.float32,
    uv: vector.V2F,
    uf: vector.V2F,
    m: numpy.float32,
) -> HitRecord:
    """Makes a hit record.

    Args:
        p: The position of the hit.
        n: The normal of the hit.
        t: How close the hit is to maximum ray length.
        uv: The texture coordinates of the hit.
        uf: The checkerboard frequency of the thing hit.
        m: What got hit: SPHERE or RECTANGLE.

    Returns:
        A representation of a ray tracer hit."""

    return (p, n, t, uv, uf, m)
