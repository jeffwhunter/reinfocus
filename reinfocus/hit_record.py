# pylint: disable=no-member
# type: ignore

"""Hit records hold information about rays hitting objects."""

import numba as nb

from numba import cuda

P = 0
N = 1
T = 2
UV = 3
M = 4

@cuda.jit
def empty_record():
    """Makes an empty hit record."""
    return (
        cuda.float32x3(0, 0, 0),
        cuda.float32x3(0, 0, 0),
        nb.float32(0),
        cuda.float32x2(0, 0),
        nb.float32(0))

@cuda.jit
def hit_record(p, n, t, uv, m):
    """Makes a hit record.

    Args:
        p: The position of the hit.
        n: The normal of the hit.
        t: How close the hit is to maximum ray length.
        uv: The texture coordinates of the hit.
        m: What got hit: SPHERE or RECTANGLE.
    """
    return (p, n, nb.float32(t), uv, nb.float32(m))
