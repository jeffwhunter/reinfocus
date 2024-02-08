"""Methods relating to 3D rays."""

import numpy

from numba import cuda

from reinfocus.graphics import vector

Ray = tuple[vector.V3F, vector.V3F]

ORIGIN = 0
DIRECTION = 1


@cuda.jit
def ray(origin: vector.V3F, direction: vector.V3F) -> Ray:
    """Makes a 3D ray originating at origin and pointing in direction.

    Args:
        origin: The origin of the ray.
        direction: The direction of the ray.

    Returns:
        A 3D ray pointing from origin to direction."""

    return origin, direction


@cuda.jit
def point_at_parameter(r: Ray, t: numpy.float32) -> vector.V3F:
    """Returns the point at the end of r scaled by t.

    Args:
        r: The ray to scale.
        t: The amount to scale ray by.

    Returns:
        The point at the end of r scaled by t."""

    return vector.d_add_v3f((r[ORIGIN], vector.d_smul_v3f(r[DIRECTION], t)))
