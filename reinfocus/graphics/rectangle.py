"""Methods relating to ray traced rectangles."""

import numpy

from numba import cuda
from numba.cuda.cudadrv.devicearray import DeviceNDArray
from reinfocus.graphics import hit_record
from reinfocus.graphics import ray
from reinfocus.graphics import shape
from reinfocus.graphics import vector

# Indices for hit's rectangular parameters
X_MIN = 0
X_MAX = 1
Y_MIN = 2
Y_MAX = 3
Z_POS = 4
FX = 5
FY = 6

# Indices for fast_hit's rectangular parameters
FH_RADIUS = 0
FH_ZPOS = 1


def rectangle(
    x_span: vector.V2F,
    y_span: vector.V2F,
    z_pos: float,
    texture: vector.V2F = vector.v2f(16, 16),
) -> shape.CpuShape:
    """Makes a representation of a z-aligned rectangle suitable for transfer to the GPU.

    Args:
        x_span: The lower and upper extent of the rectangle in the x direction.
        y_span: The lower and upper extent of the rectangle in the y direction.
        z_pos: The position of the rectangle in the z direction.
        texture: The frequency of this rectangle's checkerboard texture.

    Returns:
        A z-aligned rectangle that's easy to transfer to a GPU."""

    return shape.CpuShape(
        numpy.array([*x_span, *y_span, z_pos, *texture], dtype=numpy.float32),
        shape.RECTANGLE,
    )


@cuda.jit
def hit(
    rectangle_parameters: DeviceNDArray,
    r: ray.Ray,
    t_min: numpy.float32,
    t_max: numpy.float32,
) -> hit_record.HitResult:
    """Determines if the ray r hits the z-aligned rectangle defined byrectangle_parameters
        between t_min and t_max, returning a hit_record contraining the details if it
        does.

    Args:
        rectangle_parameters: The left, right, bottom, and top of the rectangle, then the
            depth, then the frequency of it's checkerboard texture.
        r: The ray potentially hitting the defined z-aligned rectangle.
        t_min: The minimum of the interval on r in which we look for hits with the defined
            z-aligned rectangle.
        t_max: The maximum of the interval on r in which we look for hits with the defined
            z-aligned rectangle.

    Returns:
        A GpuHitResult where the first element is True if a hit happened, while the
            second element is a GpuHitRecord with the details of the hit, which
            is empty if there was no hit."""

    t = (rectangle_parameters[Z_POS] - r[ray.ORIGIN][2]) / r[ray.DIRECTION][2]

    if t < t_min or t > t_max:
        return (False, hit_record.empty_hit_record())

    p = ray.point_at_parameter(r, t)

    x_min = rectangle_parameters[X_MIN]
    x_max = rectangle_parameters[X_MAX]
    y_min = rectangle_parameters[Y_MIN]
    y_max = rectangle_parameters[Y_MAX]

    if p[0] < x_min or p[0] > x_max or p[1] < y_min or p[1] > y_max:
        return (False, hit_record.empty_hit_record())

    return (
        True,
        hit_record.hit_record(
            p,
            vector.d_v3f(0, 0, 1),
            numpy.float32(t),
            uv(p[0:2], x_min, x_max, y_min, y_max),
            vector.d_v2f(rectangle_parameters[FX], rectangle_parameters[FY]),
            numpy.float32(shape.RECTANGLE),
        ),
    )


@cuda.jit
def fast_hit(
    parameters: DeviceNDArray,
    r: ray.Ray,
    t_min: numpy.float32,
    t_max: numpy.float32,
) -> hit_record.HitResult:
    """Determines if the ray r hits the z-aligned rectangle defined by parameters between
    t_min and t_max, returning a hit_record contraining the details if it does.

    Args:
        parameters: The half-side-length and z position of a z-axis aligned rectangle.
        r: The ray potentially hitting the defined z-aligned rectangle.
        t_min: The minimum of the interval on r in which we look for hits with the defined
            z-aligned rectangle.
        t_max: The maximum of the interval on r in which we look for hits with the defined
            z-aligned rectangle.

    Returns:
        A GpuHitResult where the first element is True if a hit happened, while the
            second element is a GpuHitRecord with the details of the hit, which
            is empty if there was no hit."""

    radius = parameters[FH_RADIUS]
    z_pos = parameters[FH_ZPOS]

    t = (z_pos - r[ray.ORIGIN][2]) / r[ray.DIRECTION][2]

    if t < t_min or t > t_max:
        return (False, hit_record.empty_hit_record())

    p = ray.point_at_parameter(r, t)

    if p[0] < -radius or p[0] > radius or p[1] < -radius or p[1] > radius:
        return (False, hit_record.empty_hit_record())

    return (
        True,
        hit_record.hit_record(
            p,
            vector.d_v3f(0, 0, 1),
            numpy.float32(t),
            uv(p[0:2], -radius, radius, -radius, radius),
            vector.d_v2f(16.0, 16.0),
            numpy.float32(shape.RECTANGLE),
        ),
    )


@cuda.jit
def uv(
    point: vector.V2F, x_min: float, x_max: float, y_min: float, y_max: float
) -> vector.V2F:
    """Returns the texture coordinate of point in the rectangle [x|y]_[max|min].

    Args:
        point: A 2D point on the [x|y]_[min|max] rectangle.
        x_min: The lower extent of the rectangle in the x direction.
        x_max: The upper extent of the rectangle in the x direction.
        y_min: The lower extent of the rectangle in the y direction.
        y_max: The upper extent of the rectangle in the y direction.

    Returns:
        A 2D point with the texture coordinates of that point."""

    return vector.d_v2f(
        (point[0] - x_min) / (x_max - x_min),  # type: ignore
        (point[1] - y_min) / (y_max - y_min),  # type: ignore
    )
