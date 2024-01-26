"""Methods relating to ray traced rectangles."""

import numpy

from numba import cuda
from reinfocus.graphics import hit_record
from reinfocus.graphics import ray
from reinfocus.graphics import shape
from reinfocus.graphics import vector

X_MIN = 0
X_MAX = 1
Y_MIN = 2
Y_MAX = 3
Z_POS = 4
FX = 5
FY = 6


def cpu_rectangle(
    x_span: vector.C2F,
    y_span: vector.C2F,
    z_pos: float,
    texture: vector.C2F = vector.c2f(16, 16),
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
def gpu_hit_rectangle(
    rectangle_parameters: shape.GpuShapeParameters,
    r: ray.GpuRay,
    t_min: float,
    t_max: float,
) -> shape.GpuHitResult:
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

    t = (rectangle_parameters[Z_POS] - r[ray.ORIGIN].z) / r[ray.DIRECTION].z

    if t < t_min or t > t_max:
        return (False, hit_record.gpu_empty_hit_record())

    p = ray.gpu_point_at_parameter(r, t)

    x_min = rectangle_parameters[X_MIN]
    x_max = rectangle_parameters[X_MAX]
    y_min = rectangle_parameters[Y_MIN]
    y_max = rectangle_parameters[Y_MAX]

    if p.x < x_min or p.x > x_max or p.y < y_min or p.y > y_max:
        return (False, hit_record.gpu_empty_hit_record())

    return (
        True,
        hit_record.gpu_hit_record(
            p,
            vector.g3f(0, 0, 1),
            t,
            gpu_rectangle_uv(vector.g2f(p.x, p.y), x_min, x_max, y_min, y_max),
            vector.g2f(rectangle_parameters[FX], rectangle_parameters[FY]),
            shape.RECTANGLE,
        ),
    )


@cuda.jit
def gpu_rectangle_uv(
    point: vector.G2F, x_min: float, x_max: float, y_min: float, y_max: float
) -> vector.G2F:
    """Returns the texture coordinate of point in the rectangle [x|y]_[max|min].

    Args:
        point: A G2F on the [x|y]_[min|max] rectangle.
        x_min: The lower extent of the rectangle in the x direction.
        x_max: The upper extent of the rectangle in the x direction.
        y_min: The lower extent of the rectangle in the y direction.
        y_max: The upper extent of the rectangle in the y direction.

    Returns:
        A G2F with the texture coordinates of that point."""

    return vector.g2f(
        (point.x - x_min) / (x_max - x_min), (point.y - y_min) / (y_max - y_min)
    )
