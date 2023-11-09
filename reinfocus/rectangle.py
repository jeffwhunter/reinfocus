"""Methods relating to ray traced rectangles."""

import numpy as np
from numba import cuda
from reinfocus import hit_record as hit
from reinfocus import ray
from reinfocus import shape as sha
from reinfocus import vector as vec

X_MIN = 0
X_MAX = 1
Y_MIN = 2
Y_MAX = 3
Z_POS = 4

def cpu_rectangle(
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    z_pos: float
) -> sha.CpuShape:
    """Makes a representation of a z-aligned rectangle suitable for transfer to the GPU.

    Args:
        x_min: The lower extent of the rectangle in the x direction.
        x_max: The upper extent of the rectangle in the x direction.
        y_min: The lower extent of the rectangle in the y direction.
        y_max: The upper extent of the rectangle in the y direction.
        z_pos: The position of the rectangle in the z direction.

    Returns:
        A z-aligned rectangle that's easy to transfer to a GPU."""
    return sha.CpuShape(
        np.array([x_min, x_max, y_min, y_max, z_pos], dtype=np.float32),
        sha.RECTANGLE)

@cuda.jit
def gpu_hit_rectangle(
    rectangle_parameters: sha.GpuShapeParameters,
    r: ray.GpuRay,
    t_min: float,
    t_max: float
) -> sha.GpuHitResult:
    """Determines if the ray r hits the z-aligned rectangle defined byrectangle_parameters
        between t_min and t_max, returning a hit_record contraining the details if it does.

    Args:
        rectangle_parameters: The z position appened to the x and y extents of the rectangle
            being hit.
        r: The ray potentially hitting the defined z-aligned rectangle.
        t_min: The minimum of the interval on r in which we look for hits with the defined
            z-aligned rectangle.
        t_max: The maximum of the interval on r in which we look for hits with the defined
            z-aligned rectangle.

    Returns:
        A GpuHitResult where the first element is True if a hit happened, while the
            second element is a GpuHitRecord with the details of the hit, which
            is empty if there was no hit.
    """
    t = (rectangle_parameters[Z_POS] - r[ray.ORIGIN].z) / r[ray.DIRECTION].z

    if t < t_min or t > t_max:
        return (False, hit.gpu_empty_hit_record())

    x = r[ray.ORIGIN].x + t * r[ray.DIRECTION].x
    y = r[ray.ORIGIN].y + t * r[ray.DIRECTION].y

    x_min = rectangle_parameters[X_MIN]
    x_max = rectangle_parameters[X_MAX]
    y_min = rectangle_parameters[Y_MIN]
    y_max = rectangle_parameters[Y_MAX]

    if x < x_min or x > x_max or y < y_min or y > y_max:
        return (False, hit.gpu_empty_hit_record())

    return (
        True,
        hit.gpu_hit_record(
            ray.gpu_point_at_parameter(r, t),
            vec.g3f(0, 0, 1),
            t,
            vec.g2f((x - x_min) / (x_max - x_min), (y - y_min) / (y_max - y_min)),
            sha.RECTANGLE))
