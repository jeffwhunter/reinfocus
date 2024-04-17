"""Methods relating to ray traced spheres."""

import math

import numpy

from numba import cuda
from numba.cuda.cudadrv.devicearray import DeviceNDArray
from reinfocus.graphics import hit_record
from reinfocus.graphics import ray
from reinfocus.graphics import shape
from reinfocus.graphics import vector

X = 0
Y = 1
Z = 2
R = 3
FX = 4
FY = 5


def sphere(
    centre: vector.V3F, radius: float, texture: vector.V2F = vector.v2f(16, 16)
) -> shape.CpuShape:
    """Makes a representation of a sphere suitable for transfer to the GPU.

    Args:
        centre: The centre of this sphere.
        radius: The radius of this sphere.
        texture: The frequency of this sphere's checkerboard texture.

    Returns:
        A sphere that's easy to transfer to a GPU."""

    return shape.CpuShape(
        numpy.array([*centre, radius, *texture], dtype=numpy.float32), shape.SPHERE
    )


@cuda.jit
def hit(
    sphere_parameters: DeviceNDArray,
    r: ray.Ray,
    t_min: numpy.float32,
    t_max: numpy.float32,
) -> hit_record.HitResult:
    """Determines if the ray r hits the sphere defined by sphere_parameters between
        t_min and t_max, returning a hit_record containing the details if it does.

    Args:
        sphere_parameters: The vector location of the sphere, then it's radius, then the
            vector frequency of it's checkerboard texture.
        r: The ray potentially hitting the defined sphere.
        t_min: The minimum of the interval on r in which we look for hits with the
            defined sphere.
        t_max: The maximum of the interval on r in which we look for hits with the
            defined sphere.

    Returns:
        A GpuHitResult where the first element is True if a hit happened, while the
            second element is a GpuHitRecord with the details of the hit, which
            is empty if there was no hit."""

    sphere_centre = vector.d_v3f(
        sphere_parameters[X], sphere_parameters[Y], sphere_parameters[Z]
    )
    sphere_radius = sphere_parameters[R]

    oc = vector.d_sub_v3f(r[ray.ORIGIN], sphere_centre)
    a = vector.d_dot_v3f(r[ray.DIRECTION], r[ray.DIRECTION])
    b = vector.d_dot_v3f(oc, r[ray.DIRECTION])
    c = vector.d_dot_v3f(oc, oc) - sphere_radius * sphere_radius

    discriminant = b * b - a * c  # type: ignore

    if discriminant < 0:
        return (False, hit_record.empty_hit_record())

    sqrtd = math.sqrt(discriminant)

    root = (-b - sqrtd) / a  # type: ignore
    if root < t_min or t_max < root:
        root = (-b + sqrtd) / a  # type: ignore
        if root < t_min or t_max < root:
            return (False, hit_record.empty_hit_record())

    p = ray.point_at_parameter(r, root)
    n = vector.d_smul_v3f(
        vector.d_sub_v3f(p, sphere_centre), numpy.float32(1.0 / sphere_radius)
    )
    return (
        True,
        hit_record.hit_record(
            p,
            n,
            numpy.float32(root),
            uv(n),
            vector.d_v2f(sphere_parameters[FX], sphere_parameters[FY]),
            numpy.float32(shape.SPHERE),
        ),
    )


@cuda.jit
def uv(point):
    """Returns the spherical texture coordinates of any point on the unit sphere.

    Args:
        point: A 3D point on the unit sphere.

    Returns:
        A 2D point with the texture coordinates of that point."""

    return vector.d_v2f(
        (math.atan2(-point[2], point[0]) + math.pi) / math.pi,
        math.acos(-point[1]) / math.pi,
    )
