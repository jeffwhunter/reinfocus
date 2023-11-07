"""Methods relating to ray traced spheres."""

import math

import numpy as np
from numba import cuda
from reinfocus import hit_record as hit
from reinfocus import ray
from reinfocus import shape as shp
from reinfocus import types as typ
from reinfocus import vector as vec

CENTRE = 0
RADIUS = 1

def cpu_sphere(centre: typ.C3F, radius: float) -> shp.CpuShape:
    """Makes a representation of a sphere suitable for transfer to the GPU.

    Args:
        centre: The centre of this sphere.
        radius: The radius of this sphere.

    Returns:
        A sphere that's easy to transfer to a GPU."""
    return shp.CpuShape(
        np.array([*centre, radius], dtype=np.float32),
        shp.SPHERE)

@cuda.jit
def gpu_hit_sphere(
    sphere_parameters: typ.GpuShapeParameters,
    r: typ.GpuRay,
    t_min: float,
    t_max: float
) -> typ.GpuHitResult:
    """Determines if the ray r hits the sphere defined by sphere_parameters between
        t_min and t_max, returning a hit_record containing the details if it does.

    Args:
        sphere_parameters: The radius appended to the x, y, and z coordinates of
            the sphere being hit.
        r: The ray potentially hitting the defined sphere.
        t_min: The minimum of the interval on r in which we look for hits with the
            defined sphere.
        t_max: The maximum of the interval on r in which we look for hits with the
            defined sphere.

    Returns:
        A GpuHitResult where the first element is True if a hit happened, while the
            second element is a GpuHitRecord with the details of the hit, which
            is empty if there was no hit.
    """
    sphere_centre = vec.g3f(sphere_parameters[0], sphere_parameters[1], sphere_parameters[2])
    sphere_radius = sphere_parameters[3]

    oc = vec.sub_g3f(r[ray.ORIGIN], sphere_centre)
    a = vec.dot_g3f(r[ray.DIRECTION], r[ray.DIRECTION])
    b = vec.dot_g3f(oc, r[ray.DIRECTION])
    c = vec.dot_g3f(oc, oc) - sphere_radius * sphere_radius

    discriminant = b * b - a * c

    if discriminant < 0:
        return (False, hit.gpu_empty_hit_record())

    sqrtd = math.sqrt(discriminant)

    root = (-b - sqrtd) / a
    if root < t_min or t_max < root:
        root = (-b + sqrtd) / a
        if root < t_min or t_max < root:
            return (False, hit.gpu_empty_hit_record())

    p = ray.gpu_point_at_parameter(r, root)
    n = vec.div_g3f(vec.sub_g3f(p, sphere_centre), sphere_radius)
    return (True, hit.gpu_hit_record(p, n, root, gpu_sphere_uv(n), shp.SPHERE))

@cuda.jit
def gpu_sphere_uv(point):
    """Returns the spherical texture coordinates of any point on the unit sphere.
    
    Args:
        point: A G3F on the unit sphere.
        
    Returns:
        A G2F with the texture coordinates of that point."""
    return vec.g2f(
        (math.atan2(-point.z, point.x) + math.pi) / (2.0 * math.pi),
        math.acos(-point.y) / math.pi)
