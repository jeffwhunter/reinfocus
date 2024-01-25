"""Methods relating to ray traced spheres."""

import math

import numpy as np
from numba import cuda
from reinfocus.graphics import hit_record as hit
from reinfocus.graphics import ray
from reinfocus.graphics import shape as sha
from reinfocus.graphics import vector as vec

X = 0
Y = 1
Z = 2
R = 3
FX = 4
FY = 5

def cpu_sphere(
    centre: vec.C3F,
    radius: float,
    texture: vec.C2F = vec.c2f(16, 16)
) -> sha.CpuShape:
    """Makes a representation of a sphere suitable for transfer to the GPU.

    Args:
        centre: The centre of this sphere.
        radius: The radius of this sphere.
        texture: The frequency of this sphere's checkerboard texture.

    Returns:
        A sphere that's easy to transfer to a GPU."""

    return sha.CpuShape(
        np.array([*centre, radius, *texture], dtype=np.float32),
        sha.SPHERE)

@cuda.jit
def gpu_hit_sphere(
    sphere_parameters: sha.GpuShapeParameters,
    r: ray.GpuRay,
    t_min: float,
    t_max: float
) -> sha.GpuHitResult:
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
    sphere_centre = vec.g3f(sphere_parameters[X], sphere_parameters[Y], sphere_parameters[Z])
    sphere_radius = sphere_parameters[R]

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
    return (
        True,
        hit.gpu_hit_record(
            p,
            n,
            root,
            gpu_sphere_uv(n),
            vec.g2f(sphere_parameters[FX], sphere_parameters[FY]),
            sha.SPHERE))

@cuda.jit
def gpu_sphere_uv(point):
    """Returns the spherical texture coordinates of any point on the unit sphere.
    
    Args:
        point: A G3F on the unit sphere.
        
    Returns:
        A G2F with the texture coordinates of that point."""
    return vec.g2f(
        (math.atan2(-point.z, point.x) + math.pi) / math.pi,
        math.acos(-point.y) / math.pi)
