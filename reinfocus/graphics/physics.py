"""Methods that relate to the physics of light in a ray tracer."""

import math
from typing import Tuple

from numba import cuda
from numba.cuda.cudadrv import devicearray as cda
from numba.cuda.random import xoroshiro128p_uniform_float32
from reinfocus.graphics import hit_record as hit
from reinfocus.graphics import ray
from reinfocus.graphics import shape as sha
from reinfocus.graphics import vector as vec
from reinfocus.graphics import world as wor

GpuColouredRay = Tuple[ray.GpuRay, vec.G3F]

@cuda.jit
def random_in_unit_sphere(
    random_states: cda.DeviceNDArray,
    pixel_index: int
) -> vec.G3F:
    """Returns a 3D GPU vector somewhere in the unit sphere.

    Args:
        random_states: An array of RNG states.
        pixel_index: Which RNG state to use.

    Returns:
        A 3D GPU vector somewhere in the unit sphere."""
    while True:
        p = vec.sub_g3f(
            vec.smul_g3f(
                vec.g3f(
                    xoroshiro128p_uniform_float32(random_states, pixel_index), # type: ignore
                    xoroshiro128p_uniform_float32(random_states, pixel_index), # type: ignore
                    xoroshiro128p_uniform_float32(random_states, pixel_index)), # type: ignore
                2.0),
            vec.g3f(1, 1, 1))
        if vec.squared_length_g3f(p) < 1.0:
            return p

@cuda.jit
def colour_checkerboard(f: vec.G2F, uv: vec.G2F) -> vec.G3F:
    """Returns the frequency f checkerboard colour of uv.

    Args:
        f: The frequency of the checkerboard pattern.
        uv: The texture coordinate to colour.

    Returns:
        The frequency f checkerboard colour of uv."""
    return (
        vec.g3f(1, 0, 0)
        if math.sin(f.x * math.pi * uv.x) * math.sin(f.y * math.pi * uv.y) > 0 else
        vec.g3f(0, 1, 0))

@cuda.jit
def rect_scatter(
    record: hit.GpuHitRecord,
    random_states: cda.DeviceNDArray,
    pixel_index: int
) -> GpuColouredRay:
    """Returns a new ray and colour pair that results from the rectangle hit described
        by record.

    Args:
        record: A hit record describing the rectangle hit.
        random_states: An array of RNG states.
        pixel_index: Which RNG state to use.

    Returns:
        The ray scattered by, along with the colour it picked up from, the rectangle hit
        described by record."""
    return (
        ray.gpu_ray(
            record[hit.P],
            vec.add_g3f(record[hit.N], random_in_unit_sphere(random_states, pixel_index))),
        colour_checkerboard(vec.g2f(8, 8), record[hit.UV]))

@cuda.jit
def sphere_scatter(
    record: hit.GpuHitRecord,
    random_states: cda.DeviceNDArray,
    pixel_index: int
) -> GpuColouredRay:
    """Returns a new ray and colour pair that results from the sphere hit described
        by record.

    Args:
        record: A hit record describing the sphere hit.
        random_states: An array of RNG states.
        pixel_index: Which RNG state to use.

    Returns:
        The ray scattered by, along with the colour it picked up from, the sphere hit
        described by record."""
    return (
        ray.gpu_ray(
            record[hit.P],
            vec.add_g3f(record[hit.N], random_in_unit_sphere(random_states, pixel_index))),
        colour_checkerboard(vec.g2f(64, 32), record[hit.UV]))

@cuda.jit
def scatter(
    record: hit.GpuHitRecord,
    random_states: cda.DeviceNDArray,
    pixel_index: int
) -> GpuColouredRay:
    """Returns a new ray and colour pair that results from the hit described by record.

    Args:
        record: A hit record describing the ray hit.
        random_states: An array of RNG states.
        pixel_index: Which RNG state to use.

    Returns:
        The ray scattered by, along with the colour it picked up from, the hit described
        by record."""
    if record[hit.M] == sha.SPHERE:
        return sphere_scatter(record, random_states, pixel_index)

    return rect_scatter(record, random_states, pixel_index)

@cuda.jit
def find_colour(
    shapes_parameters: sha.GpuShapeParameters,
    shapes_types: sha.GpuShapeTypes,
    r: ray.GpuRay,
    random_states: cda.DeviceNDArray,
    pixel_index: int
) -> vec.G3F:
    """Returns the colour picked up by r as it bounces around the world defined by
        shapes_parameters and shapes_types.

    Args:
        shapes_parameters: The parameters of all the shapes in the world.
        shapes_types: The types of all the shapes in the world.
        r: The ray to bounce around the world.
        random_states: An array of RNG states.
        pixel_index: Which RNG state to use.

    Returns:
        The colour picked up by r as it bounced around the world defined by
        shapes_parameters and shapes_types."""
    current_ray = r
    current_attenuation = vec.g3f(1, 1, 1)

    for _ in range(50):
        did_hit, record = wor.gpu_hit_world(
            shapes_parameters,
            shapes_types,
            current_ray,
            0.001,
            1000000.0)
        if did_hit:
            current_ray, attenuation = scatter(record, random_states, pixel_index)

            current_attenuation = vec.vmul_g3f(current_attenuation, attenuation)
        else:
            unit_direction = vec.norm_g3f(current_ray[ray.DIRECTION])
            t = 0.5 * (unit_direction.y + 1.0)
            return vec.vmul_g3f(
                vec.add_g3f(
                    vec.smul_g3f(vec.g3f(1, 1, 1), 1.0 - t),
                    vec.smul_g3f(vec.g3f(.5, .7, 1), t)),
                current_attenuation)

    return vec.g3f(0, 0, 0)
