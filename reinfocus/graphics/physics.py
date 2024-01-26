"""Methods that relate to the physics of light in a ray tracer."""

import math

from numba import cuda
from numba.cuda.cudadrv import devicearray

from reinfocus.graphics import hit_record
from reinfocus.graphics import random
from reinfocus.graphics import ray
from reinfocus.graphics import shape
from reinfocus.graphics import vector
from reinfocus.graphics import world

GpuColouredRay = tuple[ray.GpuRay, vector.G3F]


@cuda.jit
def random_in_unit_sphere(
    random_states: devicearray.DeviceNDArray, pixel_index: int
) -> vector.G3F:
    """Returns a 3D GPU vector somewhere in the unit sphere.

    Args:
        random_states: An array of RNG states.
        pixel_index: Which RNG state to use.

    Returns:
        A 3D GPU vector somewhere in the unit sphere."""

    while True:
        p = vector.sub_g3f(
            vector.smul_g3f(
                vector.g3f(
                    random.uniform_float(random_states, pixel_index),
                    random.uniform_float(random_states, pixel_index),
                    random.uniform_float(random_states, pixel_index),
                ),
                2.0,
            ),
            vector.g3f(1, 1, 1),
        )
        if vector.squared_length_g3f(p) < 1.0:
            return p


@cuda.jit
def colour_checkerboard(uf: vector.G2F, uv: vector.G2F) -> vector.G3F:
    """Returns the frequency f checkerboard colour of uv.

    Args:
        uf: The frequency of the checkerboard pattern.
        uv: The texture coordinate to colour.

    Returns:
        The frequency f checkerboard colour of uv."""

    return (
        vector.g3f(1, 0, 0)
        if math.sin(uf.x * math.pi * uv.x) * math.sin(uf.y * math.pi * uv.y) > 0
        else vector.g3f(0, 1, 0)
    )


@cuda.jit
def scatter(
    record: hit_record.GpuHitRecord,
    random_states: devicearray.DeviceNDArray,
    pixel_index: int,
) -> GpuColouredRay:
    """Returns a new ray and colour pair that results from the hit described by record.

    Args:
        record: A hit record describing the ray hit.
        random_states: An array of RNG states.
        pixel_index: Which RNG state to use.

    Returns:
        The ray scattered by, along with the colour it picked up from, the hit described
        by record."""

    return (
        ray.gpu_ray(
            record[hit_record.P],
            vector.add_g3f(
                record[hit_record.N], random_in_unit_sphere(random_states, pixel_index)
            ),
        ),
        colour_checkerboard(record[hit_record.UF], record[hit_record.UV]),
    )


@cuda.jit
def find_colour(
    shapes_parameters: shape.GpuShapeParameters,
    shapes_types: shape.GpuShapeTypes,
    r: ray.GpuRay,
    random_states: devicearray.DeviceNDArray,
    pixel_index: int,
) -> vector.G3F:
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
    current_attenuation = vector.g3f(1, 1, 1)

    for _ in range(50):
        did_hit, record = world.gpu_hit_world(
            shapes_parameters, shapes_types, current_ray, 0.001, 1000000.0
        )
        if did_hit:
            current_ray, attenuation = scatter(record, random_states, pixel_index)

            current_attenuation = vector.vmul_g3f(current_attenuation, attenuation)
        else:
            unit_direction = vector.norm_g3f(current_ray[ray.DIRECTION])
            t = 0.5 * (unit_direction.y + 1.0)
            return vector.vmul_g3f(
                vector.add_g3f(
                    vector.smul_g3f(vector.g3f(1, 1, 1), 1.0 - t),
                    vector.smul_g3f(vector.g3f(0.5, 0.7, 1), t),
                ),
                current_attenuation,
            )

    return vector.g3f(0, 0, 0)
