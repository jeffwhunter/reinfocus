"""Methods that relate to the physics of light in a ray tracer."""

import math

import numpy

from numba import cuda
from numba.cuda.cudadrv import devicearray

from reinfocus.graphics import hit_record
from reinfocus.graphics import random
from reinfocus.graphics import ray
from reinfocus.graphics import shape
from reinfocus.graphics import vector
from reinfocus.graphics import world

ColouredRay = tuple[ray.Ray, vector.V3F]


@cuda.jit
def random_in_unit_sphere(
    random_states: devicearray.DeviceNDArray, pixel_index: int
) -> vector.V3F:
    """Returns a 3D vector somewhere in the unit sphere.

    Args:
        random_states: An array of RNG states.
        pixel_index: Which RNG state to use.

    Returns:
        A 3D vector somewhere in the unit sphere."""

    while True:
        p = vector.d_sub_v3f(
            vector.d_smul_v3f(
                vector.d_v3f(
                    random.uniform_float(random_states, pixel_index),
                    random.uniform_float(random_states, pixel_index),
                    random.uniform_float(random_states, pixel_index),
                ),
                numpy.float32(2.0),
            ),
            vector.d_v3f(1, 1, 1),
        )
        if numpy.less(vector.d_squared_length_v3f(p), 1.0):
            return p


@cuda.jit
def colour_checkerboard(uf: vector.V2F, uv: vector.V2F) -> vector.V3F:
    """Returns the frequency f checkerboard colour of uv.

    Args:
        uf: The frequency of the checkerboard pattern.
        uv: The texture coordinate to colour.

    Returns:
        The frequency f checkerboard colour of uv."""

    si = (uf[0] * math.pi * uv[0], uf[1] * math.pi * uv[1])  # type: ignore

    return (
        vector.d_v3f(1, 0, 0)
        if math.sin(si[0]) * math.sin(si[1]) > 0
        else vector.d_v3f(0, 1, 0)
    )


@cuda.jit
def scatter(
    record: hit_record.HitRecord,
    random_states: devicearray.DeviceNDArray,
    pixel_index: int,
) -> ColouredRay:
    """Returns a new ray and colour pair that results from the hit described by record.

    Args:
        record: A hit record describing the ray hit.
        random_states: An array of RNG states.
        pixel_index: Which RNG state to use.

    Returns:
        The ray scattered by, along with the colour it picked up from, the hit described
        by record."""

    return (
        ray.ray(
            record[hit_record.P],
            vector.d_add_v3f(
                (record[hit_record.N], random_in_unit_sphere(random_states, pixel_index))
            ),
        ),
        colour_checkerboard(record[hit_record.UF], record[hit_record.UV]),
    )


@cuda.jit
def find_colour(
    shapes_parameters: shape.GpuShapeParameters,
    shapes_types: shape.GpuShapeTypes,
    r: ray.Ray,
    random_states: devicearray.DeviceNDArray,
    pixel_index: int,
) -> vector.V3F:
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
    current_attenuation = vector.d_v3f(1, 1, 1)

    for _ in range(50):
        did_hit, record = world.hit(
            shapes_parameters,
            shapes_types,
            current_ray,
            numpy.float32(0.001),
            numpy.float32(1000000.0),
        )

        if did_hit:
            current_ray, attenuation = scatter(record, random_states, pixel_index)
            current_attenuation = vector.d_vmul_v3f(current_attenuation, attenuation)
        else:
            unit_direction = vector.d_norm_v3f(current_ray[ray.DIRECTION])
            t = 0.5 * (unit_direction[1] + 1.0)  # type: ignore
            return vector.d_vmul_v3f(
                vector.d_add_v3f(
                    (
                        vector.d_smul_v3f(vector.d_v3f(1, 1, 1), 1.0 - t),
                        vector.d_smul_v3f(vector.d_v3f(0.5, 0.7, 1), t),
                    )
                ),
                current_attenuation,
            )

    return vector.d_v3f(0, 0, 0)
