"""Contains utilities for numba based unit tests."""

import numpy

from numba import cuda

from reinfocus.graphics import hit_record
from reinfocus.graphics import physics
from reinfocus.graphics import ray
from reinfocus.graphics import shape


@cuda.jit
def flatten_ray(r: ray.Ray) -> tuple[numpy.float32, ...]:
    """Flattens a Ray into a tuple[float, ...] for easy testing.

    Args:
        r: The Ray to flatten.

    Returns:
        A tuple[float, ...] containing r's data."""

    origin = r[ray.ORIGIN]
    direction = r[ray.DIRECTION]
    return origin[0], origin[1], origin[2], direction[0], direction[1], direction[2]


@cuda.jit
def flatten_coloured_ray(r: physics.ColouredRay) -> tuple[numpy.float32, ...]:
    """Flattens a ColouredRay into a tuple[float, ...] for easy testing.

    Args:
        r: The ColouredRay to flatten.

    Returns:
        A tuple[float, ...] containing r's data."""

    return flatten_ray(r[0]) + (r[1][0], r[1][1], r[1][2])


@cuda.jit
def flatten_hit_record(hit: hit_record.HitRecord) -> tuple[numpy.float32, ...]:
    """Flattens a HitRecord into a tuple[float, ...] for easy testing.

    Args:
    hit: The HitRecord to flatten.

    Returns:
        A tuple[float, ...] containing hit's data."""

    p = hit[hit_record.P]
    n = hit[hit_record.N]
    uv = hit[hit_record.UV]
    uf = hit[hit_record.UF]
    return (
        p[0],
        p[1],
        p[2],
        n[0],
        n[1],
        n[2],
        hit[hit_record.T],
        uv[0],
        uv[1],
        uf[0],
        uf[1],
        hit[hit_record.M],
    )


@cuda.jit
def flatten_hit_result(hit_result: hit_record.HitResult) -> tuple[numpy.float32, ...]:
    """Flattens a HitResult into a tuple[float, ...] for easy testing.

    Args:
        hit_result: The HitResult to flatten.

    Return:
        A tuple[float, ...] containing hit_result's data."""

    return (numpy.float32(1.0 if hit_result[0] else 0.0),) + flatten_hit_record(
        hit_result[1]
    )
