"""Contains utilities for numba based unit tests."""

import numba

from numba import cuda

from reinfocus.graphics import hit_record
from reinfocus.graphics import physics
from reinfocus.graphics import ray
from reinfocus.graphics import shape


@cuda.jit
def flatten_ray(r: ray.GpuRay) -> tuple[float, ...]:
    """Flattens a `GpuRay` into a `tuple[float, ...]` for easy testing.

    Args:
        `r`: The `GpuRay` to flatten.

    Returns:
        A `tuple[float, ...]` containing `r`'s data."""

    origin = r[ray.ORIGIN]
    direction = r[ray.DIRECTION]
    return origin.x, origin.y, origin.z, direction.x, direction.y, direction.z


@cuda.jit
def flatten_coloured_ray(r: physics.GpuColouredRay) -> tuple[float, ...]:
    """Flattens a `GpuColouredRay` into a `tuple[float, ...]` for easy testing.

    Args:
        `r`: The `GpuColouredRay` to flatten.

    Returns:
        A `tuple[float, ...]` containing `r`'s data."""

    return flatten_ray(r[0]) + (r[1].x, r[1].y, r[1].z)


@cuda.jit
def flatten_hit_record(hit: hit_record.GpuHitRecord) -> tuple[float, ...]:
    """Flattens a `GpuHitRectord` into a `tuple[float, ...]` for easy testing.

    Args:
    `hit`: The `GpuHitRecord` to flatten.

    Returns:
        A `tuple[float, ...]` containing `hit`'s data."""

    p = hit[hit_record.P]
    n = hit[hit_record.N]
    uv = hit[hit_record.UV]
    uf = hit[hit_record.UF]
    return (
        p.x,
        p.y,
        p.z,
        n.x,
        n.y,
        n.z,
        numba.float32(hit[hit_record.T]),
        uv.x,
        uv.y,
        uf.x,
        uf.y,
        numba.float32(hit[hit_record.M]),
    )  # type: ignore


@cuda.jit
def flatten_hit_result(hit_result: shape.GpuHitResult) -> tuple[float, ...]:
    """Flattens a `GpuHitResult` into a `tuple[float, ...]` for easy testing.

    Args:
        `hit_result`: The `GpuHitResult` to flatten.

    Return:
        A `tuple[float, ...]` containing `hit_result`'s data."""

    return (numba.float32(1.0 if hit_result[0] else 0.0),) + flatten_hit_record(
        hit_result[1]
    )  # type: ignore
