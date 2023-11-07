"""Contains utilities for numba based unit tests."""

from typing import Tuple

import numba as nb
import numpy as np
from numba import cuda
from reinfocus import hit_record as hit
from reinfocus import ray
from reinfocus import types as typ

def cpu_target(ndim=3, nrow=1):
    """Makes a single vector target array for GPU test output."""
    return np.array([(0.0,) * ndim] * nrow)

@cuda.jit
def flatten_ray(r: typ.GpuRay) -> Tuple[float, float, float, float, float, float]:
    """Flattens a ray into a tuple for easy testing."""
    origin = r[ray.ORIGIN]
    direction = r[ray.DIRECTION]
    return (origin.x, origin.y, origin.z, direction.x, direction.y, direction.z)

@cuda.jit()
def flatten_hit_record(
    hit_record: typ.GpuHitRecord
) -> Tuple[float, float, float, float, float, float, float, float, float, float]:
    """Flattens a hit_record into a tuple."""
    p = hit_record[hit.P]
    n = hit_record[hit.N]
    uv = hit_record[hit.UV]
    return (
        p.x,
        p.y,
        p.z,
        n.x,
        n.y,
        n.z,
        nb.float32(hit_record[hit.T]),
        uv.x,
        uv.y,
        nb.float32(hit_record[hit.M])) # type: ignore

@cuda.jit()
def flatten_hit_result(
    hit_result: typ.GpuHitResult
) -> Tuple[float, float, float, float, float, float, float, float, float, float, float]:
    """Flattens a hit_result into a tuple."""
    return (
        (nb.float32(1. if hit_result[0] else 0.),) +
        flatten_hit_record(hit_result[1])) # type: ignore
