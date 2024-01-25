"""Contains utilities for numba based unit tests."""

from typing import Tuple

import numba as nb
import numpy as np
from numba import cuda
from reinfocus.graphics import hit_record as hit
from reinfocus.graphics import physics as phy
from reinfocus.graphics import ray
from reinfocus.graphics import shape as sha

F = float
FlattenedRay = Tuple[F, F, F, F, F, F]
FlattenedColouredRay = Tuple[F, F, F, F, F, F, F, F, F]
FlattenedHitRecord = Tuple[F, F, F, F, F, F, F, F, F, F, F, F]
FlattenedHitResult = Tuple[F, F, F, F, F, F, F, F, F, F, F, F, F]

def cpu_target(ndim=3, nrow=1):
    """Makes a single vector target array for GPU test output."""
    return np.array([(0.0,) * ndim] * nrow)

@cuda.jit
def flatten_ray(r: ray.GpuRay) -> FlattenedRay:
    """Flattens a ray into a tuple for easy testing."""
    origin = r[ray.ORIGIN]
    direction = r[ray.DIRECTION]
    return (origin.x, origin.y, origin.z, direction.x, direction.y, direction.z)

@cuda.jit
def flatten_coloured_ray(r: phy.GpuColouredRay) -> FlattenedColouredRay:
    """Flattens a coloured ray into a tuple for easy testing"""
    return flatten_ray(r[0]) + (r[1].x, r[1].y, r[1].z)

@cuda.jit
def flatten_hit_record(hit_record: hit.GpuHitRecord) -> FlattenedHitRecord:
    """Flattens a hit_record into a tuple."""
    p = hit_record[hit.P]
    n = hit_record[hit.N]
    uv = hit_record[hit.UV]
    uf = hit_record[hit.UF]
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
        uf.x,
        uf.y,
        nb.float32(hit_record[hit.M])) # type: ignore

@cuda.jit
def flatten_hit_result(hit_result: sha.GpuHitResult) -> FlattenedHitResult:
    """Flattens a hit_result into a tuple."""
    return (
        (nb.float32(1. if hit_result[0] else 0.),) +
        flatten_hit_record(hit_result[1])) # type: ignore
