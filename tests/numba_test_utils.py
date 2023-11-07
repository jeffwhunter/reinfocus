"""Contains utilities for numba based unit tests."""

import numpy as np
from numba import cuda

from reinfocus import ray

def cpu_target(ndim=3, nrow=1):
    """Makes a single vector target array for GPU test output."""
    return np.array([(0.0,) * ndim] * nrow)

@cuda.jit
def flatten_ray(r):
    """Flattens a ray into a tuple for easy testing."""
    origin = r[ray.ORIGIN]
    direction = r[ray.DIRECTION]
    return (origin.x, origin.y, origin.z, direction.x, direction.y, direction.z)
