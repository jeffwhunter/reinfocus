"""Contains utilities for numba based unit tests."""
import numpy as np

def cpu_target(ndim=3):
    """Makes a single vector target array for GPU test output."""
    return np.array([(0.0,) * ndim])
