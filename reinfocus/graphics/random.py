"""Methods that relate to generating random numbers on the graphics card."""

from numba import cuda
from numba.cuda import random
from numba.cuda.cudadrv import devicearray


def make_random_states(n: int, seed: int) -> devicearray.DeviceNDArray:
    """Makes a set of n random states with a given seed on the graphics card.

    Args:
        n: The number of random states to make.
        seed: The starting seed used to initialize the states.

    Returns:
        A set of n random states on the graphics card."""

    return random.create_xoroshiro128p_states(n, seed)


@cuda.jit
def uniform_float(states: devicearray.DeviceNDArray, index: int) -> float:
    """Return a float sampled from [0.0, 1.0) using states[index], then advance
    states[index].

    Args:
        states: An array of random states used by the random number generator.
        index: The index of the random state to use in the sampling.

    Returns:
        A float32 sampled from [0.0, 1.0) using states[index]."""

    return random.xoroshiro128p_uniform_float32(states, index)  # type: ignore
