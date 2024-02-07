"""Functions that allow for easy initialization of environment states."""

from typing import Callable

import numpy

from numpy import random

from reinfocus.environment.types import State

StateInitializer = Callable[[], State]


def uniform(low: float, high: float, size: int) -> StateInitializer:
    """Makes a function that samples the initial state from a uniform distribution
        between low and high.

    Args:
        low: The lower bound of the initial state.
        high: The upper bound of the initial state.
        size: The size of the new state vector.

    Returns:
        A function that randomly initializes states of size size between low and high."""

    return lambda: random.uniform(low, high, size).astype(numpy.float32)


def ranged(ranges: list[list[tuple[float, float]]]) -> StateInitializer:
    """Makes a function that samples the initial state from a number of uniform
        distributions listed in ranges.

    Args:
        ranges: A list of lists of ranges from which the state should be initialized. The
            n-th list is the series of ranges for the n-th state element. A state element
            with more than one range will choose between them uniformly. It will then
            draw the state element from a uniform distribution on the selected range.

    Returns:
        A function that randomly initializes states to be uniformly within the listed
        ranges."""

    return lambda: numpy.array(
        [random.uniform(*r[random.choice(len(r))]) for r in ranges]
    )
