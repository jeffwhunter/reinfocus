'''Objects that mask the output of environments and calculate the resulting spaces.'''

import numpy

from gymnasium import spaces
from numpy.typing import NDArray

class ObservationFilter:
    '''Filters observations from a gymnasium box space into a subspace according to a
        mask.'''

    def __init__(
        self,
        low: float,
        high: float,
        dim: int,
        mask: set[int] | None = None
    ):
        '''Creates an observation filter for a dim-dimensional box observation space
            between low and high. Will return observations whose indices aren't included
            in mask.

        Args:
            low: The lower bound of the unfiltered observation space.
            high: The upper bound of the unfiltered observation space.
            dim: The dimension of the unfiltered observation space.
            mask: The indices of the elements of the observation that will be hidden.'''

        if mask is None:
            mask = set[int]()

        lm = len(mask)

        if lm >= dim:
            raise IndexError(f'mask {mask} has sze > {dim - 1}.')

        if lm > 0:
            if max(mask) >= dim:
                raise IndexError(f'mask {mask} has element > {dim - 1}.')

            if min(mask) < 0:
                raise IndexError(f'mask {mask} has element < 0.')

        self._takes = numpy.delete(numpy.arange(dim), list(mask))
        self._space = spaces.Box(low, high, (len(self._takes),), dtype=numpy.float32)

    def __call__(self, observation: NDArray) -> NDArray:
        '''Filters an observation down to the set subspace.

        Args:
            observation: The observation to filter.

        Returns:
            The filtered observation.'''

        return numpy.take(observation, self._takes)

    def observation_space(self) -> spaces.Box:
        '''Returns the newly filtered observation space.

        Returns:
            The newly filtered observation space.'''

        return self._space
