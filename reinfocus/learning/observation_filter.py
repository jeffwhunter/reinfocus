'''Objects that mask the output of environments and calculate the resulting spaces.'''

import typing

import gymnasium as gym
import numpy as np
import numpy.typing as npt

GymVector = typing.SupportsFloat | npt.NDArray[typing.Any]

T = typing.TypeVar('T', bound=np.generic)
Observation = npt.NDArray[T]

class ObservationFilter(typing.Generic[T]):
    '''Filters observations from a gymnasium box space into a subspace according to a
        mask.'''

    def __init__(
        self,
        low: GymVector,
        high: GymVector,
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

        self._takes = np.delete(np.arange(dim), list(mask))
        self._space = gym.spaces.Box(low, high, (len(self._takes),), dtype=np.float32)

    def __call__(self, observation: Observation) -> Observation:
        return np.take(observation, self._takes)

    def observation_space(self) -> gym.spaces.Box:
        '''Returns the newly filtered observation space.

        Returns:
            The newly filtered observation space.'''
        return self._space
