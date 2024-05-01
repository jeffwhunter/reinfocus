"""Functions that allow for easy initialization of environment states."""

from collections.abc import Collection, Sequence
from typing import Generic, Protocol

import numpy

from numpy import random
from numpy.typing import NDArray

from reinfocus.environments.types import StateT_co


class IStateInitializer(Protocol, Generic[StateT_co]):
    # pylint: disable=too-few-public-methods, unnecessary-ellipsis
    """The base that state initializers must follow."""

    def initialize(self, num_envs: int) -> StateT_co:
        """Returns a batch of new states.

        Args:
            num_envs: The number of states to initialize.

        Returns:
            A batch of num_envs states."""

        ...


class RangedInitializer(IStateInitializer):
    # pylint: disable=too-few-public-methods
    """A state initializer that produces new states where the elements are uniformly
    distributed within some range randomly chosen from some list of ranges. For example,
    it could initialize the first element of a state to be uniformly distributed in
    [-1, 1], while the second element could have a 50% chance of being uniformly
    distributed in either [-1, -0.5] or [0.5, 1]."""

    def __init__(self, ranges: Collection[Sequence[tuple[float, float]]]):
        """Creates a RangedInitializer.

        Args:
            ranges: A collection of sequences of ranges that control where state elements
                can initialize. The outer collection is per state element, so the first
                sequence in the collection controls the first element of the state, the
                second controls the second state element, etc. The inner sequences contain
                possible ranges that element can be initialized in, which will be selected
                from randomly each initialization. The example described in the class
                header would be specified with [[(-1, 1)], [(-1, -0.5), (0.5, 1)]]."""

        self._generator = random.Generator(random.PCG64DXSM())
        self._ranges = ranges

    def initialize(self, num_envs: int) -> NDArray[numpy.float32]:
        """Returns new uniformly distributed states within some uniformly chosen ranges.

        Args:
            num_envs: The number of states to initialize.

        Returns:
            A batch of uniformly distributed states in some uniformly chosen ranges."""

        return numpy.array(
            [
                [
                    self._generator.uniform(*self._generator.choice(r))
                    for r in self._ranges
                ]
                for _ in range(num_envs)
            ],
            dtype=numpy.float32,
        )
