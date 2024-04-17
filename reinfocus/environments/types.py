"""Basic types that many aspects of the focus environments refer to."""

from __future__ import annotations

from typing import Protocol, TypeVar

import numpy
from numpy.typing import NDArray


class IState(Protocol):
    # pylint: disable=too-few-public-methods, unnecessary-ellipsis
    """The base that states of vectorized environments must follow."""

    def __setitem__(self, key: NDArray[numpy.bool_], value: IState):
        """Sets some subset of the state to a new value.

        Args:
            key: An array of bools defining which elements of this state should be set.
            value: A substate that should be applied to the elements defined in key."""

        ...


StateT = TypeVar("StateT", bound=IState)
StateT_co = TypeVar("StateT_co", bound=IState, covariant=True)
StateT_contra = TypeVar("StateT_contra", bound=IState, contravariant=True)

ActionT = TypeVar("ActionT", bound=numpy.generic)
ActionT_contra = TypeVar("ActionT_contra", bound=numpy.generic, contravariant=True)

ObservationT = TypeVar("ObservationT", bound=numpy.generic)
ObservationT_co = TypeVar("ObservationT_co", bound=numpy.generic, covariant=True)
ObservationT_contra = TypeVar(
    "ObservationT_contra", bound=numpy.generic, contravariant=True
)
