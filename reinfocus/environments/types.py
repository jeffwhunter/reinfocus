"""Basic types that many aspects of the focus environments refer to."""

from __future__ import annotations

from typing import Protocol, TypeVar, Generic

import numpy
from numpy.typing import NDArray


T = TypeVar("T")


class IState(Protocol, Generic[T]):
    # pylint: disable=too-few-public-methods, unnecessary-ellipsis
    """The base that states of vectorized environments must follow."""

    def __getitem__(self, key: NDArray[numpy.bool_]) -> T:
        """Gets some substate.

        Args:
            key: An array of bools defining which elements should be in the substate.

        Returns:
            The defined substate."""

        ...

    def __setitem__(self, key: NDArray[numpy.bool_], value: T):
        """Sets some substate.

        Args:
            key: An array of bools defining which elements should be in the substate.
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
