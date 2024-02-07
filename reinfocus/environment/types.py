"""Basic types that many aspects of the focus environments refer to."""

import enum

from typing import TypeVar

import numpy
from numpy.typing import NDArray


ActionT = TypeVar("ActionT", float, int)

Observation = NDArray[numpy.float32]

StateElement = numpy.float32
State = NDArray[StateElement]


class OI(enum.IntEnum):
    """The indices of Observation."""

    TARGET = 0
    LENS = 1
    FOCUS = 2


class SI(enum.IntEnum):
    """The indices of State."""

    TARGET = 0
    LENS = 1
