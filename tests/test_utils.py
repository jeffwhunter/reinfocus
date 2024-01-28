"""Contains utilities for array based unit tests."""

from collections.abc import Sequence
from typing import TypeVar

import numpy

from numpy import testing
from numpy.typing import NDArray

CollectionT = TypeVar("CollectionT", bound=Sequence | NDArray)


def all_close(a: CollectionT, b: CollectionT, atol=1e-7, msg=None):
    """Asserts that two arrays are fairly close. An exception containing `msg` will be
    thrown if either array differs from the other in any place by more than `atol`.

    Args:
        `a`: One of the lists to compare.
        `b`: The other list to compare.
        `atol`: The absolute tolerance of the differene.
        `msg`: The message emitted if `a` and `b` differ."""

    testing.assert_allclose(
        numpy.asarray(a),
        numpy.asarray(b),
        atol=atol,
        err_msg="" if msg is None else msg,
    )


def differ(a: CollectionT, b: CollectionT, atol=1e-7, msg=None):
    """Asserts that two arrays differ in some way. An exception containing `msg` will be
    thrown if neither array differs from the other by more than `atol`.

    Args:
        `a`: One of the lists to compare.
        `b`: The other list to compare.
        `atol`: The absolute tolerance of the differene.
        `msg`: The message emitted if `a` and `b` do not differ."""

    testing.assert_raises(
        AssertionError,
        testing.assert_allclose,
        numpy.asarray(a),
        numpy.asarray(b),
        atol=atol,
        err_msg="" if msg is None else msg,
    )
