"""Classes that allow updating of GPU side data only when necessary."""

import abc

from collections.abc import Collection
from typing import Generic, TypeVar

import numpy

from numpy.typing import NDArray

DeviceDataT = TypeVar("DeviceDataT")


class DeviceData(Generic[DeviceDataT], abc.ABC):
    # pylint: disable=unnecessary-ellipsis
    """A collection of data, transformed from some original data, that will only transform
    if the original data changes."""

    def __init__(self):
        """Create a DeviceData."""

        self._data = None
        self._d_device_data = None

    def __len__(self) -> int:
        """Returns the length of the DeviceData.

        Returns:
            An integer equal to the length of the last set of original data, or zero if no
            transformations have been done."""

        return len(self._data) if self._data is not None else 0

    def device_data(self) -> DeviceDataT:
        """Returns the transformed data.

        Returns:
            The latest set of transformed data, or a thrown AssertionError if no
            transformations have been done."""

        assert self._d_device_data is not None

        return self._d_device_data

    def update(self, data: Collection[float]):
        """Checks if data has changed since the last update, and, if it has, updates the
        transformed data.

        Args:
            data: The new set of original data, which will transform into new data only if
                it has changed, or no transformations have been done."""

        data = numpy.asarray(data, dtype=numpy.float32)

        if (
            self._data is not None
            and self._data.shape == data.shape
            and all(self._data == data)
        ):
            return

        self._data = data

        self._d_device_data = self._make_device_data(data)

    @abc.abstractmethod
    def _make_device_data(self, data: NDArray[numpy.float32]) -> DeviceDataT:
        """Does some expensive transform, like sending data to the GPU.

        Args:
            The data that should be transformed into that returned by device_data.

        Returns:
            The transformed data, suitable for return by device_data."""

        ...
