"""Methods and classes related to environment-wise histories."""

from collections.abc import Collection

import numpy

from numpy.typing import NDArray


class Histories:
    """A collection of ring-buffer-like objects than can be individually reset."""

    def __init__(self, num_histories: int, max_n: int):
        """Creates a Histories.

        Args:
            num_histories: The number of ring-buffer-like histories to initialize.
            max_n: The maximum length of each history."""

        self.data = numpy.full((num_histories, max_n), numpy.nan, dtype=numpy.float32)

    def get_history(self, index: int) -> NDArray[numpy.float32]:
        """Gets all non-nan values in the specified ring-buffer-like history object.

        Args:
            index: The specific history to retreive.

        Returns:
            A numpy array containing all the non-nan values from the history specified by
            index."""

        return self.data[index, numpy.where(~numpy.isnan(self.data[index]))].flatten()

    def most_recent_events(self) -> NDArray[numpy.float32]:
        """Gets the most recent event from each history.

        Returns:
            The most recent event from each history."""

        return self.data[:, -1]

    def append_events(self, events: Collection[float]):
        """Appends a numpy array of events to histories, one event per history.

        Args:
            events: A numpy array of events to append to each history.

        Returns:
            A new history created by dropping the first event from the current history,
            and appending the new one at the end."""

        events = numpy.asarray(events)

        self.data = numpy.hstack(
            [self.data[:, 1:], events.reshape(self.data.shape[0], 1)],
            dtype=numpy.float32,
        )

    def reset(self, dones: Collection[bool]):
        """Resets some specified histories.

        Args:
            dones: A numpy array of bools specifying which histories to reset."""

        dones = numpy.asarray(dones)

        self.data[dones] = numpy.full(
            (dones.sum(), self.data.shape[1]), numpy.nan, dtype=numpy.float32
        )
