"""Objects that decide when an episode ends, and how much score to destribute."""

from typing import Protocol

from reinfocus.environment.types import SI, State


class EpisodeEnder(Protocol):
    # pylint: disable=unnecessary-ellipsis
    """An interface for all EpisodeEnders to follow."""

    def is_early_end(self, state: State) -> bool:
        """Returns True if an episode is over early, False otherwise.

        Args:
            state: The state of some environment to check for an early end.

        Returns:
            True if the episode is over early, and False otherwise."""

        ...

    def reset(self):
        """Must be called to let the EpisodeEnder know a new episode has started."""

        ...

    def status(self) -> str:
        """Returns a string that represents how close the episode is to ending.

        Returns:
            A string that represents how close the episode is to ending."""

        ...


class OnTargetEpisodeEnder(EpisodeEnder):
    """An EpisodeEnder that ends an episode early if the two elements of state stay within
    a distance of early_end_radius for early_end_steps steps."""

    def __init__(self, early_end_radius: float, early_end_steps: int = 10):
        """Creates an OnTargetEpisodeEnder.

        Args:
            early_end_radius: How close the two elements of the state have to come to end
                the episode early.
            early_end_steps: How many steps the two elements have to stay close to each
                other for the episode to end early.

        Returns:
            An OnTargetEpisodeEnder."""

        self._radius = early_end_radius
        self._steps = early_end_steps
        self._counter = 0

    def is_early_end(self, state: State) -> bool:
        """Returns a value that represents if an episode has ended early because the two
        elements of the state stayed close to each other for long enough.

        Args:
            state: The state of some environment to check for an early end.

        Returns:
            True if the two elements of state have been close enough for long enough, and
            False otherwise."""

        if abs(state[SI.TARGET] - state[SI.LENS]) < self._radius:
            self._counter += 1
        else:
            self._counter = 0

        return self._counter >= self._steps

    def reset(self):
        """Lets the OnTargetEpisodeEnder know that a new episode has begun."""

        self._counter = 0

    def status(self) -> str:
        """Returns a string that shows how many steps the state's elements have been close
        enough for, and how many steps are needed to end early.

        Returns:
            A string that shows how many steps the state has been on target for, and how
            many are left to end early."""

        return f"on target {self._counter} / {self._steps}" if self._counter > 0 else ""