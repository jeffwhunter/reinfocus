"""Objects that inspect batches of states to determine when episodes have ended."""

from typing import Generic, Protocol

import numpy

from numpy.typing import NDArray

from reinfocus.environments.types import StateT_contra


class IEpisodeEnder(Protocol, Generic[StateT_contra]):
    # pylint: disable=unnecessary-ellipsis
    """The base that episode enders must follow."""

    def step(self, states: StateT_contra):
        """Informs the episode ender of the new state of some timestep. Should only be
        called once per timestep.

        Args:
            states: The new states that were reached on the current timestep."""

        ...

    def is_terminated(self) -> NDArray[numpy.bool_]:
        """Answers if the ongoing episodes have terminated. Termination happens when the
        underlying MDP reaches some sort of terminal state (reach goal state, time limit
        in time limited MDPs, etc).

        Returns:
            A numpy array of one boolean per environment, where each element is True if
            that environment's episode has terminated, False otherwise."""

        ...

    def is_truncated(self) -> NDArray[numpy.bool_]:
        """Answers if the ongoing episodes have truncated. Truncation happens when some
        condition outside the MDP ends learning (robot reaching edge of practical
        simulation, time limits in an infinite horizon problem, etc).

        Returns:
            A numpy array of one boolean per environment, where each element is True if
            that environment's episode has truncated, False otherwise."""

        ...

    def reset(self, dones: NDArray[numpy.bool_] | None = None):
        """Informs the episode ender that some episodes have restarted.

        Args:
            dones: None, or a numpy array of one boolean per environment, where each
                element is True if that environment has just been reset. If None, all
                environments are considered reset."""

        ...

    def status(self, index: int) -> str:
        """Returns a string describing how close some environment is to ending.

        Args:
            index: The index of the environment that will have it's status returned.

        Returns:
            A string that describes the progress of some environment towards it's end."""

        ...


class EndlessEpisodeEnder(IEpisodeEnder):
    # pylint: disable=unnecessary-ellipsis
    def __init__(self, num_envs: int):
        """Creates an EndlessEpisodeEnder.

        Args:
            num_envs: The number of environments that this episode ender will handle."""

        self._num_envs = num_envs

    def step(self, states: NDArray[numpy.float32]):
        """Informs the episode ender of the new state of some timestep. Should only be
        called once per timestep.

        Args:
            states: A numpy array with shape (num_environments, num_state_elements)."""

        ...

    def is_terminated(self) -> NDArray[numpy.bool_]:
        return numpy.full(self._num_envs, False)

    def is_truncated(self) -> NDArray[numpy.bool_]:
        return numpy.full(self._num_envs, False)

    def reset(self, dones: NDArray[numpy.bool_] | None = None): ...

    def status(self, index: int) -> str:
        return ""


class OnTargetEpisodeEnder(IEpisodeEnder):
    """A episode ender that handles states in the form of numpy arrays. Will truncate an
    episode if two elements come within some distance for some number of timesteps or at
    some maximum number of timesteps."""

    def __init__(
        self,
        num_envs: int,
        check_indices: tuple[int, int],
        early_end_radius: float,
        early_end_steps: int = 10,
        max_episode_steps: int = numpy.iinfo(numpy.int32).max,
    ):
        # pylint: disable=too-many-arguments
        """Creates an OnTargetEpisodeEnder.

        Args:
            num_envs: The number of environments that this episode ender will handle.
            check_indices: The two indices of each environment's state that will, when
                close enough for long enough, terminate the episode.
            early_end_radius: How close the two elements must be to terminate the episode.
            early_end_steps: How long the two elements must be close before the episode
                terminates.
            max_episode_steps: How many steps the episode will be truncated after."""

        self._num_envs = num_envs
        self._check_indices = check_indices
        self._radius = early_end_radius
        self._early_end_steps = early_end_steps
        self._max_episode_steps = max_episode_steps
        self._total_steps = numpy.zeros(num_envs, dtype=numpy.int32)
        self._on_target_steps = numpy.zeros(num_envs, dtype=numpy.int32)

    def step(self, states: NDArray[numpy.float32]):
        """Informs the episode ender of the new state of some timestep. Should only be
        called once per timestep.

        Args:
            states: A numpy array with shape (num_environments, num_state_elements)."""

        self._total_steps += 1

        on_targets = (
            abs(states[:, self._check_indices[0]] - states[:, self._check_indices[1]])
            < self._radius
        )
        self._on_target_steps[on_targets] += 1
        self._on_target_steps[numpy.invert(on_targets)] = 0

    def is_terminated(self) -> NDArray[numpy.bool_]:
        """Always returns an array representing that no episodes have terminated, as the
        focus problem has an unlimited time horizon.

        Returns:
            A numpy array of one boolean per environment, where each element is False."""

        return numpy.full(self._num_envs, False)

    def is_truncated(self) -> NDArray[numpy.bool_]:
        """Returns an array representing if episodes have truncated for being on target
        for long enough, or if they have reached a time limit.

        Returns:
            A numpy array of one boolean per environment, where each element is True if
            that environment has been on target for long enough, or had enough timesteps
            pass, and False otherwise."""

        return numpy.logical_or(
            self._on_target_steps >= self._early_end_steps,
            self._total_steps >= self._max_episode_steps,
        )

    def reset(self, dones: NDArray[numpy.bool_] | None = None):
        """Informs the episode ender tha some episodes have restarted.

        Args:
            dones: None, or a numpy array of one boolean per environment, where each
                element is True if that environment has just been reset. If None, all
                environments are considered reset.
        """

        if dones is None:
            dones = numpy.full(self._num_envs, True)

        self._total_steps[dones] = 0
        self._on_target_steps[dones] = 0

    def status(self, index: int) -> str:
        """Returns a string with the number of steps on target and the number of on target
        steps needed to end early, for some specified environment.

        Args:
            index: The index of the environment that will have it's status returned.

        Returns:
            A string that describes the progress of some environment towards it's end."""

        on_step = self._on_target_steps[index]

        return f"on target {on_step} / {self._early_end_steps}" if on_step > 0 else ""
