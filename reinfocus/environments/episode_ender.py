"""Objects that inspect batches of states to determine when episodes have ended."""

from typing import Any, Generic, Protocol

import numpy

from numpy.typing import NDArray

from reinfocus import histories
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


class DivergingEnder(IEpisodeEnder):
    # pylint: disable=unnecessary-ellipsis
    """An episode ender that handles states in the form of numpy arrays. Will truncate an
    episode if two elements of the state spend enough non-consecutive steps away from each
    other that are larger than some threshold."""

    def __init__(
        self,
        num_envs: int,
        check_indices: tuple[int, int],
        threshold: float,
        early_end_steps: int = 10,
    ):
        """Creates a DivergingEnder.

        Args:
            num_envs: The number of environments that this episode ender will handle.
            check_indices: The two indices of each environment's state that, having
                spent enough steps divergine, will terminate the episode.
            early_end_steps: How long the two elements must be diver for before the
            episode truncates."""

        self._num_envs = num_envs
        self._check_indices = check_indices
        self._threshold = threshold
        self._early_end_steps = early_end_steps
        self._diverging_steps = numpy.zeros(num_envs, dtype=numpy.int32)
        self._last_diff = numpy.full(num_envs, numpy.inf, dtype=numpy.float32)

    def step(self, states: Any):
        """Informs the episode ender of the new state of some timestep. Should only be
        called once per timestep.

        Args:
            states: A numpy array with shape (num_environments, num_state_elements)."""

        diff = abs(states[:, self._check_indices[0]] - states[:, self._check_indices[1]])

        diverging = diff > self._last_diff + self._threshold

        self._diverging_steps[diverging] += 1

        self._last_diff = diff

    def is_terminated(self) -> NDArray[numpy.bool_]:
        """Always returns an array representing that no episodes have terminated, as the
        focus problem has an unlimited time horizon.

        Returns:
            A numpy array of one boolean per environment, where each element is False."""

        return numpy.full(self._num_envs, False)

    def is_truncated(self) -> NDArray[numpy.bool_]:
        """Returns an array representing if episodes have truncated for having certain
        state elements diverge for long enough.

        Returns:
            A numpy array of one boolean per environment, where each element is True if
            that environment has had some specific state elments diverging for long
            enough, and False otherwise."""

        return self._diverging_steps >= self._early_end_steps

    def reset(self, dones: NDArray[numpy.bool_] | None = None):
        """Informs the episode ender tha some episodes have restarted.

        Args:
            dones: None, or a numpy array of one boolean per environment, where each
                element is True if that environment has just been reset. If None, all
                environments are considered reset."""

        if dones is None:
            dones = numpy.full(self._num_envs, True)

        self._diverging_steps[dones] = 0
        self._last_diff[dones] = numpy.inf

    def status(self, index: int) -> str:
        """Returns a string with the number of steps in total that two elements of the
        state have spent diverging, for some specified environment.

        Args:
            index: The index of the environment that will have it's status returned.

        Returns:
            A string that describes the progress of some environment towards it's end."""

        divering_steps = self._diverging_steps[index]

        return (
            f"diverging {divering_steps} / {self._early_end_steps}"
            if divering_steps > 0
            else ""
        )


class EndlessEnder(IEpisodeEnder):
    # pylint: disable=unnecessary-ellipsis
    """An episode ender that handles any state and will never truncate or terminate."""

    def __init__(self, num_envs: int):
        """Creates an EndlessEnder.

        Args:
            num_envs: The number of environments that this episode ender will handle."""

        self._num_envs = num_envs

    def step(self, states: Any):
        """Informs the episode ender of the new state of some timestep. Should only be
        called once per timestep.

        Args:
            states: A numpy array with shape (num_environments, num_state_elements)."""

        ...

    def is_terminated(self) -> NDArray[numpy.bool_]:
        """Always returns an array representing that no episodes have terminated, as the
        focus problem has an unlimited time horizon.

        Returns:
            A numpy array of one boolean per environment, where each element is False."""

        return numpy.full(self._num_envs, False)

    def is_truncated(self) -> NDArray[numpy.bool_]:
        """Always returns an array representing that no episodes have truncated.

        Returns:
            A numpy array of one boolean per environment, where each element is False."""

        return numpy.full(self._num_envs, False)

    def reset(self, dones: NDArray[numpy.bool_] | None = None):
        """Informs the episode ender tha some episodes have restarted.

        Args:
            dones: None, or a numpy array of one boolean per environment, where each
                element is True if that environment has just been reset. If None, all
                environments are considered reset."""

        ...

    def status(self, index: int) -> str:
        """Returns a string with the status of the ender, for some specified environment.

        Args:
            index: The index of the environment that will have it's status returned.

        Returns:
            An empty string, as this ender will never progress towards an end."""

        return ""


class OnTargetEnder(IEpisodeEnder):
    """An episode ender that handles states in the form of numpy arrays. Will truncate an
    episode if two elements come within some distance for some number of timesteps."""

    def __init__(
        self,
        num_envs: int,
        check_indices: tuple[int, int],
        early_end_radius: float,
        early_end_steps: int = 10,
    ):
        # pylint: disable=too-many-arguments
        """Creates an OnTargetEnder.

        Args:
            num_envs: The number of environments that this episode ender will handle.
            check_indices: The two indices of each environment's state that will, when
                close enough for long enough, terminate the episode.
            early_end_radius: How close the two elements must be to terminate the episode.
            early_end_steps: How long the two elements must be close before the episode
                truncates."""

        self._num_envs = num_envs
        self._check_indices = check_indices
        self._radius = early_end_radius
        self._early_end_steps = early_end_steps
        self._on_target_steps = numpy.zeros(num_envs, dtype=numpy.int32)

    def step(self, states: NDArray[numpy.float32]):
        """Informs the episode ender of the new state of some timestep. Should only be
        called once per timestep.

        Args:
            states: A numpy array with shape (num_environments, num_state_elements)."""

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

        return self._on_target_steps >= self._early_end_steps

    def reset(self, dones: NDArray[numpy.bool_] | None = None):
        """Informs the episode ender tha some episodes have restarted.

        Args:
            dones: None, or a numpy array of one boolean per environment, where each
                element is True if that environment has just been reset. If None, all
                environments are considered reset."""

        if dones is None:
            dones = numpy.full(self._num_envs, True)

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


class StoppedEnder(IEpisodeEnder):
    """An episode ender that handles states in the form of numpy arrays. Will truncate an
    episode if some element of the state hasn't moved more than some threshold over some
    amount of prior steps. That is, the episode will end if some element of the state has
    'stopped' for long enough."""

    def __init__(
        self,
        num_envs: int,
        check_index: int,
        early_end_span: float,
        early_end_steps: int = 10,
    ):
        """Creates a StoppedEnder.

        Args:
            num_envs: The number of environments that this episode ender will handle.
            check_index: The index of each environment's state that, when stopped, end the
                episode.
            early_end_span: The maximum distance states can move from earlier ones while
                still being 'stopped'.
            early_end_steps: How many steps the elements of the state must remain
                'stopped' before the episode ends."""

        self._num_envs = num_envs
        self._check_index = check_index
        self._early_end_span = early_end_span
        self._early_end_steps = early_end_steps
        self._moves = histories.Histories(num_envs, early_end_steps + 1)

    def step(self, states: NDArray[numpy.float32]):
        """Informs the episode ender of the new state of some timestep. Should only be
        called once per timestep.

        Args:
            states: A numpy array with shape (num_environments, num_state_elements)."""

        self._moves.append_events(states[:, self._check_index])

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

        return (
            abs(numpy.nanmax(self._moves.data, 1) - numpy.nanmin(self._moves.data, 1))
            < self._early_end_span
        ) & ~numpy.any(numpy.isnan(self._moves.data), 1)

    def reset(self, dones: NDArray[numpy.bool_] | None = None):
        """Informs the episode ender tha some episodes have restarted.

        Args:
            dones: None, or a numpy array of one boolean per environment, where each
                element is True if that environment has just been reset. If None, all
                environments are considered reset."""

        if dones is None:
            dones = numpy.full(self._num_envs, True)

        self._moves.reset(dones)

    def status(self, index: int) -> str:
        """Returns a string with the number of steps on target and the number of on target
        steps needed to end early, for some specified environment.

        Args:
            index: The index of the environment that will have it's status returned.

        Returns:
            A string that describes the progress of some environment towards it's end."""

        moves = self._moves.data[index]

        top = bottom = moves[-1]

        for i, move in enumerate(moves[self._early_end_steps - 1 :: -1]):
            if numpy.isnan(move):
                return self._status_message(i)

            if move < bottom:
                bottom = move
            elif move > top:
                top = move

            if top - bottom > self._early_end_span:
                return self._status_message(i)

        return self._status_message(self._early_end_steps)

    def _status_message(self, n_stopped: int):
        """Returns an appropriate status message for various amounts of steps.

        Args:
            n_stopped: How many stopped steps this status message is reporting.

        Returns:
            An empty string if n_stopped is zero, and an appropriate status string
            otherwise."""

        if n_stopped == 0:
            return ""

        return f"stopped {n_stopped} / {self._early_end_steps}"
