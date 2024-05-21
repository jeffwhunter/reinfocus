"""Functions that produce rewards from Observations."""

from __future__ import annotations

from collections.abc import Callable
from typing import Generic, Protocol

import numpy

from numpy.typing import NDArray

from reinfocus.environments.types import ObservationT_contra, StateT_contra


class IEpisodeRewarder(Protocol, Generic[ObservationT_contra, StateT_contra]):
    # pylint: disable=too-few-public-methods, unnecessary-ellipsis
    """The base that episode rewarders must follow."""

    def reset(
        self,
        states: StateT_contra,
        observations: NDArray[ObservationT_contra],
        indices: NDArray[numpy.bool_] | None = None,
    ):
        """Informs the episode rewarder that some episodes have restarted.

        Args:
            states: The first states of the new episodes that reset marks the start of.
            indices: None, or a numpy array of one boolean per environment, where each
                element is True if that environment has just been reset. If None, all
                environments are considered reset."""

        ...

    def reward(
        self,
        states: StateT_contra,
        observations: NDArray[ObservationT_contra],
    ) -> NDArray[numpy.float32]:
        """Produces a batch of rewards, earned from takings actions, which lead to
        new_states, which produced new_observations.

        Args:
            states: The states that resulted from taking actions.
            observations: The observations seen during states.

        Returns:
            A numpy array of rewards, one per environment."""

        ...


class BaseRewarder(IEpisodeRewarder):
    # pylint: disable=unnecessary-ellipsis
    """An episode rewarder that can be combined with arithmetic operations."""

    def __add__(self, other: IEpisodeRewarder) -> BaseRewarder:
        """Combines this with another rewarder to produce a third. The outputs of the new
        rewarder will be those of the initial two combined with plus."""

        return OpRewarder(self, other, numpy.add)

    def __mul__(self, other: IEpisodeRewarder) -> BaseRewarder:
        """Combines this with another rewarder to produce a third. The outputs of the new
        rewarder will be those of the initial two combined with times."""

        return OpRewarder(self, other, numpy.multiply)

    def reset(
        self,
        states: NDArray[numpy.float32],
        observations: NDArray[numpy.float32],
        indices: NDArray[numpy.bool_] | None = None,
    ):
        """Informs the episode rewarder that some episodes have restarted.

        Args:
            states: The first states of the new episodes that reset marks the start of.
            indices: None, or a numpy array of one boolean per environment, where each
                element is True if that environment has just been reset. If None, all
                environments are considered reset."""

        ...


class DeltaRewarder(BaseRewarder):
    # pylint: disable=too-few-public-methods
    """A rewarder that produces a reward proportional to a change in state."""

    def __init__(
        self,
        check_index: int,
        scale: float,
        reward: float = -1.0,
    ):
        """Creates a DeltaRewarder.

        Args:
            check_index: The index of the state element of interest.
            scale: The distance the state has to move to earn reward.
            reward: The reward when the change in state is equal to scale."""

        super().__init__()

        self._check_index = check_index
        self._scale = scale
        self._reward = reward

        self._old_states = None

    def reset(
        self,
        states: NDArray[numpy.float32],
        observations: NDArray[numpy.float32],
        indices: NDArray[numpy.bool_] | None = None,
    ):
        """Informs the episode rewarder that some episodes have restarted.

        Args:
            states: The first states of the new episodes that reset marks the start of.
            indices: None, or a numpy array of one boolean per environment, where each
                element is True if that environment has just been reset. If None, all
                environments are considered reset."""

        if self._old_states is not None and indices is not None:
            self._old_states[indices] = states[:, self._check_index]
        else:
            self._old_states = states[:, self._check_index]

    def reward(
        self,
        states: NDArray[numpy.float32],
        observations: NDArray[numpy.float32],
    ) -> NDArray[numpy.float32]:
        """Produces a batch of rewards, where the emitted rewards are proportional to the
        change in some state element.

        Args:
            states: The states that resulted from taking actions.
            observations: The observations seen during states.

        Returns:
            A numpy array of rewards which are proportional to the distance some element
            of the state has moved."""

        assert self._old_states is not None

        reward = (
            abs(states[:, self._check_index] - self._old_states)
            * self._reward
            / self._scale
        )

        self._old_states = states[:, self._check_index]

        return reward


class DistanceRewarder(BaseRewarder):
    # pylint: disable=too-few-public-methods
    """A rewarder that produces a reward proportional to the distance between two specific
    elements of the state."""

    def __init__(
        self,
        check_indices: tuple[int, int],
        span: float,
        low: float = -1.0,
        high: float = 0.0,
    ):
        """Creates a DistanceRewarder.

        Args:
            check_indices: The reward will be proportional to the distance between the
                state elements with these indices.
            span: How far away the lens has to be from the target to earn low reward.
            low: The reward when the lens is a distance of span from the target.
            high: The reward when the lens is on the target."""

        super().__init__()

        self._check_indices = check_indices
        self._span = span
        self._low = low
        self._high = high

    def reward(
        self,
        states: NDArray[numpy.float32],
        observations: NDArray[numpy.float32],
    ) -> NDArray[numpy.float32]:
        """Produces a batch of rewards, where the rewards will be proportional to the
        distance between two specific elements of the state.

        Args:
            states: The states that resulted from taking actions.
            observations: The observations seen during states.

        Returns:
            A numpy array of rewards which are proportional to the distance between two
            specific elements of the state."""

        return (
            1
            - abs(states[:, self._check_indices[0]] - states[:, self._check_indices[1]])
            / self._span
        ) * (self._high - self._low) + self._low


class ObservationRewarder(BaseRewarder):
    # pylint: disable=too-few-public-methods
    """A rewarder that produces rewards by copying them from a specific element of the
    observation."""

    def __init__(self, reward_observation_index: int):
        """Creates an ObservationRewarder.

        Args:
            reward_observation_index: The index of the observation element to return as a
                reward."""

        super().__init__()

        self._reward_observation_index = reward_observation_index

    def reward(
        self,
        states: NDArray[numpy.float32],
        observations: NDArray[numpy.float32],
    ) -> NDArray[numpy.float32]:
        """Produces a batch of rewards, where the rewards are copied from some element of
        the observations.

        Args:
            states: The states that resulted from taking actions.
            observations: The observations seen during states.

        Returns:
            A numpy array of rewards copied from observations."""

        return observations[:, self._reward_observation_index]


class OnTargetRewarder(BaseRewarder):
    # pylint: disable=too-few-public-methods
    """A rewarder that produces two different rewards, produced when two elements of the
    state are and aren't close enough."""

    def __init__(
        self,
        check_indices: tuple[int, int],
        span: float,
        off: float = 0.0,
        on: float = 1.0,
    ):
        """Creates an OnTargetRewarder.

        Args:
            check_indices: The two indices of the elements of the state that, when close
                enough, cause this rewarder to emit the on target reward.
            span: How close the two elements need to be for the on target reward to be
                produced.
            off: The reward emitted when the two state elements differ by more than span.
            on: The reward emitted when the two state elements are closer than span."""

        super().__init__()

        self._check_indices = check_indices
        self._span = span
        self._off = off
        self._delta = on - off

    def reward(
        self,
        states: NDArray[numpy.float32],
        observations: NDArray[numpy.float32],
    ) -> NDArray[numpy.float32]:
        """Produces a batch of rewards, where the rewards will take one value if two
        elements of the state are close enough, and another value otherwise.

        Args:
            states: The states that resulted from taking actions.
            observations: The observations seen during states.

        Returns:
            A numpy array of rewards which take on of two values depending on if two
            specific elements of the state are or aren't close enough."""

        return (
            abs(states[:, self._check_indices[0]] - states[:, self._check_indices[1]])
            < self._span
        ) * self._delta + self._off


class OpRewarder(BaseRewarder):
    """An episode rewarder that combines rewards from other rewarders using some
    arithmetic operation."""

    def __init__(
        self,
        l_rewarder: IEpisodeRewarder,
        r_rewarder: IEpisodeRewarder,
        op: Callable[
            [NDArray[numpy.float32], NDArray[numpy.float32]], NDArray[numpy.float32]
        ],
    ):
        """Creates an OpRewarder.

        Args:
            l_rewarder: One of the two child rewarders whose output will be combined to
                produce the rewarder of this ender.
            r_rewarder: One of the two child rewarders whose output will be combined to
                produce the rewarder of this ender.
            op: The arithmetic operation that will combine the rewards from the child
                rewarders."""

        super().__init__()

        self._l_rewarder = l_rewarder
        self._r_rewarder = r_rewarder
        self._op = op

    def reset(
        self,
        states: NDArray[numpy.float32],
        observations: NDArray[numpy.float32],
        indices: NDArray[numpy.bool_] | None = None,
    ):
        """Informs the child episode rewarders that some episodes have restarted.

        Args:
            states: The first states of the new episodes that reset marks the start of.
            indices: None, or a numpy array of one boolean per environment, where each
                element is True if that environment has just been reset. If None, all
                environments are considered reset."""

        self._l_rewarder.reset(states, observations, indices)
        self._r_rewarder.reset(states, observations, indices)

    def reward(
        self,
        states: NDArray[numpy.float32],
        observations: NDArray[numpy.float32],
    ) -> NDArray[numpy.float32]:
        """Produces a batch of rewards, where the emitted rewards are arithmetically
        combined from the rewards of the child rewarders.

        Args:
            states: The states that resulted from taking actions.
            observations: The observations seen during states.

        Returns:
            The arithmetically combined rewards from the two child rewarders."""

        return self._op(
            self._l_rewarder.reward(states, observations),
            self._r_rewarder.reward(states, observations),
        )


class StoppedRewarder(BaseRewarder):
    # pylint: disable=too-few-public-methods
    """A rewarder that produces a reward if some element of the state is stopped."""

    def __init__(
        self,
        check_index: int,
        threshold: float,
        reward: float = 1.0,
    ):
        """Creates a StoppedRewarder.

        Args:
            check_index: The index of the state element of interest.
            threshold: The maximum distance the state element can move and still earn
                reward.
            reward: The potentially emitted reward."""

        super().__init__()

        self._check_index = check_index
        self._threshold = abs(threshold)
        self._reward = reward

        self._old_states = None

    def reset(
        self,
        states: NDArray[numpy.float32],
        observations: NDArray[numpy.float32],
        indices: NDArray[numpy.bool_] | None = None,
    ):
        """Informs the child episode rewarders that some episodes have restarted.

        Args:
            states: The first states of the new episodes that reset marks the start of.
            indices: None, or a numpy array of one boolean per environment, where each
                element is True if that environment has just been reset. If None, all
                environments are considered reset."""

        if self._old_states is not None and indices is not None:
            self._old_states[indices] = states[:, self._check_index]
        else:
            self._old_states = states[:, self._check_index]

    def reward(
        self,
        states: NDArray[numpy.float32],
        observations: NDArray[numpy.float32],
    ) -> NDArray[numpy.float32]:
        """Produces a batch of rewards, where the rewards will be emitted if some state
        element changes less than some threshold.

        Args:
            states: The states that resulted from taking actions.
            observations: The observations seen during states.

        Returns:
            A numpy array of potentially emitted rewards."""

        assert self._old_states is not None

        reward = (
            abs(states[:, self._check_index] - self._old_states) < self._threshold
        ) * self._reward

        self._old_states = states[:, self._check_index]

        return reward
