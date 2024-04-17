"""Functions that produce rewards from Observations."""

from typing import Generic, Protocol

import numpy

from numpy.typing import NDArray

from reinfocus.environments.types import (
    ActionT,
    ActionT_contra,
    ObservationT_contra,
    StateT_contra,
)


class IRewarder(Protocol, Generic[ActionT_contra, ObservationT_contra, StateT_contra]):
    # pylint: disable=too-few-public-methods, unnecessary-ellipsis
    """The base that episode rewarders must follow."""

    def reward(
        self,
        states: StateT_contra,
        observations: NDArray[ObservationT_contra],
        actions: NDArray[ActionT_contra],
    ) -> NDArray[numpy.float32]:
        """Produces a batch of rewards, earned from takings actions, which lead to
        new_states, which produced new_observations.

        Args:
            states: The states that resulted from taking actions.
            observations: The observations seen during states.
            actions: The actions that lead to states and observations.

        Returns:
            A numpy array of rewards, one per environment."""

        ...


class DistanceRewarder(IRewarder, Generic[ActionT]):
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

        self._check_indices = check_indices
        self._span = span
        self._low = low
        self._high = high

    def reward(
        self,
        states: NDArray[numpy.float32],
        observations: NDArray[numpy.float32],
        actions: NDArray[ActionT],
    ) -> NDArray[numpy.float32]:
        """Produces a batch of rewards, where the rewards will be proportional to the
        distance between two specific elements of the state.

        Args:
            states: The states that resulted from taking actions.
            observations: The observations seen during states.
            actions: The actions that lead to states and observations.

        Returns:
            A numpy array of rewards which are proportional to the distance between two
            specific elements of the state."""

        return (
            1
            - abs(states[:, self._check_indices[0]] - states[:, self._check_indices[1]])
            / self._span
        ) * (self._high - self._low) + self._low


class ObservationElementRewarder(IRewarder, Generic[ActionT]):
    # pylint: disable=too-few-public-methods
    """A rewarder that produces rewards by copying them from a specific element of the
    observation."""

    def __init__(self, reward_observation_index: int):
        """Creates an ObservationElementRewarder.

        Args:
            reward_observation_index: The index of the observation element to return as a
                reward."""

        self._reward_observation_index = reward_observation_index

    def reward(
        self,
        states: NDArray[numpy.float32],
        observations: NDArray[numpy.float32],
        actions: NDArray[ActionT],
    ) -> NDArray[numpy.float32]:
        """Produces a batch of rewards, where the rewards are copied from some element of
        the observations.

        Args:
            states: The states that resulted from taking actions.
            observations: The observations seen during states.
            actions: The actions that lead to states and observations.

        Returns:
            A numpy array of rewards copied from observations."""

        return observations[:, self._reward_observation_index]


class OnTargetRewarder(IRewarder, Generic[ActionT]):
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

        self._check_indices = check_indices
        self._span = span
        self._off = off
        self._on = on

    def reward(
        self,
        states: NDArray[numpy.float32],
        observations: NDArray[numpy.float32],
        actions: NDArray[ActionT],
    ) -> NDArray[numpy.float32]:
        """Produces a batch of rewards, where the rewards will take one value if two
        elements of the state are close enough, and another value otherwise.

        Args:
            states: The states that resulted from taking actions.
            observations: The observations seen during states.
            actions: The actions that lead to states and observations.

        Returns:
            A numpy array of rewards which take on of two values depending on if two
            specific elements of the state are or aren't close enough."""

        rewards = numpy.full(states.shape[0], self._off)

        rewards[
            abs(states[:, self._check_indices[0]] - states[:, self._check_indices[1]])
            < self._span
        ] = self._on

        return rewards
