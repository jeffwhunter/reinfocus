"""An interface for objects that transform some batch of states according to some batch of
actions."""

import abc

from collections.abc import Collection
from typing import Generic, Protocol

import numpy

from gymnasium import spaces
from gymnasium.vector import utils
from numpy.typing import NDArray

from reinfocus.environments.types import ActionT, ActionT_contra, StateT


class IStateTransformer(Generic[ActionT_contra, StateT], Protocol):
    # pylint: disable=too-few-public-methods, unnecessary-ellipsis
    """The base that state transformers must follow."""

    action_space: spaces.Space
    single_action_space: spaces.Space

    def transform(self, states: StateT, actions: NDArray[ActionT_contra]) -> StateT:
        """Returns the new states that result from enacting actions on states.

        Args:
            states: A batch of states upon which to act.
            actions: The actions to take upon those states.

        Returns:
            The new states that result from enacting actions on states."""

        ...


class StateTransformer(IStateTransformer, abc.ABC, Generic[ActionT, StateT]):
    # pylint: disable=too-few-public-methods, unnecessary-ellipsis
    """A state transformer for POMDPs with identical action spaces."""

    def __init__(self, num_envs: int, single_action_space: spaces.Space):
        """Creates a StateTransformer.

        Args:
            num_envs: The number of states this state transformer with transform.
            single_action_space: The action space each POMDP conforms to."""

        self.single_action_space = single_action_space
        self.action_space = utils.batch_space(single_action_space, num_envs)

    @abc.abstractmethod
    def transform(self, states: StateT, actions: NDArray[ActionT]) -> StateT:
        """Returns the new states that result from enacting actions on states.

        Args:
            states: A batch of states upon which to act.
            actions: The actions to take upon those states.

        Returns:
            The new states that result from enacting actions on states."""

        ...


class ContinuousJumpTransformer(StateTransformer):
    # pylint: disable=too-few-public-methods
    """A state transformer that takes actions within [-1, 1] and sets elements of the
    state to positions within the state space proportional to the actions' positions
    within the action space."""

    def __init__(
        self,
        num_envs: int,
        move_index: int,
        limits: tuple[float, float],
        stop_threshold: float = 0.1,
    ):
        """Creates a ContinuousJumpTransformer.

        Args:
            num_envs: The number of states this state transformer will transform.
            move_index: The index of the state element to move.
            limits: The lower and upper bounds that elements of the state may move to.
            stop_threshold: Actions that result in moves under this magnitude are
                stopped."""

        super().__init__(num_envs, spaces.Box(-1, 1, dtype=numpy.float32))

        self._limits = limits
        self._move_index = move_index
        self._stop_threshold = abs(stop_threshold)

    def transform(
        self, states: NDArray[numpy.float32], actions: NDArray[numpy.float32]
    ) -> NDArray[numpy.float32]:
        """Returns the new states that result from moving one element of each state to
        positions within the state space that are proportional to the actions' positions
        within the action space.

        Args:
            states: A batch of states upon which to act.
            actions: The continuous actions to take upon those states.

        Returns:
            The new states that result from enacting actions on states."""

        new_states = states.copy()

        actions = (actions.flatten() + 1) / 2.0

        moved_states = actions * (self._limits[1] - self._limits[0]) + self._limits[0]

        moved = abs(new_states[:, self._move_index] - moved_states) > self._stop_threshold

        new_states[moved, self._move_index] = moved_states[moved]

        return new_states


class ContinuousMoveTransformer(StateTransformer):
    # pylint: disable=too-few-public-methods
    """A state transformer that takes actions within [-1, 1] and moves elements of the
    state some distance proportional to the action."""

    def __init__(
        self,
        num_envs: int,
        move_index: int,
        limits: tuple[float, float],
        speed: float,
        stop_threshold: float = 0.1,
    ):
        # pylint: disable=too-many-arguments
        """Creates a ContinuousMoveTransformer.

        Args:
            num_envs: The number of states this state transformer will transform.
            move_index: The index of the state element to move.
            limits: The lower and upper bounds that elements of the state may move to.
            speed: How far an action of -1 or 1 will move the state.
            stop_threshold: Actions that result in moves under this magnitude are
                stopped."""

        super().__init__(num_envs, spaces.Box(-1, 1, dtype=numpy.float32))

        self._limits = limits
        self._move_index = move_index
        self._speed = speed
        self._stop_threshold = abs(stop_threshold)

    def transform(
        self, states: NDArray[numpy.float32], actions: NDArray[numpy.float32]
    ) -> NDArray[numpy.float32]:
        """Returns the new states that result from moving one element of each state some
        distance proportional to each action.

        Args:
            states: A batch of states upon which to act.
            actions: The continuous actions to take upon those states.

        Returns:
            The new states that result from enacting actions on states."""

        new_states = states.copy()

        actions = numpy.clip(actions.flatten(), -1, 1) * self._speed

        new_states[:, self._move_index] += (abs(actions) > self._stop_threshold) * actions
        new_states = numpy.clip(new_states, *self._limits)

        return new_states


class DiscreteJumpTransformer(StateTransformer):
    # pylint: disable=too-few-public-methods
    """A state transformer that takes actions from some set, where each action will set
    some element of the state to some specific position."""

    def __init__(
        self,
        num_envs: int,
        move_index: int,
        limits: tuple[float, float],
        action_set: Collection[float],
    ):
        """Creates a DiscreteJumpTransformer.

        Args:
            num_envs: The number of states this state transformer will transform.
            move_index: The index of the state element to move.
            limits: The lower and upper bounds that elements of the state may move to.
            actions: Where each action will set the specified state element."""

        super().__init__(num_envs, spaces.Discrete(len(action_set)))

        self._limits = limits
        self._move_index = move_index
        self._action_set = numpy.asarray(action_set, dtype=numpy.float32)

    def transform(
        self, states: NDArray[numpy.float32], actions: NDArray[numpy.int32]
    ) -> NDArray[numpy.float32]:
        """Returns the new states that result from setting one element of each state to
        some specific position chosen from a set of possible positions.

        Args:
            states: A batch of states upon which to act.
            actions: The discrete actions to take upon those states.

        Returns:
            The new states that result from enacting actions on states."""

        new_states = states.copy()

        new_states[:, self._move_index] = self._action_set[actions.flatten()]
        new_states = numpy.clip(new_states, *self._limits)

        return new_states


class DiscreteMoveTransformer(StateTransformer):
    # pylint: disable=too-few-public-methods
    """A state transformer that takes actions from some set, where each action will move
    some element of the state by a specific distance."""

    def __init__(
        self,
        num_envs: int,
        move_index: int,
        limits: tuple[float, float],
        action_set: Collection[float],
    ):
        """Creates a DiscreteMoveTransformer.

        Args:
            num_envs: The number of states this state transformer will transform.
            move_index: The index of the state element to move.
            limits: The lower and upper bounds that elements of the state may move to.
            actions: How far each of the actions move the specified state element."""

        super().__init__(num_envs, spaces.Discrete(len(action_set)))

        self._limits = limits
        self._move_index = move_index
        self._action_set = numpy.asarray(action_set)

    def transform(
        self, states: NDArray[numpy.float32], actions: NDArray[numpy.int32]
    ) -> NDArray[numpy.float32]:
        """Returns the new states that result from moving one element of each state some
        distance chosen from a set of possible moves.

        Args:
            states: A batch of states upon which to act.
            actions: The discrete actions to take upon those states.

        Returns:
            The new states that result from enacting actions on states."""

        new_states = states.copy()

        new_states[:, self._move_index] += self._action_set[actions.flatten()]
        new_states = numpy.clip(new_states, *self._limits)

        return new_states
