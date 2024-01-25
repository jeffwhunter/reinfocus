"""Objects that compute the results of various types of dynamics on general states."""

import abc

from typing import Callable, Generic, TypeVar

import numpy

from gymnasium import spaces
from numpy.typing import NDArray

StateElement = numpy.float32
State = NDArray[StateElement]

Action = TypeVar("Action", bound=numpy.number)


class Dynamics(abc.ABC, Generic[Action]):
    """Generic state dynamics controllers and their associated action spaces."""

    def __init__(
        self, action_space: spaces.Space, limits: State, update: Callable[[Action], State]
    ):
        """Creates a Dynamics controller.

        Args:
            action_space: The gymnasium action space these dynamics respond to.
            limits: The bounds of the state's elements.
            update: Returns the change in state that results from some action."""

        self._space = action_space
        self._limits = limits
        self._update = update

    def __call__(self, state: State, action: Action) -> State:
        """Returns the new state that results from enacting action in state.

        Args:
            state: The old state on which to act.
            action: The action to take in that state.

        Returns:
            The new state that results from enacting action in state."""

        return numpy.clip(state + self._update(action), *self._limits, dtype=StateElement)

    def action_space(self) -> spaces.Space:
        """Returns the action space these dynamics respond to.

        Returns:
            The action space these dynamics respond to."""

        return self._space


def make_continuous_update(speed: float) -> Callable[[numpy.float32], State]:
    """Creates a function that returns how the state changes given some floating point
    action.

    Args:
        speed: The fastest speed possible. How far a 1 or -1 will travel.

    Returns:
        A function that returns how the state changes given some floating point action."""

    mask = numpy.array([0.0, 1.0], dtype=StateElement)

    return lambda action: numpy.multiply(numpy.clip(action, -1.0, 1.0), speed) * mask


def make_continuous_dynamics(
    limits: tuple[float, float], speed: float
) -> Dynamics[numpy.float32]:
    """Creates a dynamics system that moves between the limits at speed. Actions of 1 and
    -1 move at the given speed towards the max and min limit, respectively. Actions of 0
    keep the state in the same place.

    Args:
        limits: A tuple containing the lower and upper bound of the state's elements.
        speed: The fastest speed possible. How far a 1 or -1 will travel.

    Returns: A dynamics system that moves between the limits at speed."""

    return Dynamics(
        spaces.Box(-1.0, 1.0, (1,), dtype=numpy.float32),
        numpy.array(limits),
        make_continuous_update(speed),
    )


def make_discrete_update(actions: list[float]) -> Callable[[numpy.int32], State]:
    """Creates a function that returns how the state changes given some action picked from
    actions.

    Args:
        actions: The set of actions that can be taken.

    Returns:
        A function that returns how the state changes given some action picked from
        actions."""

    mask = numpy.array([0.0, 1.0], dtype=StateElement)

    return lambda action: numpy.float32(actions[action]) * mask


def make_discrete_dynamics(
    limits: tuple[float, float], actions: list[float]
) -> Dynamics[numpy.int32]:
    """Creates a dynamics system that moves between the limits with a number of fixed
    steps. Actions are indices to actions, which are the distances in the state space each
    action moves. For example, an action of 2 moves a distance of actions[2].

    Args:
        limits: A tuple containing the lower and upper bound of the state's elements.
        actions: The list of actions these dynamics should model.

    Returns:
        A dynamics system that moves between the limits with a number of fixed steps."""

    return Dynamics(
        spaces.Discrete(len(actions)),
        numpy.array(limits),
        make_discrete_update(actions),
    )