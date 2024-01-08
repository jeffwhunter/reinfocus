'''Objects that compute the results of various types of dynamics on general states.'''

import abc
import typing

import gymnasium as gym
import numpy as np
import numpy.typing as npt

StateElement = np.float32
State = npt.NDArray[StateElement]

Action = typing.TypeVar('Action', bound=np.number)

class Dynamics(abc.ABC, typing.Generic[Action]):
    '''Generic state dynamics controllers and their associated action spaces.'''

    def __init__(
        self,
        action_space: gym.spaces.Space,
        limits: State,
        update: typing.Callable[[Action], State]
    ):
        '''Creates a Dynamics controller.

        Args:
            action_space: The gymnasium action space these dynamics respond to.
            lower: The lower bound of the state's elements.
            upper: The upper bound of the state's elements.
            update: Returns the next state that results from some action.'''

        self._space = action_space
        self._limits = limits
        self._update = update

    def __call__(self, state: State, action: Action) -> State:
        '''Returns the new state that results from enacting action in state.

        Args:
            state: The old state on which to act.
            action: The action to take in that state.

        Returns:
            The new state that results from enacting action in state.'''

        return np.clip(state + self._update(action), *self._limits, dtype=StateElement)

    def action_space(self) -> gym.spaces.Space:
        '''Returns the action space these dynamics respond to.

        Returns:
            The action space these dynamics respond to.'''
        return self._space

def make_continuous_dynamics(
    limits: typing.Tuple[float, float],
    speed: float
) -> Dynamics[np.float32]:
    '''Creates a dynamics system that moves between the limits at speed. Actions of 1 and
        -1 move at the given speed towards the max and min limit, respectively. Actions
        of 0 keep the state in the same place.

        Args:
            limits: A tuple containing the lower and upper bound of the state's elements.
            speed: How far a 1 or -1 will travel.'''
    return Dynamics(
        gym.spaces.Box(-1., 1., (1,), dtype=np.float32),
        np.array(limits),
        lambda action:
            np.multiply(np.clip(action, -1, 1), np.float32(speed)) *
            np.array([0., 1.], dtype=StateElement))

def make_discrete_dynamics(
    limits: typing.Tuple[float, float],
    actions: list[float]
) -> Dynamics[np.int32]:
    '''Creates a dynamics system that moves between the limits with a number of fixed
        steps. Actions are indices to actions, which are the distances in the state space
        each action moves. For example, an action of 2 moves a distance of actions[2].

        Args:
            limits: A tuple containing the lower and upper bound of the state's elements.
            actions: The list of actions these dynamics should model.'''
    return Dynamics(
        gym.spaces.Discrete(len(actions)),
        np.array(limits),
        lambda action:
            np.float32(actions[action]) *
            np.array([0., 1.], dtype=StateElement))
