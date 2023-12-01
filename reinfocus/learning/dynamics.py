'''Objects that compute the results of various types of dynamics on general states.'''

import abc
import typing

import gymnasium as gym
import numpy as np
import numpy.typing as npt

State = npt.NDArray[np.float32]

Action = typing.TypeVar('Action')

class Dynamics(abc.ABC, typing.Generic[Action]):
    '''Base class for the state dynamics controllers which also figure out appropriate
        action spaces.'''

    def __init__(
        self,
        action_space: gym.spaces.Space,
        low: float,
        high: float,
        mask: State):
        '''Creates a Dynamics object with the appropriate action space.

        Args:
            action_space: The gymnasium action space these dynamics respond to.
            low: The lower bound of the state's elements.
            high: The upper bound of the state's elements.
            mask: How much each element of the state should update.'''

        self._space = action_space
        self._low = low
        self._high = high
        self._mask = mask

    def __call__(self, state: State, action: Action) -> State:
        '''Returns the new state that results from enacting action in state.

        Args:
            state: The old state on which to act.
            action: The action to take in that state.

        Returns:
            The new state that results from enacting action in state.'''
        return np.clip(
            state + self._state_update(action) * self._mask,
            self._low,
            self._high,
            dtype=np.float32)

    def action_space(self) -> gym.spaces.Space:
        '''Returns the action space these dynamics respond to.

        Returns:
            The action space these dynamics respond to.'''
        return self._space

    @abc.abstractmethod
    def _state_update(self, action: Action) -> np.float32:
        '''Returns the distance that the state should move for the given action.

        Args:
            action: The action which will move the state.

        Returns:
            The distance that state should move.'''

class ContinuousDynamics(Dynamics[float]):
    # pylint: disable=too-few-public-methods
    '''Dynamics that move the state between limits. Actions of 1 and -1 will send the
        state towards those max and min limits, respectively, at some given speed, while
        actions of 0 will keep the state motionless.'''

    def __init__(
        self,
        low: float,
        high: float,
        speed: float,
        mask: State):
        '''Creates a dynamics system that moves between low and high at speed. Actions of
            1 and -1 move at the given speed towards the max and min limit, respectively.
            Actions of 0 keep the state in the same place.

        Args:
            low: The lower bound of the state's elements.
            high: The upper bound of the state's elements.
            speed: How far a 1 or -1 will travel.
            mask: How much each element of the state should update.'''

        super().__init__(gym.spaces.Box(-1., 1., (1,), dtype=np.float32), low, high, mask)

        self._speed = speed

    def _state_update(self, action: np.float32) -> np.float32:
        '''Returns a state update with distance of speed * action.

        Args:
            action: The action to which the move responds to, between -1 and 1.

        Returns:
            A move distance of speed * action.'''
        return np.clip(action, -1, 1) * self._speed

class DiscreteDynamics(Dynamics[int]):
    # pylint: disable=too-few-public-methods
    '''Dynamics that move the state between limits. Actions are indices of a given set of
        movements.'''

    def __init__(
        self,
        low: float,
        high: float,
        actions: list[float],
        mask: State):
        '''Creates a dynamics system that moves between low and high with a number of
            fixed steps.

        Args:
            low: The lower bound of the state's elements.
            high: The upper bound of the state's elements.
            actions: The various fixed steps.
            mask: How much each element of the state should update.'''

        super().__init__(gym.spaces.Discrete(len(actions)), low, high, mask)

        self._actions = actions

    def _state_update(self, action: np.int32) -> np.float32:
        '''Returns a state update with the distance that has index action.

        Args:
            action: The index of the movement to return.

        Returns:
            The indexed movement.'''
        return np.float32(self._actions[action])
