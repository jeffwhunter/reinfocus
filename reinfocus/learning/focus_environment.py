"""Reinforcement learning environments that simulate focusing a camera."""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, SupportsFloat, Tuple, TypeVar, Union

import gymnasium as gym
import numpy as np
import numpy.typing as npt

from gymnasium import spaces

import reinfocus.graphics.render as ren
import reinfocus.graphics.world as wor

from reinfocus import vision as vis

TARGET = 0
LENS = 1
FOCUS = 2

Observation = TypeVar("Observation")
ObservationNormer = Callable[[Observation], Observation]
Rewarder = Callable[[Observation], float]

def make_observation_normer(
    mid: Observation,
    scale: Observation
) -> ObservationNormer:
    """Makes a function that scales inputs to [-1., 1.].

    Args:
        mid: The midpoint of the range of inputs.
        scale: Half the range of possible inputs.

    Returns:
        A function that scales inputs to [-1., 1.]."""
    return lambda x: (x - mid) / scale

def make_lens_distance_penalty(span: float) -> Rewarder:
    """Makes a function that returns a penalty that increases the further the lens gets
        from the target. The penalty is 0 when the lens is on target, and 1 when the lens
         is as far as possible from the target.

    Args:
        span: The range of possible lens positions.

    Returns:
        A function that returns a penalty when the lens is off target."""
    return lambda o: -abs(o[TARGET] - o[LENS]) / span

def make_lens_on_target_reward(radius: float) -> Rewarder:
    """Makes a function that returns a reward when the lens is within some distance of
        the target. The reward is 1 when the lens is within that distance, and 0
        otherwise.

    Args:
        radius: How close the lens has to get to the target before a reward is returned.

    Returns:
        A function that returns a reward when the lens is within some distance of the
        target."""
    return lambda o: 1 if abs(o[TARGET] - o[LENS]) < radius else 0

def make_focus_reward() -> Rewarder:
    """Makes a function that returns a reward equal to the focus value.

    Returns:
        A function that returns a reward equal to the focus value."""
    return lambda o: o[FOCUS]

def render_and_measure(world: wor.World, focus_distance: float) -> float:
    """Renders then measures the focus value of world when focused on the plane at
        focus_distance.

    Args:
        world: The world to render.
        focus_distance: The distance from the camera of the focus plane.

    Returns:
        A measure of how in focus the given scene is, with higher values implying a
        better focus."""
    return vis.focus_value(
        ren.render(frame_shape=(150, 150), world=world, focus_distance=focus_distance))

def pretty_render(world: wor.World, focus_distance: float) -> npt.NDArray:
    """Renders a high resolution image, intended for human consumption, of world when
        focused on a plane at focus_distance.

    Args:
        world: The world to render.
        focus_distance: The distance from the camera of the focus plane.

    Returns:
        An image of world when focused on a plane focus_distance units away."""
    return ren.render(frame_shape=(600, 600), world=world, focus_distance=focus_distance)

def find_focus_value_min_and_max(
    min_pos: float = 1.0,
    max_pos: float = 10.0,
    measurement_steps: int = 91
) -> Tuple[float, float]:
    """Finds the minimum and maximum possible focus values by scanning through a number
        of scenes, calculating their focus values, and returning their min and max.

    Args:
        min_pos: The minimum lens and target positions.
        max_pos: The maximum lens and target positions.
        measurement_steps: How many steps taken to scan the space of positions.

    Returns:
        min: The minimum focus value.
        max: The maximum focus value."""
    space = np.linspace(min_pos, max_pos, measurement_steps)

    def make_render_and_measure_world(target, focus):
        return render_and_measure(wor.one_rect_world(target), focus_distance=focus)

    focus_values = [make_render_and_measure_world(i, i) for i in space]
    focus_values.append(make_render_and_measure_world(space[0], space[-1]))
    focus_values.append(make_render_and_measure_world(space[-1], space[0]))

    return min(focus_values), max(focus_values)

class RewardType(Enum):
    """An enum that represents the various reward types."""
    PENALTY = 1
    TARGET = 2
    FOCUS = 3

class FocusEnvironment(gym.Env):
    """A reinforcement learning environment that simulates focusing a camera by rendering
        simple scenes in a ray tracer."""
    metadata = {"render_modes": ["rgb_array"]}

    @dataclass
    class HelperFunctions:
        """A simple helper to reduce the amount of instance attributes of FocusEnvironment.

        Args:
            normer: A function that normalizes observations.
            rewarder: A function that returns rewards."""
        normer: ObservationNormer
        rewarder: Rewarder

    def __init__(
        self,
        render_mode: Union[str, None]=None,
        limits: Tuple[float, float]=(1.0, 10.0),
        reward_type: RewardType=RewardType.PENALTY
    ):
        """Constructor for the focus environment.

        Args:
            render_mode: The render mode for the environment, either "rgb_array" or None.
            limits: The low and high limits of the lens and target positions.
            reward_type: Which type of reward this environment should emit."""
        self._limits = limits

        min_focus_value, max_focus_value = find_focus_value_min_and_max(
            self._limits[0],
            self._limits[1],
            91)

        low = np.array([limits[0], limits[0], min_focus_value])
        high = np.array([limits[1], limits[1], max_focus_value])

        if reward_type == RewardType.PENALTY:
            rewarder = make_lens_distance_penalty(2.)
        elif reward_type == RewardType.TARGET:
            rewarder = make_lens_on_target_reward(.1)
        else:
            rewarder = make_focus_reward()

        self._helpers = FocusEnvironment.HelperFunctions(
            make_observation_normer((low + high) / 2, (high - low) / 2),
            rewarder)

        self.action_space = spaces.Box(-1., 1., (1,))
        self.observation_space = spaces.Box(-1., 1., (3,))

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self._state = None
        self._world = None

    def reset(
        self,
        *,
        seed: Union[int, None]=None,
        options: Union[Dict[str, Any], None]=None
    ) -> Tuple[npt.NDArray, Dict[str, Any]]:
        """Resets the environment to an initial internal state, returning an initial observation.

        Args:
            seed: The seed used to initialize the environment.
            options: Additional information to specify how the environment is reset.

        Returns:
            observation: An observation of the initial state.
            info: An unused information dictionary."""
        super().reset(seed=seed)

        self._state = np.random.uniform(self._limits[0], self._limits[1], 2)

        self._world = wor.one_rect_world(self._state[0])

        return self._get_obs(), {}

    def step(
        self,
        action: float
    ) -> Tuple[npt.NDArray, SupportsFloat, bool, bool, Dict[str, Any]]:
        """Run one timestep of the environment's dynamics using action.

        Args:
            action: The action to which the environment's dynamics will respond.

        Returns:
            observation: An observation of the new state.
            reward: The reward the agent earned from the new state.
            terminated: Whether the agent reaches some terminal state.
            truncated: Whether the episode was ended early, as in a time limit.
            info: An unused information dictionary.
            done: (Deprecated) Has the episode ended."""
        assert self._state is not None

        self._state[1] += action * (self._limits[1] - self._limits[0])
        self._state[1] = np.clip(self._state[1], self._limits[0], self._limits[1])

        observation = self._get_obs()

        return (
            observation,
            self._helpers.rewarder(observation),
            False,
            False,
            {})

    def render(self) -> Union[None, npt.NDArray]:
        """Returns a suitable rendering of the environment given the rendering mode.

        Returns:
            An image of the environment if the rendering mode is "rgb_array", otherwise
            None."""
        assert self._state is not None
        assert self._world is not None

        if self.render_mode == "rgb_array":
            return pretty_render(self._world, self._state[1])

        return None

    def close(self):
        """Called after the environment has been finished to clean up any resources."""

    def _get_obs(self) -> npt.NDArray:
        """Gets the current observation of the environment.

        Returns:
            The current observation of the environment."""
        assert self._state is not None
        assert self._world is not None

        return np.clip(
            self._helpers.normer(
                np.array([
                    self._state[0],
                    self._state[1],
                    render_and_measure(self._world, self._state[1])])),
            -1 * np.ones(3),
            np.ones(3))

    def __str__(self) -> str:
        """Returns a string representation of this environment.

        Returns:
            A string representation of this environment."""
        result = 'FocusEnvironment() '
        result += f'{self.observation_space}'
        return result