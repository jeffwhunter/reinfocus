"""Reinforcement learning environments that simulates focusing a camera."""

import dataclasses
import enum
import typing

import gymnasium as gym
import numpy as np
import numpy.typing as npt

import reinfocus.graphics.render as ren
import reinfocus.graphics.world as wor
import reinfocus.learning.observation_filter as fil
import reinfocus.vision as vis

TARGET = 0
LENS = 1
FOCUS = 2

State = npt.NDArray[np.float32]
Action = typing.TypeVar('Action')
Observation = npt.NDArray[np.float32]

StateDynamics = typing.Callable[[State, Action], State]
StateInitializer = typing.Callable[[], State]
ObservationNormer = typing.Callable[[Observation], Observation]
Rewarder = typing.Callable[[Observation], float]

def make_continuous_dynamics(low: float, high: float, speed: float = 1.0) -> StateDynamics:
    """Makes a function that moves the focus plane between low and high. Actions of 1 and
        -1 will send the plane speed * (high - low) towards the right or left. Actions of
        0 will keep the plane in the same place. Actions in between will blend between
        these three.

    Args:
        low: The lower bound of the focus plane's range of motion.
        high: The upper bound of the focus plane's range of motion.
        speed: The maximum proportion of the distance high - low one move may make.

    Returns:
        A function that returns the new state that results after performing the given
            action."""
    r = high - low
    return lambda state, action: np.clip(
        state + np.clip(action, -1, 1) * r * speed * np.array([0, 1]),
        low,
        high,
        dtype=np.float32)

def make_discrete_dynamics(low: float, high: float, actions: list[float]) -> StateDynamics:
    """Makes a function that moves the focus plane between low and high. Actions must be
        valid indices of actions, if they are, the indexed move will be taken, within the
        limits of low and high.

    Args:
        low: The lower bound of the focus plane's range of motion.
        high: The upper bound of the focus plane's range of motion.
        actions: The various movements the focus plane can take.

    Returns:
        A function that returns the new state that results after performing the given
        action."""
    return lambda state, action: np.clip(
        state + actions[action] * np.array([0, 1]),
        low,
        high,
        dtype=np.float32)

def make_uniform_initializer(low: float, high: float, size: int) -> StateInitializer:
    """Makes a function that samples the initial state from a uniform distribution
        between low and high.

    Args:
        low: The lower bound of the initial state.
        high: The upper bound of the initial state.
        size: The size of the new state vector.

    Returns:
        A function that randomly initializes states of size size between low and high."""
    return lambda: np.random.uniform(low, high, size).astype(np.float32)

def make_ranged_initializer(
    ranges: list[list[tuple[float, float]]]
) -> StateInitializer:
    """Makes a function that samples the initial state from a number of uniform
        distributions listed in ranges.

    Args:
        ranges: A list of lists of ranges from which the state should be initialized. The
            n-th list is the series of ranges for the n-th state element. A state element
            with more than one range will choose between them uniformly. It will then
            draw the state element from a uniform distribution on the selected range.

    Returns:
        A function that randomly initializes states to be uniformly within the listed
        ranges."""

    return lambda: np.array(
        [np.random.uniform(*r[np.random.choice(len(r))]) for r in ranges])

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

def find_focus_value_limits(
    min_pos: float = 1.0,
    max_pos: float = 10.0,
    measurement_steps: int = 91
) -> tuple[float, float]:
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

class InitializerType(enum.Enum):
    """An enum that represents the ways the target's position can be initialized."""
    UNIFORM = 1
    DEVIATED = 2

class ObservableType(enum.Enum):
    """An enum that represents the ways this environment can mask it's observations."""
    FULL = 1
    NO_TARGET = 2
    ONLY_FOCUS = 3

class RewardType(enum.Enum):
    """An enum that represents the various reward types."""
    PENALTY = 1
    TARGET = 2
    FOCUS = 3

class FocusEnvironment(gym.Env, typing.Generic[Action]):
    """A reinforcement learning environment that simulates focusing a camera by rendering
        simple scenes in a ray tracer."""
    metadata = {"render_modes": ["rgb_array"]}

    @dataclasses.dataclass
    class HelperFunctions:
        """A simple helper to reduce the amount of instance attributes of FocusEnvironment.

        Args:
            dynamics: A function that calculates the next state from the current state
                and action.
            initializer: A function that initializes the state on reset.
            normer: A function that normalizes observations.
            filter: A function that filters out unobservable observations.
            rewarder: A function that returns rewards."""
        dynamics: StateDynamics
        initializer: StateInitializer
        filter: fil.ObservationFilter
        normer: ObservationNormer
        rewarder: Rewarder

    @dataclasses.dataclass
    class Modes:
        """The various modes the FocusEnvironment can function in.

        Args:
            initializer_type: Controls state initialization on reset. UNIFORM initializes
                the target anywhere in it's range, while DEVIATED initializes the target
                uniformly in ranges that avoid the middle and ends of the target's range.
            observable_type: Controls which features are observable. FULL shows all
                features, NO_TARGET hides the target, and ONLY_FOCUS hides the target and
                lens positions.
            reward_type: Controls how the reward is calculated. PENALTY returns a
                negative penalty that scales from 0 to -1 depending on how far the focus
                plane is from the target. TARGET returns a 0 unless the focus plane is
                no futher from the target than 5% of the target's range. FOCUS returns
                the focus value from the scene as the reward."""
        initializer_type: InitializerType = InitializerType.UNIFORM
        observable_type: ObservableType = ObservableType.FULL
        reward_type: RewardType = RewardType.PENALTY

    def __init__(
        self,
        render_mode: str | None = None,
        limits: tuple[float, float] = (1.0, 10.0),
        modes: Modes = Modes()
    ):
        """Constructor for the focus environment.

        Args:
            render_mode: The render mode for the environment, either "rgb_array" or None.
            limits: The low and high limits of the lens and target positions.
            reward_type: Which type of reward this environment should emit."""

        min_focus_value, max_focus_value = find_focus_value_limits(
            limits[0],
            limits[1],
            91)

        low = np.array([limits[0], limits[0], min_focus_value], dtype=np.float32)
        high = np.array([limits[1], limits[1], max_focus_value], dtype=np.float32)

        if modes.reward_type == RewardType.PENALTY:
            rewarder = make_lens_distance_penalty(2.)
        elif modes.reward_type == RewardType.TARGET:
            rewarder = make_lens_on_target_reward(.1)
        else:
            rewarder = make_focus_reward()

        self.action_space = gym.spaces.Box(-1., 1., (1,), dtype=np.float32)

        if modes.observable_type == ObservableType.ONLY_FOCUS:
            mask = {0, 1}
        elif modes.observable_type == ObservableType.NO_TARGET:
            mask = {0}
        else:
            mask = set[int]()

        if modes.initializer_type == InitializerType.UNIFORM:
            initializer = make_uniform_initializer(limits[0], limits[1], 2)
        else:
            r = limits[1] - limits[0]
            initializer = make_ranged_initializer([
                [
                    (limits[0] + .1 * r, limits[0] + .25 * r),
                    (limits[0] + .75 * r, limits[0] + .9 * r)],
                [limits]])

        self._helpers = FocusEnvironment.HelperFunctions(
            make_continuous_dynamics(*limits),
            initializer,
            fil.ObservationFilter(-1., 1., 3, mask),
            make_observation_normer((low + high) / 2, (high - low) / 2),
            rewarder)

        self.observation_space = self._helpers.filter.observation_space()

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self._state = np.zeros(2, dtype=np.float32)
        self._world = None

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, typing.Any] | None = None
    ) -> tuple[Observation, dict[str, typing.Any]]:
        """Resets the environment to an initial internal state, returning an initial observation.

        Args:
            seed: The seed used to initialize the environment.
            options: Additional information to specify how the environment is reset.

        Returns:
            observation: An observation of the initial state.
            info: An unused information dictionary."""
        super().reset(seed=seed)

        self._state = self._helpers.initializer()

        self._world = wor.one_rect_world(self._state[0])

        return self._helpers.filter(self._get_obs()), {}

    def step(
        self,
        action: float
    ) -> tuple[Observation, typing.SupportsFloat, bool, bool, dict[str, typing.Any]]:
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

        self._state = self._helpers.dynamics(self._state, action)

        observation = self._get_obs()

        return (
            self._helpers.filter(observation),
            self._helpers.rewarder(observation),
            False,
            False,
            {})

    def render(self) -> None | npt.NDArray:
        """Returns a suitable rendering of the environment given the rendering mode.

        Returns:
            An image of the environment if the rendering mode is "rgb_array", otherwise
            None."""
        assert self._world is not None

        if self.render_mode == "rgb_array":
            return pretty_render(self._world, self._state[1])

        return None

    def close(self):
        """Called after the environment has been finished to clean up any resources."""

    def _get_obs(self) -> Observation:
        """Gets the current observation of the environment.

        Returns:
            The current observation of the environment."""
        assert self._world is not None

        return np.clip(
            self._helpers.normer(
                np.array(
                    [
                        self._state[0],
                        self._state[1],
                        render_and_measure(self._world, self._state[1])],
                    np.float32)),
            -1 * np.ones(3, dtype=np.float32),
            np.ones(3, dtype=np.float32),
            dtype=np.float32)
