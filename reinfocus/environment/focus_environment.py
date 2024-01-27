"""Reinforcement learning environments that simulates focusing a camera."""

import dataclasses
import enum
import functools

from typing import Any, Callable, Generic, SupportsFloat, TypeVar

import gymnasium
import numpy

from numpy import random
from numpy.typing import NDArray

from reinfocus.graphics import render
from reinfocus.graphics import world
from reinfocus.environment import dynamics
from reinfocus.environment import observation_filter
from reinfocus import vision

TARGET = 0
LENS = 1
FOCUS = 2

State = NDArray[numpy.float32]
Action = TypeVar("Action", bound=numpy.number)
Observation = NDArray[numpy.float32]

StateInitializer = Callable[[], State]
ObservationNormer = Callable[[Observation], Observation]
Rewarder = Callable[[Observation], float]


def make_uniform_initializer(low: float, high: float, size: int) -> StateInitializer:
    """Makes a function that samples the initial state from a uniform distribution
        between low and high.

    Args:
        low: The lower bound of the initial state.
        high: The upper bound of the initial state.
        size: The size of the new state vector.

    Returns:
        A function that randomly initializes states of size size between low and high."""

    return lambda: random.uniform(low, high, size).astype(numpy.float32)


def make_ranged_initializer(ranges: list[list[tuple[float, float]]]) -> StateInitializer:
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

    return lambda: numpy.array(
        [random.uniform(*r[random.choice(len(r))]) for r in ranges]
    )


def make_observation_normer(mid: Observation, scale: Observation) -> ObservationNormer:
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


def render_and_measure(render_world: world.World, focus_distance: float) -> float:
    """Renders then measures the focus value of world when focused on the plane at
        focus_distance.

    Args:
        render_world: The world to render.
        focus_distance: The distance from the camera of the focus plane.

    Returns:
        A measure of how in focus the given scene is, with higher values implying a
        better focus."""

    return vision.focus_value(
        render.render(
            frame_shape=(150, 150), cpu_world=render_world, focus_distance=focus_distance
        )
    )


def pretty_render(render_world: world.World, focus_distance: float) -> NDArray:
    """Renders a high resolution image, intended for human consumption, of world when
        focused on a plane at focus_distance.

    Args:
        render_world: The world to render.
        focus_distance: The distance from the camera of the focus plane.

    Returns:
        An image of world when focused on a plane focus_distance units away."""

    return render.render(
        frame_shape=(600, 600), cpu_world=render_world, focus_distance=focus_distance
    )


@functools.cache
def find_focus_value_limits(
    min_pos: float = 1.0, max_pos: float = 10.0, measurement_steps: int = 91
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

    space = numpy.linspace(min_pos, max_pos, measurement_steps)

    focus_values = [
        render_and_measure(world.one_rect_world(world.ShapeParameters(distance=i)), i)
        for i in space
    ]
    focus_values.append(
        render_and_measure(
            world.one_rect_world(world.ShapeParameters(distance=space[0])), space[-1]
        )
    )
    focus_values.append(
        render_and_measure(
            world.one_rect_world(world.ShapeParameters(distance=space[-1])), space[0]
        )
    )

    return min(focus_values), max(focus_values)


class DynamicsType(enum.Enum):
    """The dynamics types."""

    CONTINUOUS = 1
    DISCRETE = 2


class InitializerType(enum.Enum):
    """How the target's position can be initialized."""

    UNIFORM = 1
    DEVIATED = 2


class ObservableType(enum.Enum):
    """How this environment can mask it's observations."""

    FULL = 1
    NO_TARGET = 2
    ONLY_FOCUS = 3


class RewardType(enum.Enum):
    """The various reward types."""

    PENALTY = 1
    TARGET = 2
    FOCUS = 3


class FocusEnvironment(gymnasium.Env, Generic[Action]):
    """A reinforcement learning environment that simulates focusing a camera by rendering
    simple scenes in a ray tracer."""

    metadata = {"render_modes": ["rgb_array"]}

    @dataclasses.dataclass
    class HelperFunctions:
        """A helper to reduce the amount of instance attributes of FocusEnvironment.

        Args:
            dynamics: A function that calculates the next state from the current state
                and action.
            initializer: A function that initializes the state on reset.
            normer: A function that normalizes observations.
            filter: A function that filters out unobservable observations.
            rewarder: A function that returns rewards."""

        dynamics: dynamics.Dynamics
        initializer: StateInitializer
        filter: observation_filter.ObservationFilter
        normer: ObservationNormer
        rewarder: Rewarder

    @dataclasses.dataclass
    class Modes:
        """The various modes the FocusEnvironment can function in.

        Args:
            dynamics_type: Controls how the state responds to actions. CONTINUOUS accepts
                actions as floats centered around 0. DISCRETE accepts a number of indices
                to various movements.
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

        dynamics_type: DynamicsType = DynamicsType.CONTINUOUS
        initializer_type: InitializerType = InitializerType.UNIFORM
        observable_type: ObservableType = ObservableType.FULL
        reward_type: RewardType = RewardType.PENALTY

    def __init__(
        self,
        render_mode: str | None = None,
        limits: tuple[float, float] = (1.0, 10.0),
        modes: Modes = Modes(),
    ):
        """Constructor for the focus environment.

        Args:
            render_mode: The render mode for the environment, either "rgb_array" or None.
            limits: The low and high limits of the lens and target positions.
            reward_type: Which type of reward this environment should emit."""

        min_focus_value, max_focus_value = find_focus_value_limits(*limits, 91)

        low = numpy.array([limits[0], limits[0], min_focus_value], dtype=numpy.float32)
        high = numpy.array([limits[1], limits[1], max_focus_value], dtype=numpy.float32)

        diff = limits[1] - limits[0]

        if modes.dynamics_type == DynamicsType.CONTINUOUS:
            dynamics_function = dynamics.make_continuous_dynamics(limits, diff * 0.1)
        else:
            step = diff * 0.01
            dynamics_function = dynamics.make_discrete_dynamics(
                limits, [0, step, -step, 5 * step, -5 * step, 10 * step, -10 * step]
            )

        if modes.initializer_type == InitializerType.UNIFORM:
            initializer = make_uniform_initializer(limits[0], limits[1], 2)
        else:
            r = limits[1] - limits[0]
            initializer = make_ranged_initializer(
                [
                    [
                        (limits[0] + 0.1 * r, limits[0] + 0.25 * r),
                        (limits[0] + 0.75 * r, limits[0] + 0.9 * r),
                    ],
                    [limits],
                ]
            )

        if modes.observable_type == ObservableType.ONLY_FOCUS:
            mask = {0, 1}
        elif modes.observable_type == ObservableType.NO_TARGET:
            mask = {0}
        else:
            mask = set[int]()

        if modes.reward_type == RewardType.PENALTY:
            rewarder = make_lens_distance_penalty(2.0)
        elif modes.reward_type == RewardType.TARGET:
            rewarder = make_lens_on_target_reward(0.1)
        else:
            rewarder = make_focus_reward()

        self._helpers = FocusEnvironment.HelperFunctions(
            dynamics_function,
            initializer,
            observation_filter.ObservationFilter(-1.0, 1.0, 3, mask),
            make_observation_normer((low + high) / 2, (high - low) / 2),
            rewarder,
        )

        self.observation_space = self._helpers.filter.observation_space()
        self.action_space = self._helpers.dynamics.action_space()

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self._state = numpy.zeros(2, dtype=numpy.float32)
        self._world = None

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Observation, dict[str, Any]]:
        """Resets the environment to an initial internal state, returning an initial
            observation.

        Args:
            seed: The seed used to initialize the environment.
            options: Additional information to specify how the environment is reset.

        Returns:
            observation: An observation of the initial state.
            info: An unused information dictionary."""

        super().reset(seed=seed)

        self._state = self._helpers.initializer()

        self._world = world.one_rect_world(world.ShapeParameters(distance=self._state[0]))

        return self._helpers.filter(self._get_obs()), {}

    def step(
        self, action: Action
    ) -> tuple[Observation, SupportsFloat, bool, bool, dict[str, Any]]:
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
            {},
        )

    def render(self) -> None | NDArray:
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

        return numpy.clip(
            self._helpers.normer(
                numpy.array(
                    [
                        self._state[0],
                        self._state[1],
                        render_and_measure(self._world, self._state[1]),
                    ],
                    numpy.float32,
                )
            ),
            -1 * numpy.ones(3, dtype=numpy.float32),
            numpy.ones(3, dtype=numpy.float32),
            dtype=numpy.float32,
        )
