"""Reinforcement learning environments that simulates focusing a camera."""

import dataclasses

from typing import Any, Generic, SupportsFloat

import gymnasium
import numpy

from numpy.typing import NDArray

from reinfocus.environment import dynamics
from reinfocus.environment import episode_ender
from reinfocus.environment import observation_filter
from reinfocus.environment import observation_producer
from reinfocus.environment import observation_rewarder
from reinfocus.environment import state_initializer
from reinfocus.environment import visualization
from reinfocus.environment.types import ActionT, Observation, OI, SI
from reinfocus.graphics import world


@dataclasses.dataclass
class FocusEnvironmentDependencies(Generic[ActionT]):
    """Dependencies for a FocusEnvironment.

    Args:
        dynamics_function: A function that calculates the next state from the current
            state and action.
        ender: Decided if the episode ends early and how much bonus reward ending is
            worth.
        initializer: A function that initializes the state on reset.
        obs_filter: A function that filters out unobservable observations.
        obs_producer: A function that produces observations from target, lens, and world
            states.
        rewarder: A function that returns rewards.
        visualizer: Produces visualizations of the move history intended for humans."""

    dynamics_function: dynamics.Dynamics[ActionT]
    ender: episode_ender.EpisodeEnder
    initializer: state_initializer.StateInitializer
    obs_filter: observation_filter.ObservationFilter
    obs_producer: observation_producer.FocusObservationProducer
    rewarder: observation_rewarder.Rewarder
    visualizer: visualization.FocusHistoryVisualizer


class FocusEnvironment(gymnasium.Env, Generic[ActionT]):
    """A reinforcement learning environment that simulates focusing a camera by rendering
    simple scenes in a ray tracer."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 4}

    def __init__(
        self,
        dependencies: FocusEnvironmentDependencies[ActionT],
        render_mode: str | None = None,
    ):
        """Constructor for the focus environment.

        Args:
            dependencies: The various dependencies this FocusEnvironment needs to operate.
            render_mode: The render mode for the environment, either "rgb_array" or
                None."""

        self._dependencies = dependencies

        self.observation_space = self._dependencies.obs_filter.observation_space()
        self.action_space = self._dependencies.dynamics_function.action_space()

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

        self._state = self._dependencies.initializer()

        self._dependencies.ender.reset()

        self._world = world.one_rect_world(
            world.ShapeParameters(distance=self._state[SI.TARGET])
        )

        observation = self._get_obs()

        self._dependencies.visualizer.reset()
        self._dependencies.visualizer.add_step(
            (self._state[SI.LENS], observation[OI.FOCUS])
        )

        return self._dependencies.obs_filter(observation), {}

    def step(
        self, action: ActionT
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

        self._state = self._dependencies.dynamics_function(self._state, action)

        observation = self._get_obs()

        self._dependencies.visualizer.add_step(
            (self._state[SI.LENS], observation[OI.FOCUS])
        )

        return (
            self._dependencies.obs_filter(observation),
            self._dependencies.rewarder(self._state, observation),
            self._dependencies.ender.is_early_end(self._state),
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
            return self._dependencies.visualizer.visualize(
                self._world, self._state[SI.TARGET], self._dependencies.ender.status()
            )

        return None

    def close(self):
        """Called after the environment has been finished to clean up any resources."""

    def _get_obs(self) -> Observation:
        """Gets the current observation of the environment.

        Returns:
            The current observation of the environment."""

        assert self._world is not None

        return self._dependencies.obs_producer.produce_observation(
            self._state, self._world
        )
