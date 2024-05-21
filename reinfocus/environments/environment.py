"""Reinforcement learning environments that simulates focusing a camera."""

from typing import Any, Generic, SupportsFloat

import gymnasium
import numpy

from numpy.typing import NDArray

from reinfocus.environments import episode_ender
from reinfocus.environments import episode_rewarder
from reinfocus.environments import episode_visualizer
from reinfocus.environments import state_observer
from reinfocus.environments import state_initializer
from reinfocus.environments import state_transformer
from reinfocus.environments.types import ActionT, ObservationT, StateT


class Environment(gymnasium.Env, Generic[ActionT, ObservationT, StateT]):
    # pylint: disable=too-many-instance-attributes
    """A generic environment that can be flexibly specialized."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 4}

    def __init__(
        self,
        ender: episode_ender.IEpisodeEnder[StateT],
        initializer: state_initializer.IStateInitializer[StateT],
        observer: state_observer.IStateObserver[ObservationT, StateT],
        rewarder: episode_rewarder.IEpisodeRewarder[ObservationT, StateT],
        transformer: state_transformer.IStateTransformer[ActionT, StateT],
        visualizer: episode_visualizer.IEpisodeVisualizer[ObservationT, StateT],
        render_mode: str | None = None,
    ):
        # pylint: disable=too-many-arguments
        """Creates an Environment.

        Args:
            ender: The episode ender that will control when the episodes of this
                environment are truncated or terminated.
            initializer: The initializer that will initialize new episodes' states.
            observer: The observer that will produce this environment's observations.
            rewarder: The rewarder that will produce this environment's rewards.
            transformer: The transformer that will transform states from one step to the
                next according to some actions.
            visualizer: The visualizer that will visualize the performace of agents in
                this environment.
            render_mode: The render mode for the environment, either "rgb_array" or
                None."""

        self._ender = ender
        self._initializer = initializer
        self._observer = observer
        self._rewarder = rewarder
        self._transformer = transformer
        self._visualizer = visualizer

        self.observation_space = observer.single_observation_space
        self.action_space = transformer.single_action_space

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self._state = None

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[ObservationT, dict[str, Any]]:
        """Resets the environment to an initial internal state, returning an initial
        observation.

        Args:
            seed: The seed used to initialize the environment.
            options: Additional information to specify how the environment is reset.

        Returns:
            observation: An observation of the initial state.
            info: An unused information dictionary."""

        super().reset(seed=seed)

        self._state = self._initializer.initialize(1)

        self._ender.reset(self._state)

        observations = self._observer.reset(self._state)

        self._rewarder.reset(self._state, observations)

        if self.render_mode == "rgb_array":
            self._visualizer.reset(self._state, observations)

        return observations[0], {}

    def step(
        self, action: ActionT
    ) -> tuple[ObservationT, SupportsFloat, bool, bool, dict[str, Any]]:
        """Run one timestep of the environment's dynamics using action.

        Args:
            action: The action to which the environment's dynamics will respond.

        Returns:
            observation: An observation of the new state.
            reward: The reward the agent earned from the new state.
            terminated: Whether the agent reaches some terminal state.
            truncated: Whether the episode was ended early, as in a time limit.
            info: An unused information dictionary."""

        assert self._state is not None

        actions = numpy.array([action])
        self._state = self._transformer.transform(self._state, actions)

        self._ender.step(self._state)

        observations = self._observer.observe(self._state)

        if self.render_mode == "rgb_array":
            self._visualizer.step(self._state, observations)

        return (
            observations[0],
            self._rewarder.reward(self._state, observations)[0],
            self._ender.is_terminated()[0],
            self._ender.is_truncated()[0],
            {},
        )

    def render(self) -> None | NDArray:
        """Returns a suitable rendering of the environment given the rendering mode.

        Returns:
            An image of the environment if the rendering mode is "rgb_array", otherwise
            None."""

        if self.render_mode == "rgb_array":
            return self._visualizer.visualize()

        return None
