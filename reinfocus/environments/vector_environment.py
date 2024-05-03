"""Reinforcement learning environments that simulate focusing a camera."""

from typing import Any, Generic

import numpy

from gymnasium.experimental import vector
from numpy.typing import NDArray

from reinfocus.environments import episode_ender
from reinfocus.environments import episode_rewarder
from reinfocus.environments import episode_visualizer
from reinfocus.environments import state_observer
from reinfocus.environments import state_initializer
from reinfocus.environments import state_transformer
from reinfocus.environments.types import ActionT, ObservationT, StateT


class VectorEnvironment(vector.VectorEnv, Generic[ActionT, ObservationT, StateT]):
    # pylint: disable=too-many-instance-attributes
    """A generic stack of environments that can be flexibly specialized."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 4}

    def __init__(
        self,
        ender: episode_ender.IEpisodeEnder[StateT],
        initializer: state_initializer.IStateInitializer[StateT],
        observer: state_observer.IStateObserver[ObservationT, StateT],
        rewarder: episode_rewarder.IEpisodeRewarder[ActionT, ObservationT, StateT],
        transformer: state_transformer.IStateTransformer[ActionT, StateT],
        visualizer: episode_visualizer.IEpisodeVisualizer[ObservationT, StateT],
        num_envs: int = 2,
        render_mode: str | None = None,
    ):
        # pylint: disable=too-many-arguments
        """Creates a VectorEnvironment.

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
            num_envs: The number of individual environments this vectorized environment
                will simulate.
            render_mode: The render mode for the environment, either "rgb_array" or
                None."""

        super().__init__()

        self._ender = ender
        self._initializer = initializer
        self._observer = observer
        self._rewarder = rewarder
        self._transformer = transformer
        self._visualizer = visualizer

        self.num_envs = num_envs

        self.action_space = transformer.action_space
        self.observation_space = observer.observation_space
        self.single_action_space = transformer.single_action_space
        self.single_observation_space = observer.single_observation_space

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self._state = None

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[NDArray[ObservationT], dict[str, Any]]:
        """Resets the environment to an initial internal state, returning an initial
        observation.

        Args:
            seed: The seed used to initialize the environment.
            options: Additional information to specify how the environment is reset.

        Returns:
            observation: An observation of the initial state.
            info: An unused information dictionary."""

        super().reset(seed=seed)

        self._state = self._initializer.initialize(self.num_envs)

        self._ender.reset(self._state)

        observations = self._observer.reset(self._state, None)

        if self.render_mode == "rgb_array":
            self._visualizer.reset()
            self._visualizer.step(self._state, observations)

        return observations, {}

    def step(self, actions: NDArray[ActionT]) -> tuple[
        NDArray[ObservationT],
        NDArray[numpy.float32],
        NDArray[numpy.bool_],
        NDArray[numpy.bool_],
        dict[str, Any],
    ]:
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
        self._state = self._transformer.transform(self._state, actions)

        self._ender.step(self._state)

        observations = self._observer.observe(self._state)

        rewards = self._rewarder.reward(actions, self._state, observations)

        terminated = self._ender.is_terminated()
        truncated = self._ender.is_truncated()

        done = terminated | truncated

        if any(done):
            new_state = self._initializer.initialize(done.sum())

            self._state[done] = new_state

            self._ender.reset(new_state, done)

            observations[done] = self._observer.reset(new_state, done)

            if self.render_mode == "rgb_array":
                self._visualizer.reset(done)

        if self.render_mode == "rgb_array":
            self._visualizer.step(self._state, observations)

        return (
            observations,
            rewards,
            terminated,
            truncated,
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
