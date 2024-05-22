"""An experimental shim that allows stable baselines to use vectorized reinfocus
environments."""

from collections.abc import Iterable, Sequence
from typing import Any, Type

import gymnasium
import numpy

from gymnasium.experimental import vector
from numpy.typing import NDArray
from stable_baselines3.common import monitor
from stable_baselines3.common import vec_env
from stable_baselines3.common.vec_env import base_vec_env
from stable_baselines3.common.vec_env import vec_monitor

from reinfocus.environments import vector_environment


class SB3Wrapper(base_vec_env.VecEnv):
    """This is a wrapper that lets reinfocus vector environments play nicely with
    stable baselines. Reinfocus vector environments follow gymnasium's vectorized api,
    which has some differences to stable baselines'."""

    def __init__(self, env: vector.VectorEnv, render_mode: str | None):
        """Creates an SB3Wrapper.

        Args:
            env: The reinfocus vector environment to wrap.
            render_mode: The render mode this environment will pretend to be."""

        if not isinstance(env, vector_environment.VectorEnvironment):
            raise NotImplementedError

        self._env = env

        super().__init__(
            env.num_envs,
            env.single_observation_space,
            env.single_action_space,
        )

        self.render_mode = render_mode

        self._actions = None

    def step_async(self, actions: numpy.ndarray):
        """Tell all the environments to start taking a step with the given actions. Call
        step_wait() to get the results of the step.

        Args:
            actions: The actions to begin taking."""

        self._actions = actions

    def reset(self) -> base_vec_env.VecEnvObs:
        """Reset all the environments and return an array of observations, or a tuple of
        observation arrays."""

        return self._env.reset()[0]

    def step_wait(self) -> base_vec_env.VecEnvStepReturn:
        """Wait for the step taken with step_async().

        Returns:
            observation: An observation of the new state.
            reward: The reward the agent earned from the new state.
            dones: True if the episode ended, False otherwise.
            info: An unused information dictionary."""

        assert self._actions is not None

        obs, rewards, terminated, truncated, info_dict = self._env.step(self._actions)
        dones = terminated | truncated

        infos = []

        for i in range(self.num_envs):
            infos.append(
                {
                    key: info_dict[key][i]
                    for key in info_dict.keys()
                    if isinstance(info_dict[key], numpy.ndarray)
                }
            )
            if dones[i]:
                infos[i]["terminal_observation"] = obs[i]

        return (obs, rewards, dones, infos)

    def close(self):
        """Clean up the environment's resources."""

        self._env.close()

    def get_attr(
        self, attr_name: str, indices: base_vec_env.VecEnvIndices = None
    ) -> list:
        """Return attribute from vectorized environment.

        Args:
            attr_name: The name of the attribute whose value to return.
            indices: Indices of envs to get attribute from.

        Returns:
            List of values of 'attr_name' in all the indexed environments."""

        if hasattr(self._env, attr_name):
            return [getattr(self._env, attr_name)] * self._get_result_length(indices)

        raise NotImplementedError(f"{attr_name}, {indices}")

    def set_attr(
        self, attr_name: str, value: Any, indices: base_vec_env.VecEnvIndices = None
    ):
        """Set attribute inside vectorized environments.

        Args:
            attr_name: The name of the attribute to assign new value.
            value: Value to assign to attr_name.
            indices: Indices of envs to assign value."""

        raise NotImplementedError(f"{attr_name}, {value}, {indices}")

    def env_method(
        self,
        method_name: str,
        *method_args,
        indices: base_vec_env.VecEnvIndices = None,
        **method_kwargs,
    ) -> list[Any]:
        """Call instance methods of vectorized environments.

        Args:
            method_name: The instance method to call.
            method_args: The arguments to pass to the method.
            indices: The environments to call the method on.
            method_kwargs: The keyword arguments to pass to the method.

        Returns:
            The return values from calling the method on the given environments."""

        raise NotImplementedError(
            f"{method_name}, {method_args}, {indices}, {method_kwargs}"
        )

    def env_is_wrapped(
        self,
        wrapper_class: Type[gymnasium.Wrapper],
        indices: base_vec_env.VecEnvIndices = None,
    ) -> list[bool]:
        """Check if worker environments are wrapped with a given wrapper.

        Args:
            wrapper_class: The wrapper to check for.
            indices: The environments to check.

        Returns:
            Always returns False as this doesn't seem to be needed for reinfocus."""

        return [False] * self._get_result_length(indices)

    def _get_result_length(self, indices: base_vec_env.VecEnvIndices) -> int:
        """Checks how many environments are being indexed.

        Args:
            indices: The indices indexing the environments.

        Returns:
            An integer counting how many environments were indices by the indices."""

        if isinstance(indices, int):
            return 1

        if isinstance(indices, Iterable):
            return len(list(indices))

        return self._env.num_envs

    def get_images(self) -> Sequence[NDArray[numpy.uint8] | None]:
        """Return RGB images from each environment when available.

        Returns:
            An array of images of the wrapped environments."""

        return [self._env.render()]


def rewrapper(naive_vec_env: base_vec_env.VecEnv) -> base_vec_env.VecEnv:
    """A function that pretends to be a vector wrapper so that it can unwrap a stable
    baselines3 vector environment and recreate it using a different constructor.

    Args:
        naive_vec_env: The vector environment that stable baselines3 created.

    Returns:
        A vector environment created with gymnasium's 'custom' vectorization mode."""

    if not isinstance(naive_vec_env, vec_env.DummyVecEnv):
        return naive_vec_env

    wrapper = naive_vec_env.envs[0]

    vector_kwargs = {}

    if wrapper.spec is None:
        return naive_vec_env

    if wrapper.spec.max_episode_steps is not None:
        vector_kwargs["max_episode_steps"] = wrapper.spec.max_episode_steps

    render_mode = "rgb_array" if wrapper.render_mode == "human" else None

    vector_kwargs["render_mode"] = render_mode

    result = SB3Wrapper(
        gymnasium.make_vec(
            wrapper.spec,
            naive_vec_env.num_envs,
            vectorization_mode="custom",
            vector_kwargs=vector_kwargs,
        ),
        render_mode,
    )

    if isinstance(wrapper, monitor.Monitor):
        result = vec_monitor.VecMonitor(result, wrapper.EXT, wrapper.info_keywords)

    return result
