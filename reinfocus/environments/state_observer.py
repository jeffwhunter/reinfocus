"""Functions that normalize environment observations."""

import abc
import functools

from collections.abc import Sequence
from typing import Generic, Protocol, SupportsFloat

import numpy

from gymnasium import spaces
from gymnasium.vector import utils
from numpy.typing import NDArray

from reinfocus import vision

from reinfocus.environments.types import (
    ObservationT_co,
    StateT_contra,
)
from reinfocus.graphics import render
from reinfocus.graphics import world


class IStateObserver(Protocol, Generic[ObservationT_co, StateT_contra]):
    # pylint: disable=too-few-public-methods, unnecessary-ellipsis
    """The base that state observers must follow."""

    observation_space: spaces.Box
    single_observation_space: spaces.Box

    def observe(self, states: StateT_contra) -> NDArray[ObservationT_co]:
        """Returns a batch of observations of state.

        Args:
            states: The state of some batch of POMDPs.

        Returns:
            A batch of observations, one per state."""

        ...

    def reset(
        self, states: StateT_contra, dones: NDArray[numpy.bool_] | None = None
    ) -> NDArray[ObservationT_co]:
        """Informs the state observer that some episodes have restarted.

        Args:
            states: The first states of the new episode that reset marks the start of.
            dones: None, or a numpy array of one boolean per environment, where each
                element is True if that environment has just been reset. If None, all
                environments are considered reset.

        Returns:
            A batch of observations from only the reset states."""

        ...


class BaseObserver(IStateObserver, abc.ABC):
    # pylint: disable=too-few-public-methods, unnecessary-ellipsis
    """A state observer that produces observations within some range."""

    def __init__(
        self,
        num_envs: int,
        min_obs: SupportsFloat | NDArray[numpy.float32],
        max_obs: SupportsFloat | NDArray[numpy.float32],
    ):
        """Creates a BaseObserver.

        Args:
            num_envs: The number of environments this observer will observe.
            min_obs: The minimum observation possible.
            max_obs: The maximum observation possible."""

        self.single_observation_space = spaces.Box(min_obs, max_obs, dtype=numpy.float32)
        self.observation_space = utils.batch_space(
            self.single_observation_space, num_envs
        )  # type: ignore

    @abc.abstractmethod
    def observe(self, states: NDArray[numpy.float32]) -> NDArray[numpy.float32]:
        """Returns a batch of observations of state.

        Args:
            states: The state of some batch of POMDPs.

        Returns:
            A batch of observations, one per state."""

        ...

    def reset(
        self, states: NDArray[numpy.float32], dones: NDArray[numpy.bool_] | None = None
    ) -> NDArray[numpy.float32]:
        """Informs the base episode ender that some episodes have restarted.

        Args:
            states: The first states of the new episode that reset marks the start of.
            dones: None, or a numpy array of one boolean per environment, where each
                element is True if that environment has just been reset. If None, all
                environments are considered reset.

        Returns:
            A batch of observations from only the reset states."""

        if dones is None:
            dones = numpy.full(self.observation_space.shape[0], True)

        return self.observe(states)[dones]


class WrapperObserver(BaseObserver):
    """A state observer that produces observations from other observers."""

    def __init__(
        self,
        observers: Sequence[BaseObserver],
        min_obs: SupportsFloat | NDArray[numpy.float32],
        max_obs: SupportsFloat | NDArray[numpy.float32],
    ):
        """Creates a WrapperObserver.

        Args:
            observers: The other observers from which the observations of this observer
                will be produced.
            min_obs: The minimum observation possible.
            max_obs: The maximum observation possible."""

        observers_num_envs = set(
            observer.observation_space.shape[0] for observer in observers
        )
        assert (
            len(observers_num_envs) == 1
        ), "Appended observers must have the same number of environments"

        super().__init__(observers_num_envs.pop(), min_obs, max_obs)

        self._observers = observers

    def reset(
        self, states: NDArray[numpy.float32], dones: NDArray[numpy.bool_] | None = None
    ) -> NDArray[numpy.float32]:
        """Informs the episode ender that some episodes have restarted. Also resets all
        the wrapped observers.

        Args:
            states: The first states of the new episode that reset marks the start of.
            dones: None, or a numpy array of one boolean per environment, where each
                element is True if that environment has just been reset. If None, all
                environments are considered reset.

        Returns:
            A batch of observations from only the reset states."""

        return numpy.hstack(
            [observer.reset(states, dones) for observer in self._observers],
            dtype=numpy.float32,
        )

    def wrapped_observations(
        self, states: NDArray[numpy.float32]
    ) -> NDArray[numpy.float32]:
        """Returns the stacked observations of state from all the wrapped observers.

        Args:
            states: The state of some batch of POMDPs.

        Returns:
            Observations from the wrapped observers, stacked along the second dimension,
            in the order the wrapped observers were passed into the constructor."""

        return numpy.hstack(
            [observer.observe(states) for observer in self._observers],
            dtype=numpy.float32,
        )


class DeltaObserver(WrapperObserver):
    # pylint: disable=too-few-public-methods
    """An observer that observers changes in the observations of other observers."""

    def __init__(
        self,
        observers: BaseObserver | Sequence[BaseObserver],
        include_original: bool = False,
        max_change: SupportsFloat | NDArray[numpy.float32] | None = None,
    ):
        """Creates a DeltaObserver.

        Args:
            observers: The wrapped observers, whose observations the DeltaObserver will
                observe changes in.
            include_original: Whether or not to include the wrapped observers'
                observations in the observation alongside the change.
            max_change: The maximum amount of change possible. If None, will be
                calculated from the wrapped observation space. If multidimensional,
                numpy.nan entries will be calculated from the wrapped observation
                space."""

        if not isinstance(observers, Sequence):
            observers = [observers]

        n_observations = sum(
            observer.single_observation_space.shape[0] for observer in observers
        )

        wrapped_highs = numpy.hstack(
            [observer.single_observation_space.high for observer in observers],
            dtype=numpy.float32,
        )
        wrapped_lows = numpy.hstack(
            [observer.single_observation_space.low for observer in observers],
            dtype=numpy.float32,
        )

        if max_change is None:
            diff = wrapped_highs - wrapped_lows
        elif isinstance(max_change, numpy.ndarray):
            not_nan = numpy.isfinite(max_change)

            diff = wrapped_highs - wrapped_lows
            diff[not_nan] = max_change[not_nan]
        else:
            diff = numpy.full(n_observations, max_change, dtype=numpy.float32)

        if include_original:
            low = numpy.append(wrapped_lows, -diff)
            high = numpy.append(wrapped_highs, diff)
        else:
            low = -diff
            high = diff

        super().__init__(observers, low, high)

        self._include_original = include_original

        self._old_wrapped_observations = numpy.full(
            (self.observation_space.shape[0], n_observations),
            numpy.nan,
            dtype=numpy.float32,
        )

    def observe(self, states: NDArray[numpy.float32]) -> NDArray[numpy.float32]:
        """Produces a batch of observations which are the changes in, and optionally
        include, the observations from the wrapped observers.

        Args:
            states: The state of some batch of POMDPs.

        Returns:
            A batch of observations copied directly from some element of the state, one
            per state."""

        wrapped_observations = self.wrapped_observations(states)

        observations = wrapped_observations - self._old_wrapped_observations

        if self._include_original:
            observations = numpy.hstack(
                [wrapped_observations, observations],
                dtype=numpy.float32,
            )

        self._old_wrapped_observations = wrapped_observations

        return observations

    def reset(
        self, states: NDArray[numpy.float32], dones: NDArray[numpy.bool_] | None = None
    ) -> NDArray[numpy.float32]:
        """Informs the episode ender that some episodes have restarted. Also resets all
        the wrapped observers.

        Args:
            states: The first states of the new episode that reset marks the start of.
            dones: None, or a numpy array of one boolean per environment, where each
                element is True if that environment has just been reset. If None, all
                environments are considered reset.

        Returns:
            A batch of observations from only the reset states."""

        if dones is None:
            dones = numpy.full(self.observation_space.shape[0], True)

        wrapped_observations = super().reset(states, dones)

        observations = numpy.zeros(wrapped_observations.shape, dtype=numpy.float32)

        if self._include_original:
            observations = numpy.hstack(
                [wrapped_observations, observations], dtype=numpy.float32
            )

        self._old_wrapped_observations[dones] = wrapped_observations

        return observations


@functools.cache
def cached_focus_extrema(ends: tuple[float, float], frame_height: int):
    """A cached function that finds the maximum and minimum focus over some range of
    positions for the target and focus plane. Will only run once per set of inputs,
    subsequent runs will used the cached result. Checks for minimum focus when the target
    and focus place are at opposide sides of the range. Checks for maximum focus when the
    target and focus plane are in the same position, at various places within the range.

    Args:
        ends: A tuple containing the minimum and maximum (respectively) positions that
            the target and focus plane can exist between.
        frame_height: The height of the images in which the focus extrema should be found.

    Returns:
        A tuple containing the minimum and maximum (respectively) focus values that are
        possible for the given range of possible positions."""

    max_targets = numpy.linspace(*ends, 11)

    renderer = render.FastRenderer()
    renderer.update_targets(numpy.append(ends, max_targets))
    renderer.update_focus_planes(numpy.append(ends[::-1], max_targets))

    focus_values = vision.focus_values(renderer.render(frame_height))

    return min(focus_values[0:2]), max(focus_values[2:13])


class FocusObserver(BaseObserver):
    # pylint: disable=too-few-public-methods
    """A state observer that calculates the focus value of a simple rendered scene, where
    the state has the location of a target and focus plane within that scene."""

    def __init__(
        self,
        num_envs: int,
        target_index: int,
        focus_plane_index: int,
        ends: tuple[float, float],
        renderer: render.FastRenderer,
        frame_height: int = 300,
    ):
        # pylint: disable=too-many-arguments
        """Creates a FocusObserver.

        Args:
            num_envs: The number of environments this observer will observe.
            target_index: The index of the location of the target in the state.
            focus_plane_index: The index of the location of the focus plane in the state.
            ends: The minimum and maximum possible positions for the target and focus
                plane.
            renderer: The render used to produce images of the scene.
            frame_height: How high in pixels the rendered images should be.
        """

        min_focus, max_focus = cached_focus_extrema(ends, frame_height)

        super().__init__(num_envs, min_focus, max_focus)

        self._target_index = target_index
        self._focus_plane_index = focus_plane_index
        self._renderer = renderer
        self._frame_height = frame_height

    def observe(self, states: NDArray[numpy.float32]) -> NDArray[numpy.float32]:
        """Produces a batch of observations calulated from the focus values of simple
        rendered scenes defined by the state.

        Args:
            states: The state of some batch of POMDPs where the state has the location of a
                target and focus plane.

        Returns:
            A batch of observations calculated from the focus values of scenes defined by
            the state."""

        self._renderer.update_targets(states[:, self._target_index])
        self._renderer.update_focus_planes(states[:, self._focus_plane_index])

        return numpy.reshape(
            vision.focus_values(self._renderer.render(self._frame_height)),
            self.observation_space.shape,
        )


class IndexedElementObserver(BaseObserver):
    # pylint: disable=too-few-public-methods
    """A scalar state observer that simply returns some element of the state directly as
    it's observation."""

    def __init__(self, num_envs: int, element_index: int, min_obs: float, max_obs: float):
        """Creates an IndexedElementObserver.

        Args:
            num_envs: The number of environments this observer will observe.
            element_index: The index of the state element to return as an observation.
            min_obs: The minimum observation possible.
            max_obs: The maximum observation possible."""

        super().__init__(num_envs, min_obs, max_obs)
        self._element_index = element_index

    def observe(self, states: NDArray[numpy.float32]) -> NDArray[numpy.float32]:
        """Produces a batch of observations copied directly from elements of the state.

        Args:
            states: The state of some batch of POMDPs.

        Returns:
            A batch of observations copied directly from some element of the state, one
            per state."""

        return states[:, self._element_index].reshape(self.observation_space.shape)


class NormalizedObserver(WrapperObserver):
    # pylint: disable=too-few-public-methods
    """A state observer that appends then normalizes the output of some scalar observers
    to [-1, 1]."""

    def __init__(self, observers: BaseObserver | Sequence[BaseObserver]):
        """Creates a NormalizedObserver.

        Args:
            observers: A sequence of scalar observers whose output will form the output of
                this observer, appened in the same order they're passed in here.
        """

        if not isinstance(observers, Sequence):
            observers = [observers]

        n_observations = sum(
            observer.single_observation_space.shape[0] for observer in observers
        )

        super().__init__(
            observers,
            numpy.ones(n_observations, dtype=numpy.float32) * -1,
            numpy.ones(n_observations, dtype=numpy.float32),
        )

        observer_spans = numpy.vstack(
            [
                numpy.hstack(
                    [observer.single_observation_space.low for observer in observers],
                    dtype=numpy.float32,
                ),
                numpy.hstack(
                    [observer.single_observation_space.high for observer in observers],
                    dtype=numpy.float32,
                ),
            ],
            dtype=numpy.float32,
        )

        self._mid = numpy.average(observer_spans, axis=0)

        div_spans = observer_spans / 2

        diff_div = numpy.diff(div_spans, axis=0)

        self._scale = diff_div.reshape(n_observations)

    def observe(self, states: NDArray[numpy.float32]) -> NDArray[numpy.float32]:
        """Produces a batch of appended and normalized observations from a number of child
        observers.

        Args:
            states: The state of some batch of POMDPs.

        Returns:
            A batch of observations normalized to [-1, 1], one per state, where each
            pre-normalized observation is formed from appending the observations of this
            observer's children."""

        return self._normalize(self.wrapped_observations(states))

    def reset(
        self, states: NDArray[numpy.float32], dones: NDArray[numpy.bool_] | None = None
    ) -> NDArray[numpy.float32]:
        """Informs the episode ender that some episodes have restarted. Also resets all
        the wrapped observers and assembles, then normalizes, their return values.

        Args:
            states: The first states of the new episode that reset marks the start of.
            dones: None, or a numpy array of one boolean per environment, where each
                element is True if that environment has just been reset. If None, all
                environments are considered reset.

        Returns:
            A batch of observations normalized to [-1, 1], one per state, where each
            pre-normalized observation is formed from appending the reset observations of
            this observer's children."""

        return self._normalize(super().reset(states, dones))

    def _normalize(self, values: NDArray[numpy.float32]) -> NDArray[numpy.float32]:
        """Normalizes some values.

        Args:
            values: The values to normalize.

        Returns:
            The values scaled to [-1, 1]."""

        return numpy.clip((values - self._mid) / self._scale, -1, 1, dtype=numpy.float32)
