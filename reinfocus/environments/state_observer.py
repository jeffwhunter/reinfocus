"""Functions that normalize environment observations."""

import abc
import functools

from collections.abc import Sequence
from typing import Generic, Protocol

import numpy

from gymnasium import spaces
from numpy.typing import NDArray

from reinfocus import vision

from reinfocus.environments.types import (
    ObservationT,
    ObservationT_co,
    StateT,
    StateT_contra,
)
from reinfocus.graphics import render
from reinfocus.graphics import world


class IStateObserver(Protocol, Generic[ObservationT_co, StateT_contra]):
    # pylint: disable=too-few-public-methods, unnecessary-ellipsis
    """The base that state observers must follow."""

    observation_space: spaces.Box
    single_observation_space: spaces.Box

    def observe(self, state: StateT_contra) -> NDArray[ObservationT_co]:
        """Returns a batch of observations of state.

        Args:
            state: The state of some batch of POMDPs.

        Returns:
            A batch of observations, one per state."""

        ...


class ScalarObserver(IStateObserver, abc.ABC, Generic[ObservationT, StateT]):
    # pylint: disable=too-few-public-methods, unnecessary-ellipsis
    """A state observer that produces scalar observations within some range."""

    def __init__(self, num_envs: int, min_obs: float, max_obs: float):
        self.single_observation_space = spaces.Box(min_obs, max_obs, dtype=numpy.float32)
        self.observation_space = spaces.Box(
            min_obs, max_obs, (num_envs,), dtype=numpy.float32
        )
        """Creates a ScalarObserver.

        Args:
            num_envs: The number of environments this observer will observe.
            min_obs: The minimum observation possible.
            max_obs: The maximum observation possible."""

    @abc.abstractmethod
    def observe(self, state: StateT) -> NDArray[ObservationT]:
        """Returns a batch of observations of state.

        Args:
            state: The state of some batch of POMDPs.

        Returns:
            A batch of scalar observations, one per state."""

        ...


class NormalizedObserver(IStateObserver, Generic[ObservationT, StateT]):
    # pylint: disable=too-few-public-methods
    """A state observer that appends then normalizes the output of some scalar observers
    to [-1, 1]."""

    def __init__(self, observers: Sequence[ScalarObserver]):
        """Creates a NormalizedObserver.

        Args:
            observers: A sequence of scalar observers whose output will form the output of
                this observer, appened in the same order they're passed in here.
        """

        num_envs = set(observer.observation_space.shape[0] for observer in observers)
        assert (
            len(num_envs) == 1
        ), "Appended observers must have the same number of environments"

        self._observers = observers
        n_observers = len(observers)

        observer_spans = numpy.array(
            [
                [
                    observer.single_observation_space.low,
                    observer.single_observation_space.high,
                ]
                for observer in self._observers
            ]
        ).reshape((n_observers, 2))

        self._mid = numpy.average(observer_spans, axis=1)
        self._scale = numpy.diff(observer_spans / 2).reshape(n_observers)

        self.single_observation_space = spaces.Box(
            -1, 1, (n_observers,), dtype=numpy.float32
        )
        self.observation_space = spaces.Box(
            -1, 1, (num_envs.pop(), n_observers), dtype=numpy.float32
        )

    def observe(self, state: NDArray[numpy.float32]) -> NDArray[numpy.float32]:
        """Produces a batch of appended and normalized observations from a number of child
        observers.

        Args:
            state: The state of some batch of POMDPs.

        Returns:
            A batch of observations normalized to [-1, 1], one per state, where each
            pre-normalized observation is formed from appending the out of this observer's
            children."""

        raw_observation = numpy.array(
            [observer.observe(state) for observer in self._observers]
        ).T
        return numpy.clip(
            (raw_observation - self._mid) / self._scale, -1, 1, dtype=numpy.float32
        )


class IndexedElementObserver(ScalarObserver):
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

    def observe(self, state: NDArray[numpy.float32]) -> NDArray[numpy.float32]:
        """Produces a batch of observations copied directly from elements of the state.

        Args:
            state: The state of some batch of POMDPs.

        Returns:
            A batch of observations copied directly from some element of the state, one
            per state."""
        return state[:, self._element_index]


@functools.cache
def cached_focus_extrema(ends: tuple[float, float], frame_shape: tuple[int, int]):
    """A cached function that finds the maximum and minimum focus over some range of
    positions for the target and focus plane. Will only run once per set of inputs,
    subsequent runs will used the cached result. Checks for minimum focus when the target
    and focus place are at opposide sides of the range. Checks for maximum focus when the
    target and focus plane are in the same position, at various places within the range.

    Args:
        ends: A tuple containing the minimum and maximum (respectively) positions that
            the target and focus plane can exist between.
        frame_shape: The resolution at which the focus extrema should be found.

    Returns:
        A tuple containing the minimum and maximum (respectively) focus values that are
        possible for the given range of possible positions."""

    max_targets = numpy.linspace(*ends, 11)

    worlds = world.FocusWorlds(13)
    worlds.update_targets(numpy.append(ends, max_targets))

    focus_values = vision.focus_values(
        render.fast_render(
            worlds,
            numpy.append(tuple(reversed(ends)), max_targets),
            frame_shape=frame_shape,
        )
    )

    return min(focus_values[0:2]), max(focus_values[2:13])


class FocusObserver(ScalarObserver):
    # pylint: disable=too-few-public-methods
    """A scalar state observer that calculates the focus value of a simple rendered scene,
    where the state has the location of a target and focus plane within that scene."""

    _frame_shape = (300, 300)

    def __init__(
        self,
        num_envs: int,
        target_index: int,
        focus_plane_index: int,
        ends: tuple[float, float],
        worlds: world.FocusWorlds,
    ):
        # pylint: disable=too-many-arguments
        """Creates a FocusObserver.

        Args:
            num_envs: The number of environments this observer will observe.
            target_index: The index of the location of the target in the state.
            focus_plane_index: The index of the location of the focus plane in the state.
            ends: The minimum and maximum possible positions for the target and focus
                plane.
            worlds: These worlds will be rendered, and the focus value of the resulting
                images will be returned as observations.
        """

        min_focus, max_focus = cached_focus_extrema(ends, FocusObserver._frame_shape)

        super().__init__(num_envs, min_focus, max_focus)

        self._target_index = target_index
        self._focus_plane_index = focus_plane_index
        self._worlds = worlds

    def observe(self, state: NDArray[numpy.float32]) -> NDArray[numpy.float32]:
        """Produces a batch of observations calulated from the focus values of simple
        rendered scenes defined by the state.

        Args:
            state: The state of some batch of POMDPs where the state has the location of a
                target and focus plane.

        Returns:
            A batch of observations calculated from the focus values of scenes defined by
            the state."""

        self._worlds.update_targets(state[:, self._target_index])

        return numpy.asarray(
            vision.focus_values(
                render.fast_render(
                    self._worlds,
                    state[:, self._focus_plane_index],
                    FocusObserver._frame_shape,
                )
            )
        )
