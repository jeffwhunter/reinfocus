"""Examples of custom implementations of Environment and VectorEnvironment."""

import numpy

from reinfocus.graphics import world

from reinfocus.environments import environment
from reinfocus.environments import episode_ender
from reinfocus.environments import episode_rewarder
from reinfocus.environments import state_initializer
from reinfocus.environments import state_observer
from reinfocus.environments import state_transformer

from reinfocus.environments import vector_environment
from reinfocus.environments import visualization


class DiscreteSteps(environment.Environment):
    # pylint: disable=too-few-public-methods
    """An environment with the following properties:
    * It's state is [target position, focus position] initialized in [5, 10].
    * It's observations are [focus position, focus value].
    * It has 11 actions that move the focus position [-.5, -.4, ..., .4, .5] in [5, 10].
    * It rewards each action with the focus value from the new observation."""

    def __init__(self, render_mode: str | None = None):
        """Creates a DiscreteSteps.

        Args:
            render_mode: 'rgb_array' or None. If 'rgb_array', a nice visualization will be
                generated each time render is called. If None, None will be returned by
                render."""

        ends = (5.0, 10.0)

        # Indices of the elements of the state
        target_position_s_index = 0
        focus_position_s_index = 1

        # Indices of the elements of the observation
        focus_value_o_index = 1

        worlds = world.FocusWorlds(1)

        super().__init__(
            ender=episode_ender.EndlessEpisodeEnder(1),
            initializer=state_initializer.RangedInitializer([[ends]] * 2),
            observer=state_observer.NormalizedObserver(
                [
                    state_observer.IndexedElementObserver(
                        1, focus_position_s_index, *ends
                    ),
                    state_observer.FocusObserver(
                        1,
                        target_position_s_index,
                        focus_position_s_index,
                        ends,
                        worlds,
                    ),
                ]
            ),
            rewarder=episode_rewarder.ObservationElementRewarder(focus_value_o_index),
            transformer=state_transformer.DiscreteMoveTransformer(
                1, focus_position_s_index, ends, numpy.linspace(-0.5, 0.5, 11)
            ),
            visualizer=visualization.FocusHistoryVisualizer(
                1,
                target_position_s_index,
                focus_position_s_index,
                focus_value_o_index,
                worlds,
                ends,
            ),
            render_mode=render_mode,
        )


class VectorDiscreteSteps(vector_environment.VectorEnvironment):
    # pylint: disable=too-few-public-methods
    """A vectorized environment with the following properties:
    * It's state is [target position, focus position] initialized in [5, 10].
    * It's observations are [focus position, focus value].
    * It has 11 actions that move the focus position [-.5, -.4, ..., .4, .5] in [5, 10].
    * It rewards each action with the focus value from the new observation."""

    def __init__(self, num_envs: int = 1, render_mode: str | None = None):
        """Creates a VectorDiscreteSteps.

        Args:
            num_envs: The number of single environments this environment contains.
            render_mode: 'rgb_array' or None. If 'rgb_array', a nice visualization will be
                generated each time render is called. If None, None will be returned by
                render."""

        ends = (5.0, 10.0)

        # Indices of the elements of the state
        target_position_s_index = 0
        focus_position_s_index = 1

        # Indices of the elements of the observation
        focus_value_o_index = 1

        worlds = world.FocusWorlds(num_envs)

        super().__init__(
            ender=episode_ender.EndlessEpisodeEnder(num_envs),
            initializer=state_initializer.RangedInitializer([[ends]] * 2),
            observer=state_observer.NormalizedObserver(
                [
                    state_observer.IndexedElementObserver(
                        num_envs, focus_position_s_index, *ends
                    ),
                    state_observer.FocusObserver(
                        num_envs,
                        target_position_s_index,
                        focus_position_s_index,
                        ends,
                        worlds,
                    ),
                ]
            ),
            rewarder=episode_rewarder.ObservationElementRewarder(focus_value_o_index),
            transformer=state_transformer.DiscreteMoveTransformer(
                num_envs, focus_position_s_index, ends, numpy.linspace(-0.5, 0.5, 11)
            ),
            visualizer=visualization.FocusHistoryVisualizer(
                num_envs,
                target_position_s_index,
                focus_position_s_index,
                focus_value_o_index,
                worlds,
                ends,
            ),
            num_envs=num_envs,
            render_mode=render_mode,
        )
