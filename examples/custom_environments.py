"""Examples of custom implementations of Environment and VectorEnvironment."""

import numpy

from reinfocus.environments import environment
from reinfocus.environments import episode_ender
from reinfocus.environments import episode_rewarder
from reinfocus.environments import episode_visualizer
from reinfocus.environments import state_initializer
from reinfocus.environments import state_observer
from reinfocus.environments import state_transformer
from reinfocus.environments import vector_environment
from reinfocus.graphics import render


class DiscreteSteps(environment.Environment):
    # pylint: disable=too-few-public-methods
    """An environment with the following properties:
    * It's state is [target position, focus plane] initialized in [5, 10].
    * Observations: [focus plane, focus value, focus plane change, focus value change].
    * It has 11 actions that move the focus plane [-.5, -.4, ..., .4, .5] in [5, 10].
    * It rewards each action with the sum of the following rewards:
      * -1 for every .5 the focus plane moves.
      * The focus value.
      * 1 for any action that brings the focus plane within .25 of the target.
    * It ends if the target and focus plane widen more than .5 distance 10 times."""

    def __init__(self, render_mode: str | None = None):
        """Creates a DiscreteSteps.

        Args:
            render_mode: 'rgb_array' or None. If 'rgb_array', a nice visualization will be
                generated each time render is called. If None, None will be returned by
                render."""

        ends = (5.0, 10.0)

        target_radius = 0.25
        max_focus_position_move = 0.5

        # Indices of the elements of the state
        target_position_s_index = 0
        focus_position_s_index = 1

        # Indices of the elements of the observation
        # focus_position_o_index = 0
        focus_value_o_index = 1
        # focus_position_change_o_index = 2
        # focus_value_change_o_index = 3

        renderer = render.FastRenderer()

        ender = episode_ender.DivergingEnder(
            1,
            (target_position_s_index, focus_position_s_index),
            max_focus_position_move * 0.1,
            early_end_steps=3,
        )

        super().__init__(
            ender=ender,
            initializer=state_initializer.RangedInitializer([[ends]] * 2),
            observer=state_observer.NormalizedObserver(
                state_observer.DeltaObserver(
                    [
                        state_observer.IndexedElementObserver(
                            1, focus_position_s_index, *ends
                        ),
                        state_observer.FocusObserver(
                            1,
                            target_position_s_index,
                            focus_position_s_index,
                            ends,
                            renderer,
                        ),
                    ],
                    True,
                    numpy.array([max_focus_position_move, numpy.nan]),
                )
            ),
            rewarder=episode_rewarder.SumRewarder(
                episode_rewarder.DeltaRewarder(focus_position_s_index, target_radius * 2),
                episode_rewarder.ObservationRewarder(focus_value_o_index),
                episode_rewarder.OnTargetRewarder(
                    (target_position_s_index, focus_position_s_index), target_radius
                ),
            ),
            transformer=state_transformer.DiscreteMoveTransformer(
                1,
                focus_position_s_index,
                ends,
                numpy.linspace(-max_focus_position_move, max_focus_position_move, 11),
            ),
            visualizer=episode_visualizer.HistoryVisualizer(
                1,
                target_position_s_index,
                focus_position_s_index,
                focus_value_o_index,
                renderer,
                ends,
                ender=ender,
                target_radius=target_radius,
            ),
            render_mode=render_mode,
        )


class VectorDiscreteSteps(vector_environment.VectorEnvironment):
    # pylint: disable=too-few-public-methods
    """A vectorized environment with the following properties:
    * It's state is [target position, focus plane] initialized in [5, 10].
    * Observations: [focus plane, focus value, focus plane change, focus value change].
    * It has 11 actions that move the focus plane [-.5, -.4, ..., .4, .5] in [5, 10].
    * It rewards each action with the sum of the following rewards:
      * -1 for every .5 the focus plane moves.
      * The focus value.
      * 1 for any action that brings the focus plane within .25 of the target.
    * It ends if the target and focus plane widen more than .5 distance 10 times."""

    def __init__(self, num_envs: int = 1, render_mode: str | None = None):
        """Creates a VectorDiscreteSteps.

        Args:
            num_envs: The number of single environments this environment contains.
            render_mode: 'rgb_array' or None. If 'rgb_array', a nice visualization will be
                generated each time render is called. If None, None will be returned by
                render."""

        ends = (5.0, 10.0)

        target_radius = 0.25
        max_focus_position_move = 0.5

        # Indices of the elements of the state
        target_position_s_index = 0
        focus_position_s_index = 1

        # Indices of the elements of the observation
        # focus_position_o_index = 0
        focus_value_o_index = 1
        # focus_position_change_o_index = 2
        # focus_value_change_o_index = 3

        renderer = render.FastRenderer()

        ender = episode_ender.DivergingEnder(
            num_envs,
            (target_position_s_index, focus_position_s_index),
            max_focus_position_move * 0.1,
            early_end_steps=3,
        )

        super().__init__(
            ender=ender,
            initializer=state_initializer.RangedInitializer([[ends]] * 2),
            observer=state_observer.NormalizedObserver(
                state_observer.DeltaObserver(
                    [
                        state_observer.IndexedElementObserver(
                            num_envs, focus_position_s_index, *ends
                        ),
                        state_observer.FocusObserver(
                            num_envs,
                            target_position_s_index,
                            focus_position_s_index,
                            ends,
                            renderer,
                        ),
                    ],
                    True,
                    numpy.array([max_focus_position_move, numpy.nan]),
                )
            ),
            rewarder=episode_rewarder.SumRewarder(
                episode_rewarder.DeltaRewarder(focus_position_s_index, target_radius * 2),
                episode_rewarder.ObservationRewarder(focus_value_o_index),
                episode_rewarder.OnTargetRewarder(
                    (target_position_s_index, focus_position_s_index), target_radius
                ),
            ),
            transformer=state_transformer.DiscreteMoveTransformer(
                num_envs,
                focus_position_s_index,
                ends,
                numpy.linspace(-max_focus_position_move, max_focus_position_move, 11),
            ),
            visualizer=episode_visualizer.HistoryVisualizer(
                num_envs,
                target_position_s_index,
                focus_position_s_index,
                focus_value_o_index,
                renderer,
                ends,
                ender=ender,
                target_radius=target_radius,
            ),
            num_envs=num_envs,
            render_mode=render_mode,
        )


class ContinuousJumps(environment.Environment):
    # pylint: disable=too-few-public-methods
    """An environment with the following properties:
    * It's state is [target position, focus plane] initialized in [5, 10].
    * Observations: [focus plane, focus value, focus plane change, focus value change].
    * It takes actions in [-1, 1] which jump the focus plane to anywhere in [5, 10].
    * It rewards each action with the sum of the following rewards:
      * -1 for every .5 the focus plane moves.
      * The focus value.
      * 1 for any action that brings the focus plane within .25 of the target.
    * It ends if the target and focus plane widen more than .5 distance 10 times."""

    def __init__(self, render_mode: str | None = None):
        """Creates a ContinuousJumps.

        Args:
            render_mode: 'rgb_array' or None. If 'rgb_array', a nice visualization will be
                generated each time render is called. If None, None will be returned by
                render."""

        ends = (5.0, 10.0)

        target_radius = 0.25
        max_focus_position_move = 0.5

        # Indices of the elements of the state
        target_position_s_index = 0
        focus_position_s_index = 1

        # Indices of the elements of the observation
        # focus_position_o_index = 0
        focus_value_o_index = 1
        # focus_position_change_o_index = 2
        # focus_value_change_o_index = 3

        renderer = render.FastRenderer()
        ender = episode_ender.DivergingEnder(
            1,
            (target_position_s_index, focus_position_s_index),
            max_focus_position_move * 0.1,
            early_end_steps=3,
        )

        super().__init__(
            ender=ender,
            initializer=state_initializer.RangedInitializer([[ends]] * 2),
            observer=state_observer.NormalizedObserver(
                state_observer.DeltaObserver(
                    [
                        state_observer.IndexedElementObserver(
                            1, focus_position_s_index, *ends
                        ),
                        state_observer.FocusObserver(
                            1,
                            target_position_s_index,
                            focus_position_s_index,
                            ends,
                            renderer,
                        ),
                    ],
                    True,
                    numpy.array([max_focus_position_move, numpy.nan]),
                )
            ),
            rewarder=episode_rewarder.SumRewarder(
                episode_rewarder.DeltaRewarder(focus_position_s_index, target_radius * 2),
                episode_rewarder.ObservationRewarder(focus_value_o_index),
                episode_rewarder.OnTargetRewarder(
                    (target_position_s_index, focus_position_s_index), target_radius
                ),
            ),
            transformer=state_transformer.ContinuousJumpTransformer(
                1,
                focus_position_s_index,
                ends,
            ),
            visualizer=episode_visualizer.HistoryVisualizer(
                1,
                target_position_s_index,
                focus_position_s_index,
                focus_value_o_index,
                renderer,
                ends,
                ender=ender,
                target_radius=target_radius,
            ),
            render_mode=render_mode,
        )
