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
    * States are initialized in [5, 10] and have this form:
      [target position, focus plane]
    * Observations are normalized to [-1, 1] and have this form:
      [focus plane, focus value, focus plane change, focus value change]
    * 13 actions move the focus plane within [5, 10] by taking steps of these sizes:
      [-5, -2.5, -1.25, ..., -5 / 2 ** 5, 0, 5 / 2 ** 5, ..., 1.25, 2.5, 5]
    * Actions are rewarded with the sum of the following rewards:
      * -1 for every .5 the focus plane moves.
      * The focus value.
      * 1 for any action that brings the focus plane within .25 of the target.
    * Episodes end if the target and focus plane widen more than .125 three times."""

    def __init__(self, render_mode: str | None = None):
        """Creates a DiscreteSteps.

        Args:
            render_mode: 'rgb_array' or None. If 'rgb_array', a nice visualization will be
                generated each time render is called. If None, None will be returned by
                render."""

        ends = (5.0, 10.0)

        target_radius = 0.25
        max_focus_position_move = 5.0

        n_move_sizes = 6
        moves = max_focus_position_move / 2.0 ** numpy.arange(n_move_sizes)

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
            target_radius / 2,
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
            rewarder=episode_rewarder.DeltaRewarder(
                focus_position_s_index, target_radius * 2
            )
            + episode_rewarder.ObservationRewarder(focus_value_o_index)
            + episode_rewarder.OnTargetRewarder(
                (target_position_s_index, focus_position_s_index), target_radius
            ),
            transformer=state_transformer.DiscreteMoveTransformer(
                1,
                focus_position_s_index,
                ends,
                numpy.concatenate([-moves, [0], moves[::-1]]),
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
    r"""A environment exactly similar to DiscreteSteps, but which has been 'vectorized'
    (it runs multiple environments in parallel) to increase efficiency.

    Vectorized environments are currently experiemental: you should be able to apply and
    remove the vectorized wrapped without affecting training, but in practice,
    hyperparameters tuned on a vectorized environment are unsuccessful on non-vectorized
    environments, and vice versa. So, if you're going to use vectorized environments, use
    them for both hyperparameter optimization and training. To enable them, add a
    `vector_entry_point` to the environment's registration as shown in
    `examples\\__init__.py`, then add `vec_env_wrapper` hyperparameters as shown in
    `examples\ppo_lstm_untuned.yml` and `examples\ppo_lstm_tuned.yml`.

    The configuraiton files are already set up to use this environment with ppo_lstm, but
    you can very easily use this with ppo as well. To do so, as it already has a
    registered `vector_entry_point`, all you need to do is add a `vec_env_wrapper` in
    `ppo_untuned.yml` pointing at vector_shim.rewrapper, then run
    `optimize_hyperparameters.py`. Be sure to include the same `vec_env_wrapper` in the
    tuned parameters as well.

    This environment has the following properties:
    * States are initialized in [5, 10] and have this form:
      [target position, focus plane]
    * Observations are normalized to [-1, 1] and have this form:
      [focus plane, focus value, focus plane change, focus value change]
    * 13 actions move the focus plane within [5, 10] by taking steps of these sizes:
      [-5, -2.5, -1.25, ..., -5 / 2 ** 5, 0, 5 / 2 ** 5, ..., 1.25, 2.5, 5]
    * Actions are rewarded with the sum of the following rewards:
      * -1 for every .5 the focus plane moves.
      * The focus value.
      * 1 for any action that brings the focus plane within .25 of the target.
    * Episodes end if the target and focus plane widen more than .125 three times."""

    def __init__(
        self,
        max_episode_steps: int = 20,
        num_envs: int = 1,
        render_mode: str | None = None,
    ):
        """Creates a VectorDiscreteSteps.

        Args:
            max_episode_steps: Episodes will truncate after this many steps. Gymnasium and
                stable baselines don't have a convenient TimeLimit wrapper for vectorized
                environments, so vectorized environments must be told directly how long to
                last.
            num_envs: The number of single environments this environment contains.
            render_mode: 'rgb_array' or None. If 'rgb_array', a nice visualization will be
                generated each time render is called. If None, None will be returned by
                render."""

        ends = (5.0, 10.0)

        target_radius = 0.25
        max_focus_position_move = 5.0

        n_move_sizes = 6
        moves = max_focus_position_move / 2.0 ** numpy.arange(n_move_sizes)

        # Indices of the elements of the state
        target_position_s_index = 0
        focus_position_s_index = 1

        # Indices of the elements of the observation
        # focus_position_o_index = 0
        focus_value_o_index = 1
        # focus_position_change_o_index = 2
        # focus_value_change_o_index = 3

        renderer = render.FastRenderer()

        ender = episode_ender.TimeLimitEnder(
            num_envs, max_episode_steps
        ) | episode_ender.DivergingEnder(
            num_envs,
            (target_position_s_index, focus_position_s_index),
            target_radius / 2,
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
            rewarder=episode_rewarder.DeltaRewarder(
                focus_position_s_index, target_radius * 2
            )
            + episode_rewarder.ObservationRewarder(focus_value_o_index)
            + episode_rewarder.OnTargetRewarder(
                (target_position_s_index, focus_position_s_index), target_radius
            ),
            transformer=state_transformer.DiscreteMoveTransformer(
                num_envs,
                focus_position_s_index,
                ends,
                numpy.concatenate([-moves, [0], moves[::-1]]),
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
    """Intended to show how environments might be customized, this environment is exactly
    similar to DiscreteSteps in every respect except for these:
    * Actions in [-1, 1] move the focus plane to any position in [5, 10] if the move is
        larger than .125, and don't move otherwise.
    * Actions are rewarded with the sum of the following rewards:
      * The focus value.
      * 1 for actions keep the focus plane stopped within .25 of the target.

    It's remaining properties are:
    * States are initialized in [5, 10] and have this form:
      [target position, focus plane]
    * Observations are normalized to [-1, 1] and have this form:
      [focus plane, focus value, focus plane change, focus value change]
    * Episodes end if the target and focus plane widen more than .125 three times."""

    def __init__(self, render_mode: str | None = None):
        """Creates a ContinuousJumps.

        Args:
            render_mode: 'rgb_array' or None. If 'rgb_array', a nice visualization will be
                generated each time render is called. If None, None will be returned by
                render."""

        ends = (5.0, 10.0)

        target_radius = 0.25
        max_focus_position_move = 5.0
        min_focus_position_move = target_radius / 2

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
            min_focus_position_move,
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
            rewarder=episode_rewarder.ObservationRewarder(focus_value_o_index)
            + episode_rewarder.StoppedRewarder(
                focus_position_s_index, min_focus_position_move
            )
            * episode_rewarder.OnTargetRewarder(
                (target_position_s_index, focus_position_s_index), target_radius
            ),
            transformer=state_transformer.ContinuousJumpTransformer(
                1,
                focus_position_s_index,
                ends,
                target_radius / 2.0,
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
