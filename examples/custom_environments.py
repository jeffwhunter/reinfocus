"""Examples of custom implementations of FocusEnvironment."""

import numpy

from gymnasium.envs import registration

from reinfocus.environment import dynamics
from reinfocus.environment import episode_ender
from reinfocus.environment import focus_environment
from reinfocus.environment import observation_filter
from reinfocus.environment import observation_producer
from reinfocus.environment import observation_rewarder
from reinfocus.environment import state_initializer
from reinfocus.environment import visualization


registration.register(
    id="SimpleDiscreteEnviroment-v0",
    entry_point="custom_environments:SimpleDiscreteEnviroment",
    max_episode_steps=200,
)


registration.register(
    id="ContinuousLeftOrRight-v0",
    entry_point="custom_environments:ContinuousLeftOrRight",
    max_episode_steps=200,
)


class SimpleDiscreteEnviroment(focus_environment.FocusEnvironment[int]):
    """An environment with the following properties:
    * It's target can start, and it's lens can start and move, anywhere in [5.0, 10.0].
    * It ends early if the lens is within 0.125 of the target for 10 steps.
    * It has 11 discrete actions evenly spread in [-0.5, 0.5] (-0.5, -0.4, ..., 0.4, 0.5).
    * It penalizes each step based on the distance between the lens and target."""

    def __init__(self, render_mode: str | None = None):
        """Creates a SimpleDiscreteEnviroment.

        Args:
            render_mode: 'rgb_array' or None. If 'rgb_array', a nice visualization will be
                generated each time render is called. If None, None will be returned by
                render.

        Returns:
            A SimpleDiscreteEnviroment."""

        ends = (5.0, 10.0)
        diff = ends[1] - ends[0]

        largest_move = diff * 0.1
        target_radius = largest_move / 4

        super().__init__(
            focus_environment.FocusEnvironmentDependencies(
                dynamics_function=dynamics.discrete(
                    ends, list(numpy.linspace(-largest_move, largest_move, 11))
                ),
                ender=episode_ender.OnTargetEpisodeEnder(target_radius),
                initializer=state_initializer.uniform(ends[0], ends[1], 2),
                obs_filter=observation_filter.ObservationFilter(-1.0, 1.0, 3, {0}),
                obs_producer=observation_producer.from_ends(ends),
                rewarder=observation_rewarder.distance(diff),
                visualizer=visualization.FocusHistoryVisualizer(
                    ends, target_radius=target_radius
                ),
            ),
            render_mode=render_mode,
        )


class ContinuousLeftOrRight(focus_environment.FocusEnvironment[float]):
    """An environment with the following properties:
    * It's target can start anywhere in [5.125, 6.25] or [8.75, 9.875].
    * It's lens can start anywhere in [6.5, 8.5] and move anywhere in [5.0, 10.0]
    * It ends early if the lens is within 0.125 of the target for 10 steps.
    * It takes continuous steps in [-0.5, 0.5].
    * It earns a reward of -1 every step it's not on target."""

    def __init__(self, render_mode: str | None = None):
        """Creates a ContinuousLeftOrRight.

        Args:
            render_mode: 'rgb_array' or None. If 'rgb_array', a nice visualization will be
                generated each time render is called. If None, None will be returned by
                render.

        Returns:
            A ContinuousLeftOrRight."""

        ends = (5.0, 10.0)
        diff = ends[1] - ends[0]
        quarter = diff / 4

        largest_move = diff * 0.1
        target_radius = largest_move / 4

        super().__init__(
            focus_environment.FocusEnvironmentDependencies(
                dynamics_function=dynamics.continuous(ends, largest_move),
                ender=episode_ender.OnTargetEpisodeEnder(0.5),
                initializer=state_initializer.ranged(
                    [
                        [
                            (ends[0] + target_radius, ends[0] + quarter),
                            (ends[1] - quarter, ends[1] - target_radius),
                        ],
                        [(ends[0] + 3 * largest_move, ends[1] - 3 * largest_move)],
                    ]
                ),
                obs_filter=observation_filter.ObservationFilter(-1.0, 1.0, 3, {0}),
                obs_producer=observation_producer.from_ends(ends),
                rewarder=observation_rewarder.on_target(target_radius, -1, 0),
                visualizer=visualization.FocusHistoryVisualizer(
                    ends, target_radius=target_radius
                ),
            ),
            render_mode=render_mode,
        )
