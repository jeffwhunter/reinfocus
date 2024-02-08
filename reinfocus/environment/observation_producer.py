"""Functions that normalize environment observations."""

import numpy

from reinfocus.environment import focus_instrument
from reinfocus.environment import observation_normer
from reinfocus.environment.types import Observation, State, SI
from reinfocus.graphics import world


class FocusObservationProducer:
    # pylint: disable=too-few-public-methods
    """Returns observations that include both elements of the state (target and lens
    positions), followed by the focus value of a scene rendered with a target and lens in
    those position, normalized by some amount."""

    def __init__(
        self,
        obs_normer: observation_normer.ObservationNormer,
        measure_focus: focus_instrument.FocusInstrument,
    ):
        """Creates a FocusObservationProducer.

        Args:
            obs_normer: A function that normalizes observations.
            measure_focus: A function that measures how in focus a world is when viewed
                with a focus plane."""

        self._obs_normer = obs_normer
        self._measure_focus = measure_focus

    def produce_observation(self, state: State, world_data: world.World) -> Observation:
        """Returns an observation of state and world_data.

        Args:
            state: A tuple containing the target and lens positions.
            world_data: The GPU data needed to render the environment.

        Returns:
            A tuple containing the state and measured focus of the world in that state,
            normalized by some amount, then clipped to [-1, 1]."""

        return numpy.clip(
            self._obs_normer(
                numpy.array(
                    [*state, self._measure_focus(world_data, state[SI.LENS])],
                    numpy.float32,
                )
            ),
            -1 * numpy.ones(3, dtype=numpy.float32),
            numpy.ones(3, dtype=numpy.float32),
            dtype=numpy.float32,
        )


def from_ends(ends: tuple[float, float]) -> FocusObservationProducer:
    """Creates a FocusObservationProducer that maps the state from ends[0] to -1 and
    ends[1] to 1.0, and maps the focus to a range where it's highest value is 1 and lowest
    is -1.

    Args:
        ends: The values of the state that will be mapped to [-1, 1].

    Returns:
        A FocusObservationProducer that maps the state and focus to a range of [-1, 1]."""

    return FocusObservationProducer(
        obs_normer=observation_normer.from_spans(
            [ends, ends, focus_instrument.focus_extrema(ends)]
        ),
        measure_focus=focus_instrument.render_and_measure,
    )
