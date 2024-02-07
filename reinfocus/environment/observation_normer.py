"""Functions that normalize environment observations."""

from typing import Callable

import numpy

from reinfocus.environment.types import Observation


ObservationNormer = Callable[[Observation], Observation]


def make_observation_normer(mid: Observation, scale: Observation) -> ObservationNormer:
    """Makes a function that scales inputs to [-1., 1.].

    Args:
        mid: The midpoint of the range of inputs.
        scale: Half the range of possible inputs.

    Returns:
        A function that scales inputs to [-1., 1.]."""

    return lambda x: (x - mid) / scale


def from_spans(spans: list[tuple[float, float]]) -> ObservationNormer:
    """Makes an ObservationNormer that norms each dimension according to the span in the
    corresponding dimension of spans.

    Args:
        spans: The list of tuples controlling the normalization of each dimension. Each
            tuple contains two values that will be mapped to -1.0 and 1.0, respectively.

    Returns:
        An ObservationNormer that norms according to the given spans."""

    spans_a = numpy.array(spans)

    return make_observation_normer(
        numpy.average(spans_a, axis=1), numpy.diff(spans_a).reshape(len(spans)) / 2
    )
