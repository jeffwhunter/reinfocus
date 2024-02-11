"""Functions that produce rewards from Observations."""

from typing import Callable

from reinfocus.environment.types import Observation, OI, SI, State


Rewarder = Callable[[State, Observation], float]


def distance(span: float, low: float = -1.0, high: float = 0.0) -> Rewarder:
    """Makes a function that returns a reward that scales linearly as the lens moves away
    from the target. The reward is high when the lens is on target, and low when the lens
    is a distance of span from the target.

    Args:
        span: How far away the lens has to be from the target to earn low reward.
        low: The reward when the lens is a distance of span from the target.
        high: The reward when the lens is on the target.

    Returns:
        A function that returns a reward that goes from high to low as the lens moves from
        the target to a distance of span."""

    diff = high - low

    return lambda s, _: (1.0 - abs(s[SI.TARGET] - s[SI.LENS]) / span) * diff + low


def on_target(span: float, off: float = 0.0, on: float = 1.0) -> Rewarder:
    """Makes a function that returns on reward when the lens is within span of the
    target, and off reward otherwise.

    Args:
        span: How close away the lens has to be from the target to earn on reward.
        off: The reward when the lens is more than span distance from the target.
        on: The reward when the lens is less than span distance from the target.

    Returns:
        A function that returns on reward when the lens is within span of the
        target, and off reward otherwise."""

    return lambda s, _: on if abs(s[SI.TARGET] - s[SI.LENS]) < span else off


def focus() -> Rewarder:
    """Makes a function that returns a reward equal to the focus value.

    Returns:
        A function that returns a reward equal to the focus value."""

    return lambda _, o: o[OI.FOCUS]
