"""Functions that produce rewards from Observations."""

from typing import Callable

from reinfocus.environment.types import Observation, OI


Rewarder = Callable[[Observation], float]


def distance_penalty(span: float) -> Rewarder:
    """Makes a function that returns a penalty that increases the further the lens gets
        from the target. The penalty is 0 when the lens is on target, and 1 when the lens
         is a distance of `span` from the target.

    Args:
        `span`: The range of possible lens positions.

    Returns:
        A function that returns a penalty that increases as the lens gets further from the
        target."""

    return lambda o: -abs(o[OI.TARGET] - o[OI.LENS]) / span


def on_target_reward(radius: float) -> Rewarder:
    """Makes a function that returns a reward when the lens is within some distance of
        the target. The reward is 1 when the lens is within that distance, and 0
        otherwise.

    Args:
        radius: How close the lens has to get to the target before a reward is returned.

    Returns:
        A function that returns a reward when the lens is within some distance of the
        target."""

    return lambda o: 1 if abs(o[OI.TARGET] - o[OI.LENS]) < radius else 0


def focus_reward() -> Rewarder:
    """Makes a function that returns a reward equal to the focus value.

    Returns:
        A function that returns a reward equal to the focus value."""

    return lambda o: o[OI.FOCUS]
