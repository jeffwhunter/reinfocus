"""Function that render the world and measure it for focus."""

import functools

from typing import Callable

import numpy

from reinfocus import vision
from reinfocus.graphics import render
from reinfocus.graphics import world


FocusInstrument = Callable[[world.World, float], float]


def render_and_measure(world_data: world.World, focus_distance: float) -> float:
    """Renders then measures the focus value of world when focused on the plane at
        focus_distance.

    Args:
        world_data: The world to render.
        focus_distance: The distance from the camera of the focus plane.

    Returns:
        A measure of how in focus the given scene is, with higher values implying a
        better focus."""

    return vision.focus_value(
        render.render(
            frame_shape=(300, 300), world_data=world_data, focus_distance=focus_distance
        )
    )


@functools.cache
def focus_extrema(
    ends: tuple[float, float],
    measurement_steps: int = 91,
) -> tuple[float, float]:
    """Finds the minimum and maximum possible focus values by scanning through a number
        of scenes, calculating their focus values, and returning their min and max.

    Args:
        ends: The minimum and maximum lens and target positions.
        measurement_steps: How many steps taken to scan the space of positions.

    Returns:
        min: The minimum focus value.
        max: The maximum focus value."""

    space = numpy.linspace(*ends, measurement_steps)

    focus_values = [
        render_and_measure(world.one_rect_world(world.ShapeParameters(distance=i)), i)
        for i in space
    ]
    focus_values.append(
        render_and_measure(
            world.one_rect_world(world.ShapeParameters(distance=space[0])), space[-1]
        )
    )
    focus_values.append(
        render_and_measure(
            world.one_rect_world(world.ShapeParameters(distance=space[-1])), space[0]
        )
    )

    return min(focus_values), max(focus_values)
