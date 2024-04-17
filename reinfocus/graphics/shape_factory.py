"""Defines the possible shapes to render."""

import math

from collections.abc import Sequence
from typing import NamedTuple

from reinfocus.graphics import rectangle
from reinfocus.graphics import shape
from reinfocus.graphics import sphere
from reinfocus.graphics import vector


class ShapeParameters(NamedTuple):
    """Defines all the necessary information for a shape in one of these worlds.

    Args:
        distance: How far away from the origin the shape is.
        size: How large the shape is. If zero r_size will be used.
        r_size: How many degrees of FOV this object should take up when size is zero.
        texture_f: How often the checkerboard of this object changes in x and y."""

    distance: float = 10.0
    size: float = 0.0
    r_size: float = 20.0
    texture_f: tuple[int, int] = (16, 16)


def get_absolute_size(parameters: ShapeParameters) -> float:
    """Returns the actual size of a shape defined with parameters.

    Args:
        parameters: The parameters defining the shape.

    Returns:
        The actual size of some shape defined by parameters."""

    if parameters.size != 0.0:
        return parameters.size

    return parameters.distance * math.tan(math.radians(parameters.r_size / 2))


def one_sphere(
    parameters: ShapeParameters = ShapeParameters(),
) -> Sequence[shape.CpuShape]:
    """Creates one sphere on the z axis.

    Args:
        parameters: The parameters for the sphere.

    Returns:
        One sphere suitable for transfer to the GPU."""

    return [
        sphere.sphere(
            vector.v3f(0, 0, -parameters.distance),
            get_absolute_size(parameters),
            vector.v2f(*parameters.texture_f),
        )
    ]


def two_sphere(
    left_parameters: ShapeParameters = ShapeParameters(20.0),
    right_parameters: ShapeParameters = ShapeParameters(5.0),
) -> Sequence[shape.CpuShape]:
    """Creates spheres at different distances on the left and right.

    Args:
        left_parameters: The parameters for the left sphere.
        right_parameters: The parameters for the right sphere.

    Returns:
        Two spheres suitable for transfer to the GPU."""

    distance_to_offset = math.tan(math.radians(15))

    return [
        sphere.sphere(
            vector.v3f(
                -left_parameters.distance * distance_to_offset,
                0,
                -left_parameters.distance,
            ),
            get_absolute_size(left_parameters),
            vector.v2f(*left_parameters.texture_f),
        ),
        sphere.sphere(
            vector.v3f(
                right_parameters.distance * distance_to_offset,
                0,
                -right_parameters.distance,
            ),
            get_absolute_size(right_parameters),
            vector.v2f(*right_parameters.texture_f),
        ),
    ]


def one_rect(parameters: ShapeParameters = ShapeParameters()) -> Sequence[shape.CpuShape]:
    """Creates one rectangle on the z axis.

    Args:
        parameters: The parameters for the rectangle.

    Returns:
        One rectangle suitable for transfer to the GPU."""

    size = get_absolute_size(parameters)

    return [
        rectangle.rectangle(
            vector.v2f(-size, size),
            vector.v2f(-size, size),
            -parameters.distance,
            vector.v2f(*parameters.texture_f),
        )
    ]


def two_rect(
    left_parameters: ShapeParameters = ShapeParameters(20.0),
    right_parameters: ShapeParameters = ShapeParameters(5.0),
) -> Sequence[shape.CpuShape]:
    """Creates rectangles at different distances on the left and right.

    Args:
        left_parameters: The parameters for the left rectangle.
        right_parameters: The parameters for the right rectangle.

    Returns:
        Two rectangles suitable for transfer to the GPU."""

    distance_to_offset = math.tan(math.radians(15))

    left_offset = left_parameters.distance * distance_to_offset
    left_size = get_absolute_size(left_parameters)

    right_offset = right_parameters.distance * distance_to_offset
    right_size = get_absolute_size(right_parameters)

    return [
        rectangle.rectangle(
            vector.v2f(-left_offset - left_size, -left_offset + left_size),
            vector.v2f(-left_size, left_size),
            -left_parameters.distance,
            vector.v2f(*left_parameters.texture_f),
        ),
        rectangle.rectangle(
            vector.v2f(right_offset - right_size, right_offset + right_size),
            vector.v2f(-right_size, right_size),
            -right_parameters.distance,
            vector.v2f(*right_parameters.texture_f),
        ),
    ]


def mixed(
    left_parameters: ShapeParameters = ShapeParameters(5.0),
    right_parameters: ShapeParameters = ShapeParameters(),
) -> Sequence[shape.CpuShape]:
    """Creates a sphere on the left at a different distance from the rectangle on the
    right.

    Args:
        left_parameters: The parameters for the left sphere.
        right_parameters: The parameters for the right sphere.

    Returns:
        A sphere and rectangle suitable for transfer to the GPU."""

    distance_to_offset = math.tan(math.radians(15))

    left_size = get_absolute_size(left_parameters)

    right_offset = right_parameters.distance * distance_to_offset
    right_size = get_absolute_size(right_parameters)

    return [
        sphere.sphere(
            vector.v3f(
                -left_parameters.distance * distance_to_offset,
                0,
                -left_parameters.distance,
            ),
            left_size,
            vector.v2f(*left_parameters.texture_f),
        ),
        rectangle.rectangle(
            vector.v2f(right_offset - right_size, right_offset + right_size),
            vector.v2f(-right_size, right_size),
            -right_parameters.distance,
            vector.v2f(*right_parameters.texture_f),
        ),
    ]
