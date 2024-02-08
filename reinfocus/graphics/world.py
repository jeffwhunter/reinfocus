"""Methods relating to representing the world in a ray tracer."""

import math

from typing import NamedTuple

import numpy

from numba import cuda
from numba.cuda.cudadrv.devicearray import DeviceNDArray
from reinfocus.graphics import hit_record
from reinfocus.graphics import ray
from reinfocus.graphics import rectangle
from reinfocus.graphics import shape
from reinfocus.graphics import sphere
from reinfocus.graphics import vector


class World:
    """Represents the world of a ray tracer in a way easy to transfer to the GPU."""

    def __init__(self, *shapes: shape.CpuShape):
        """Constructor for the World.

        Args:
            shapes: The shapes this world will hold."""

        self._device_shape_parameters = self._make_device_shape_parameters(shapes)
        self._device_shape_types = self._make_device_shape_types(shapes)

    def _make_device_shape_parameters(
        self, shapes: tuple[shape.CpuShape, ...]
    ) -> DeviceNDArray:
        """Makes a device array containing the parameters of each shape in this world.

        Args:
            shapes: The shapes whose parameters will be sent to the device.

        Returns:
            A device array containing the parameters of each shape in this world."""

        parameters = numpy.zeros(
            shape=(len(shapes), max(len(s.parameters) for s in shapes))
        )

        for i, s in enumerate(shapes):
            parameters[i, : len(s.parameters)] = s.parameters

        return cuda.to_device(parameters)

    def device_shape_parameters(self) -> DeviceNDArray:
        """Returns a device array containing the parameters of each shape in this world.

        Returns:
            A device array containing the parameters of each shape in this world."""

        return self._device_shape_parameters

    def _make_device_shape_types(
        self, shapes: tuple[shape.CpuShape, ...]
    ) -> DeviceNDArray:
        """Makes a device array containing the type of each shape in this world.

        Args:
            shapes: The shapes whose types will be sent to the device.

        Returns:
            A device array containing the type of each shape in this world."""

        return cuda.to_device(numpy.array([s.type for s in shapes]))

    def device_shape_types(self) -> DeviceNDArray:
        """Returns a device array containing the type of each shape in this world.

        Returns:
            A device array containing the type of each shape in this world."""

        return self._device_shape_types


@cuda.jit
def hit(
    shapes_parameters: DeviceNDArray,
    shapes_types: DeviceNDArray,
    r: ray.Ray,
    t_min: numpy.float32,
    t_max: numpy.float32,
) -> hit_record.HitResult:
    """Determines if the ray r hits any of the shapes defined by shape_parameters between
        t_min and t_max, according to their types in shapes_types, returning a hit_record
        containing the details if it does.

    Args:
        shapes_parameters: The parameters of each shape in this world.
        shapes_types: The type of each shape in this world.
        r: The ray potentially hitting the defined world.
        t_min: The minimum of the interval on r in which we look for hits with the world.
        t_max: The maximum of the interval on r in which we look for hits with the world.

    Returns:
        A GpuHitResult where the first element is True if a hit happened, while the second
            element is a GpuHitRecord with the details of the hit, which is empty if there
            was no hit."""

    hit_anything = False
    closest_so_far = t_max
    record = hit_record.empty_hit_record()
    for shape_parameters, shape_type in zip(shapes_parameters, shapes_types):
        h = False
        temp_record = hit_record.empty_hit_record()

        if shape_type == shape.SPHERE:
            h, temp_record = sphere.hit(shape_parameters, r, t_min, closest_so_far)
        else:
            h, temp_record = rectangle.hit(shape_parameters, r, t_min, closest_so_far)

        if h:
            hit_anything = True
            closest_so_far = temp_record[hit_record.T]
            record = temp_record

    return hit_anything, record


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


def one_sphere_world(parameters: ShapeParameters = ShapeParameters()) -> World:
    """Makes a world with one sphere on the z axis.

    Args:
        parameters: The parameters for the sphere.

    Returns:
        A world with one sphere on the z axis."""

    return World(
        sphere.sphere(
            vector.v3f(0, 0, -parameters.distance),
            get_absolute_size(parameters),
            vector.v2f(*parameters.texture_f),
        )
    )


def two_sphere_world(
    left_parameters: ShapeParameters = ShapeParameters(20.0),
    right_parameters: ShapeParameters = ShapeParameters(5.0),
) -> World:
    """Makes a world with spheres at different distances on the left and right.

    Args:
        left_parameters: The parameters for the left sphere.
        right_parameters: The parameters for the right sphere.

    Returns:
        A world with spheres at different distances on the left and right."""

    distance_to_offset = math.tan(math.radians(15))

    return World(
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
    )


def one_rect_world(parameters: ShapeParameters = ShapeParameters()) -> World:
    """Makes a world with one rectangle on the z axis.

    Args:
        parameters: The parameters for the rectangle.

    Returns:
        A world with one rectangle on the z axis."""

    size = get_absolute_size(parameters)

    return World(
        rectangle.rectangle(
            vector.v2f(-size, size),
            vector.v2f(-size, size),
            -parameters.distance,
            vector.v2f(*parameters.texture_f),
        )
    )


def two_rect_world(
    left_parameters: ShapeParameters = ShapeParameters(20.0),
    right_parameters: ShapeParameters = ShapeParameters(5.0),
) -> World:
    """Makes a world with rectangles at different distances on the left and right.

    Args:
        left_parameters: The parameters for the left rectangle.
        right_parameters: The parameters for the right rectangle.

    Returns:
        A world with rectangles at different distances on the left and right."""

    distance_to_offset = math.tan(math.radians(15))

    left_offset = left_parameters.distance * distance_to_offset
    left_size = get_absolute_size(left_parameters)

    right_offset = right_parameters.distance * distance_to_offset
    right_size = get_absolute_size(right_parameters)

    return World(
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
    )


def mixed_world(
    left_parameters: ShapeParameters = ShapeParameters(5.0),
    right_parameters: ShapeParameters = ShapeParameters(),
) -> World:
    """Makes a world with a sphere on the left at a different distance from the rectangle
        on the right.

    Args:
        left_parameters: The parameters for the left sphere.
        right_parameters: The parameters for the right sphere.

    Returns:
        A world with a sphere on the left at a different distance from the rectangle on
            the right."""

    distance_to_offset = math.tan(math.radians(15))

    left_size = get_absolute_size(left_parameters)

    right_offset = right_parameters.distance * distance_to_offset
    right_size = get_absolute_size(right_parameters)

    return World(
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
    )
