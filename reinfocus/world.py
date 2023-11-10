"""Methods relating to representing the world in a ray tracer."""

import math

import numpy as np
from numba import cuda
from numba.cuda.cudadrv import devicearray as cda
from reinfocus import hit_record as hit
from reinfocus import ray
from reinfocus import rectangle as rec
from reinfocus import shape as sha
from reinfocus import sphere as sph
from reinfocus import vector as vec

class World:
    """Represents the world of a ray tracer in a way easy to transfer to the GPU."""
    def __init__(self, *shapes: sha.CpuShape):
        self.shapes = shapes
        self.__device_shape_parameters = self.__make_device_shape_parameters()
        self.__device_shape_types = self.__make_device_shape_types()

    def __make_device_shape_parameters(self) -> cda.DeviceNDArray:
        """Makes a device array containing the parameters of each shape in this world.

        Returns:
            A device array containing the parameters of each shape in this world."""
        parameters = np.zeros(shape=(
            len(self.shapes),
            max(len(shape.parameters) for shape in self.shapes)))

        for i, shape in enumerate(self.shapes):
            parameters[i, :len(shape.parameters)] = shape.parameters

        return cuda.to_device(parameters)

    def device_shape_parameters(self) -> cda.DeviceNDArray:
        """Returns a device array containing the parameters of each shape in this world.

        Returns:
            A device array containing the parameters of each shape in this world."""
        return self.__device_shape_parameters

    def __make_device_shape_types(self) -> cda.DeviceNDArray:
        """Makes a device array containing the type of each shape in this world.

        Returns:
            A device array containing the type of each shape in this world."""
        return cuda.to_device(np.array([shape.type for shape in self.shapes]))

    def device_shape_types(self) -> cda.DeviceNDArray:
        """Returns a device array containing the type of each shape in this world.

        Returns:
            A device array containing the type of each shape in this world."""
        return self.__device_shape_types

@cuda.jit
def gpu_hit_world(
    shapes_parameters: cda.DeviceNDArray,
    shapes_types: cda.DeviceNDArray,
    r: ray.GpuRay,
    t_min: float,
    t_max: float
) -> sha.GpuHitResult:
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
    record = hit.gpu_empty_hit_record()
    for shape_parameters, shape_type in zip(shapes_parameters, shapes_types):
        h = False
        temp_record = hit.gpu_empty_hit_record()

        if shape_type == sha.SPHERE:
            h, temp_record = sph.gpu_hit_sphere(
                shape_parameters,
                r,
                t_min,
                closest_so_far)
        else:
            h, temp_record = rec.gpu_hit_rectangle(
                shape_parameters,
                r,
                t_min,
                closest_so_far)

        if h:
            hit_anything = True
            closest_so_far = temp_record[hit.T]
            record = temp_record

    return hit_anything, record

def one_sphere_world(distance: float=10.0, r_size=20.0):
    """Makes a world with one sphere at (0, 0, -distance).

    Args:
        distance: How far away from the origin the sphere should be.
        r_size: How large should the sphere be in degrees of fov.

    Returns:
        A world with one sphere at (0, 0, -distance)."""
    return World(
        sph.cpu_sphere(
            vec.c3f(0, 0, -distance),
            distance * math.tan(math.radians(r_size / 2))))

def two_sphere_world(left_distance: float=20.0, right_distance: float=5.0):
    """Makes a world with spheres at different distances on the left and right.

    Args:
        left_distance: How far away from the origin the left sphere should be.
        right_distance: How far away from the origin the right sphere should be.

    Returns:
        A world with spheres at different distances on the left and right."""
    distance_to_radius = math.tan(math.radians(10))
    distance_to_offset = math.tan(math.radians(15))

    return World(
        sph.cpu_sphere(
            (-left_distance * distance_to_offset, 0, -left_distance),
            left_distance * distance_to_radius),
        sph.cpu_sphere(
            (right_distance * distance_to_offset, 0, -right_distance),
            right_distance * distance_to_radius))

def one_rect_world(distance: float=10.0, r_size=20.0):
    """Makes a world with one rectangle at (0, 0, -distance).

    Args:
        distance: How far away from the origin the rectangle should be.
        r_size: How large should the rectangle be in degrees of fov.

    Returns:
        A world with one rectangle at (0, 0, -distance)."""
    radius = distance * math.tan(math.radians(r_size / 2))
    return World(rec.cpu_rectangle(-radius, radius, -radius, radius, -distance))

def two_rect_world(left_distance: float=20.0, right_distance: float=5.0):
    """Makes a world with rectangles at different distances on the left and right.

    Args:
        left_distance: How far away from the origin the left rectangle should be.
        right_distance: How far away from the origin the right rectangle should be.

    Returns:
        A world with rectangles at different distances on the left and right."""
    distance_to_radius = math.tan(math.radians(10))
    distance_to_offset = math.tan(math.radians(15))

    left_offset = left_distance * distance_to_offset
    left_radius = left_distance * distance_to_radius

    right_offset = right_distance * distance_to_offset
    right_radius = right_distance * distance_to_radius

    return World(
        rec.cpu_rectangle(
            -left_offset - left_radius / 2,
            -left_offset + left_radius / 2,
            -left_radius / 2,
            left_radius / 2,
            -left_distance),
        rec.cpu_rectangle(
            right_offset - right_radius / 2,
            right_offset + right_radius / 2,
            -right_radius / 2,
            right_radius / 2,
            -right_distance))

def mixed_world(left_distance: float=20.0, right_distance: float=5.0):
    """Makes a world with shapes at different distances on the left and right.

    Args:
        left_distance: How far away from the origin the left sphere should be.
        right_distance: How far away from the origin the right rectangle should be.

    Returns:
        A world with shapes at different distances on the left and right."""
    distance_to_radius = math.tan(math.radians(10))
    distance_to_offset = math.tan(math.radians(15))

    right_offset = right_distance * distance_to_offset
    right_radius = right_distance * distance_to_radius

    return World(
        sph.cpu_sphere(
            (-left_distance * distance_to_offset, 0, -left_distance),
            left_distance * distance_to_radius),
        rec.cpu_rectangle(
            right_offset - right_radius / 2,
            right_offset + right_radius / 2,
            -right_radius / 2,
            right_radius / 2,
            -right_distance))
