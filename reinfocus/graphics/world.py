"""Methods relating to representing the world in a ray tracer."""

from collections.abc import Collection

import numpy

from numba import cuda
from numba.cuda.cudadrv.devicearray import DeviceNDArray

from reinfocus.graphics import hit_record
from reinfocus.graphics import ray
from reinfocus.graphics import rectangle
from reinfocus.graphics import shape
from reinfocus.graphics import shape_factory
from reinfocus.graphics import sphere

GpuWorld = tuple[DeviceNDArray, DeviceNDArray, DeviceNDArray]

# GpuWorld indices
MW_PARAMETERS = 0
MW_TYPES = 1
MW_ENV_SIZES = 2


class Worlds:
    """A collection of sets of shapes that can be conveniently transfered to the GPU."""

    def __init__(self, *env_shapes: Collection[shape.CpuShape]):
        """Creates a Worlds.

        Args:
            env_shapes: A collection of collections of shapes, each element specifying the
                shapes in each environment."""

        environment_sizes = numpy.array(
            [len(shapes) for shapes in env_shapes], dtype=numpy.int32
        )

        self._d_environment_sizes = cuda.to_device(environment_sizes)

        self._num_envs = len(env_shapes)
        most_shapes = max(environment_sizes)

        parameters = numpy.zeros(
            (
                self._num_envs,
                most_shapes,
                max(max(len(s.parameters) for s in shapes) for shapes in env_shapes),
            ),
            dtype=numpy.float32,
        )

        for env_i, shapes in enumerate(env_shapes):
            for shape_i, s in enumerate(shapes):
                parameters[env_i, shape_i, : len(s.parameters)] = s.parameters

        self._d_shape_params = cuda.to_device(parameters)

        shape_types = numpy.zeros((self._num_envs, most_shapes), dtype=numpy.int32)

        for env_i, shapes in enumerate(env_shapes):
            shape_types[env_i, : len(shapes)] = [s.shape_type for s in shapes]

        self._d_shape_types = cuda.to_device(shape_types)

    def __len__(self) -> int:
        """How many worlds are contained in this collection of worlds.

        Returns:
            The number of worlds contained in this collection."""

        return self._num_envs

    def device_data(self) -> GpuWorld:
        """Returns a tuple containing all the properties of these worlds.

        Returns:
            A tuple containing all the properties of these worlds."""

        return (self._d_shape_params, self._d_shape_types, self._d_environment_sizes)


class FocusWorlds:
    """A collection of sets of shapes that can be conveniently transfered to the GPU.
    Reduces the amount of GPU data needed by assuming all environments only contain one
    z-axis aligned square each."""

    def __init__(self, num_envs: int):
        """Creates a FocusWorlds.

        Args:
            num_envs: How many worlds this collection should hold."""

        self._targets = numpy.full(num_envs, numpy.nan, dtype=numpy.float32)
        self._d_parameters = None

    def device_data(self) -> DeviceNDArray:
        """Returns a device array containing the parameters of each shape in this world.

        Returns:
            A device array containing the parameters of each shape in this world."""

        return self._d_parameters

    def update_targets(self, new_targets: Collection[float], r_size: float = 20):
        """Updates the position of the various targets in each world.

        Args:
            new_targets: How far along the negative z-axis each target should be
                positioned.
            r_size: How many degrees of field of view each target should occupy."""

        new_targets = numpy.asarray(new_targets)

        if all(self._targets == new_targets):
            return

        self._targets = new_targets

        self._d_parameters = cuda.to_device(
            numpy.array(
                [
                    [
                        shape_factory.get_absolute_size(
                            shape_factory.ShapeParameters(target, r_size=r_size)
                        ),
                        -target,
                    ]
                    for target in new_targets
                ],
                dtype=numpy.float32,
            )
        )


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
