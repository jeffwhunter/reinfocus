"""Methods relating to 3D rays."""

from numba import cuda

from reinfocus.graphics import vector

GpuRay = tuple[vector.G3F, vector.G3F]

ORIGIN = 0
DIRECTION = 1


@cuda.jit
def gpu_ray(origin: vector.G3F, direction: vector.G3F) -> GpuRay:
    """Makes a 3D ray on the GPU originating at origin and pointing in direction.

    Args:
        origin: The origin of the ray.
        direction: The direction of the ray.

    Returns:
        A 3D GPU ray pointing from origin to direction."""

    return origin, direction


@cuda.jit
def cpu_to_gpu_ray(origin: vector.C3F, direction: vector.C3F) -> GpuRay:
    """Converts a CPU to a GPU 3D ray.

    Args:
        origin: The origin of the ray.
        direction: The direction of the ray.

    Returns:
        A 3D GPU ray pointing from origin to direction."""

    return vector.c3f_to_g3f(origin), vector.c3f_to_g3f(direction)


@cuda.jit
def gpu_point_at_parameter(ray: GpuRay, t: float) -> vector.G3F:
    """Returns the point at the end of ray scaled by t.

    Args:
        ray: The ray to scale.
        t: The amount to scale ray by.

    Returns:
        The point at the end of ray scaled by t."""

    return vector.add_g3f(ray[ORIGIN], vector.smul_g3f(ray[DIRECTION], t))
