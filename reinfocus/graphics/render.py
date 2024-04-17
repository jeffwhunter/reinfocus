"""Methods relating to ray tracing."""

from collections.abc import Collection

import numpy

from numba import cuda
from numba.cuda.cudadrv.devicearray import DeviceNDArray
from numpy.typing import NDArray

from reinfocus.graphics import camera
from reinfocus.graphics import cutil
from reinfocus.graphics import physics
from reinfocus.graphics import random
from reinfocus.graphics import vector
from reinfocus.graphics import world


def make_render_target(frame_shape: tuple[int, ...] = (300, 600)) -> DeviceNDArray:
    """Creates a render target on the GPU; reduces the number of linter ignores.

    Args:
        frame_shape: The shape of the frame to create.

    Returns:
        A render target on the GPU."""

    return cuda.device_array(frame_shape + (3,), dtype=numpy.uint8)  # type: ignore


@cuda.jit
def device_render(
    frames: DeviceNDArray,
    cameras: camera.GpuCameras,
    samples_per_pixel: int,
    random_states: DeviceNDArray,
    gpu_world: world.GpuWorld,
):
    """Uses the GPU to ray trace images of the gpu_world into frames using cameras' views.

    Args:
        frames: The frames into which the images will be rendered.
        cameras: The cameras with which the images will be rendered.
        samples_per_pixel: How many rays to fire per pixel.
        random_states: An array of RNG states.
        gpu_world: The world to render."""

    index = cutil.cube_index()
    if cutil.outside_shape(index, frames.shape):
        return

    h, w = frames.shape[1:3]
    e, y, x = index

    pixel_index = e * h * w + y * w + x

    colour = vector.d_v3f(0, 0, 0)

    num_shapes = gpu_world[world.MW_ENV_SIZES][e]

    for _ in range(samples_per_pixel):
        colour = vector.d_add_v3f(
            (
                colour,
                physics.find_colour(
                    gpu_world[world.MW_PARAMETERS][e, :num_shapes],
                    gpu_world[world.MW_TYPES][e, :num_shapes],
                    camera.get_ray(
                        camera.from_cameras(cameras, e),
                        numpy.float32(
                            (x + random.uniform_float(random_states, pixel_index)) / w
                        ),
                        numpy.float32(
                            (y + random.uniform_float(random_states, pixel_index)) / h
                        ),
                        random_states,
                        pixel_index,
                    ),
                    random_states,
                    pixel_index,
                ),
            )
        )

    frames[index] = vector.d_smul_v3f(colour, numpy.float32(255.0 / samples_per_pixel))


@cuda.jit
def fast_device_render(
    frame: DeviceNDArray,
    cameras: camera.FastGpuCameras,
    samples_per_pixel: int,
    random_states: DeviceNDArray,
    shapes: DeviceNDArray,
):
    """Uses the GPU to ray trace images of the worlds defined in shapes into frames using
    cameras' views. Only renders one rectangle per environment in order to reduce memory
    use and runtime.

    Args:
        frames: The frames into which the images will be rendered.
        cameras: The cameras with which the images will be rendered.
        samples_per_pixel: How many rays to fire per pixel.
        random_states: An array of RNG states.
        shapes: The properties of the rectangles to render."""

    index = cutil.cube_index()
    if cutil.outside_shape(index, frame.shape):
        return

    h, w = frame.shape[1:3]
    e, y, x = index

    pixel_index = e * h * w + y * w + x

    colour = vector.d_v3f(0, 0, 0)

    for _ in range(samples_per_pixel):
        colour = vector.d_add_v3f(
            (
                colour,
                physics.fast_find_colour(
                    shapes[e],
                    camera.get_ray(
                        camera.from_fast_cameras(cameras, e),
                        numpy.float32(
                            (x + random.uniform_float(random_states, pixel_index)) / w
                        ),
                        numpy.float32(
                            (y + random.uniform_float(random_states, pixel_index)) / h
                        ),
                        random_states,
                        pixel_index,
                    ),
                    random_states,
                    pixel_index,
                ),
            )
        )

    frame[index] = vector.d_smul_v3f(colour, numpy.float32(255.0 / samples_per_pixel))


def render(
    world_data: world.Worlds,
    cameras: camera.Cameras,
    frame_shape: tuple[int, int] = (300, 600),
    block_shape: tuple[int, int, int] = (1, 16, 16),
    samples_per_pixel: int = 100,
) -> NDArray[numpy.uint8]:
    """Returns ray traced images of world_data from the point of view of cameras.

    Args:
        world_data: The shapes that will be rendered in each image.
        cameras: The cameras from whose view each image will be rendered.
        frame_shape: The shape of each image to render.
        block_shape: The shape of the GPU block used to render the images.
        samples_per_pixel: The number of rays that will resolve each pixel.

    Returns:
        Ray traced images of world_data from the point of view of cameras."""

    grid_shape = (len(world_data),) + frame_shape

    frame = make_render_target(grid_shape)

    cutil.launcher(device_render, grid_shape, block_shape)(
        frame,
        cameras.device_data(),
        samples_per_pixel,
        random.make_random_states(int(numpy.prod(grid_shape)), 0),
        world_data.device_data(),
    )

    return frame.copy_to_host().astype(numpy.uint8)


def fast_render(
    world_data: world.FocusWorlds,
    focus_distances: Collection[float],
    frame_shape: tuple[int, int] = (300, 600),
    block_shape: tuple[int, int, int] = (1, 16, 16),
    samples_per_pixel: int = 100,
) -> NDArray[numpy.uint8]:
    """Returns ray traced images of world_data with the focus plane of each defined by
    focus_distances.

    Args:
        world_data: The shapes that will be rendered in each image.
        focus_distances: Where the plane of perfect focus will be located in each image.
        frame_shape: The shape of each image to render.
        block_shape: The shape of the GPU block used to render the images.
        samples_per_pixel: The number of rays that will resolve each pixel.

    Returns:
        Ray traced images of world_data focused on planes at focus_distances."""

    num_envs = len(focus_distances)

    grid_shape = (num_envs,) + frame_shape

    frame = make_render_target(grid_shape)

    cameras = camera.FastCameras(focus_distances, frame_shape[1] / frame_shape[0])

    cutil.launcher(fast_device_render, grid_shape, block_shape)(
        frame,
        cameras.device_data(),
        samples_per_pixel,
        random.make_random_states(int(numpy.prod(grid_shape)), 0),
        world_data.device_data(),
    )

    return frame.copy_to_host().astype(numpy.uint8)
