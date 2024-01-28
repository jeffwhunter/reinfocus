"""Methods relating to ray tracing."""

import numpy

from numba import cuda
from numba.cuda.cudadrv.devicearray import DeviceNDArray
from numpy.typing import NDArray

from reinfocus.graphics import camera
from reinfocus.graphics import cutil
from reinfocus.graphics import physics
from reinfocus.graphics import random
from reinfocus.graphics import shape
from reinfocus.graphics import vector
from reinfocus.graphics import world


@cuda.jit
def device_render(
    frame: NDArray,
    cpu_camera: camera.CpuCamera,
    samples_per_pixel: int,
    random_states: DeviceNDArray,
    shapes: shape.GpuShapes,
):
    """Uses the GPU to ray trace an image of the world defined in shapes into frame using
        camera's view.

    Args:
        frame: The frame into which the image will be rendered.
        cpu_camera: The camera with which the image will be rendered.
        samples_per_pixel: How many rays to fire per pixel.
        random_states: An array of RNG states.
        shapes: The shapes to render."""

    index = cutil.grid_index()
    if cutil.outside_shape(index, frame.shape):
        return

    height, width = frame.shape[:2]
    y, x = index

    pixel_index = y * width + x

    colour = vector.g3f(0, 0, 0)

    gpu_camera = camera.cast_to_gpu_camera(cpu_camera)

    for _ in range(samples_per_pixel):
        colour = vector.add_g3f(
            colour,
            physics.find_colour(
                shapes[shape.PARAMETERS],
                shapes[shape.TYPES],
                camera.get_ray(
                    gpu_camera,
                    (x + random.uniform_float(random_states, pixel_index)) / width,
                    (y + random.uniform_float(random_states, pixel_index)) / height,
                    random_states,
                    pixel_index,
                ),
                random_states,
                pixel_index,
            ),
        )

    frame[index] = vector.g3f_to_c3f(vector.div_g3f(colour, samples_per_pixel))


def render(
    frame_shape: tuple[int, int] = (300, 600),
    block_shape: tuple[int, int] = (16, 16),
    cpu_world: world.World = world.mixed_world(),
    samples_per_pixel: int = 100,
    focus_distance: float = 10.0,
) -> NDArray:
    """Returns a ray traced image, focused on a plane at focus_distance, of world, made
        on the GPU, to the CPU.

    Args:
        frame_shape: The shape of the image to render.
        block_shape: The shape of the GPU block used to render the image.
        cpu_world: The world to render.
        samples_per_pixel: How many rays to fire per pixel.
        focus_distance: How far away should the plane of perfect focus be from the view.

    Returns:
        A ray traced image, focused on a plane at focus_distance, of world."""

    frame = cuda.device_array(frame_shape + (3,), dtype=numpy.float32)

    cutil.launcher(device_render, frame_shape, block_shape)(
        frame,
        camera.cpu_camera(
            camera.CameraOrientation(
                vector.c3f(0, 0, -10), vector.c3f(0, 0, 0), vector.c3f(0, 1, 0)
            ),
            camera.CameraView(frame_shape[1] / frame_shape[0], 30.0),
            camera.CameraLens(0.1, focus_distance),
        ),
        samples_per_pixel,
        random.make_random_states(frame_shape[0] * frame_shape[1], 0),
        (cpu_world.device_shape_parameters(), cpu_world.device_shape_types()),
    )

    return frame.copy_to_host()
