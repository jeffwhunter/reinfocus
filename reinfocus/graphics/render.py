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
    cam: camera.Camera,
    samples_per_pixel: int,
    random_states: DeviceNDArray,
    shapes: shape.GpuShapes,
):
    """Uses the GPU to ray trace an image of the world defined in shapes into frame using
        camera's view.

    Args:
        frame: The frame into which the image will be rendered.
        cam: The camera with which the image will be rendered.
        samples_per_pixel: How many rays to fire per pixel.
        random_states: An array of RNG states.
        shapes: The shapes to render."""

    index = cutil.grid_index()
    if cutil.outside_shape(index, frame.shape):
        return

    h, w = frame.shape[:2]
    y, x = index

    pixel_index = y * w + x

    colour = vector.d_v3f(0, 0, 0)

    for _ in range(samples_per_pixel):
        colour = vector.d_add_v3f(
            (
                colour,
                physics.find_colour(
                    shapes[shape.PARAMETERS],
                    shapes[shape.TYPES],
                    camera.get_ray(
                        cam,
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


def make_render_target(frame_shape: tuple[int, int] = (300, 600)) -> DeviceNDArray:
    """Creates a render target on the GPU; reduces the number of linter ignores.

    Args:
        frame_shape: The shape of the frame to create.

    Returns:
        A render target on the GPU."""

    return cuda.device_array(frame_shape + (3,), dtype=numpy.uint8)  # type: ignore


def render(
    frame_shape: tuple[int, int] = (300, 600),
    block_shape: tuple[int, int] = (16, 16),
    world_data: world.World = world.mixed_world(),
    samples_per_pixel: int = 100,
    focus_distance: float = 10.0,
) -> NDArray:
    """Returns a ray traced image, focused on a plane at focus_distance, of world, made
        on the GPU, to the CPU.

    Args:
        frame_shape: The shape of the image to render.
        block_shape: The shape of the GPU block used to render the image.
        world_data: The world to render.
        samples_per_pixel: How many rays to fire per pixel.
        focus_distance: How far away should the plane of perfect focus be from the view.

    Returns:
        A ray traced image, focused on a plane at focus_distance, of world."""

    frame = make_render_target(frame_shape)

    cutil.launcher(device_render, frame_shape, block_shape)(
        frame,
        camera.camera(
            camera.CameraOrientation(
                vector.v3f(0, 0, -10), vector.v3f(0, 0, 0), vector.v3f(0, 1, 0)
            ),
            camera.CameraView(frame_shape[1] / frame_shape[0], 30.0),
            camera.CameraLens(0.1, focus_distance),
        ),
        samples_per_pixel,
        random.make_random_states(frame_shape[0] * frame_shape[1], 0),
        (world_data.device_shape_parameters(), world_data.device_shape_types()),
    )

    return frame.copy_to_host()
