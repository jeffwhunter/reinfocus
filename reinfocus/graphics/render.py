"""Methods relating to ray tracing."""

import math

import numpy as np
from numba import cuda
from numba.cuda.cudadrv import devicearray as cda
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32
from reinfocus.graphics import camera as cam
from reinfocus.graphics import physics as phy
from reinfocus.graphics import shape as sha
from reinfocus.graphics import vector as vec
from reinfocus.graphics import world as wor

GpuFrame = cda.DeviceNDArray

@cuda.jit
def device_render(
    frame: GpuFrame,
    camera: cam.CpuCamera,
    samples_per_pixel: int,
    random_states: cda.DeviceNDArray,
    shapes: sha.GpuShapes
):
    """Uses the GPU to ray trace an image of the world defined in shapes into frame using
        camera's view.

    Args:
        frame: The frame into which the image will be rendered.
        camera: The camera with which the image will be rendered.
        samples_per_pixel: How many rays to fire per pixel.
        random_states: An array of RNG states.
        shapes: The shapes to render."""
    # pylint: disable=no-value-for-parameter,comparison-with-callable

    y, x = cuda.grid(2) # type: ignore

    height, width = frame.shape[:2]

    if y >= height or x >= width:
        return

    pixel_index = y * width + x

    colour = vec.g3f(0, 0, 0)

    for _ in range(samples_per_pixel):
        u = (x + xoroshiro128p_uniform_float32(random_states, pixel_index)) / width
        v = (y + xoroshiro128p_uniform_float32(random_states, pixel_index)) / height

        r = cam.get_ray(cam.to_gpu_camera(camera), u, v, random_states, pixel_index)

        colour = vec.add_g3f(
            colour,
            phy.find_colour(
                shapes[sha.PARAMETERS],
                shapes[sha.TYPES],
                r,
                random_states,
                pixel_index))

    frame[y, x] = vec.g3f_to_c3f(vec.div_g3f(colour, samples_per_pixel))

def make_device_frame(x: int, y: int) -> cda.DeviceNDArray:
    """Returns an empty image of shape frame_shape in GPU memory.

    Args:
        x: The width of the image in pixels.
        y: The height of the image in pixels.

    Returns:
        An empty image of shape frame_shape in GPU memory."""
    return cuda.to_device([[(np.float32(0), ) * 3] * y] * x)

def render(
    frame_shape=(300, 600),
    block_shape=(16, 16),
    world=wor.mixed_world(),
    samples_per_pixel=100,
    focus_distance=10.0
):
    """Returns a ray traced image, focused on a plane at focus_distance, of world, made
        on the GPU, to the CPU.

    Args:
        frame_shape: The shape of the image to render.
        block_shape: The shape of the GPU block used to render the image.
        world: The world to render.
        samples_per_pixel: How many rays to fire per pixel.
        focus_distance: How far away should the plane of perfect focus be from the view.

    Returns:
        A ray traced image, focused on a plane at focus_distance, of world."""
    d_frame = make_device_frame(*frame_shape)

    device_render[ # type: ignore
        tuple(math.ceil(f / b) for f, b in zip(frame_shape, block_shape)),
        block_shape
    ](
        d_frame,
        cam.cpu_camera(
            cam.CameraOrientation(
                vec.c3f(0, 0, -10),
                vec.c3f(0, 0, 0),
                vec.c3f(0, 1, 0)),
            cam.CameraView(frame_shape[1] / frame_shape[0], 30.0),
            cam.CameraLens(0.1, focus_distance)),
        samples_per_pixel,
        create_xoroshiro128p_states(frame_shape[0] * frame_shape[1], seed=0),
        (world.device_shape_parameters(), world.device_shape_types()))

    return d_frame.copy_to_host()
