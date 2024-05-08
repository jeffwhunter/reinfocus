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


class FastRenderer:
    # pylint: disable=too-few-public-methods
    """Produces images of focus scenes, and semi-efficiently transfers the data
    involved."""

    def __init__(
        self,
        block_shape: tuple[int, int, int] = (1, 16, 16),
        samples_per_pixel: int = 100,
        r_size: float = 20,
    ):
        """Creates a FastRenderer.

        Args:
            block_shape: The shape of the GPU block used to render the images.
            samples_per_pixel: The number of rays that will resolve each pixel."""

        self._block_shape = block_shape
        self._samples_per_pixel = samples_per_pixel

        self._cameras = camera.FastCameras()
        self._worlds = world.FastWorlds(r_size=r_size)

        self._random_states = None

    def update_targets(self, targets: Collection[float]):
        """Updates the positions of the checkerboard targets in each environment.

        Args:
            targets: A list of floats, one per environment, containing the z position of
                that environment's target."""

        self._worlds.update(targets)

    def update_focus_planes(self, focus_planes: Collection[float]):
        """Updates the positions of the focus plane in each environment.

        Args:
            focus_planes: A list of floats, one per environment, containing the z position
                of that environment's focus plane."""

        self._cameras.update(focus_planes)

    def render(self, frame_height: int) -> NDArray[numpy.uint8]:
        """Produces ray traced images of some number of simple focus scenes.

        Args:
            frame_height: The height in pixels of the produced images.

        Returns:
            Ray traced images of simple focus scenes."""

        grid_shape = (len(self._worlds), frame_height, frame_height)

        self._make_random_states(grid_shape)

        frame = make_render_target(grid_shape)

        cutil.launcher(FastRenderer._device_render, grid_shape, self._block_shape)(
            frame,
            self._worlds.device_data(),
            self._cameras.device_data(),
            self._samples_per_pixel,
            self._random_states,
        )

        return frame.copy_to_host().astype(numpy.uint8)

    @staticmethod
    @cuda.jit
    def _device_render(
        frames: DeviceNDArray,
        shapes: DeviceNDArray,
        cameras: camera.FastGpuCameras,
        samples_per_pixel: int,
        random_states: DeviceNDArray,
    ):
        """Uses the GPU to ray trace images of the worlds defined in shapes into frames using
        cameras' views. Only renders one rectangle per environment in order to reduce memory
        use and runtime.

        Args:
            frames: The frames into which the images will be rendered.
            shapes: The properties of the rectangles to render.
            cameras: The cameras with which the images will be rendered.
            samples_per_pixel: How many rays to fire per pixel.
            random_states: An array of RNG states."""

        index = cutil.cube_index()
        if cutil.outside_shape(index, frames.shape):
            return

        h, w = frames.shape[1:3]
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

        frames[index] = vector.d_smul_v3f(
            colour, numpy.float32(255.0 / samples_per_pixel)
        )

    def _make_random_states(self, grid_shape: tuple[int, int, int]):
        """Creates enough random states for each pixel in grid_shape.

        Args:
            grid_shape: The shape of the elements that each need a random state."""

        total_size = int(numpy.prod(grid_shape))

        if self._random_states is None or len(self._random_states) < total_size:
            self._random_states = random.make_random_states(total_size, 0)
