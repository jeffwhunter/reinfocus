"""Methods that reduce the number of disabled lint rules involved in numba cuda code."""

from typing import Callable, TypeVar

import numba
import numpy

from numba import cuda

Shape = int | tuple[int, ...]

KernelT = TypeVar("KernelT", bound=Callable[..., None])
ShapeT = TypeVar("ShapeT", bound=Shape)


def enough_blocks(shape: ShapeT, block_shape: ShapeT) -> Shape:
    """Returns the number of blocks necessary to cover shape if each block has the shape
    block_shape.

    Args:
        shape: The shape of whatever you want to divide into blocks.
        block_shape: The shape of each block.

    Returns:
        The shape of the grid of blocks necessary to cover shape."""

    blocks = numpy.ceil(numpy.divide(shape, block_shape))

    if isinstance(shape, tuple):
        return tuple(map(int, blocks))

    return int(blocks)


def constant_like(n: int, shape: Shape) -> Shape:
    """Returns a Shape with the same length as shape, but filled with n.

    Args:
        n: The constant to fill the new Shape with.
        shape: The Shape whose length the new Shape will have.

    Returns:
        A new Shape with the same length as shape, but filled with n."""

    if isinstance(shape, int):
        return n

    return len(shape) * (n,)


def launcher(
    kernel: KernelT, shape: ShapeT, block_shape: ShapeT | None = None
) -> KernelT:
    """Returns a callable that invokes kernel on the gpu with enough blocks of size
    block_shape to cover shape.

    Args:
        kernel: The numba cuda kernel to invoke on the device.
        shape: The shape of the block of threads used in this execution. Can be either an
            int or a tuple of ints.
        block_shape: The shape of the blocks of threads this execution will be divided
            into. Blocks must be large enough to fill the execution units they're assigned
            to. Must be the same type as shape or None. If None, it will default to a
            shape with the same number of dimensions as shape, but with lengths of 16 on
            all sides. Read more at http://docs.nvidia.com/cuda/cuda-c-programming-guide.

    Returns:
        A callable that invokes kernel on the gpu."""

    actual_block_shape = constant_like(16, shape) if block_shape is None else block_shape

    return kernel[enough_blocks(shape, actual_block_shape), actual_block_shape]  # type: ignore


@cuda.jit
def outside_shape(index: Shape, shape: tuple[int, ...]) -> bool:
    """Checks if index is outside shape.

    Args:
        index: The index to check.
        shape: The shape in which to check the index.

    Returns:
        True if index >= shape, False otherwise."""

    if isinstance(index, numba.int32):  # type: ignore
        return index >= shape[0]  # type: ignore

    if isinstance(index, tuple):
        for i, s in zip(index, shape):
            if i >= s:
                return True

    return False


@cuda.jit
def line_index() -> int:
    # pylint: disable=no-value-for-parameter
    """Returns the index of the current thread in the line of currently running threads.

    Returns:
        The linear thread index."""

    return cuda.grid(1)  # type: ignore


@cuda.jit
def grid_index() -> tuple[int, int]:
    """Returns the index of the current thread in the grid of currently running threads.

    Returns:
        The cartesian thread indices in the x and y directions."""

    return (
        cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x,  # type: ignore
        cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y,  # type: ignore
    )
