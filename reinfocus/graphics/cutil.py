"""Methods that reduce the number of disabled lint rules involved in numba cuda code."""

from typing import Callable, TypeVar

from numba import cuda

CallableT = TypeVar("CallableT", bound=Callable)


def launcher(kernel: CallableT, block_size: tuple[int, int]) -> CallableT:
    """Returns a callable that invokes kernel on the gpu with a block of threads of size
    block_size.

    Args:
        kernel: The numba cuda kernel to invoke on the device.
        block_size: A tuple defining the blocks per grid, and threads per block.

    Returns:
        A callable that invokes kernel on the gpu."""

    return kernel[*block_size]  # type: ignore


@cuda.jit
def line_index() -> int:
    # pylint: disable=no-value-for-parameter
    """Returns the index of the current thread in the line of currently running threads.

    Returns:
        The linear thread index."""

    return cuda.grid(1)  # type: ignore
