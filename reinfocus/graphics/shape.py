"""Defines the possible shapes to render."""

import dataclasses

import numpy

from numpy.typing import NDArray

SPHERE = 0
RECTANGLE = 1


@dataclasses.dataclass
class CpuShape:
    """Convenience class for transferring 'polymorphic' geometry to the GPU.

    Args:
        parameters: The necessary parameters to bounce a ray off the shape.
        shape_type: The 'polymorphic' 'type' of the shape."""

    parameters: NDArray[numpy.float32]
    shape_type: int
