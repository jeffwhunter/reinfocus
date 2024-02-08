"""Defines the possible shapes to render."""

import dataclasses

import numpy

from numba.cuda.cudadrv.devicearray import DeviceNDArray
from numpy.typing import NDArray

CpuShapeParameters = NDArray[numpy.float32]
CpuShapeTypes = NDArray[numpy.int32]
GpuShapeParameters = DeviceNDArray
GpuShapeTypes = DeviceNDArray
GpuShapes = tuple[GpuShapeParameters, GpuShapeTypes]

PARAMETERS = 0
TYPES = 1

SPHERE = 0
RECTANGLE = 1


@dataclasses.dataclass
class CpuShape:
    """Convenience class for transferring 'polymorphic' geometry to the GPU.

    Args:
        parameters: The necessary parameters to bounce a ray off the shape.
        type: The 'polymorphic' type of the shape."""

    parameters: CpuShapeParameters
    type: int
