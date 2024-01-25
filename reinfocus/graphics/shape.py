"""Defines the possible shapes to render."""

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import numpy.typing as npt
from numba.cuda.cudadrv import devicearray as cda
from reinfocus.graphics import hit_record as hit

GpuHitResult = Tuple[bool, hit.GpuHitRecord]

CpuShapeParameters = npt.NDArray[np.float32]
CpuShapeTypes = npt.NDArray[np.int32]
GpuShapeParameters = cda.DeviceNDArray
GpuShapeTypes = cda.DeviceNDArray
GpuShapes = Tuple[GpuShapeParameters, GpuShapeTypes]

PARAMETERS = 0
TYPES = 1

SPHERE = 0
RECTANGLE = 1


@dataclass
class CpuShape:
    """Convenience class for transferring 'polymorphic' geometry to the GPU.

    Args:
        parameters: The necessary parameters to bounce a ray off the shape.
        type: The 'polymorphic' type of the shape."""

    parameters: CpuShapeParameters
    type: int
