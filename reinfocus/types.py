# pylint: disable=no-member

"""Defines a system of types."""

from typing import NewType, Tuple

import numba as nb
import numpy as np
import numpy.typing as npt
from numba import cuda

C2F = Tuple[float, float]
C3F = Tuple[float, float, float]
G2F = NewType('G2F', cuda.float32x2) # type: ignore
G3F = NewType('G3F', cuda.float32x3) # type: ignore

GpuHitRecord = Tuple[G3F, G3F, nb.float32, G2F, nb.float32] # type: ignore
GpuHitResult = Tuple[bool, GpuHitRecord]

GpuRay = Tuple[G3F, G3F]

GpuColouredRay = Tuple[GpuRay, G3F]

CpuCamera = Tuple[
    C3F,
    C3F,
    C3F,
    C3F,
    C3F,
    C3F,
    C3F,
    float]

GpuCamera = Tuple[
    G3F,
    G3F,
    G3F,
    G3F,
    G3F,
    G3F,
    G3F,
    float]

GpuShapeParameters = npt.NDArray[np.float32]
