# pylint: disable=no-member

"""Defines a system of types."""

from typing import NewType, Tuple

import numba as nb
from numba import cuda

C2F = Tuple[float, float]
C3F = Tuple[float, float, float]
G2F = NewType('G2F', cuda.float32x2) # type: ignore
G3F = NewType('G3F', cuda.float32x3) # type: ignore

GpuHitRecord = Tuple[G3F, G3F, nb.float32, G2F, nb.float32] # type: ignore
GpuRay = Tuple[G3F, G3F]


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
