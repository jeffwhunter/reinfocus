# pylint: disable=no-member

"""Methods relating to [2|3]D vectors."""

import math

from typing import Tuple

from numba import cuda

C2F = Tuple[float, float]
C3F = Tuple[float, float, float]
G2F = cuda.float32x2 # type: ignore
G3F = cuda.float32x3 # type: ignore

def c2f(x: float, y: float) -> C2F:
    """Makes a 2D vector on the CPU."""
    return (x, y)

@cuda.jit
def empty_g2f() -> G2F:
    """Makes an empty 2D vector on the GPU."""
    return cuda.float32x2(0, 0) # type: ignore

@cuda.jit
def g2f(x: float, y: float) -> G2F:
    """Makes a 2D vector on the GPU."""
    return cuda.float32x2(x, y) # type: ignore

def c3f(x: float, y: float, z: float) -> C3F:
    """Makes a 3D vector on the CPU."""
    return (x, y, z)

@cuda.jit
def empty_g3f() -> G3F:
    """Makes an empty 3D vector on the GPU."""
    return cuda.float32x3(0, 0, 0) # type: ignore

@cuda.jit
def g3f(x: float, y: float, z: float) -> G3F:
    """Makes a 3D vector on the GPU."""
    return cuda.float32x3(x, y, z) # type: ignore

@cuda.jit
def g2f_to_c2f(vector: G2F) -> C2F:
    """Converts a 2D vector from GPU to CPU."""
    return (vector.x, vector.y)

@cuda.jit
def c2f_to_g2f(vector: C2F) -> G2F:
    """Converts a 2D vector from CPU to GPU."""
    return cuda.float32x2(vector[0], vector[1]) # type: ignore

@cuda.jit
def g3f_to_c3f(vector: G3F) -> C3F:
    """Converts a 3D vector from GPU to CPU."""
    return (vector.x, vector.y, vector.z)

@cuda.jit
def c3f_to_g3f(vector: C3F) -> G3F:
    """Converts a 3D vector from CPU to GPU."""
    return cuda.float32x3(vector[0], vector[1], vector[2]) # type: ignore

@cuda.jit
def add_g3f(a: G3F, b: G3F) -> G3F:
    """Adds two 3D GPU vectors."""
    return cuda.float32x3(a.x + b.x, a.y + b.y, a.z + b.z) # type: ignore

def add3_c3f(a: C3F, b: C3F, c: C3F) -> C3F:
    """Adds three 3D CPU vectors."""
    return (a[0] + b[0] + c[0], a[1] + b[1] + c[1], a[2] + b[2] + c[2])

@cuda.jit
def add3_g3f(a: G3F, b: G3F, c: G3F) -> G3F:
    """Adds three 3D GPU vectors."""
    return cuda.float32x3(a.x + b.x + c.x, a.y + b.y + c.y, a.z + b.z + c.z) # type: ignore

@cuda.jit
def neg_g3f(v: G3F) -> G3F:
    """Negates a 3D GPU vector."""
    return cuda.float32x3(-v.x, -v.y, -v.z) # type: ignore

@cuda.jit
def sub_g2f(lhs: G3F, rhs: G3F) -> G3F:
    """Subtracts 2D GPU vectors."""
    return cuda.float32x2(lhs.x - rhs.x, lhs.y - rhs.y) # type: ignore

def sub_c3f(lhs: C3F, rhs: C3F) -> C3F:
    """Subtracts 3D CPU vectors."""
    return (lhs[0] - rhs[0], lhs[1] - rhs[1], lhs[2] - rhs[2])

@cuda.jit
def sub_g3f(lhs: G3F, rhs: G3F) -> G3F:
    """Subtracts 3D GPU vectors."""
    return cuda.float32x3(lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z) # type: ignore

@cuda.jit
def smul_g2f(vector: G2F, scalar: float) -> G2F:
    """Multiplies a 2D GPU vector by a scalar."""
    return cuda.float32x2(vector.x * scalar, vector.y * scalar) # type: ignore

def smul_c3f(vector: C3F, scalar: float) -> C3F:
    """Multiplies a 3D CPU vector by a scalar."""
    return (vector[0] * scalar, vector[1] * scalar, vector[2] * scalar)

@cuda.jit
def smul_g3f(vector: G3F, scalar: float) -> G3F:
    """Multiplies a 3DGPU vector by a scalar."""
    return cuda.float32x3(vector.x * scalar, vector.y * scalar, vector.z * scalar) # type: ignore

@cuda.jit
def vmul_g3f(a: G3F, b: G3F) -> G3F:
    """Returns the Hadamard product of two 3D GPU vectors."""
    return cuda.float32x3(a.x * b.x, a.y * b.y, a.z * b.z) # type: ignore

def div_c3f(vector: C3F, scalar: float) -> C3F:
    """Divides a 3D CPU vector by a scalar."""
    return (vector[0] / scalar, vector[1] / scalar, vector[2] / scalar)

@cuda.jit
def div_g3f(vector: G3F, scalar: float) -> G3F:
    """Divides a 3D GPU vector by a scalar."""
    return cuda.float32x3(vector.x / scalar, vector.y / scalar, vector.z / scalar) # type: ignore

@cuda.jit
def dot_g2f(lhs: G2F, rhs: G2F) -> float:
    """Returns the dot product of two 2D GPU vectors."""
    return lhs.x * rhs.x + lhs.y * rhs.y

@cuda.jit
def dot_g3f(lhs: G3F, rhs: G3F) -> float:
    """Returns the dot product of two 3D GPU vectors."""
    return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z

def cross_c3f(lhs: C3F, rhs: C3F) -> C3F:
    """Returns the cross product of two 3D CPU vectors."""
    return c3f(
        lhs[1] * rhs[2] - lhs[2] * rhs[1],
        lhs[2] * rhs[0] - lhs[0] * rhs[2],
        lhs[0] * rhs[1] - lhs[1] * rhs[0])

@cuda.jit
def cross_g3f(lhs: G3F, rhs: G3F) -> G3F:
    """Returns the cross product of two 3D GPU vectors."""
    return cuda.float32x3( # type: ignore
        lhs.y * rhs.z - lhs.z * rhs.y,
        lhs.z * rhs.x - lhs.x * rhs.z,
        lhs.x * rhs.y - lhs.y * rhs.x)

@cuda.jit
def squared_length_g3f(vector: G3F) -> float:
    """Returns the squared length of a 3D GPU vector."""
    return vector.x * vector.x + vector.y * vector.y + vector.z * vector.z

def length_c3f(vector: C3F) -> float:
    """Returns the length of a 3D CPU vector."""
    return math.sqrt(vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2])

@cuda.jit
def length_g3f(vector: G3F) -> float:
    """Returns the length of a 3D GPU vector."""
    return math.sqrt(squared_length_g3f(vector))

def norm_c3f(vector: C3F) -> C3F:
    """Normalizes a 3D CPU vector."""
    return div_c3f(vector, length_c3f(vector))

@cuda.jit
def norm_g3f(vector: G3F) -> G3F:
    """Normalizes a 3D GPU vector."""
    return div_g3f(vector, length_g3f(vector))
