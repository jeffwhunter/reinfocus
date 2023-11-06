# pylint: disable=no-member
# type: ignore

"""Methods relating to 3D vectors."""

import math
import numba as nb
import numpy as np

from numba import cuda

def cpu_vector(x, y, z):
    """Makes a 3D vector on the CPU."""
    return (np.float32(x), np.float32(y), np.float32(z))

@cuda.jit
def gpu_vector(x, y, z):
    """Makes a 3D vector on the GPU."""
    return cuda.float32x3(x, y, z)

@cuda.jit
def to_cpu_vector(vector):
    """Converts a 3D vector from GPU to CPU."""
    return (np.float32(vector.x), np.float32(vector.y), np.float32(vector.z))

@cuda.jit
def to_gpu_vector(vector):
    """Converts a 3D vector from CPU to GPU."""
    return cuda.float32x3(vector[0], vector[1], vector[2])

@cuda.jit
def gpu_add(a, b):
    """Adds two GPU vectors."""
    return cuda.float32x3(a.x + b.x, a.y + b.y, a.z + b.z)

def cpu_add3(a, b, c):
    """Adds three CPU vectors."""
    return (a[0] + b[0] + c[0], a[1] + b[1] + c[1], a[2] + b[2] + c[2])

@cuda.jit
def gpu_add3(a, b, c):
    """Adds three GPU vectors."""
    return cuda.float32x3(a.x + b.x + c.x, a.y + b.y + c.y, a.z + b.z + c.z)

@cuda.jit
def gpu_neg(v):
    """Negates a GPU vector."""
    return cuda.float32x3(-v.x, -v.y, -v.z)

def cpu_sub(lhs, rhs):
    """Subtracts CPU vectors."""
    return (lhs[0] - rhs[0], lhs[1] - rhs[1], lhs[2] - rhs[2])

@cuda.jit
def gpu_sub(lhs, rhs):
    """Subtracts GPU vectors."""
    return cuda.float32x3(lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z)

def cpu_smul(vector, scalar):
    """Multiplies a CPU vector by a scalar."""
    scalar = nb.float32(scalar)
    return (vector[0] * scalar, vector[1] * scalar, vector[2] * scalar)

@cuda.jit
def gpu_smul(vector, scalar):
    """Multiplies a GPU vector by a scalar."""
    scalar = nb.float32(scalar)
    return cuda.float32x3(vector.x * scalar, vector.y * scalar, vector.z * scalar)

@cuda.jit
def gpu_vmul(a, b):
    """Returns the Hadamard product of two GPU vectors."""
    return cuda.float32x3(a.x * b.x, a.y * b.y, a.z * b.z)

def cpu_div(vector, scalar):
    """Divides a CPU vector by a scalar."""
    scalar = np.float32(scalar)
    return (vector[0] / scalar, vector[1] / scalar, vector[2] / scalar)

@cuda.jit
def gpu_div(vector, scalar):
    """Divides a GPU vector by a scalar."""
    scalar = nb.float32(scalar)
    return cuda.float32x3(vector.x / scalar, vector.y / scalar, vector.z / scalar)

@cuda.jit
def gpu_dot(lhs, rhs):
    """Returns the dot product of two GPU vectors."""
    return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z

def cpu_cross(lhs, rhs):
    """Returns the cross product of two CPU vectors."""
    return cpu_vector(
        lhs[1] * rhs[2] - lhs[2] * rhs[1],
        lhs[2] * rhs[0] - lhs[0] * rhs[2],
        lhs[0] * rhs[1] - lhs[1] * rhs[0])

@cuda.jit
def gpu_cross(lhs, rhs):
    """Returns the cross product of two GPU vectors."""
    return cuda.float32x3(
        lhs.y * rhs.z - lhs.z * rhs.y,
        lhs.z * rhs.x - lhs.x * rhs.z,
        lhs.x * rhs.y - lhs.y * rhs.x)

@cuda.jit
def gpu_squared_length(vector):
    """Returns the squared length of a GPU vector."""
    return vector.x * vector.x + vector.y * vector.y + vector.z * vector.z

def cpu_length(vector):
    """Returns the length of a CPU vector."""
    return math.sqrt(vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2])

@cuda.jit
def gpu_length(vector):
    """Returns the length of a GPU vector."""
    return math.sqrt(gpu_squared_length(vector))

def cpu_norm_vector(vector):
    """Normalizes a CPU vector."""
    return cpu_div(vector, cpu_length(vector))

@cuda.jit
def gpu_norm_vector(vector):
    """Normalizes a GPU vector."""
    return gpu_div(vector, gpu_length(vector))
