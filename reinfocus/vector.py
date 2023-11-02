"""Methods relating to 3D vectors."""
import math
import numba as nb
import numpy as np

from numba import cuda

R = 0
G = 1
B = 2

X = 0
Y = 1
Z = 2

def cpu_vector(x, y, z):
    """Creates a 3D vector on the CPU."""
    return (np.float32(x), np.float32(y), np.float32(z))

@cuda.jit
def gpu_vector(x, y, z):
    """Creates a 3D vector on the GPU."""
    return (nb.float32(x), nb.float32(y), nb.float32(z))

@cuda.jit
def gpu_add(a, b):
    """Adds two GPU vectors."""
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])

@cuda.jit
def gpu_add3(a, b, c):
    """Adds three GPU vectors."""
    return (a[0] + b[0] + c[0], a[1] + b[1] + c[1], a[2] + b[2] + c[2])

def cpu_add3(a, b, c):
    """Adds three CPU vectors."""
    return (a[0] + b[0] + c[0], a[1] + b[1] + c[1], a[2] + b[2] + c[2])

@cuda.jit
def gpu_neg(v):
    """Negates a GPU vector."""
    return gpu_vector(-v[X], -v[Y], -v[Z])

@cuda.jit
def gpu_sub(lhs, rhs):
    """Subtracts GPU vectors."""
    return (lhs[0] - rhs[0], lhs[1] - rhs[1], lhs[2] - rhs[2])

def cpu_sub(lhs, rhs):
    """Subtracts CPU vectors."""
    return (lhs[0] - rhs[0], lhs[1] - rhs[1], lhs[2] - rhs[2])

@cuda.jit
def gpu_smul(vector, scalar):
    """Multiplies a GPU vector by a scalar."""
    scalar = nb.float32(scalar)
    return (vector[0] * scalar, vector[1] * scalar, vector[2] * scalar)

def cpu_smul(vector, scalar):
    """Multiplies a CPU vector by a scalar."""
    scalar = nb.float32(scalar)
    return (vector[0] * scalar, vector[1] * scalar, vector[2] * scalar)

@cuda.jit
def gpu_vmul(a, b):
    """Returns the Hadamard product of two GPU vectors."""
    return gpu_vector(a[0] * b[0], a[1] * b[1], a[2] * b[2])

@cuda.jit
def gpu_div(vector, scalar):
    """Divides a GPU vector by a scalar."""
    scalar = nb.float32(scalar)
    return (vector[0] / scalar, vector[1] / scalar, vector[2] / scalar)

def cpu_div(vector, scalar):
    """Divides a CPU vector by a scalar."""
    scalar = np.float32(scalar)
    return (vector[0] / scalar, vector[1] / scalar, vector[2] / scalar)

@cuda.jit
def gpu_dot(lhs, rhs):
    """Returns the dot product of two GPU vectors."""
    return lhs[0] * rhs[0] + lhs[1] * rhs[1] + lhs[2] * rhs[2]

@cuda.jit
def gpu_cross(lhs, rhs):
    """Returns the cross product of two GPU vectors."""
    return gpu_vector(
        lhs[1] * rhs[2] - lhs[2] * rhs[1],
        lhs[2] * rhs[0] - lhs[0] * rhs[2],
        lhs[0] * rhs[1] - lhs[1] * rhs[0])

def cpu_cross(lhs, rhs):
    """Returns the cross product of two CPU vectors."""
    return cpu_vector(
        lhs[1] * rhs[2] - lhs[2] * rhs[1],
        lhs[2] * rhs[0] - lhs[0] * rhs[2],
        lhs[0] * rhs[1] - lhs[1] * rhs[0])

@cuda.jit
def gpu_squared_length(vector):
    """Returns the squared length of a GPU vector."""
    return vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2]

@cuda.jit
def gpu_length(vector):
    """Returns the length of a GPU vector."""
    return math.sqrt(gpu_squared_length(vector))

def cpu_length(vector):
    """Returns the length of a CPU vector."""
    return math.sqrt(vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2])

@cuda.jit
def gpu_norm_vector(vector):
    """Normalizes a GPU vector."""
    return gpu_div(vector, gpu_length(vector))

def cpu_norm_vector(vector):
    """Normalizes a CPU vector."""
    return cpu_div(vector, cpu_length(vector))