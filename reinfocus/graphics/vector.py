"""Methods relating to [2|3]D vectors."""

import math

import numpy

from numba import cuda
from numpy import linalg

V2F = tuple[numpy.float32, numpy.float32]
V3F = tuple[numpy.float32, numpy.float32, numpy.float32]
V3UI = tuple[numpy.uint8, numpy.uint8, numpy.uint8]


def v2f(x: float = 0.0, y: float = 0.0) -> V2F:
    """Makes a 2D float vector.

    Args:
        x: The x coordinate of the vector.
        y: The y coordinate of the vector.

    Returns:
        The 2D float vector."""

    return (numpy.float32(x), numpy.float32(y))


@cuda.jit
def d_v2f(x: float, y: float) -> V2F:
    """Makes a 2D float vector on the device.

    Args:
        x: The x coordinate of the vector.
        y: The y coordinate of the vector.

    Returns:
        The 2D float vector."""

    return (numpy.float32(x), numpy.float32(y))


def v3f(x: float = 0.0, y: float = 0.0, z: float = 0.0) -> V3F:
    """Makes a 3D float vector.

    Args:
        x: The x coordinate of the vector.
        y: The y coordinate of the vector.
        z: The z coordinate of the vector.

    Returns:
        The 3D float vector."""

    return (numpy.float32(x), numpy.float32(y), numpy.float32(z))


@cuda.jit
def d_v3f(x: float, y: float, z: float) -> V3F:
    """Makes a 3D float vector on the device.

    Args:
        x: The x coordinate of the vector.
        y: The y coordinate of the vector.
        z: The z coordinate of the vector.

    Returns:
        The 3D float vector."""

    return (numpy.float32(x), numpy.float32(y), numpy.float32(z))


@cuda.jit
def d_v3f_to_v3ui(v: V3F) -> V3UI:
    """Casts a 3D float vector to an unsigned int vector on the device.

    Args:
        v: The vector to cast to uints.

    Returns:
        A 3D unsigned int vector."""

    return (numpy.uint8(v[0]), numpy.uint8(v[1]), numpy.uint8(v[2]))


def add_v3f(summands: tuple[V3F, ...]) -> V3F:
    """Adds an arbitrary number of 3D float vectors.

    Args:
        summands: The vectors to add.

    Returns:
        The sum of summands."""

    s = numpy.sum(summands, axis=0)

    return (s[0], s[1], s[2])


@cuda.jit
def d_add_v3f(summands: tuple[V3F, ...]) -> V3F:
    """Adds an arbitrary number of 3D float vectors on the device.

    Args:
        summands: The vectors to add.

    Returns:
        The sum of summands."""

    x = y = z = numpy.float32(0.0)

    for summand in summands:
        x += summand[0]  # type: ignore
        y += summand[1]  # type: ignore
        z += summand[2]  # type: ignore

    return (x, y, z)


@cuda.jit
def d_sub_v2f(a: V2F, b: V2F) -> V2F:
    """Subtracts 2D float vectors on the device.

    Args:
        a: The vector to subtract from.
        b: The vector to subtract.

    Returns:
        The vector a - b."""

    return (a[0] - b[0], a[1] - b[1])  # type: ignore


def sub_v3f(a: V3F, b: V3F) -> V3F:
    """Subtracts 3D float vectors.

    Args:
        a: The vector to subtract from.
        b: The vector to subtract.

    Returns:
        The vector a - b."""

    r = numpy.subtract(a, b)

    return (r[0], r[1], r[2])


@cuda.jit
def d_sub_v3f(a: V3F, b: V3F) -> V3F:
    """Subtracts 3D float vectors on the device.

    Args:
        a: The vector to subtract from.
        b: The vector to subtract.

    Returns:
        The vector a - b."""

    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])  # type: ignore


@cuda.jit
def d_smul_v2f(v: V2F, s: numpy.float32) -> V2F:
    """Multiplies a 2D float vector by a scalar on the device.

    Args:
        v: The vector to multiply.
        s: The scalar to multiply by.

    Returns:
        The vector v * s."""

    return (v[0] * s, v[1] * s)  # type: ignore


def smul_v3f(v: V3F, s: float) -> V3F:
    """Multiplies a 3D float vector by a scalar.

    Args:
        v: The vector to multiply.
        s: The scalar to multiply by.

    Returns:
        The vector v * s."""

    r = numpy.multiply(v, s)

    return (r[0], r[1], r[2])


@cuda.jit
def d_smul_v3f(v: V3F, s: numpy.float32) -> V3F:
    """Multiplies a 3D float vector by a scalar on the device.

    Args:
        v: The vector to multiply.
        s: The scalar to multiply by.

    Returns:
        The vector v * s."""

    return (
        numpy.float32(v[0] * s),  # type: ignore
        numpy.float32(v[1] * s),  # type: ignore
        numpy.float32(v[2] * s),  # type: ignore
    )


@cuda.jit
def d_vmul_v3f(a: V3F, b: V3F) -> V3F:
    """Returns the Hadamard product of two 3D float vectors on the device.

    Args:
        a: The left vector to multiply.
        b: The right vector to multiply.

    Returns:
        The Hadamard product of a and b."""

    return (a[0] * b[0], a[1] * b[1], a[2] * b[2])  # type: ignore


@cuda.jit
def d_dot_v2f(a: V2F, b: V2F) -> numpy.float32:
    """Returns the dot product of two 2D float vectors on the device.

    Args:
        a: The left vector to dot.
        b: The right vector to dot.

    Returns:
        The dot product of a and b."""

    return a[0] * b[0] + a[1] * b[1]  # type: ignore


@cuda.jit
def d_dot_v3f(a: V3F, b: V3F) -> numpy.float32:
    """Returns the dot product of two 3D float vectors on the device.

    Args:
        a: The left vector to dot.
        b: The right vector to dot.

    Returns:
        The dot product of a and b."""

    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]  # type: ignore


def cross_v3f(a: V3F, b: V3F) -> V3F:
    """Returns the cross product of two 3D float vectors.

    Args:
        a: The left vector to cross.
        b: The right vector to cross.

    Returns:
        The cross product of a and b."""

    c = tuple(numpy.cross(numpy.asarray(a), numpy.asarray(b)))
    return (c[0], c[1], c[2])


@cuda.jit
def d_cross_v3f(a: V3F, b: V3F) -> V3F:
    """Returns the cross product of two 3D float vectors on the device.

    Args:
        a: The left vector to cross.
        b: The right vector to cross.

    Returns:
        The cross product of a and b."""

    return (
        a[1] * b[2] - a[2] * b[1],  # type: ignore
        a[2] * b[0] - a[0] * b[2],  # type: ignore
        a[0] * b[1] - a[1] * b[0],  # type: ignore
    )


@cuda.jit
def d_squared_length_v3f(vector: V3F) -> numpy.float32:
    """Returns the squared length of a 3D float vector on the device.

    Args:
        vector: The vector to measure.

    Returns:
        The squared length of vector."""

    return (
        numpy.float32(vector[0] ** 2)  # type: ignore
        + numpy.float32(vector[1] ** 2)  # type: ignore
        + numpy.float32(vector[2] ** 2)  # type: ignore
    )


def length_v3f(vector: V3F) -> float:
    """Returns the length of a 3D float vector.

    Args:
        vector: The vector to measure.

    Returns:
        The length of vector."""

    return float(linalg.norm(numpy.asarray(vector)))


@cuda.jit
def d_length_v3f(vector: V3F) -> numpy.float32:
    """Returns the length of a 3D float vector on the device.

    Args:
        vector: The vector to measure.

    Returns:
        The length of vector."""

    return numpy.float32(math.sqrt(d_squared_length_v3f(vector)))


def norm_v3f(vector: V3F) -> V3F:
    """Normalizes a 3D float vector.

    Args:
        vector: The vector to normalize.

    Returns:
        The normalized vector."""

    return smul_v3f(vector, 1.0 / length_v3f(vector))


@cuda.jit
def d_norm_v3f(vector: V3F) -> V3F:
    """Normalizes a 3D float vector on the device.

    Args:
        vector: The vector to normalize.

    Returns:
        The normalized vector."""

    return d_smul_v3f(vector, numpy.float32(1.0) / d_length_v3f(vector))  # type: ignore
