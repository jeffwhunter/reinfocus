# pylint: disable=no-member
"""Methods relating to [2|3]D vectors."""

import math

from numba import cuda

C2F = tuple[float, float]
C3F = tuple[float, float, float]
G2F = cuda.float32x2  # type: ignore
G3F = cuda.float32x3  # type: ignore


def c2f(x: float, y: float) -> C2F:
    """Makes a 2D vector on the CPU.

    Args:
        x: The x coordinate of the vector.
        y: The y coordinate of the vector.

    Returns:
        The 2D CPU vector."""

    return (x, y)


@cuda.jit
def empty_g2f() -> G2F:
    """Makes an empty 2D vector on the GPU.

    Returns:
        The empty 2D GPU vector."""

    return cuda.float32x2(0, 0)  # type: ignore


@cuda.jit
def g2f(x: float, y: float) -> G2F:
    """Makes a 2D vector on the GPU.

    Args:
        x: The x coordinate of the vector.
        y: The y coordinate of the vector.

    Returns:
        The 2D GPU vector."""

    return cuda.float32x2(x, y)  # type: ignore


def c3f(x: float, y: float, z: float) -> C3F:
    """Makes a 3D vector on the CPU.

    Args:
        x: The x coordinate of the vector.
        y: The y coordinate of the vector.
        z: The z coordinate of the vector.

    Returns:
        The 3D CPU vector."""

    return (x, y, z)


@cuda.jit
def empty_g3f() -> G3F:
    """Makes an empty 3D vector on the GPU.

    Returns:
        The empty 3D GPU vector."""

    return cuda.float32x3(0, 0, 0)  # type: ignore


@cuda.jit
def g3f(x: float, y: float, z: float) -> G3F:
    """Makes a 3D vector on the GPU.

    Args:
        x: The x coordinate of the vector.
        y: The y coordinate of the vector.
        z: The z coordinate of the vector.

    Returns:
        The 3D GPU vector."""

    return cuda.float32x3(x, y, z)  # type: ignore


@cuda.jit
def g2f_to_c2f(vector: G2F) -> C2F:
    """Converts a 2D vector from GPU to CPU.

    Args:
        vector: The vector to convert.

    Returns:
        The newly converted C2F."""

    return (vector.x, vector.y)


@cuda.jit
def c2f_to_g2f(vector: C2F) -> G2F:
    """Converts a 2D vector from CPU to GPU.

    Args:
        vector: The vector to convert.

    Returns:
        The newly converted G2F."""

    return cuda.float32x2(vector[0], vector[1])  # type: ignore


@cuda.jit
def g3f_to_c3f(vector: G3F) -> C3F:
    """Converts a 3D vector from GPU to CPU.

    Args:
        vector: The vector to convert.

    Returns:
        The newly converted C3F."""

    return (vector.x, vector.y, vector.z)


@cuda.jit
def c3f_to_g3f(vector: C3F) -> G3F:
    """Converts a 3D vector from CPU to GPU.

    Args:
        vector: The vector to convert.

    Returns:
        The newly converted G3F."""

    return cuda.float32x3(vector[0], vector[1], vector[2])  # type: ignore


@cuda.jit
def add_g3f(a: G3F, b: G3F) -> G3F:
    """Adds two 3D GPU vectors.

    Args:
        a: The left vector.
        b: The right vector.

    Returns:
        The sum of a and b."""

    return cuda.float32x3(a.x + b.x, a.y + b.y, a.z + b.z)  # type: ignore


def add3_c3f(a: C3F, b: C3F, c: C3F) -> C3F:
    """Adds three 3D CPU vectors.

    Args:
        a: The left vector.
        b: The middle vector.
        c: The right vector.

    Returns:
        The sum of a, b, and c."""

    return (a[0] + b[0] + c[0], a[1] + b[1] + c[1], a[2] + b[2] + c[2])


@cuda.jit
def add3_g3f(a: G3F, b: G3F, c: G3F) -> G3F:
    """Adds three 3D GPU vectors.

    Args:
        a: The left vector.
        b: The middle vector.
        c: The right vector.

    Returns:
        The sum of a, b, and c."""

    return cuda.float32x3(  # type: ignore
        a.x + b.x + c.x,
        a.y + b.y + c.y,
        a.z + b.z + c.z,
    )


@cuda.jit
def neg_g3f(v: G3F) -> G3F:
    """Negates a 3D GPU vector.

    Args:
        v: The vector to negate.

    Returns:
        The negation of v."""

    return cuda.float32x3(-v.x, -v.y, -v.z)  # type: ignore


@cuda.jit
def sub_g2f(a: G3F, b: G3F) -> G3F:
    """Subtracts 2D GPU vectors.

    Args:
        a: The vector to subtract from.
        b: The vector to subtract.

    Returns:
        The vector a - b."""

    return cuda.float32x2(a.x - b.x, a.y - b.y)  # type: ignore


def sub_c3f(a: C3F, b: C3F) -> C3F:
    """Subtracts 3D CPU vectors.

    Args:
        a: The vector to subtract from.
        b: The vector to subtract.

    Returns:
        The vector a - b."""

    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


@cuda.jit
def sub_g3f(a: G3F, b: G3F) -> G3F:
    """Subtracts 3D GPU vectors.

    Args:
        a: The vector to subtract from.
        b: The vector to subtract.

    Returns:
        The vector a - b."""

    return cuda.float32x3(a.x - b.x, a.y - b.y, a.z - b.z)  # type: ignore


@cuda.jit
def smul_g2f(vector: G2F, scalar: float) -> G2F:
    """Multiplies a 2D GPU vector by a scalar.

    Args:
        vector: The vector to multiply.
        scalar: The scalar to multiply by.

    Returns:
        A copy of vector scaled by scalar."""

    return cuda.float32x2(vector.x * scalar, vector.y * scalar)  # type: ignore


def smul_c3f(vector: C3F, scalar: float) -> C3F:
    """Multiplies a 3D CPU vector by a scalar.

    Args:
        vector: The vector to multiply.
        scalar: The scalar to multiply by.

    Returns:
        A copy of vector scaled by scalar."""

    return (vector[0] * scalar, vector[1] * scalar, vector[2] * scalar)


@cuda.jit
def smul_g3f(vector: G3F, scalar: float) -> G3F:
    """Multiplies a 3DGPU vector by a scalar.

    Args:
        vector: The vector to multiply.
        scalar: The scalar to multiply by.

    Returns:
        A copy of vector scaled by scalar."""

    return cuda.float32x3(  # type: ignore
        vector.x * scalar,
        vector.y * scalar,
        vector.z * scalar,
    )


@cuda.jit
def vmul_g3f(a: G3F, b: G3F) -> G3F:
    """Returns the Hadamard product of two 3D GPU vectors.

    Args:
        a: The left vector to multiply.
        b: The right vector to multiply.

    Returns:
        The Hadamard product of a and b."""

    return cuda.float32x3(a.x * b.x, a.y * b.y, a.z * b.z)  # type: ignore


def div_c3f(vector: C3F, scalar: float) -> C3F:
    """Divides a 3D CPU vector by a scalar.

    Args:
        vector: The vector to divide.
        scalar: The scalar to divide by.

    Returns:
        The quotient of a divided by b."""

    return (vector[0] / scalar, vector[1] / scalar, vector[2] / scalar)


@cuda.jit
def div_g3f(vector: G3F, scalar: float) -> G3F:
    """Divides a 3D GPU vector by a scalar.

    Args:
        vector: The vector to divide.
        scalar: The scalar to divide by.

    Returns:
        The quotient of a divided by b."""

    return cuda.float32x3(  # type: ignore
        vector.x / scalar,
        vector.y / scalar,
        vector.z / scalar,
    )


@cuda.jit
def dot_g2f(a: G2F, b: G2F) -> float:
    """Returns the dot product of two 2D GPU vectors.

    Args:
        a: The left vector to dot.
        b: The right vector to dot.

    Returns:
        The dot product of a and b."""

    return a.x * b.x + a.y * b.y


@cuda.jit
def dot_g3f(a: G3F, b: G3F) -> float:
    """Returns the dot product of two 3D GPU vectors.

    Args:
        a: The left vector to dot.
        b: The right vector to dot.

    Returns:
        The dot product of a and b."""

    return a.x * b.x + a.y * b.y + a.z * b.z


def cross_c3f(a: C3F, b: C3F) -> C3F:
    """Returns the cross product of two 3D CPU vectors.

    Args:
        a: The left vector to cross.
        b: The right vector to cross.

    Returns:
        The cross product of a and b."""

    return c3f(
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


@cuda.jit
def cross_g3f(a: G3F, b: G3F) -> G3F:
    """Returns the cross product of two 3D GPU vectors.

    Args:
        a: The left vector to cross.
        b: The right vector to cross.

    Returns:
        The cross product of a and b."""

    return cuda.float32x3(  # type: ignore
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x,
    )


@cuda.jit
def squared_length_g3f(vector: G3F) -> float:
    """Returns the squared length of a 3D GPU vector.

    Args:
        vector: The vector to measure.

    Returns:
        The squared length of vector."""

    return vector.x * vector.x + vector.y * vector.y + vector.z * vector.z


def length_c3f(vector: C3F) -> float:
    """Returns the length of a 3D CPU vector.

    Args:
        vector: The vector to measure.

    Returns:
        The length of vector."""

    return math.sqrt(
        vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2]
    )


@cuda.jit
def length_g3f(vector: G3F) -> float:
    """Returns the length of a 3D GPU vector.

    Args:
        vector: The vector to measure.

    Returns:
        The length of vector."""

    return math.sqrt(squared_length_g3f(vector))


def norm_c3f(vector: C3F) -> C3F:
    """Normalizes a 3D CPU vector.

    Args:
        vector: The vector to normalize.

    Returns:
        The normalized vector."""

    return div_c3f(vector, length_c3f(vector))


@cuda.jit
def norm_g3f(vector: G3F) -> G3F:
    """Normalizes a 3D GPU vector.

    Args:
        vector: The vector to normalize.

    Returns:
        The normalized vector."""

    return div_g3f(vector, length_g3f(vector))
