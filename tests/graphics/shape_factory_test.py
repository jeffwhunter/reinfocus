"""Contains tests for reinfocus.graphics.shape_factory."""

from collections.abc import Sequence

from numba.cuda.testing import unittest
from numpy import testing

from reinfocus.graphics import shape
from reinfocus.graphics import shape_factory


def _types(cpu_shapes: Sequence[shape.CpuShape]) -> Sequence[int]:
    """Extracts the shape types from a colleciton of shapes."""

    return [cpu_shape.shape_type for cpu_shape in cpu_shapes]


class OneSphereTest(unittest.TestCase):
    """Test cases for reinfocus.graphics.shape_factory.one_sphere."""

    def test_types(self):
        """Tests that one_sphere creates one sphere."""

        testing.assert_allclose(_types(shape_factory.one_sphere()), [shape.SPHERE])


class TwoSphereTest(unittest.TestCase):
    """Test cases for reinfocus.graphics.shape_factory.two_sphere."""

    def test_types(self):
        """Tests that two_sphere creates two spheres."""

        testing.assert_allclose(_types(shape_factory.two_sphere()), [shape.SPHERE] * 2)


class OneRectTest(unittest.TestCase):
    """Test cases for reinfocus.graphics.shape_factory.one_rect."""

    def test_types(self):
        """Tests that one_rect creates one rectangle."""

        testing.assert_allclose(_types(shape_factory.one_rect()), [shape.RECTANGLE])


class TwoRectTest(unittest.TestCase):
    """Test cases for reinfocus.graphics.shape_factory.two_rect."""

    def test_types(self):
        """Tests that two_rect creates two rectangles."""

        testing.assert_allclose(_types(shape_factory.two_rect()), [shape.RECTANGLE] * 2)


class MixedTest(unittest.TestCase):
    """Test cases for reinfocus.graphics.shape_factory.mixed."""

    def test_types(self):
        """Tests that mixed creates one rectangle and one sphere."""

        types = set(_types(shape_factory.mixed()))

        self.assertLessEqual(types, set([shape.RECTANGLE, shape.SPHERE]))
        self.assertIs(len(types), 2)


if __name__ == "__main__":
    unittest.main()
