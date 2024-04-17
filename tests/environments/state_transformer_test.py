"""Contains tests for reinfocus.environments.state_transformer."""

import unittest

import numpy

from numpy import testing

from reinfocus.environments import state_transformer


class ContinuousJumpTransformerTest(unittest.TestCase):
    """TestCase for reinfocus.environments.state_transformer.ContinuousJumpTransformer."""

    def test_transform(self):
        """Tests that ContinuousJumpTransformer properly jumps elements of the state."""

        state = numpy.array([[10, 5], [7.5, 5], [5, 5]])

        transform_left = state_transformer.ContinuousJumpTransformer(3, 0, (5, 10))

        testing.assert_allclose(
            transform_left.transform(state, numpy.array([[-1], [0], [1]])),
            [[5, 5], [7.5, 5], [10, 5]],
        )

        transform_right = state_transformer.ContinuousJumpTransformer(3, 1, (5, 10))

        testing.assert_allclose(
            transform_right.transform(state, numpy.array([[1], [0], [-1]])),
            [[10, 10], [7.5, 7.5], [5, 5]],
        )


class ContinuousMoveTransformerTest(unittest.TestCase):
    """TestCase for reinfocus.environments.state_transformer.ContinuousMoveTransformer."""

    def test_transform(self):
        """Tests that ContinuousMoveTransformer properly moves elements of the state."""

        state = numpy.array([[1, 0], [1, 0.5], [1, 1]])

        fast = state_transformer.ContinuousMoveTransformer(3, 1, (0, 1), 1)

        testing.assert_allclose(
            fast.transform(state, numpy.array([[-1], [-1], [-1]])),
            [[1, 0], [1, 0], [1, 0]],
        )
        testing.assert_allclose(
            fast.transform(state, numpy.array([[0.5], [0], [-0.5]])),
            [[1, 0.5], [1, 0.5], [1, 0.5]],
        )
        testing.assert_allclose(
            fast.transform(state, numpy.array([1, 1, 1])), [[1, 1], [1, 1], [1, 1]]
        )
        testing.assert_allclose(
            fast.transform(state, numpy.array([0, 0.1, 0])), [[1, 0], [1, 0.6], [1, 1]]
        )

        num_envs = 7

        testing.assert_allclose(
            state_transformer.ContinuousMoveTransformer(
                num_envs, 0, (0, 1), 0.1
            ).transform(
                numpy.tile([0.5, 1], (num_envs, 1)),
                numpy.array([-2, -1, -0.1, 0, 0.1, 1, 2]),
            ),
            [[0.4, 1], [0.4, 1], [0.49, 1], [0.5, 1], [0.51, 1], [0.6, 1], [0.6, 1]],
        )


class DiscreteJumpTransformerTest(unittest.TestCase):
    """TestCase for reinfocus.environments.state_transformer.DiscreteJumpTransformer."""

    def test_transform(self):
        """Tests that DiscreteJumpTransformer properly jumps elements of the state."""

        state = numpy.array([[5, 4], [4.5, 4], [4, 4]])

        transform_left = state_transformer.DiscreteJumpTransformer(
            3, 0, (4, 5), numpy.linspace(4, 5, 5)
        )

        testing.assert_allclose(
            transform_left.transform(state, numpy.array([[0], [1], [2]])),
            [[4, 4], [4.25, 4], [4.5, 4]],
        )

        transform_right = state_transformer.DiscreteJumpTransformer(
            3, 1, (4, 5), numpy.linspace(4, 5, 5)
        )

        testing.assert_allclose(
            transform_right.transform(state, numpy.array([[2], [3], [4]])),
            [[5, 4.5], [4.5, 4.75], [4, 5]],
        )


class DiscreteMoveTransformerTest(unittest.TestCase):
    """TestCase for reinfocus.environments.state_transformer.DiscreteMoveTransformer."""

    def test_transform(self):
        """Tests that DiscreteMoveTransformer properly moves elements of the state."""

        state = numpy.array([[1, 0], [1, 0.5], [1, 1]])

        transform_left = state_transformer.DiscreteMoveTransformer(
            3, 0, (0, 1), [-0.5, 0, 0.5]
        )

        testing.assert_allclose(
            transform_left.transform(state, numpy.array([[0], [1], [2]])),
            [[0.5, 0], [1, 0.5], [1, 1]],
        )

        testing.assert_allclose(
            transform_left.transform(state, numpy.array([[2], [1], [0]])),
            [[1, 0], [1, 0.5], [0.5, 1]],
        )

        transform_right = state_transformer.DiscreteMoveTransformer(
            3, 1, (0, 1), [-0.5, 0, 0.5]
        )

        testing.assert_allclose(
            transform_right.transform(state, numpy.full((3, 1), 0)),
            [[1, 0], [1, 0], [1, 0.5]],
        )

        testing.assert_allclose(
            transform_right.transform(state, numpy.full((3, 1), 1)),
            [[1, 0], [1, 0.5], [1, 1]],
        )

        testing.assert_allclose(
            transform_right.transform(state, numpy.full((3, 1), 2)),
            [[1, 0.5], [1, 1], [1, 1]],
        )


if __name__ == "__main__":
    unittest.main()
