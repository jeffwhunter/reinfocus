"""Contains tests for reinfocus.environment.observation_normer."""

import unittest

import numpy

from reinfocus.environment import observation_normer
from tests import test_utils


class ObservationNormerTest(unittest.TestCase):
    """TestCases for reinfocus.environment.observation_normer."""

    def test_make_observation_normer(self):
        """Tests that make_observation_normer creates a normer that norms as expected."""

        normer = observation_normer.make_observation_normer(
            numpy.array([1]), numpy.array([2])
        )
        test_utils.all_close(normer(numpy.array([0])), [-0.5])
        test_utils.all_close(normer(numpy.array([1])), [0])
        test_utils.all_close(normer(numpy.array([2])), [0.5])

    def test_from_spans(self):
        """Tests that from_spans creates a normer that norms each element as expected."""

        two_normer = observation_normer.from_spans([(-10.0, -5.0), (2.0, 4.0)])

        test_utils.all_close(two_normer(numpy.array([-10.0, 2.0])), [-1.0, -1.0])
        test_utils.all_close(two_normer(numpy.array([-7.5, 3.0])), [0.0, 0.0])
        test_utils.all_close(two_normer(numpy.array([-5.0, 4.0])), [1.0, 1.0])

        four_normer = observation_normer.from_spans(
            [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 4.0)]
        )

        test_utils.all_close(
            four_normer(numpy.linspace(0.0, 3.0, 4)), numpy.full(4, -1.0)
        )
        test_utils.all_close(four_normer(numpy.linspace(0.5, 3.5, 4)), numpy.zeros(4))
        test_utils.all_close(four_normer(numpy.linspace(1.0, 4.0, 4)), numpy.ones(4))


if __name__ == "__main__":
    unittest.main()
