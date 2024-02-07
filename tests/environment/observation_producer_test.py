"""Contains tests for reinfocus.environment.observation_producer."""

import unittest

from unittest import mock

import numpy

from reinfocus.environment import observation_producer
from tests import test_utils


class ObservationProducerTest(unittest.TestCase):
    """TestCases for reinfocus.environment.observation_producer."""

    def test_produce_observation(self):
        """Tests that produce_observation produces clipped values returned by the
        observation normer."""

        testee = observation_producer.FocusObservationProducer(
            obs_normer=lambda obs: obs, measure_focus=lambda _1, _2: 8.0
        )

        test_utils.all_close(
            testee.produce_observation(numpy.array([-10.0, 0.5]), mock.Mock()),
            [-1.0, 0.5, 1.0],
        )

        test_utils.all_close(
            testee.produce_observation(numpy.array([-0.5, 1.5]), mock.Mock()),
            [-0.5, 1.0, 1.0],
        )

        test_utils.all_close(
            testee.produce_observation(numpy.array([10.0, -1.5]), mock.Mock()),
            [1.0, -1.0, 1.0],
        )


if __name__ == "__main__":
    unittest.main()
