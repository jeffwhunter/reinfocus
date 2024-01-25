'''Contains tests for reinfocus.learning.observation_filer.'''

import unittest

import numpy

from reinfocus.environment import observation_filter
from tests import test_utils

class ObservationFilterTest(unittest.TestCase):
    '''TestCases for reinfocus.learning.observation_filter.'''

    def test_long_mask_throws(self):
        '''Tests that a mask as long as the number of dimensions will raise an error.'''

        with self.assertRaises(IndexError):
            observation_filter.ObservationFilter(0, 0, 3, set(range(3)))

    def test_negative_mask_throws(self):
        '''Tests that a mask with a negative element will raise en error.'''

        with self.assertRaises(IndexError):
            observation_filter.ObservationFilter(0, 0, 3, {-1})

    def test_large_mask_throws(self):
        '''Tests that a mask with an element as large as the number of dimensions will
            raise en error.'''

        with self.assertRaises(IndexError):
            observation_filter.ObservationFilter(0, 0, 3, {3})

    def test_observation_space_size(self):
        '''Tests that the observation space is the correct size for the given mask.'''

        self.assertEqual(
            observation_filter.ObservationFilter(0, 0, 3).observation_space().shape,
            (3,))

        for m in range(3):
            self.assertEqual(
                observation_filter.ObservationFilter(0, 0, 3, {m})
                    .observation_space()
                    .shape,
                (2,))

        for p in range(3):
            m = set(numpy.delete(numpy.arange(3), p))
            self.assertEqual(
                observation_filter.ObservationFilter(0, 0, 3, m)
                    .observation_space()
                    .shape,
                (1,))

    def test_filter_filters(self):
        '''Tests that ObservationFilters properly filter observations.'''

        o = numpy.array([1, 2, 3])

        test_utils.arrays_close(
            self,
            observation_filter.ObservationFilter(0, 0, 3)(o),
            o)

        for m in range(3):
            test_utils.arrays_close(
                self,
                observation_filter.ObservationFilter(0, 0, 3, {m})(o),
                numpy.delete(o, m))

        for p in range(3):
            m = numpy.delete(numpy.arange(3), p)
            test_utils.arrays_close(
                self,
                observation_filter.ObservationFilter(0, 0, 3, set(m))(o),
                numpy.delete(o, m))

if __name__ == '__main__':
    unittest.main()
