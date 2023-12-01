'''Contains tests for reinfocus.learning.observation_filer.'''

import unittest

import numpy as np

import reinfocus.learning.observation_filter as fil
import tests.test_utils as tu

class ObservationFilterTest(unittest.TestCase):
    '''TestCases for reinfocus.learning.observation_filter.'''

    def test_long_mask_throws(self):
        '''Tests that a mask as long as the number of dimensions will raise an error.'''
        with self.assertRaises(IndexError):
            fil.ObservationFilter(0, 0, 3, set(range(3)))

    def test_negative_mask_throws(self):
        '''Tests that a mask with a negative element will raise en error.'''
        with self.assertRaises(IndexError):
            fil.ObservationFilter(0, 0, 3, {-1})

    def test_large_mask_throws(self):
        '''Tests that a mask with an element as large as the number of dimensions will
            raise en error.'''
        with self.assertRaises(IndexError):
            fil.ObservationFilter(0, 0, 3, {3})

    def test_observation_space_size(self):
        '''Tests that the observation space is the correct size for the given mask.'''
        self.assertEqual(fil.ObservationFilter(0, 0, 3).observation_space().shape, (3,))

        for m in range(3):
            self.assertEqual(
                fil.ObservationFilter(0, 0, 3, {m}).observation_space().shape,
                (2,))

        for p in range(3):
            m = set(np.delete(np.arange(3), p))
            self.assertEqual(
                fil.ObservationFilter(0, 0, 3, m).observation_space().shape,
                (1,))

    def test_filter_filters(self):
        '''Tests that ObservationFilters properly filter observations.'''
        o = np.array([1, 2, 3])

        tu.arrays_close(self, fil.ObservationFilter(0, 0, 3)(o), o)

        for m in range(3):
            tu.arrays_close(
                self,
                fil.ObservationFilter(0, 0, 3, {m})(o),
                np.delete(o, m))

        for p in range(3):
            m = np.delete(np.arange(3), p)
            tu.arrays_close(
                self,
                fil.ObservationFilter(0, 0, 3, set(m))(o),
                np.delete(o, m))

if __name__ == '__main__':
    unittest.main()
