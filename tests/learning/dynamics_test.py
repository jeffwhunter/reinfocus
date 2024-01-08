'''Contains tests for reinfocus.learning.dynamics.'''

import unittest

import numpy as np

import reinfocus.learning.dynamics as dyn
import tests.test_utils as tu

class DynamicsTest(unittest.TestCase):
    '''TestCases for reinfocus.learning.dynamics.'''

    def test_continuous_dynamics(self):
        '''Tests that ContinuousDynamics mvoes the state with the expected speed.'''
        fast_dynamics = dyn.make_continuous_dynamics((0, 1), 1)

        state = np.array([1, .5])
        tu.arrays_close(self, fast_dynamics(state, np.float32(1.)), [1, 1])
        tu.arrays_close(self, fast_dynamics(state, np.float32(.75)), [1, 1])
        tu.arrays_close(self, fast_dynamics(state, np.float32(.5)), [1, 1])
        tu.arrays_close(self, fast_dynamics(state, np.float32(.25)), [1, .75])
        tu.arrays_close(self, fast_dynamics(state, np.float32(.1)), [1, .6])
        tu.arrays_close(self, fast_dynamics(state, np.float32(0)), [1, .5])
        tu.arrays_close(self, fast_dynamics(state, np.float32(-.1)), [1, .4])
        tu.arrays_close(self, fast_dynamics(state, np.float32(-.25)), [1, .25])
        tu.arrays_close(self, fast_dynamics(state, np.float32(-.5)), [1, 0])
        tu.arrays_close(self, fast_dynamics(state, np.float32(-.75)), [1, 0])
        tu.arrays_close(self, fast_dynamics(state, np.float32(-1)), [1, 0])

        state = np.array([1, 1])
        tu.arrays_close(self, fast_dynamics(state, np.float32(1)), [1, 1])
        tu.arrays_close(self, fast_dynamics(state, np.float32(0)), [1, 1])
        tu.arrays_close(self, fast_dynamics(state, np.float32(-.5)), [1, .5])
        tu.arrays_close(self, fast_dynamics(state, np.float32(-1)), [1, 0])

        state = np.array([1, 0])
        tu.arrays_close(self, fast_dynamics(state, np.float32(1)), [1, 1])
        tu.arrays_close(self, fast_dynamics(state, np.float32(.5)), [1, .5])
        tu.arrays_close(self, fast_dynamics(state, np.float32(0)), [1, 0])
        tu.arrays_close(self, fast_dynamics(state, np.float32(-1)), [1, 0])

        slow_dynamics = dyn.make_continuous_dynamics((0, 1), .1)

        state = np.array([1, .5])
        tu.arrays_close(self, slow_dynamics(state, np.float32(2)), [1, .6])
        tu.arrays_close(self, slow_dynamics(state, np.float32(1)), [1, .6])
        tu.arrays_close(self, slow_dynamics(state, np.float32(.1)), [1, .51])
        tu.arrays_close(self, slow_dynamics(state, np.float32(0)), [1, .5])
        tu.arrays_close(self, slow_dynamics(state, np.float32(-.1)), [1, .49])
        tu.arrays_close(self, slow_dynamics(state, np.float32(-1)), [1, .4])
        tu.arrays_close(self, slow_dynamics(state, np.float32(-2)), [1, .4])

    def test_discrete_dynamics(self):
        '''Tests that ContinuousDynamics mvoes the state with the expected actions.'''

        dynamics = dyn.make_discrete_dynamics((0, 1), [-1., -.5, -.1, 0, .1, .5, 1])

        state = np.array([1, .5])
        tu.arrays_close(self, dynamics(state, np.int32(0)), [1, 0.])
        tu.arrays_close(self, dynamics(state, np.int32(1)), [1, 0.])
        tu.arrays_close(self, dynamics(state, np.int32(2)), [1, .4])
        tu.arrays_close(self, dynamics(state, np.int32(3)), [1, .5])
        tu.arrays_close(self, dynamics(state, np.int32(4)), [1, .6])
        tu.arrays_close(self, dynamics(state, np.int32(5)), [1, 1.])
        tu.arrays_close(self, dynamics(state, np.int32(6)), [1, 1.])

        state = np.array([1, .0])
        tu.arrays_close(self, dynamics(state, np.int32(0)), [1, 0.])
        tu.arrays_close(self, dynamics(state, np.int32(1)), [1, 0.])
        tu.arrays_close(self, dynamics(state, np.int32(2)), [1, 0.])
        tu.arrays_close(self, dynamics(state, np.int32(3)), [1, 0.])
        tu.arrays_close(self, dynamics(state, np.int32(4)), [1, .1])
        tu.arrays_close(self, dynamics(state, np.int32(5)), [1, .5])
        tu.arrays_close(self, dynamics(state, np.int32(6)), [1, 1.])

        state = np.array([1, 1.])
        tu.arrays_close(self, dynamics(state, np.int32(0)), [1, 0.])
        tu.arrays_close(self, dynamics(state, np.int32(1)), [1, .5])
        tu.arrays_close(self, dynamics(state, np.int32(2)), [1, .9])
        tu.arrays_close(self, dynamics(state, np.int32(3)), [1, 1.])
        tu.arrays_close(self, dynamics(state, np.int32(4)), [1, 1.])
        tu.arrays_close(self, dynamics(state, np.int32(5)), [1, 1.])
        tu.arrays_close(self, dynamics(state, np.int32(6)), [1, 1.])

if __name__ == '__main__':
    unittest.main()
