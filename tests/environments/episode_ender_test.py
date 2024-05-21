"""Contains tests for reinfocus.environments.episode_ender."""

import unittest

from typing import Any
from unittest import mock

import numpy

from numpy import testing
from numpy.typing import NDArray

from reinfocus.environments import episode_ender


class BaseEnderTest(unittest.TestCase):
    """Test cases for reinfocus.environments.episode_ender.BaseEnder."""

    class _BaseEnder(episode_ender.BaseEnder):
        # pylint: disable=unnecessary-ellipsis
        """A minimal implementation of BaseEnder to allow testing of it."""

        def __init__(
            self,
            status: str = "",
            terminated: NDArray[numpy.bool_] = numpy.array([False]),
            truncated: NDArray[numpy.bool_] = numpy.array([False]),
        ):
            """Creates a _BaseEnder.

            Args:
                terminated: The termination signal to emit.
                truncated: The truncation signal to emit."""

            super().__init__()

            self._status = status
            self._terminated = terminated
            self._truncated = truncated

        def step(self, states: Any):
            """Required to implement IEpisodeEnder.

            Args:
                states: The states, which will be ignored."""

            ...

        def is_terminated(self) -> NDArray[numpy.bool_]:
            """Emits the preconfigured termination signal.

            Returns:
                The termination signal configured in the initializer."""

            return self._terminated

        def is_truncated(self) -> NDArray[numpy.bool_]:
            """Emits the preconfigured truncation signal.

            Returns:
                The truncation signal configured in the initializer."""

            return self._truncated

        def reset(self, states: Any, indices: NDArray[numpy.bool_] | None = None):
            """Required to implement IEpisodeEnder.

            Args:
                states: The states, which will be ignored.
                indices: Which episodes have ended, which will be ignored."""

            ...

        def status(self, index: int) -> str:
            """Emits the preconfigured status message.

            Returns:
                The status message configured in the initializer."""

            return self._status

    def test_and(self):
        """Tests that BaseEnders can be combined with &."""

        testee = BaseEnderTest._BaseEnder(
            terminated=numpy.array([True, True, False, False]),
            truncated=numpy.array([True, False, True, False]),
        ) & BaseEnderTest._BaseEnder(
            terminated=numpy.array([False, True, False, True]),
            truncated=numpy.array([False, False, True, True]),
        )

        testing.assert_allclose(testee.is_terminated(), [False, True, False, False])
        testing.assert_allclose(testee.is_truncated(), [False, False, True, False])

    def test_or(self):
        """Tests that BaseEnders can be combined with |."""

        testee = BaseEnderTest._BaseEnder(
            terminated=numpy.array([True, True, False, False]),
            truncated=numpy.array([True, False, True, False]),
        ) | BaseEnderTest._BaseEnder(
            terminated=numpy.array([False, True, False, True]),
            truncated=numpy.array([False, False, True, True]),
        )

        testing.assert_allclose(testee.is_terminated(), [True, True, False, True])
        testing.assert_allclose(testee.is_truncated(), [True, False, True, True])


class DivergingEnderTest(unittest.TestCase):
    """Test cases for reinfocus.environments.episode_ender.DivergingEnder."""

    def test_is_terminated(self):
        """Tests that is_terminated is always False (ie: the MDP has no end state)."""

        num_envs = 3

        testee = episode_ender.DivergingEnder(num_envs, (0, 1), 0, 1)

        testee.reset(numpy.array([[-1, -1], [0, 0], [1, 1]]))

        testee.step(numpy.array([[-1, -0.5], [0, -0.5], [1.5, 1]]))
        testing.assert_allclose(testee.is_terminated(), [False] * num_envs)

    def test_is_truncated_diverge(self):
        """Tests that is_truncated correctly ends the episode when state elements
        diverge for long enough."""

        num_envs = 3

        testee = episode_ender.DivergingEnder(num_envs, (0, 1), 0, 2)

        testee.reset(numpy.array([[-1, -1], [0, 0], [1, 1]]))

        testee.step(numpy.array([[-1, -0.5], [0, -0.5], [1.5, 1]]))
        testing.assert_allclose(testee.is_truncated(), [False] * num_envs)

        testee.step(numpy.array([[-1, -0.6], [0, -0.6], [1.5, 0.5]]))
        testing.assert_allclose(testee.is_truncated(), [False, True, True])

        testee.step(numpy.array([[-1, -0.5], [0, -0.6], [1.5, 0.5]]))
        testing.assert_allclose(testee.is_truncated(), [True] * num_envs)

    def test_is_truncated_threshold(self):
        """Tests that is_truncated only ends the episode when state elements diverge
        farther than the threshold."""

        num_envs = 3

        testee = episode_ender.DivergingEnder(num_envs, (0, 1), 0.25, 2)

        testee.reset(numpy.array([[-1, -1], [0, 0], [1, 1]]))

        testee.step(numpy.array([[-1, -0.5], [0, -0.5], [1.5, 1]]))
        testing.assert_allclose(testee.is_truncated(), [False] * num_envs)

        testee.step(numpy.array([[-1, -0.6], [0, -0.6], [1.5, 0.5]]))
        testing.assert_allclose(testee.is_truncated(), [False, False, True])

        testee.step(numpy.array([[-1, -0.5], [0, -0.9], [1.5, 0.5]]))
        testing.assert_allclose(testee.is_truncated(), [False, True, True])

    def test_reset(self):
        """Tests that truncation responds appropriately after a reset."""

        num_envs = 3

        testee = episode_ender.DivergingEnder(num_envs, (0, 1), 0, 2)

        testee.reset(numpy.array([[-1, -1], [0, 0], [1, 1]]))

        testee.step(numpy.array([[-1, -0.5], [0, -0.5], [1.5, 1]]))
        testing.assert_allclose(testee.is_truncated(), [False] * num_envs)

        testee.reset(numpy.array([[1.5, 1]]), numpy.array([False, False, True]))

        testee.step(numpy.array([[-1, -0.6], [0, -0.6], [1.5, 0.5]]))
        testing.assert_allclose(testee.is_truncated(), [False, True, False])

        testee.step(numpy.array([[-1, -0.5], [0, -0.6], [2, 0.5]]))
        testing.assert_allclose(testee.is_truncated(), [True, True, True])

    def test_status(self):
        """Tests status returns a string that shows how close the episode is to ending."""

        num_envs = 3

        testee = episode_ender.DivergingEnder(num_envs, (0, 1), 0, 2)

        testee.reset(numpy.array([[-1, -1], [0, 0], [1, 1]]))

        testee.step(numpy.array([[-1, -0.5], [0, -0.5], [1.5, 1]]))
        self.assertEqual(
            [testee.status(i) for i in range(num_envs)], ["diverging 1 / 2"] * num_envs
        )

        testee.reset(numpy.array([[1.5, 0.5]]), numpy.array([False, False, True]))

        testee.step(numpy.array([[-1, -0.6], [0, -0.6], [1.5, 0.5]]))
        self.assertEqual(
            [testee.status(i) for i in range(num_envs)],
            ["diverging 1 / 2", "diverging 2 / 2", ""],
        )

        testee.step(numpy.array([[-1, -0.5], [0, -0.6], [1.5, 0.5]]))
        self.assertEqual(
            [testee.status(i) for i in range(num_envs)],
            ["diverging 2 / 2", "diverging 2 / 2", ""],
        )


class EndlessEnderTest(unittest.TestCase):
    """Test cases for reinfocus.environments.episode_ender.EndlessEnder."""

    def test_never_ends(self):
        """Tests that is_terminated and is_truncated are always False."""

        num_envs = 5

        testee = episode_ender.EndlessEnder(num_envs)

        testee.step(numpy.zeros(num_envs, dtype=numpy.float32))

        testing.assert_allclose(testee.is_terminated(), [False] * num_envs)
        testing.assert_allclose(testee.is_truncated(), [False] * num_envs)


class OnTargetEnderTest(unittest.TestCase):
    """Test cases for reinfocus.environments.episode_ender.OnTargetEnder."""

    def test_is_terminated(self):
        """Tests that is_terminated is always False (ie: the MDP has no end state)."""

        num_envs = 3

        testee = episode_ender.OnTargetEnder(num_envs, (0, 1), 1, 1)

        testee.reset(numpy.array([[-1, -1], [0, 0], [1, 1]]))

        testee.step(numpy.array([[-1, -1], [0, 0], [1, 1]]))
        testing.assert_allclose(testee.is_terminated(), [False] * num_envs)

    def test_is_truncated_on_target(self):
        """Tests that is_truncated correctly ends the episode when on target."""

        num_envs = 3

        testee = episode_ender.OnTargetEnder(num_envs, (0, 1), 2, 1)

        testee.reset(numpy.array([[0, 2], [0, 1], [0, 1]]))

        testee.step(numpy.array([[0, 2], [0, 2], [0, 1]]))
        testing.assert_allclose(testee.is_truncated(), [False, False, True])

    def test_check_indices(self):
        """Tests that truncation does not depend on the check indices."""

        testee = episode_ender.OnTargetEnder(2, (3, 7), 2, 1)

        testee.reset(numpy.array([[0, 0, 0, 1, 0, 0, 0, 3], [0, 0, 0, 1, 0, 0, 0, 2]]))

        testee.step(numpy.array([[0, 0, 0, 1, 0, 0, 0, 3], [0, 0, 0, 1, 0, 0, 0, 2]]))
        testing.assert_allclose(testee.is_truncated(), [False, True])

    def test_reset(self):
        """Tests that truncation responds appropriately after a reset."""

        num_envs = 2

        testee = episode_ender.OnTargetEnder(num_envs, (0, 1), 2, 2)

        testee.reset(numpy.array([[0, 1], [0, 1]]))

        testee.step(numpy.array([[0, 1], [0, 1]]))
        testing.assert_allclose(testee.is_truncated(), [False] * num_envs)

        testee.reset(numpy.array([[0, 1]]), numpy.array([True, False]))

        testee.step(numpy.array([[0, 1], [0, 1]]))
        testing.assert_allclose(testee.is_truncated(), [False, True])

    def test_status(self):
        """Tests status returns a string that shows how close the episode is to ending."""

        num_envs = 3

        testee = episode_ender.OnTargetEnder(num_envs, (0, 1), 1.5, 2)

        testee.reset(numpy.array([[0, 2], [0, 1], [0, 1]]))

        self.assertEqual([testee.status(i) for i in range(num_envs)], [""] * num_envs)

        testee.step(numpy.array([[0, 2], [0, 1], [0, 1]]))

        self.assertEqual(
            [testee.status(i) for i in range(num_envs)],
            ["", "on target 1 / 2", "on target 1 / 2"],
        )

        testee.step(numpy.array([[0, 2], [0, 2], [0, 1]]))

        self.assertEqual(
            [testee.status(i) for i in range(num_envs)], ["", "", "on target 2 / 2"]
        )


class OpEnderTest(unittest.TestCase):
    """Test cases for reinfocus.environments.episode_ender.OpEnder."""

    def test_step(self):
        """Tests that OpEnder passes the step signal to it's child enders."""

        l_ender = mock.Mock()
        r_ender = mock.Mock()

        states = numpy.ones((3, 2), dtype=numpy.float32)

        episode_ender.OpEnder(l_ender, r_ender, lambda l, r: l | r).step(states)

        l_ender.step.assert_called_once_with(states)
        r_ender.step.assert_called_once_with(states)

    def test_terminated(self):
        """Tests that OpEnder properly combines the termination signals from it's child
        enders."""

        num_envs = 5

        l_ender = mock.Mock()
        l_ender.is_terminated.return_value = numpy.arange(num_envs) % 2 == 0

        r_ender = mock.Mock()
        r_ender.is_terminated.return_value = numpy.arange(num_envs) % 2 == 1

        testing.assert_allclose(
            episode_ender.OpEnder(l_ender, r_ender, lambda l, r: l | r).is_terminated(),
            [True] * num_envs,
        )

    def test_truncated(self):
        """Tests that OpEnder properly combines the truncation signals from it's child
        enders."""

        num_envs = 5

        l_ender = mock.Mock()
        l_ender.is_truncated.return_value = numpy.arange(num_envs) % 2 == 0

        r_ender = mock.Mock()
        r_ender.is_truncated.return_value = numpy.arange(num_envs) % 2 == 1

        testing.assert_allclose(
            episode_ender.OpEnder(l_ender, r_ender, lambda l, r: l & r).is_truncated(),
            [False] * num_envs,
        )

    def test_reset(self):
        """Tests that OpEnder passes the reset signal to it's child enders."""

        l_ender = mock.Mock()
        r_ender = mock.Mock()

        states = numpy.ones((3, 2), dtype=numpy.float32)
        indices = numpy.arange(3) % 2 == 0

        episode_ender.OpEnder(l_ender, r_ender, lambda l, r: l | r).reset(states, indices)

        l_ender.reset.assert_called_once_with(states, indices)
        r_ender.reset.assert_called_once_with(states, indices)

    def test_status(self):
        """Tests that OpEnder properly combines the status messages from it's child
        enders."""

        a_ender = mock.Mock()
        a_ender.status.return_value = "A"

        b_ender = mock.Mock()
        b_ender.status.return_value = "B"

        e_ender = mock.Mock()
        e_ender.status.return_value = ""

        self.assertEqual(
            episode_ender.OpEnder(a_ender, b_ender, lambda l, r: l & r).status(0),
            "A, B",
        )

        self.assertEqual(
            episode_ender.OpEnder(b_ender, a_ender, lambda l, r: l & r).status(0),
            "B, A",
        )

        self.assertEqual(
            episode_ender.OpEnder(a_ender, e_ender, lambda l, r: l & r).status(0),
            "A",
        )

        self.assertEqual(
            episode_ender.OpEnder(e_ender, b_ender, lambda l, r: l & r).status(0),
            "B",
        )


class StoppedEnderTest(unittest.TestCase):
    """Test cases for reinfocus.environments.episode_ender.StoppedEnder."""

    def test_is_terminated(self):
        """Tests that is_terminated is always False (ie: the MDP has no end state)."""

        num_envs = 3

        testee = episode_ender.StoppedEnder(num_envs, 1, 0.5, 1)

        testee.reset(numpy.array([[-1, -1], [0, 0], [1, 1]]))

        testee.step(numpy.array([[-1, -1], [0, 0], [1, 1]]))
        testing.assert_allclose(testee.is_terminated(), [False] * num_envs)

    def test_is_truncated_stopped(self):
        """Tests that is_truncated correctly ends the episode when stopped."""

        num_envs = 4

        testee = episode_ender.StoppedEnder(num_envs, 0, 0.5, 1)

        testee.reset(numpy.array([[1, 0], [2, 0], [3, 0], [4, 0]]))

        testee.step(numpy.array([[0.6, 0], [1.4, 0], [3.6, 0], [4.4, 0]]))
        testing.assert_allclose(testee.is_truncated(), [True, False, False, True])

    def test_check_index(self):
        """Tests that truncation does not depend on the check index."""

        num_envs = 4

        testee = episode_ender.StoppedEnder(num_envs, 1, 0.5, 1)

        testee.reset(numpy.array([[0, 1], [0, 2], [0, 3], [0, 4]]))

        testee.step(numpy.array([[0, 0.4], [0, 1.6], [0, 3.4], [0, 4.6]]))
        testing.assert_allclose(testee.is_truncated(), [False, True, True, False])

    def test_early_end_steps(self):
        """Tests that truncation responds appropriately to early_end_steps."""

        num_envs = 4

        testee = episode_ender.StoppedEnder(num_envs, 0, 0.5, 2)

        testee.reset(numpy.array([[1, 0], [2, 0], [3, 0], [4, 0]]))

        testee.step(numpy.array([[0.6, 0], [1.4, 0], [3.6, 0], [4.4, 0]]))
        testing.assert_allclose(testee.is_truncated(), [False] * num_envs)

        testee.step(numpy.array([[0.6, 0], [1.4, 0], [3.6, 0], [4.4, 0]]))
        testing.assert_allclose(testee.is_truncated(), [True, False, False, True])

    def test_slow_move(self):
        """Tests that states which move farther than the threshold over multiple steps
        don't end."""

        num_envs = 4

        testee = episode_ender.StoppedEnder(num_envs, 0, 0.5, 2)

        testee.reset(numpy.array([[1, 0], [2, 0], [3, 0], [4, 0]]))

        testee.step(numpy.array([[0.7, 0], [2.3, 0], [2.8, 0], [4.2, 0]]))
        testing.assert_allclose(testee.is_truncated(), [False] * num_envs)

        testee.step(numpy.array([[0.4, 0], [2.6, 0], [2.6, 0], [4.4, 0]]))
        testing.assert_allclose(testee.is_truncated(), [False, False, True, True])

    def test_reset(self):
        """Tests that truncation responds appropriately after a reset."""

        num_envs = 4

        testee = episode_ender.StoppedEnder(num_envs, 0, 0.5, 2)

        testee.reset(numpy.array([[1, 0], [2, 0], [3, 0], [4, 0]]))

        testee.step(numpy.array([[0.6, 0], [1.6, 0], [3.4, 0], [4.4, 0]]))
        testing.assert_allclose(testee.is_truncated(), [False] * num_envs)

        testee.reset(
            numpy.array([[0.6, 0], [5.4, 0]]), numpy.array([True, False, True, False])
        )

        testee.step(numpy.array([[0.6, 0], [1.6, 0], [5.4, 0], [4.4, 0]]))
        testing.assert_allclose(testee.is_truncated(), [False, True, False, True])

        testee.step(numpy.array([[0.6, 0], [1.6, 0], [5.4, 0], [4.4, 0]]))
        testing.assert_allclose(testee.is_truncated(), [True] * num_envs)

    def test_status(self):
        """Tests status returns a string that shows how close the episode is to ending."""

        num_envs = 4

        testee = episode_ender.StoppedEnder(num_envs, 0, 0.5, 2)

        testee.reset(numpy.array([[1, 0], [2, 0], [3, 0], [4, 0]]))

        self.assertEqual([testee.status(i) for i in range(num_envs)], [""] * num_envs)

        testee.step(numpy.array([[0.7, 0], [1.8, 0], [3.2, 0], [4.3, 0]]))

        self.assertEqual(
            [testee.status(i) for i in range(num_envs)],
            ["stopped 1 / 2"] * num_envs,
        )

        testee.step(numpy.array([[0.4, 0], [1.6, 0], [3.4, 0], [4.9, 0]]))

        self.assertEqual(
            [testee.status(i) for i in range(num_envs)],
            ["stopped 1 / 2", "stopped 2 / 2", "stopped 2 / 2", ""],
        )


class TimeLimitEnderTest(unittest.TestCase):
    """Test cases for reinfocus.environments.episode_ender.TimeLimitEnder."""

    def test_is_terminated(self):
        """Tests that is_terminated is always False (ie: the MDP has no end state)."""

        num_envs = 3

        testee = episode_ender.TimeLimitEnder(num_envs, 0)

        testee.reset(numpy.array([[-1, -1], [0, 0], [1, 1]]))

        testee.step(numpy.array([[-1, -1], [0, 0], [1, 1]]))
        testing.assert_allclose(testee.is_terminated(), [False] * num_envs)

    def test_is_truncated_stopped(self):
        """Tests that is_truncated correctly ends the episode after a number of steps."""

        num_envs = 4

        testee = episode_ender.TimeLimitEnder(num_envs, 1)

        testee.reset(numpy.array([[1, 0], [2, 0], [3, 0], [4, 0]]))

        testee.step(numpy.array([[1, 0], [2, 0], [3, 0], [4, 0]]))
        testing.assert_allclose(testee.is_truncated(), [True] * num_envs)

    def test_early_end_steps(self):
        """Tests that truncation responds appropriately to max_steps."""

        num_envs = 4

        testee = episode_ender.TimeLimitEnder(num_envs, 2)

        testee.reset(numpy.array([[1, 0], [2, 0], [3, 0], [4, 0]]))

        testee.step(numpy.array([[1, 0], [2, 0], [3, 0], [4, 0]]))
        testing.assert_allclose(testee.is_truncated(), [False] * num_envs)

        testee.step(numpy.array([[1, 0], [2, 0], [3, 0], [4, 0]]))
        testing.assert_allclose(testee.is_truncated(), [True] * num_envs)

    def test_reset(self):
        """Tests that truncation responds appropriately after a reset."""

        num_envs = 4

        testee = episode_ender.TimeLimitEnder(num_envs, 2)

        testee.reset(numpy.array([[1, 0], [2, 0], [3, 0], [4, 0]]))

        testee.step(numpy.array([[1, 0], [2, 0], [3, 0], [4, 0]]))
        testing.assert_allclose(testee.is_truncated(), [False] * num_envs)

        testee.reset(
            numpy.array([[1, 0], [3, 0]]), numpy.array([True, False, True, False])
        )

        testee.step(numpy.array([[1, 0], [2, 0], [3, 0], [4, 0]]))
        testing.assert_allclose(testee.is_truncated(), [False, True, False, True])

        testee.step(numpy.array([[1, 0], [2, 0], [3, 0], [4, 0]]))
        testing.assert_allclose(testee.is_truncated(), [True] * num_envs)

    def test_status(self):
        """Tests status returns a string that shows how close the episode is to ending."""

        num_envs = 4

        testee = episode_ender.TimeLimitEnder(num_envs, 2)

        testee.reset(numpy.array([[1, 0], [2, 0], [3, 0], [4, 0]]))

        testee.step(numpy.array([[1, 0], [2, 0], [3, 0], [4, 0]]))
        self.assertSequenceEqual(
            [testee.status(i) for i in range(num_envs)], ["step 1 / 2"] * num_envs
        )

        testee.reset(
            numpy.array([[1, 0], [3, 0]]), numpy.array([True, False, True, False])
        )

        testee.step(numpy.array([[1, 0], [2, 0], [3, 0], [4, 0]]))
        self.assertSequenceEqual(
            [testee.status(i) for i in range(num_envs)],
            ["step 1 / 2", "step 2 / 2", "step 1 / 2", "step 2 / 2"],
        )

        testee.step(numpy.array([[1, 0], [2, 0], [3, 0], [4, 0]]))
        self.assertSequenceEqual(
            [testee.status(i) for i in range(num_envs)],
            ["step 2 / 2", "step 3 / 2", "step 2 / 2", "step 3 / 2"],
        )


if __name__ == "__main__":
    unittest.main()
