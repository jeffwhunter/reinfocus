"""Contains tests for reinfocus.graphics.device_data."""

import unittest

from typing import Any
from unittest import mock

import numpy

from numpy import testing

from reinfocus.graphics import device_data


class DeviceDataTest(unittest.TestCase):
    """Test cases for reinfocus.graphics.device_data.DeviceData."""

    class _DeviceData(device_data.DeviceData):

        def __init__(self, make_device_data: mock.Mock):
            super().__init__()

            self._m_mock = make_device_data

        def _make_device_data(self, data: Any) -> Any:
            return self._m_mock(data)

    def test_update_filters_data(self):
        """Tests that update will not transform when it's input does not change."""

        target = [1, 2, 3]

        make_device_data = mock.Mock(return_value=target)

        testee = DeviceDataTest._DeviceData(make_device_data)

        testee.update(target)
        testee.update(target)

        make_device_data.assert_called_once()

        testing.assert_allclose(testee.device_data(), target)

    def test_update_transforms_new_data(self):
        """Tests that update will transform when it's input changes."""

        targets = numpy.array([[1, 2, 3], [2, 3, 4]], dtype=numpy.float32)

        make_device_data = mock.Mock(side_effect=targets)

        testee = DeviceDataTest._DeviceData(make_device_data)

        for target in targets:
            testee.update(target)
            make_device_data.assert_called_with(target)

        testing.assert_allclose(testee.device_data(), targets[-1])


if __name__ == "__main__":
    unittest.main()
