"""Methods related to measuring images."""

from collections.abc import Sequence

import cv2
import numpy

from numpy.typing import NDArray


def focus_value(image: NDArray) -> float:
    # pylint: disable=no-member
    """Returns a number that represents how 'in focus' image is, with larger numbers
    implying a better focus.

    Args:
        image: An RGB image.

    Returns:
        A number that represents how 'in focus' image is, with larger numbers implying
        a better focus."""

    return cv2.Laplacian(
        cv2.medianBlur(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), 3), cv2.CV_8U
    ).var()


def focus_values(images: NDArray[numpy.uint8]) -> Sequence[float]:
    """Returns numbers that represent how 'in focus' images are, with larger numbers
    implying better focus.

    Args:
        images: A number of RGB images.

    Returns:
        Numbers that represent how 'in focus' images are, with larger numbers implying
        a better focus."""

    return [focus_value(image) for image in images]
