"""Methods related to measuring images."""

import cv2

from numpy.typing import NDArray

def focus_value(image: NDArray) -> float:
    # pylint: disable=c-extension-no-member
    """Returns a number that represents how 'in focus' image is, with larger numbers
        implying a better focus.

    Args:
        image: An RGB image.

    Returns:
        A number that represents how 'in focus' image is, with larger numbers implying
        a better focus."""

    return cv2.Laplacian(
        cv2.medianBlur(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), 3), cv2.CV_32F).var()
