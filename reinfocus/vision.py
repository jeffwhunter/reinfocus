"""Methods related to measuring images."""

import cv2 as cv
import numpy.typing as npt

def focus_value(image: npt.NDArray) -> float:
    """Returns a number that represents how 'in focus' image is, with larger numbers
        implying a better focus.

    Args:
        image: An RGB image.

    Returns:
        A number that represents how 'in focus' image is, with larger numbers implying
        a better focus."""
    # pylint: disable=c-extension-no-member
    return cv.Laplacian(
        cv.medianBlur(cv.cvtColor(image, cv.COLOR_RGB2GRAY), 3), cv.CV_32F).var()
