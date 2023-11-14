"""Methods that evaluate how well a set of ... evaluations ... did."""

from typing import List

import numpy as np
import numpy.typing as npt
import scipy.stats

def iqm_interval(evaluations: List[float]) -> npt.NDArray:
    """Returns an array containing the IQM, the 25th percentile, and the 75th percentile
        of evalutations.

    Args:
        evaluations: The list of evaluations over which the stats will be calculated.

    Returns:
        The iqm and 25th and 75th percentiles of evaluations."""

    return np.array([
        scipy.stats.trim_mean(evaluations, proportiontocut=0.25),
        np.percentile(evaluations, 100 * 0.25),
        np.percentile(evaluations, 100 * (1 - 0.25))])
