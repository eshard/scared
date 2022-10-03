"""Pattern detection methods.

Provides sliding metrics to detect a 'pattern' into a 'trace', with len(pattern) < len(trace).

correlation: Pearson correlation metric. Highest score is the better.
distance: Euclidean distance metric. Lowest score is the better.
bcdc: BCDC metric. Lowest score is the better.

"""

import numpy as _np
from scipy import signal as _signal

from .base import cast_array
from .moving_operators import moving_sum, moving_mean


def correlation(trace, pattern):
    """Pattern matching using Pearson correlation.

    This function performs a sliding Pearson correlation between trace and pattern.
    Highest score is the better.

    Args:
        trace (numpy.ndarray): 1-D signal.
        pattern (numpy.ndarray): 1-D pattern with len(pattern) < len(trace).

    Returns:
        (numpy.ndarray): 1-D ndarray with shape (len(trace-len(pattern)+1), with -1 < scores < 1.
        Highest score is the better.

    Example:
        >>> pattern = np.random.randint(0, 256, (100))
        >>> trace = np.random.randint(0, 256, (1000))
        >>> trace[533:633] = pattern

        >>> correlation = pattern_detection.corr(trace, pattern)
        >>> correlation.shape
        (901, )
        >>> np.argmax(correlation)
        533

    """
    trace, pattern = _check_and_cast_args(trace, pattern)

    n = len(pattern)
    ex = moving_mean(trace, n)
    ey = _np.mean(pattern)
    x2 = moving_sum(trace**2, n)
    y2 = _np.sum(pattern**2)
    xy = _signal.correlate(trace, pattern, 'valid')

    numerator = xy - n * ex * ey
    denominator = _np.sqrt(x2 - n * ex**2) * _np.sqrt(y2 - n * ey**2)
    return numerator / denominator


def distance(trace, pattern):
    """Pattern matching using euclidean distance.

    This function performs a sliding euclidean distance between trace and pattern.
    Lowest score is the better.

    Args:
        trace (numpy.ndarray): 1-D signal.
        pattern (numpy.ndarray): 1-D pattern with len(pattern) < len(trace).

    Returns:
        (numpy.ndarray): 1-D ndarray with shape (len(trace-len(pattern)+1), with -1 < scores < 1.
        Lowest score is the better.

    Example:
        >>> pattern = np.random.randint(0, 256, (100))
        >>> trace = np.random.randint(0, 256, (1000))
        >>> trace[533:633] = pattern

        >>> distance = pattern_detection.dist(trace, pattern)
        >>> distance.shape
        (901, )
        >>> np.argmin(distance)
        533

    """
    trace, pattern = _check_and_cast_args(trace, pattern)

    tmp1 = moving_sum(trace**2, len(pattern)) + _np.sum(pattern**2)
    tmp2 = _signal.correlate(trace, pattern, 'valid')
    return _np.sqrt(_np.abs(tmp1 - 2 * tmp2))


def bcdc(trace, pattern):
    """Pattern matching using BCDC.

    This function performs a sliding BCDC between trace and pattern.
    Lowest score is the better.

    Args:
        trace (numpy.ndarray): 1-D signal.
        pattern (numpy.ndarray): 1-D pattern with len(pattern) < len(trace).

    Returns:
        (numpy.ndarray): 1-D ndarray with shape (len(trace-len(pattern)+1), with -1 < scores < 1.
        Lowest score is the better.

    Example:
        >>> pattern = np.random.randint(0, 256, (100))
        >>> trace = np.random.randint(0, 256, (1000))
        >>> trace[533:633] = pattern

        >>> bcdc_scores = pattern_detection.bcdc(trace, pattern)
        >>> bcdc_scores.shape
        (901, )
        >>> np.argmin(bcdc_scores)
        533

    """
    trace, pattern = _check_and_cast_args(trace, pattern)

    ex2 = moving_sum(trace**2, len(pattern))
    ey2 = _np.sum(pattern**2)
    exy = _signal.correlate(trace, pattern, 'valid')
    ex = moving_sum(trace, len(pattern))
    ey = _np.sum(pattern)
    n = len(pattern)

    tmp1 = ex2 + ey2 - 2 * exy
    tmp2 = ((ex - ey) / n)**2
    numerator = tmp1 / n - tmp2
    tmp1 = ex2 + ey2 + 2 * exy
    tmp2 = ((ex + ey) / n)**2
    denominator = tmp1 / n - tmp2
    return _np.sqrt(_np.abs(numerator)) / _np.sqrt(_np.abs(denominator))


def _check_and_cast_args(trace, pattern):
    if not isinstance(trace, _np.ndarray):
        raise TypeError(f"'trace' should be a numpy ndarray, not {type(trace)}.")
    if not isinstance(pattern, _np.ndarray):
        raise TypeError(f"'pattern' should be a numpy ndarray, not {type(pattern)}.")
    if trace.ndim != 1:
        raise ValueError(f"'trace' should have one dimension, not {trace.ndim}.")
    if pattern.ndim != 1:
        raise ValueError(f"'pattern' should have one dimension, not {pattern.ndim}.")
    if len(trace) <= len(pattern):
        raise ValueError(f"len(pattern) (here {len(pattern)}) should be lower than len(trace) (here {len(trace)})")

    trace = cast_array(trace)
    pattern = cast_array(pattern)
    return trace, pattern
