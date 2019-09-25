"""Peaks detection functionnalities."""

from enum import Enum as _Enum
import warnings as _warnings

import numpy as _np

from . import _c_find_peaks


def find_peaks(data, min_peak_distance, min_peak_height):
    """Find peaks (local maxima) on data.

    Args:
        data (numpy.ndarray): 1-D ndarray.
        min_peak_distance (int): minimum distance between two maxima.
        min_peak_height (float or numpy.inf): minimum value to be considered as a maximum.

    Returns:
        (numpy.ndarray): The found peaks.

    """
    _check_data(data)
    if not isinstance(min_peak_distance, int):
        raise TypeError(f"'min_peak_distance' should be an int value, not {type(min_peak_distance)}.")
    if min_peak_distance < 0:
        raise ValueError(f"'min_peak_distance' should be positive, but it is equal to {min_peak_distance}.")
    if _np.abs(min_peak_height) != _np.inf:
        if not isinstance(min_peak_distance, float) and not isinstance(min_peak_distance, int):
            raise TypeError(f"'min_peak_distance' should be a float or int value, or numpy.inf, not {type(min_peak_distance)}.")

    float_data = data.astype('float32')
    float_result = _c_find_peaks.c_find_peaks(float_data, min_peak_distance, min_peak_height)
    return float_result.astype(int)


class Direction(_Enum):
    r"""Direction for interval detection.

    Positive: finds patterns with shape \_\_\_\|\-\-\|\_\_\_, then detects positive windows (humps) in signal.
    Negative: finds patterns with shape \-\-\-\|\_\_\|\-\-\-, then detects negative windows (hollows) in signal.

    """

    POSITIVE = -1
    NEGATIVE = 1


def find_width(data, direction, threshold, min_width, max_width=None, delta=None):
    """Find positive or negative window on signal.

    Args:
        data (numpy.ndarray): 1-D ndarray.
        direction (Direction): Direction.POSITIVE or Direction.NEGATIVE.
        threshold (float): threshold.
        min_width (int): minimum width of intervals.
        max_width (int): maximum width of intervals. (optional, default: None).
        delta (int): if specified, search intervals with width between min_width-delta and min_width+delta. (optional, default: None).

    Returns:
        (numpy.ndarray): List of couples, ([t0, t1]), with t0, t1 respectively start and end of intervals.

    """
    _check_find_width_args(data, direction, threshold, min_width, max_width, delta)

    sign = direction.value
    tmp = _np.where(sign * data >= sign * threshold)[0]
    widths = _np.diff(tmp)
    if max_width is not None:
        tmp_starts = _np.where(_np.bitwise_and(widths > min_width, widths <= max_width + 1))[0] + 1
    elif delta is not None:
        tmp_starts = _np.where(_np.bitwise_and(widths > min_width - delta, widths <= min_width + delta + 1))[0] + 1
    else:
        tmp_starts = _np.where(widths > min_width)[0] + 1
    starts = tmp[tmp_starts - 1] + 1
    ends = starts + widths[tmp_starts - 1] - 1
    return _np.vstack((starts, ends)).T


def _check_find_width_args(data, direction, threshold, min_width, max_width, delta):  # noqa: C901
    _check_data(data)
    if direction not in Direction:
        raise TypeError(f"'direction' should be a Direction object, not a {type(direction)}.")
    if not isinstance(threshold, int) and not isinstance(threshold, float):
        raise TypeError(f"'threshold' should be an int or a float, not a {type(threshold)}.")
    if not isinstance(min_width, int):
        raise TypeError(f"'min_width' should be an int, not a {type(min_width)}.")
    if min_width <= 0:
        raise ValueError(f"'min_width' should be bigger than 0, here it is {min_width}.")
    if max_width is not None:
        if not isinstance(max_width, int):
            raise TypeError(f"'max_width' should be an int, not a {type(max_width)}.")
        if max_width <= 0:
            raise ValueError(f"'max_width' should be bigger than 0, here it is {max_width}.")
    if delta is not None:
        if not isinstance(delta, int):
            raise TypeError(f"'delta' should be an int, not a {type(delta)}.")
        if delta <= 0:
            raise ValueError(f"'delta' should be bigger than 0, here it is {delta}.")
    if max_width is None and delta is not None and min_width <= delta:
        raise ValueError("'delta' argument should be lower than 'min_width'")

    if max_width is not None and delta is not None:
        _warnings.warn("'max_width' and 'delta' arguments are specified. 'delta will be ignored'")


class ExtractMode(_Enum):
    """Mode for signal extraction.

    stack: stacks all extracted data slices.
    concatenate: concatenates all extracted data slices.
    average: averages all extracted data slices.

    """

    STACK = 'stack'
    CONCATENATE = 'concatenate'
    AVERAGE = 'average'


def extract_around_indexes(data, indexes, before, after, mode=ExtractMode.STACK):
    """Extract parts of data around given indexes.

    Args:
        data (numpy.ndarray): 1-D array, the source array.
        indexes (numpy.ndarray): 1-D integer array.
        before (int): number of samples to take before each index.
        after (int): number of samples to take after each index.
        mode (ExtractMode): (ExtractMode.STACK, ExtractMode.CONCATENATE or ExtractMode.AVERAGE).
            stack: stacks all extracted data slices. Returns a ndarray with shape (len(indexes), before+after+1).
            concatenate: concatenates all extracted data slices. Returns a ndarray with shape (len(indexes) * before+after+1).
            average: averages all extracted data slices. Returns a ndarray with shape (before+after+1).

    Returns:
        (numpy.ndarray): The extracted parts of data.

    """
    _check_data(data)
    if not isinstance(indexes, _np.ndarray):
        raise TypeError(f"'indexes' should be a numpy ndarray, not {type(indexes)}.")
    if indexes.ndim != 1:
        raise ValueError(f"'indexes' should have one dimension, not {indexes.ndim}.")
    if not issubclass(indexes.dtype.type, _np.integer):
        raise ValueError(f"'indexes' should be an array of integers, not an array of {indexes.dtype.type}.")
    if not isinstance(before, int):
        raise TypeError(f"'before' should be an int, not {type(before)}.")
    if not isinstance(after, int):
        raise TypeError(f"'after' should be an int, not {type(after)}.")
    if not isinstance(mode, ExtractMode):
        raise TypeError(f"'mode' should be an ExtractMode, not {type(mode)}.")

    extended_indexes = _np.tile(_np.arange(-before, after + 1), (len(indexes), 1))
    extended_indexes = (extended_indexes.T + indexes).T
    result = _np.take(data, extended_indexes)

    if mode is ExtractMode.STACK:
        return result
    elif mode is ExtractMode.CONCATENATE:
        return result.reshape((-1,))
    elif mode is ExtractMode.AVERAGE:
        return _np.mean(result, 0)
    else:
        raise NotImplementedError(f"{mode} does is not supported by extract_around_indexes.")


def _check_data(data):
    if not isinstance(data, _np.ndarray):
        raise TypeError(f"'data' should be a numpy ndarray, not {type(data)}.")
    if data.ndim != 1:
        raise ValueError(f"'data' should have one dimension, not {data.ndim}.")
