"""Filter functionalities with mainly Butterworth filter."""

import numpy as _np
from scipy import signal as _signal
from enum import Enum as _Enum
from functools import lru_cache as _lru_cache


class FilterType(_Enum):
    """Filter types."""

    LOW_PASS = 'lowpass'
    HIGH_PASS = 'highpass'
    BAND_PASS = 'bandpass'
    BAND_STOP = 'bandstop'


def butterworth(data, frequency, cutoff, filter_type=FilterType.LOW_PASS, order=3, axis=-1, precision='float32'):
    """Butterworth digital filter.

    Design a Butterworth filter with the given parameters and apply it on 'data'.

    Args:
        data (numpy.ndarray): signal(s) to filter. 1-D array.
        frequency (int, float): the sampling rate/frequency, in Hz.
        cutoff (int, list): scalar for low and high pass filters.
            length-2 array or list for bandpass and bandstop filters.
            gives the filter critical frequencies, in Hz.
        filter_type (FilterType): FilterType.LOW_PASS, FilterType.HIGH_PASS, FilterType.BAND_PASS or FilterType.BAND_STOP (default: FilterType.LOW_PASS).
        order (int): int, the order of the filter (default: 3).
        axis (int): The axis of the input data array along which to apply the linear filter.
            The filter is applied to each subarray along this axis. (default: -1).
        precision (numpy.dtype): floating point numerical precision of output (default: 'float32').

    Returns:
        (numpy.ndarray): The output of the digital filter.

    Example:
        >>> data = np.random.randint(0, 256, (10, 10000), 'uint8')

        >>> filtered1 = filters.butterworth(data, 1e6, 0.45e6, 'low')
        >>> filtered2 = filters.butterworth(data, 1e6, [0.2e6, 0.4e6], filters.FilterType.BAND_PASS)

    """
    (data, frequency, cutoff, filter_type, order,
     axis, precision) = _butterworth_args_check(data, frequency, cutoff, filter_type, order, axis, precision)

    b_coef, a_coef = _signal_butter_wrapper(order, frequency, cutoff, filter_type)
    if b_coef.dtype != _np.dtype(precision):
        b_coef = b_coef.astype(precision)
        a_coef = a_coef.astype(precision)

    return _signal.lfilter(b_coef, a_coef, data, axis)


def _butterworth_args_check(data, frequency, cutoff, filter_type, order, axis, precision):  # noqa: C901
    if not isinstance(data, _np.ndarray):
        raise TypeError(f"'data' should be a numpy ndarray, not {type(data)}.")
    if not isinstance(frequency, int) and not isinstance(frequency, float):
        raise TypeError(f"'frequency' should be an of int or float type, not {type(frequency)}.")
    if frequency <= 0:
        raise ValueError("'frequency' should be positive.")
    if hasattr(cutoff, '__iter__'):
        cutoff = tuple(cutoff)
    else:
        cutoff = (cutoff, )
    for value in cutoff:
        if not isinstance(value, int) and not isinstance(value, float):
            raise TypeError(f"'cutoff' should be a value or a collection of values of int or float type, not {type(value)}.")
        if value <= 0:
            raise ValueError("'cutoff' value(s) should be positive.")
    if not isinstance(filter_type, FilterType):
        raise TypeError(f"'filter_type' should be an of FilterType type, not {type(filter_type)}.")
    if not isinstance(order, int):
        raise TypeError(f"'order' should be an of int type, not {type(order)}.")
    if order <= 0:
        raise ValueError("'order' should be positive.")
    if not isinstance(axis, int):
        raise TypeError(f"'axis' should be an of int type, not {type(axis)}.")
    try:
        precision = _np.dtype(precision)
    except TypeError:
        raise TypeError(f'precision should be a valid dtype, not {type(precision)}.')
    return data, frequency, cutoff, filter_type, order, axis, precision


@_lru_cache(maxsize=None)
def _signal_butter_wrapper(order, frequency, cutoff, filter_type):
    filter_type = filter_type.value
    if 'band' in filter_type:
        if len(cutoff) != 2:
            raise ValueError(f"For bandpass or bandstop filter type, cutoff should have 2 values, not {len(cutoff)}.")
        if not cutoff[1] > cutoff[0]:
            raise ValueError(f"If 'cutoff' has 2 values, the second ({cutoff[1]}) should be bigger than the first ({cutoff[0]}).")
        if not isinstance(cutoff, _np.ndarray):
            cutoff = _np.array(cutoff)
    else:
        if len(cutoff) != 1:
            raise ValueError(f"For lowpass or highpass filter type, cutoff should have 1 value, not {len(cutoff)}.")
        cutoff = cutoff[0]

    max_cutoff = _np.max(cutoff)
    if max_cutoff >= frequency / 2:
        raise ValueError("'cutoff' should be lower than frequency/2.")

    return _signal.butter(order, cutoff / (frequency / 2), filter_type)
