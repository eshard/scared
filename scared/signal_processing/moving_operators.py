"""Moving functionnalities with moving sum, mean, variance, standard deviation, skew and kurtosis."""

import numpy as _np
from .base import cast_array, pad


def moving_sum(data, window_size, axis=-1):
    """Move sum along specified axis with 'window_size' windows length.

    Args:
        data (numpy.ndarray): N-dimensional array.
        window_size (int): windows length.
        axis (int): Axis over which to move window (default: -1).

    Returns:
        (numpy.ndarray): N-dimensional array. The length on 'axis' dimension is: data.shape['axis'] - 'window_size' + 1.

    """
    _moving_argument_check(data, window_size, axis)
    if window_size < 2:
        return data
    # Referencing the target axis as 0 during the whole processing
    # We will swap back at the end
    data = _np.swapaxes(data, 0, axis)
    # Padding along axis win/2 at the beginning and win/2 at the end
    target_dimension = list(data.shape)
    target_dimension[0] += 1

    offset = _np.zeros(data.ndim, dtype=int)
    offset[0] = 1
    padded = pad(data, target_dimension, offset)
    padded = cast_array(padded, 'float64')
    # Processing
    ret = _np.cumsum(padded, axis=0)
    ret = (ret[window_size:] - ret[:-window_size])

    return _np.swapaxes(ret, axis, 0)


def moving_mean(data, window_size, axis=-1):
    """Move mean along specified axis with 'window_size' windows length.

    Args:
        data (numpy.ndarray): N-dimensional array.
        window_size (int): windows length.
        axis (int): Axis over which to move window (default: -1).

    Returns:
        (numpy.ndarray): N-dimensional array. The length on 'axis' dimension is: data.shape['axis'] - 'window_size' + 1.

    """
    _moving_argument_check(data, window_size, axis)
    return moving_sum(data, window_size, axis) / window_size


def moving_var(data, window_size, axis=-1):
    """Move variance along specified axis with 'window_size' windows length.

    Args:
        data (numpy.ndarray): N-dimensional array.
        window_size (int): windows length.
        axis (int): Axis over which to move window (default: -1).

    Returns:
        (numpy.ndarray): N-dimensional array. The length on 'axis' dimension is: data.shape['axis'] - 'window_size' + 1.

    """
    _moving_argument_check(data, window_size, axis)
    data = cast_array(data, 'float64')

    m1 = moving_mean(data, window_size, axis)
    m2 = moving_mean(data**2, window_size, axis)

    return m2 - m1**2


def moving_std(data, window_size, axis=-1):
    """Move standard deviation along specified axis with 'window_size' windows length.

    Args:
        data (numpy.ndarray): N-dimensional array.
        window_size (int): windows length.
        axis (int): Axis over which to move window (default: -1).

    Returns:
        (numpy.ndarray): N-dimensional array. The length on 'axis' dimension is: data.shape['axis'] - 'window_size' + 1.

    """
    _moving_argument_check(data, window_size, axis)
    return _np.sqrt(moving_var(data, window_size, axis))


def moving_skew(data, window_size, axis=-1):
    """Move skew along specified axis with 'window_size' windows length.

    ====== =====================================================================
    skew   = µ3 / µ2^(2/3) with µi the i^th central moment.
    µ2     = σ^2
    µ3     = m3 - 3*m2*m1 + 2*m1^3 with mi the i^th moment (m=E[X], m2 = E[X^2])
    µ3     = m3 - 3*m2*m1 + 3*m1^3 - 3*m1^3 + 2*m1^3
    µ3     = m3 - (3*m2*m1 - 3*m1^3) - 3*m1^3 + 2*m1^3
    µ3     = m3 - 3*m1*(m2 - m1^2) - m1^3  Then
    skew   = (m3 - 3*m1*σ^2 - m1^3) / (σ^2)^(3/2)
    ====== =====================================================================

    Args:
        data (numpy.ndarray): N-dimensional array.
        window_size (int): windows length.
        axis (int): Axis over which to move window (default: -1).

    Returns:
        (numpy.ndarray): N-dimensional array. The length on 'axis' dimension is: data.shape['axis'] - 'window_size' + 1.

    """
    _moving_argument_check(data, window_size, axis)
    # cast to avoid overflow during 'data**3' operation
    data = cast_array(data, 'float64')

    m1 = moving_mean(data, window_size, axis)
    m2 = moving_mean(data**2, window_size, axis)
    m3 = moving_mean(data**3, window_size, axis)
    v = m2 - m1**2

    return (m3 - 3 * m1 * v - m1**3) / v**(3 / 2)


def moving_kurtosis(data, window_size, axis=-1):
    """Move kurtosis along specified axis with 'window_size' windows length.

    ========== =================================================================================
    kurtosis   = µ4 / σ^4 with µi the i^th central moment
    µ4         = m4 - 4*m3*m1 + 6*m2*m1^2 - 3*m1^4 with mi the i^th moment (m=E[X], m2 = E[X^2])
    6*m2*m1^2  = 6*m1^2*m2 - 6*m1^4 + 6*m1^4
    6*m2*m1^2  = 6*m1^2*(m2 - m1^2) + 6*m1^4
    µ4         = m4 - 4*m3*m1 + 6*m1^2*σ^2 + 3*m1^4
    kurtosis   = (m4 - 4*m3*m1 + 6*m1^2*σ^2 + 3*m1^4) / σ^4
    ========== =================================================================================

    Args:
        data (numpy.ndarray): N-dimensional array.
        window_size (int): windows length.
        axis (int): Axis over which to move window (default: -1).

    Returns:
        (numpy.ndarray): N-dimensional array. The length on 'axis' dimension is: data.shape['axis'] - 'window_size' + 1.

    """
    _moving_argument_check(data, window_size, axis)
    # cast to avoid overflow during 'data**3' or 'data**4' operation
    data = cast_array(data, 'float64')

    m1 = moving_mean(data, window_size, axis)
    m2 = moving_mean(data**2, window_size, axis)
    m3 = moving_mean(data**3, window_size, axis)
    m4 = moving_mean(data**4, window_size, axis)
    v = m2 - m1**2

    return (m4 - 4 * m3 * m1 + 6 * v * m1**2 + 3 * m1**4) / v**2 - 3


def _moving_argument_check(data, window_size, axis):
    if not isinstance(data, _np.ndarray):
        raise TypeError(f"'data' should be a numpy ndarray, not {type(data)}.")
    if not isinstance(window_size, int):
        raise TypeError(f"'window_size' should be an of int or float type, not {type(window_size)}.")
    if window_size <= 0:
        raise ValueError(f"'window_size' should be positive.")
    if not isinstance(axis, int):
        TypeError(f"'axis' should be an of int type, not {type(axis)}.")
    if axis >= data.ndim or axis < -(data.ndim):
        raise ValueError(f"bad axis {axis} for a {data.ndim}-D array")
    if window_size > data.shape[axis]:
        raise ValueError(f"windows_size={window_size} but data shape in {axis} dimension is {data.shape[axis]}.")
