"""Frequency analysis functionalities with mainly FFT operation."""

import numpy as _np


def fft(data, frequency, axis=-1):
    """Compute rFFT on real data along the given axis.

    Args:
        data (numpy.ndarray): N-dimensional real values array.
        frequency (float): the sampling rate/frequency, in Hz.
        axis (int): The axis of the input data array along which to apply the FFT.
            FFT is applied to each subarray along this axis. (default: -1).

    Returns:
        (numpy.ndarray, numpy.ndarray, numpy.ndarray): frequencies, magnitude, phase

    Example:
        >>> import scipy.signal
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt

        >>> frequency = 10e3
        >>> f = 100
        >>> signal = scipy.signal.square(2 * np.pi * int(f) * np.linspace(0, 1, int(frequency)))

        >>> frequencies, magnitude, phase = frequency_analysis.fft(signal, frequency)

        >>> plt.plot(frequencies, magnitude)
        >>> plt.show()

        >>> plt.plot(frequencies, phase)
        >>> plt.show()

    """
    if not isinstance(data, _np.ndarray):
        raise TypeError(f"'data' should be a numpy ndarray, not {type(data)}.")
    if not isinstance(frequency, float) and not isinstance(frequency, int):
        raise TypeError(f"'frequency' should be an of int or float type, not {type(frequency)}.")
    if frequency <= 0:
        raise ValueError("'frequency' should be positive.")
    if not isinstance(axis, int):
        TypeError(f"'axis' should be an of int type, not {type(axis)}.")
    if axis >= data.ndim or axis < -(data.ndim):
        raise ValueError(f"bad axis {axis} for a {data.ndim}-D array")

    nb_points = data.shape[axis]
    frequencies = _np.fft.rfftfreq(nb_points) * frequency
    fft = _np.fft.rfft(data, axis=axis) / nb_points
    magnitude = _np.abs(fft)
    phase = _np.angle(fft) * 180 / _np.pi

    return frequencies, magnitude, phase
