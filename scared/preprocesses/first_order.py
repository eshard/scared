from ._base import Preprocess, preprocess, PreprocessError
import numpy as _np


def _center(traces, mean):
    if mean is None:
        return center(traces)
    try:
        return traces - mean
    except ValueError:
        raise PreprocessError(f'Incompatible shapes between traces {traces.shape} and mean {mean.shape}.')


@preprocess
def square(traces):
    """Returns traces square.

    Args:
        traces (numpy.ndarray): a 2 dimensional numpy array.

    Returns:
        (numpy.ndarray) square of input traces array.

    """
    return _np.square(traces, dtype=max(traces.dtype, 'float32'))


@preprocess
def serialize_bit(traces):
    """Returns serialized bit of bytes traces.

    Args:
        traces (numpy.ndarray): a 2 dimensional numpy array. If traces is not of uint8 dtype,
            it will be casted to uint8 before serializing bits.

    Returns:
        (numpy.ndarray) serialized bits along dimension 2 of input traces array.

    """
    return _np.unpackbits(traces.astype('uint8'), axis=1)


@preprocess
def fft_modulus(traces):
    """Returns the modulus of FFT of input traces.

    Due to symmetry of the modulus of FFT, only the first half of computing modulus traces are returned.

    Args:
        traces (numpy.ndarray): a 2 dimensional numpy array.

    Returns:
        (numpy.ndarray) the modulus of FFT applied on traces.

    """
    length = _np.ceil(traces.shape[1] / 2).astype('uint32')
    return _np.abs(_np.fft.fft(traces))[:, :length]


@preprocess
def center(traces):
    """Returns traces centered around the mean of all traces.

    Args:
        traces (numpy.ndarray): a 2 dimensional numpy array.

    Returns:
        (numpy.ndarray) traces subtracted of the mean on all traces.

    """
    return _center(traces, _np.nanmean(traces, axis=0, dtype=max(traces.dtype, 'float32')))


@preprocess
def standardize(traces):
    """Returns traces centered and normalized on the standard of all traces.

    Args:
        traces (numpy.ndarray): a 2 dimensional numpy array.

    Returns:
        (numpy.ndarray) traces subtracted of the mean on all traces and normalized on the standard of all traces.

    """
    return center(traces) / _np.nanstd(traces, axis=0, dtype=max(traces.dtype, 'float32'))


class StandardizeOn(Preprocess):
    """Returns a preprocess to compute standardization around provided mean and std values.

    If mean (resp. std) is not provided, the preprocess will use the mean (resp. std) value processed on traces.

    Args:
        mean (numpy.ndarray, default=None): use this array as mean value to center traces.
        std (numpy.ndarray, default=None): use this array as std value to normalize traces.
        precision (numpy.dtype, default='float32'): optional parameter to define minimum numerical precision used to perform computation.
            If input data has higher precision, it will be kept instead.

    Returns:
        (callable): preprocess function to compute standardization of traces around mean and std.

    """

    def __init__(self, mean=None, std=None, precision='float32'):
        self.mean = mean
        self.std = std
        self.precision = _np.dtype(precision)

    def __call__(self, traces):
        precision = max(traces.dtype, self.precision)
        _mean = self.mean if self.mean is not None else _np.nanmean(traces, axis=0, dtype=precision)
        _std = self.std if self.std is not None else _np.nanstd(traces, axis=0, dtype=precision)
        try:
            return (traces - _mean) / _std
        except ValueError:
            raise PreprocessError(f'Incompatible shapes between traces {traces.shape} and mean {_mean.shape} and/or std {_std.shape}.')


class CenterOn(Preprocess):
    """Returns a preprocess to compute centering traces around provided mean.

    Args:
        mean (numpy.ndarray): use this array as mean value to center traces.
        precision (numpy.dtype, default='float32'): optional parameter to define minimum numerical precision used to perform computation.
            If input data has higher precision, it will be kept instead.

    Returns:
        (callable): preprocess function to compute center of traces around mean.

    """

    def __init__(self, mean=None, precision='float32'):
        self.mean = mean
        self.precision = _np.dtype(precision)

    def __call__(self, traces):
        return _center(traces.astype(max(traces.dtype, self.precision)), self.mean)


class ToPower(Preprocess):
    def __init__(self, power: int = 1, precision='float32'):
        if not isinstance(power, (int, float)):
            raise ValueError(f'power must be an integer, not {type(power)}.')
        self.power = power
        self.precision = _np.dtype(precision)

    def __call__(self, traces):
        return _np.power(traces, self.power, dtype=max(traces.dtype, self.precision))
