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
    return traces ** 2


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
        (numpy.ndarray) traces substracted of the mean on all traces.

    """
    return _center(traces, _np.nanmean(traces, axis=0))


@preprocess
def standardize(traces):
    """Returns traces centered and normalized on the standard of all traces.

    Args:
        traces (numpy.ndarray): a 2 dimensional numpy array.

    Returns:
        (numpy.ndarray) traces substracted of the mean on all traces and normalized on the standard of all traces.

    """
    return center(traces) / _np.nanstd(traces, axis=0)


class StandardizeOn(Preprocess):
    """Returns a preprocess to compute standardization around provided mean and std values.

    If mean (resp. std) is not provided, the preprocess will use the mean (resp. std) value processed on traces.

    Args:
        mean (numpy.ndarray, default=None): use this array as mean value to center traces.
        std (numpy.ndarray, default=None): use this array as std value to normalize traces.

    Returns:
        (callable): preprocess function to compute standardization of traces around mean and std.

    """

    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def __call__(self, traces):
        _mean = self.mean if self.mean is not None else _np.nanmean(traces, axis=0)
        _std = self.std if self.std is not None else _np.nanstd(traces, axis=0)
        try:
            return (traces - _mean) / _std
        except ValueError:
            raise PreprocessError(f'Incompatible shapes between traces {traces.shape} and mean {_mean.shape} and/or std {_std.shape}.')


class CenterOn(Preprocess):
    """Returns a preprocess to compute centering traces around provided mean.

    Args:
        mean (numpy.ndarray): use this array as mean value to center traces.

    Returns:
        (callable): preprocess function to compute center of traces around mean.

    """

    def __init__(self, mean):
        self.mean = mean

    def __call__(self, traces):
        return _center(traces, self.mean)
