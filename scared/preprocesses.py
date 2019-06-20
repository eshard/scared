import functools
import numpy as _np


class PreprocessError(Exception):
    """Error related to function decorated with preprocess."""

    pass


def preprocess(function):
    """Decorator to ensure that the given function proceeds to basic verification suitable for samples preprocessing.

    A preprocess function must expect one argument traces, which is expected to be a 2 dimension Numpy array.
    It must returns a 2 dimensions Numpy array, with first dimension unchanged (number of traces processed).

    """
    @functools.wraps(function)
    def _(traces):
        if not isinstance(traces, _np.ndarray):
            raise TypeError(f'preprocess expect Numpy ndarray, not {type(traces)}.')
        if traces.ndim != 2:
            raise ValueError(f'preprocess expect 2 dimension nparray, not {traces.ndim} dimensions array.')
        result = function(traces)
        if not isinstance(result, _np.ndarray):
            raise PreprocessError(f'Preprocess {function} does not returns correct typed results, but {type(result)}.')
        if result.ndim != 2:
            raise PreprocessError(f'Preprocess {function} returns array of dimension {result.ndim}, instead of 2.')
        if result.shape[0] != traces.shape[0]:
            raise PreprocessError(f'Preprocess {function} modifies number of traces dimension.')
        return result
    return _


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


def _center(traces, mean):
    try:
        return traces - mean
    except ValueError:
        raise PreprocessError(f'Incompatible shapes between traces {traces.shape} and mean {mean.shape}.')


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
    """Returns traces centered around the mean of all traces, and normalized on the standard of all traces.

    Args:
        traces (numpy.ndarray): a 2 dimensional numpy array.

    Returns:
        (numpy.ndarray) traces substracted of the mean on all traces and normalized on the standard of all traces.

    """
    return center(traces) / _np.nanstd(traces, axis=0)


def standardize_on(mean=None, std=None):
    """Returns a preprocess to compute standardization around provided mean and std values.

    If mean (resp. std) is not provided, the preprocess will use the mean (resp. std) value processed on traces.

    Args:
        mean (numpy.ndarray, default=None): use this array as mean value to center traces.
        std (numpy.ndarray, default=None): use this array as std value to normalize traces.

    Returns:
        (callable): preprocess function to compute standardization of traces around mean and std.

    """
    def _(traces):
        _mean = mean if mean is not None else _np.nanmean(traces, axis=0)
        _std = std if std is not None else _np.nanstd(traces, axis=0)
        try:
            return (traces - _mean) / _std
        except ValueError:
            raise PreprocessError(f'Incompatible shapes between traces {traces.shape} and mean {_mean.shape} and/or std {_std.shape}.')
    return preprocess(_)


def center_on(mean):
    """Returns a preprocess to compute centering traces around provided mean.

    Args:
        mean (numpy.ndarray): use this array as mean value to center traces.

    Returns:
        (callable): preprocess function to compute center of traces around mean.

    """
    def _(traces):
        return _center(traces, mean)
    return preprocess(_)
