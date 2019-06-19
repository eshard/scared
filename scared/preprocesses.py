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
