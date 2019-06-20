import functools
import numpy as _np
import abc


class PreprocessError(Exception):
    """Error related to function preprocess decorated functions."""

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


class _MetaPreprocess(abc.ABCMeta):

    def __new__(mcls, name, bases, namespace, **kwargs):  # noqa: N804
        cls = super().__new__(mcls, name, bases, namespace, **kwargs)
        base_func = cls.__call__

        def pre_call(obj, traces):
            @preprocess
            def _(traces):
                return base_func(obj, traces)
            return _(traces)

        cls.__call__ = abc.abstractmethod(pre_call)
        return cls


class Preprocess(metaclass=_MetaPreprocess):
    """Base class to build complex preprocess callable - where simple @preprocess decorator is not enough.

    To  define a new class preprocess, inherit from this class and implements at least a `__call__` method.
    """

    @abc.abstractmethod
    def __call__(self, traces):
        pass
