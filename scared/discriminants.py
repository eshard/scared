import numpy as _np
import functools


def discriminant(function):
    """Decorator that build a discriminant from the function.

    A discriminant is a function that takes a numeric array as input and returns
    array reduced over its last dimension, applying some kind of operation on it.

    Args:
        function (callable): a function which takes a data Numpy ndarray as argument
            and returns data reduced over the last dimension.
    Returns:
        (callable): resulting discriminant function.

    """
    @functools.wraps(function)
    def disc(data, axis=-1):
        """Computes discriminant operation over the axis dimension of data array.

        Args:
            data (numpy.ndarray): a numeric numpy ndarray.
            axis (integer): axis on which the discriminant must be applied.

        """
        if not isinstance(data, _np.ndarray) or data.dtype.kind not in ('b', 'i', 'u', 'f', 'c'):
            raise TypeError(f'data must be a numeric array, not {data}.')

        if axis == -1:
            axis = len(data.shape) - 1
        results = function(data, axis=axis)
        if not isinstance(results, _np.ndarray) or results.dtype.kind not in ('b', 'i', 'u', 'f', 'c'):
            raise ValueError(f'Discriminant {function} do not preserve data type.')
        final_shape = tuple([d for i, d in enumerate(data.shape) if i != axis])
        if results.shape != final_shape:
            raise ValueError(f'Discriminant instance {function} do not returns array reduced on {axis} dimension.')
        return results

    if function.__doc__ is not None:
        disc.__doc__ += function.__doc__
    return disc


@discriminant
def nanmax(data, axis):
    """Nan Max discriminant.

    Returns:
        (numpy.ndarray) nanmax value over the last dimension.

    """
    return _np.nanmax(data, axis=axis)


@discriminant
def maxabs(data, axis):
    """Nan absolute max discriminant.

    Returns:
        (numpy.ndarray) nanmax over absolute values of the last dimension.

    """
    return _np.nanmax(_np.absolute(data), axis=axis)


@discriminant
def opposite_min(data, axis):
    """Opposite of min value.

    Returns:
        (numpy.ndarray) opposite of min value over the last dimension.

    """
    return _np.nanmax(-data, axis=axis)


@discriminant
def nansum(data, axis):
    """Sum discriminant function.

    Returns:
        (numpy.ndarray) sum over the last dimension.

    """
    return _np.nansum(data, axis=axis)


@discriminant
def abssum(data, axis):
    """Sum on absolute discriminant function.

    Returns:
        (numpy.ndarray) sum over absolute values over the last dimension.

    """
    return _np.nansum(_np.abs(data), axis=axis)
