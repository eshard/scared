"""Signal processing basic functionalities on numpy arrays."""

import numpy as _np


def pad(array, target_shape, offsets=None, pad_with=0):
    """Pad a given array with 'pad_with' values, according to the given shape.

    Args:
        array (numpy.ndarray): data to be padded.
        target_shape (list, tuple or ndarray): shape of the output.
        offsets (list, tuple or ndarray): of length array.ndim), offsets where to place 'array' (default: None). If 'None', all offsets are 0.
        pad_with (compatible with array.dtype): value used to pad (default: 0).

    Returns:
        (numpy.ndarray): A new array being filled with the correct parameters.

    Example:
        >>> a = np.array([[10, 11, 12], [13, 14, 15]])
        >>> a
        array([[10, 11, 12],
               [13, 14, 15]])

        >>> pad(a, (5, 5), offsets=[0, 0])
        array([[10, 11, 12,  0,  0],
               [13, 14, 15,  0,  0],
               [ 0,  0,  0,  0,  0],
               [ 0,  0,  0,  0,  0],
               [ 0,  0,  0,  0,  0]])

        >>> pad(a, (5, 5), offsets=[1, 2])
        array([[ 0,  0,  0,  0,  0],
               [ 0,  0, 10, 11, 12],
               [ 0,  0, 13, 14, 15],
               [ 0,  0,  0,  0,  0],
               [ 0,  0,  0,  0,  0]])

    """
    if not isinstance(array, _np.ndarray):
        raise TypeError(f"'array' should be a numpy ndarray type, not {type(array)}.")
    if offsets is None:
        offsets = tuple(0 for _ in range(array.ndim))
    if not hasattr(target_shape, '__len__'):
        raise TypeError(f"'target_shape' should be a ndarray, list or tuple, not {type(target_shape)}.")
    if not hasattr(offsets, '__len__'):
        raise TypeError(f"'offsets' should be a ndarray, list or tuple, not {type(offsets)}.")

    if len(target_shape) != array.ndim:
        raise ValueError("'target_shape' length should corresponds to the number of dimensions in 'array'.")
    if len(offsets) != array.ndim:
        raise ValueError("'offsets' length should corresponds to the number of dimensions in 'array'.")
    for dim in range(array.ndim):
        dimension_length = offsets[dim] + array.shape[dim]
        if dimension_length > target_shape[dim]:
            raise ValueError(f"'target_shape[{dim}] = {target_shape[dim]}' is too low to insert 'array' with an 'offset' {offsets[dim]}.")

    insert_here = []
    for dim in range(array.ndim):
        dimension_start = offsets[dim]
        dimension_length = offsets[dim] + array.shape[dim]
        insert_here.append(slice(dimension_start, dimension_length))
    result = _np.zeros(target_shape, dtype=array.dtype) + pad_with
    result[tuple(insert_here)] = array
    return result


def cast_array(array, dtype='float64'):
    """Cast 'array' to dtype.

    Cast only if expected dtype is different than the current one.
    Useful for performance considerations.

    Args:
        array (numpy.ndarray): ndarray to be casted.
        dtype (numpy.dtype): valid dtype (default: float64).

    Returns:
        (numpy.ndarray): The given array if no cast needed, or a new array with the correct cast.

    """
    if not isinstance(array, _np.ndarray):
        raise TypeError(f"'array' should be a numpy ndarray type, not {type(array)}.")
    try:
        return array.astype(dtype, copy=False)
    except TypeError:
        raise TypeError(f'dtype should be a valid dtype, not {dtype}.')
