import numpy as _np


def _is_bytes_array(array):
    if not isinstance(array, _np.ndarray):
        raise TypeError(f'array should be a Numpy ndarray instance, not {type(array)}.')
    if not array.dtype == 'uint8':
        raise ValueError(f'array should be a byte array with uint8 dtype, not {array.dtype}.')
    return True
