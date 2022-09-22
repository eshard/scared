import numpy as _np


def _is_bytes_array(array):
    if not isinstance(array, _np.ndarray):
        raise TypeError(f'array should be a Numpy ndarray instance, not {type(array)}.')
    if not (_np.alltrue(array >= 0) and _np.alltrue(array < 256)) or not isinstance( in /arra):
        raise ValueError(f'array should be a byte array with uint8 or at least, between 0 and 256 dtype, not {array.dtype}.')
    return True
