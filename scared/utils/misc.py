import numpy as _np
import platform as _platform


def _is_bytes_array(array):
    # Note: Integer arrays cannot contain np.nan or np.inf
    if not isinstance(array, _np.ndarray):
        raise TypeError(f'array should be a Numpy ndarray instance, not {type(array)}.')
    if array.dtype == _np.uint8:
        return True
    if array.dtype.kind not in 'ui':
        raise ValueError(f'array should be an integer array, not {array.dtype}.')
    if array.dtype.kind == 'i' and _np.min(array) < 0:
        raise ValueError(f'array should be a bytes array, i.e with values in [0, 255], but lowest value {_np.min(array)} found.')
    if array.dtype != _np.int8 and _np.max(array) > 255:
        raise ValueError(f'array should be a bytes array, i.e with values in [0, 255], but highest value {_np.max(array)} found.')
    return True


def _use_parallel():
    arch = _platform.machine().lower()
    return 'x86_64' in arch or 'amd64' in arch
