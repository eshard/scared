import numba as nb
import numpy as np


@nb.njit(parallel=True)
def _fast_astype_core_f(data, out):
    for i in nb.prange(data.shape[1]):
        out[:, i] = data[:, i]


@nb.njit(parallel=True)
def _fast_astype_core_c(data, out):
    for i in nb.prange(data.shape[0]):
        out[i] = data[i]


def _data_order(data):
    if data.flags.c_contiguous and data.flags.f_contiguous:
        return 'FC'
    if data.flags.c_contiguous:
        return 'C'
    if data.flags.f_contiguous:
        return 'F'
    return ''


def fast_astype(data, dtype='float32', order='C'):
    if not isinstance(data, np.ndarray):
        raise TypeError(f'data to cast must be a ndarray, but {type(data)} found.')
    dtype = np.dtype(dtype)
    if data.dtype == dtype and order in _data_order(data):
        return data
    if data.ndim != 2:
        return data.astype(dtype=dtype, order=order)
    if order.upper() == 'C':
        func = _fast_astype_core_c
    elif order.upper() == 'F':
        func = _fast_astype_core_f
    else:
        raise ValueError(f'Invalid order {order}, only F and C allowed.')
    out = np.empty_like(data, order=order, dtype=dtype)
    func(data, out)
    return out
