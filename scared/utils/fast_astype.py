import numba as nb

import numpy as np


@nb.njit(parallel=True)
def _fast_astype_core_F(data, out):
    for i in nb.prange(data.shape[1]):
        out[:, i] = data[:, i]


@nb.njit(parallel=True)
def _fast_astype_core_C(data, out):
    for i in nb.prange(data.shape[0]):
        out[i] = data[i]


def _data_order(data):
    if data.flags.c_contiguous and data.flags.f_contiguous:
        return 'FC'
    if data.flags.c_contiguous:
        return 'C'
    if data.flags.f_contiguous:
        return 'F'


def fast_astype(data, dtype='float32', order='C'):
    dtype = np.dtype(dtype)
    if data.dtype == dtype and order in _data_order(data):
        return data
    if data.ndim != 2:
        return data.astype(dtype=dtype, order=order)
    out = np.empty_like(data, order=order, dtype=dtype)
    if order.upper() == 'C':
        _fast_astype_core_C(data, out)
    elif order.upper() == 'F':
        _fast_astype_core_F(data, out)
    else:
        raise ValueError(f'Invalid order {order}.')
    return out
