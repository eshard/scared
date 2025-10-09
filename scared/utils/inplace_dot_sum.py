from scipy.linalg.blas import dgemm as _dgemm, sgemm as _sgemm
import numpy as _np


def _check_matrix(m, name='array'):
    if not isinstance(m, _np.ndarray):
        raise TypeError(f'`{name}` must be a ndarray, but {type(m)} found.')
    if m.ndim != 2:
        raise ValueError(f'`{name}` must be a 2D ndarray, but {m.ndim}D found.')
    if 0 in m.shape:
        raise ValueError(f'`{name}` has a null dimension. `{name}` shape is {m.shape}.')


def _have_same_dtype(*args):
    ref = args[0].dtype
    for arg in args[1:]:
        if arg.dtype != ref:
            return False
    return True


def _are_all_contiguous(*args):
    for arg in args:
        if not arg.flags.c_contiguous and not arg.flags.f_contiguous:
            return False
    return True


def inplace_dot_sum(a, b, c):
    """Do inplace c += a @ b.

    Args:
        a (ndarray): 2D ndarray
        b (ndarray): 2D ndarray
        c (ndarray): 2D ndarray

    This function is efficient only for float32 and float64 arrays. A fallback is used for any other dtype.
    All the input matrices must have the same dtype. If not, the fallback is used.

    Note:
        The calculus is performed inplace, i.e. the matrix `c` is updated during the calculus.
        Note that implies that the matrix C cannot be casted.
        So, when the fallback is triggered, it can result in memory loss if not used carefully.

    Returns:
        The matrix c = c + a @ b

    """
    _check_matrix(a, 'a'), _check_matrix(b, 'b'), _check_matrix(c, 'c')

    # Fallback if dtype is not float32/64 or dtypes mismatch
    if a.dtype not in [_np.float32, _np.float64] or not _have_same_dtype(a, b, c) or not _are_all_contiguous(a, b, c):
        c[:] = c + _np.dot(a, b)
        return
    # Select optimized GEMM function
    gemm = _sgemm if a.dtype == _np.float32 else _dgemm
    (a, transpose_a) = (a.T, True) if a.flags.c_contiguous else (a, False)
    (b, transpose_b) = (b.T, True) if b.flags.c_contiguous else (b, False)

    # Handle c contiguous: swap a and b + transpose logic to match GEMM expectations
    if c.flags.c_contiguous:  # (a @ b).T = b.T @ a.T
        c = c.T
        (tmp_b, tmp_transpose_b) = (b, not transpose_b)
        (b, transpose_b) = (a, not transpose_a)
        (a, transpose_a) = (tmp_b, tmp_transpose_b)

    gemm(alpha=1, a=a, b=b, c=c, beta=1.0, overwrite_c=True,
         trans_a=transpose_a, trans_b=transpose_b)
