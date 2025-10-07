from scipy.linalg.blas import dgemm as _dgemm, sgemm as _sgemm
import numpy as _np


def _check_matrix(m, name='array'):
    if not isinstance(m, _np.ndarray):
        raise TypeError(f'`{name}` must be a ndarray, but {type(m)} found.')
    if ndim:=m.ndim != 2:
        raise ValueError(f'`{name}` must be a 2D ndarray, but {ndim}D found.')


def _have_same_dtype(*args):
    ref = args[0].dtype
    for arg in args[1:]:
        if arg.dtype != ref:
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

    Returns:
        The matrix c = c + a @ b

    """

    if 0 in a.shape or 0 in b.shape or 0 in c.shape:
        return c
    _check_matrix(a, 'a'), _check_matrix(b, 'b'), _check_matrix(c, 'c')
    if a.dtype not in [_np.float32, _np.float64] or not _have_same_dtype(a, b, c):
        c[:] = c + _np.dot(a, b)
    gemm = _sgemm if a.dtype == _np.float32 else _dgemm
    (a, transpose_a) = (a.T, True) if a.flags.c_contiguous else (a, False)
    (b, transpose_b) = (b.T, True) if b.flags.c_contiguous else (b, False)

    if c.flags.c_contiguous:  # (a @ b).T = b.T @ a.T
        c = c.T
        (tmp_b, tmp_transpose_b) = (b, not transpose_b)
        (b, transpose_b) = (a, not transpose_a)
        (a, transpose_a) = (tmp_b, tmp_transpose_b)

    gemm(alpha=1, a=a, b=b, c=c, beta=1.0, overwrite_c=True,
         trans_a=transpose_a, trans_b=transpose_b)
