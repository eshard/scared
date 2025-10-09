import numpy as np
import pytest
import itertools
import tracemalloc
from scared.utils.inplace_dot_sum import inplace_dot_sum as _inplace_dot_sum, _check_matrix, _are_all_contiguous
import scared


def make_array(shape, dtype, order):
    if order == "C":
        arr = np.arange(np.prod(shape), dtype=dtype).reshape(shape)
        result = np.array(arr, order='C')
        assert result.flags.c_contiguous
    elif order == "F":
        arr = np.arange(np.prod(shape), dtype=dtype).reshape(shape)
        result = np.array(arr, order='F')
        assert result.flags.f_contiguous
    elif order == "N":  # non-contiguous
        arr = np.arange(np.prod(shape) * 4, dtype=dtype).reshape((shape[1] * 2, shape[0] * 2))
        result = np.array(arr, order='C').T[::2, ::2]  # slicing makes it non-contiguous
        assert not result.flags.c_contiguous and not result.flags.f_contiguous
    assert result.shape == shape
    assert result.dtype == np.dtype(dtype)
    return result


# --- Shape validation helper ------------------------------------------------
def assert_same_shape(c, expected):
    assert c.shape == expected.shape, f"Expected shape {expected.shape}, got {c.shape}"


# --- Basic validation -------------------------------------------------------
def test_check_matrix_invalid_inputs():
    with pytest.raises(TypeError, match='`a` must be a ndarray'):
        _check_matrix(42, 'a')
    with pytest.raises(ValueError, match='`a` must be a 2D ndarray'):
        _check_matrix(np.arange(5, dtype=np.float32), 'a')


def test_check_matrix_null_dim():
    with pytest.raises(ValueError, match='`a` has a null dimension'):
        _check_matrix(np.random.rand(0, 42), 'a')
    with pytest.raises(ValueError, match='`b` has a null dimension'):
        _check_matrix(np.random.rand(42, 0), 'b')


@pytest.mark.parametrize("a_order, b_order", itertools.product(["C", "F", "N"], repeat=2))
def test_are_all_contiguous(a_order, b_order):
    a = make_array((13, 42), dtype='float64', order=a_order)
    b = make_array((42, 13), dtype='float64', order=b_order)
    expected = False if "N" in [a_order, b_order] else True
    assert _are_all_contiguous(a, b) is expected


# --- Core correctness -------------------------------------------------------

def total_rel_abs_diff(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    abs_diff = np.abs(a - b)
    denom = np.abs(a)  # use values from the first array as reference
    rel_diff = np.where(denom != 0, abs_diff / denom, np.inf)
    return np.sum(rel_diff)


M, K, N = 123, 456, 789
RTOL = 5e-6
RTOL64 = 1e-12
PERCENT_PRESICION_LOSS = 50


def test_inplace_dot_sum_correctness_float32():
    dtype = np.float32
    a = make_array((M, K), dtype=dtype, order='F')
    b = make_array((K, N), dtype=dtype, order='F')
    c = make_array((M, N), dtype=dtype, order='F')
    expected64 = c.astype('float64') + (a.astype('float64') @ b.astype('float64'))
    expected = c + (a @ b)
    _inplace_dot_sum(a, b, c)
    assert total_rel_abs_diff(c, expected64) < total_rel_abs_diff(expected, expected64) * (1 + PERCENT_PRESICION_LOSS / 100)
    assert_same_shape(c, expected)
    np.testing.assert_allclose(c, expected64, rtol=RTOL)


def test_inplace_dot_sum_correctness_float64():
    dtype = np.float64
    a = make_array((M, K), dtype=dtype, order='F')
    b = make_array((K, N), dtype=dtype, order='F')
    c = make_array((M, N), dtype=dtype, order='F')
    expected = c + (a @ b)
    _inplace_dot_sum(a, b, c)
    assert_same_shape(c, expected)
    np.testing.assert_allclose(c, expected, rtol=RTOL64)


def test_inplace_dot_sum_transposition1():
    dtype = 'float32'
    a = make_array((M, K), dtype=dtype, order='F')
    c = make_array((M, N), dtype=dtype, order='F')
    e = make_array((N, K), dtype=dtype, order='F')
    expected = c + a @ e.T
    _inplace_dot_sum(a, e.T, c)
    np.testing.assert_allclose(c, expected, rtol=2 * RTOL)


def test_inplace_dot_sum_transposition2():
    dtype = 'float32'
    b = make_array((K, N), dtype=dtype, order='F')
    c = make_array((M, N), dtype=dtype, order='F')
    d = make_array((K, M), dtype=dtype, order='F')
    expected = c + d.T @ b
    _inplace_dot_sum(d.T, b, c)
    np.testing.assert_allclose(c, expected, rtol=2 * RTOL)


def test_inplace_dot_sum_transposition3():
    dtype = 'float32'
    a = make_array((M, K), dtype=dtype, order='F')
    b = make_array((K, N), dtype=dtype, order='F')
    f = make_array((N, M), dtype=dtype, order='F')
    expected = f.T + a @ b
    _inplace_dot_sum(a, b, f.T)
    np.testing.assert_allclose(f.T, expected, rtol=RTOL)


@pytest.mark.parametrize("a_dtype, b_dtype, c_dtype", itertools.product([np.float32, np.float64], repeat=3))
def test_fallback_different_dtype(a_dtype, b_dtype, c_dtype):
    a = make_array((M, K), dtype=a_dtype, order='F')
    b = make_array((K, N), dtype=b_dtype, order='F')
    c = make_array((M, N), dtype=c_dtype, order='F')
    expected64 = c.astype('float64') + (a.astype('float64') @ b.astype('float64'))
    _inplace_dot_sum(a, b, c)
    assert_same_shape(c, expected64)
    np.testing.assert_allclose(c, expected64, rtol=RTOL)


def test_null_dim():
    a = np.random.rand(M, K)
    b = np.random.rand(K, N)
    c = np.random.rand(M, N)
    a0 = np.random.rand(M, 0)
    b0 = np.random.rand(K, 0)
    c0 = np.random.rand(M, 0)
    with pytest.raises(ValueError, match='`a` has a null dimension'):
        _inplace_dot_sum(a0, b, c)
    with pytest.raises(ValueError, match='`b` has a null dimension'):
        _inplace_dot_sum(a, b0, c)
    with pytest.raises(ValueError, match='`c` has a null dimension'):
        _inplace_dot_sum(a, b, c0)


def test_sgemm_called_when_needed(mocker):
    dtype = 'float32'
    a = make_array((M, K), dtype=dtype, order='F')
    b = make_array((K, N), dtype=dtype, order='C')
    c = make_array((M, N), dtype=dtype, order='F')
    spy = mocker.spy(scared.utils.inplace_dot_sum, '_sgemm')
    _inplace_dot_sum(a, b, c)
    spy.assert_called_once()
    spy = mocker.spy(scared.utils.inplace_dot_sum, '_sgemm')
    _inplace_dot_sum(b.T, a.T, c.T)
    spy.assert_called_once()


def test_dgemm_called_when_needed(mocker):
    dtype = 'float64'
    a = make_array((M, K), dtype=dtype, order='F')
    b = make_array((K, N), dtype=dtype, order='F')
    c = make_array((M, N), dtype=dtype, order='C')
    spy = mocker.spy(scared.utils.inplace_dot_sum, '_dgemm')
    _inplace_dot_sum(a, b, c)
    spy.assert_called_once()
    spy = mocker.spy(scared.utils.inplace_dot_sum, '_dgemm')
    _inplace_dot_sum(b.T, a.T, c.T)
    spy.assert_called_once()


# --- Full order combination coverage ---------------------------------------
@pytest.mark.parametrize("a_order, b_order, c_order", itertools.product(["C", "F", "N"], repeat=3))
def test_all_memory_order_combinations(a_order, b_order, c_order):
    a = make_array((M, K), dtype='float64', order=a_order)
    b = make_array((K, N), dtype='float64', order=b_order)
    c = make_array((M, N), dtype='float64', order=c_order)
    expected = c + (a @ b)
    _inplace_dot_sum(a, b, c)
    assert_same_shape(c, expected)
    np.testing.assert_allclose(c, expected, rtol=RTOL64)


# --- Large functional coverage for dtype + order ---------------------------
@pytest.mark.parametrize("orders", itertools.product(["C", "F", "N"], repeat=3))
@pytest.mark.parametrize("dtypes", itertools.product([np.float32, np.float64, np.float128], repeat=3))
def test_all_memory_order_and_dtype_combinations(orders, dtypes):
    a = make_array((13, 42), dtype=dtypes[0], order=orders[0])
    b = make_array((42, 7), dtype=dtypes[1], order=orders[1])
    c = make_array((13, 7), dtype=dtypes[2], order=orders[2])
    expected = c + (a @ b)
    excpected_dtype = c.dtype
    _inplace_dot_sum(a, b, c)
    assert_same_shape(c, expected)
    assert c.dtype == excpected_dtype


# --- Memory gain ----------------------------------------------------------
def test_does_not_double_memory():
    a = make_array((M, K), dtype='float64', order='F')
    b = make_array((K, N), dtype='float64', order='F')
    c = make_array((M, N), dtype='float64', order='F')
    tracemalloc.start()
    _ = c + (a @ b)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak > c.nbytes * 0.8
    tracemalloc.start()
    _inplace_dot_sum(a, b, c)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak < c.nbytes * 0.01
