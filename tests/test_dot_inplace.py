import numpy as np
import pytest

from scared.utils.inplace_dot_sum import inplace_dot_sum as _inplace_dot_sum, _check_matrix
import tracemalloc
import itertools


def test_check_matrix_invalid_inputs():
    with pytest.raises(TypeError):
        _check_matrix(42, 'a')
    with pytest.raises(ValueError):
        _check_matrix(np.arange(5, dtype=np.float32), 'a')


@pytest.mark.parametrize("dtype, rtol", [(np.float32, 1e-6), (np.float64, 1e-12)])
def test_inplace_dot_sum_correctness(dtype, rtol):
    a = np.random.randn(5, 3).astype(dtype, order="F")
    b = np.random.randn(3, 4).astype(dtype, order="F")
    c = np.zeros((5, 4), dtype=dtype, order="F")

    _inplace_dot_sum(a, b, c)
    expected = a @ b

    np.testing.assert_allclose(c, expected, rtol=rtol)


def test_inplace_behavior():
    a = np.ones((2, 2), dtype=np.float32, order="F")
    b = np.ones((2, 2), dtype=np.float32, order="F")
    c = np.ones((2, 2), dtype=np.float32, order="F")

    _inplace_dot_sum(a, b, c)

    np.testing.assert_array_equal(c, np.full((2, 2), 3, dtype=np.float32))


def test_fallback_different_dtype():
    a = np.ones((2, 2), dtype=np.float32)
    b = np.ones((2, 2), dtype=np.float64)
    c = np.ones((2, 2), dtype=np.float64)

    _inplace_dot_sum(a, b, c)
    expected = np.ones_like(c) + a.astype(np.float64) @ b

    np.testing.assert_allclose(c, expected)


@pytest.mark.parametrize("order", ["C", "F"])
def test_c_and_f_contiguous(order):
    a = np.random.randn(3, 3).astype(np.float64, order=order)
    b = np.random.randn(3, 3).astype(np.float64, order=order)
    c = np.zeros((3, 3), dtype=np.float64, order=order)

    _inplace_dot_sum(a, b, c)
    expected = a @ b

    np.testing.assert_allclose(c, expected)


def test_empty_matrices():
    a = np.empty((0, 3), dtype=np.float32)
    b = np.empty((3, 4), dtype=np.float32)
    c = np.empty((0, 4), dtype=np.float32)

    _inplace_dot_sum(a, b, c)
    np.testing.assert_array_equal(c, np.empty((0, 4), dtype=np.float32))


@pytest.mark.parametrize("a_order, b_order, c_order", itertools.product(["C", "F"], repeat=3))
def test_all_memory_order_combinations(a_order, b_order, c_order):
    a = np.random.randn(3, 3).astype(np.float64, order=a_order)
    b = np.random.randn(3, 3).astype(np.float64, order=b_order)
    c = np.zeros((3, 3), dtype=np.float64, order=c_order)

    expected = a @ b
    _inplace_dot_sum(a, b, c)

    np.testing.assert_allclose(c, expected, rtol=1e-12)


# ---- Test non-contiguous arrays ----
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_non_contiguous_arrays(dtype):
    a = np.arange(12, dtype=dtype)[::2].reshape(3, 2)  # non-contiguous
    b = np.arange(6, dtype=dtype).reshape(2, 3)
    c = np.zeros((3, 3), dtype=dtype)

    _inplace_dot_sum(a, b, c)
    expected = a @ b
    assert c.shape == expected.shape
    rtol = 1e-5 if dtype == np.float32 else 1e-12
    np.testing.assert_allclose(c, expected, rtol=rtol)


@pytest.mark.parametrize("a_shape,b_shape", [((4, 3), (3, 2)), ((2, 4), (4, 3))])
def test_inplace_dot_sum_rectangular(a_shape, b_shape):
    a = np.random.randn(*a_shape).astype(np.float64, order="F")
    b = np.random.randn(*b_shape).astype(np.float64, order="F")
    c = np.zeros((a_shape[0], b_shape[1]), dtype=np.float64, order="F")

    expected = a @ b
    _inplace_dot_sum(a, b, c)
    np.testing.assert_allclose(c, expected, rtol=1e-12)


@pytest.mark.parametrize("orders", itertools.product(["C", "F"], repeat=3))
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_inplace_dot_sum_all_order_combinations(dtype, orders):
    a = np.random.randn(20, 30).astype(dtype, order=orders[0])
    b = np.random.randn(30, 10).astype(dtype, order=orders[1])
    c = np.zeros((20, 10), dtype=dtype, order=orders[2])

    expected = a @ b
    _inplace_dot_sum(a, b, c)
    np.testing.assert_allclose(c, expected, rtol=1e-5 if dtype == np.float32 else 1e-12)


def test_does_not_double_memory():
    a = np.random.randn(100, 100).astype(np.float32)
    b = np.random.randn(100, 100).astype(np.float32)
    c = np.zeros((100, 100), dtype=np.float32)

    tracemalloc.start()
    _inplace_dot_sum(a, b, c)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert peak < c.nbytes * 2
