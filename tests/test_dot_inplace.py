import numpy as np
import pytest

from scared.utils.utils import inplace_dot_sum


@pytest.mark.parametrize("dtype,rtol", [(np.float32, 1e-6), (np.float64, 1e-12)])
def test_inplace_dot_sum_correctness(dtype, rtol):
    a = np.random.randn(5, 3).astype(dtype, order="F")
    b = np.random.randn(3, 4).astype(dtype, order="F")
    c = np.zeros((5, 4), dtype=dtype, order="F")

    inplace_dot_sum(a, b, c)
    expected = a @ b

    np.testing.assert_allclose(c, expected, rtol=rtol)


def test_inplace_behavior():
    a = np.ones((2, 2), dtype=np.float32, order="F")
    b = np.ones((2, 2), dtype=np.float32, order="F")
    c = np.ones((2, 2), dtype=np.float32, order="F")

    inplace_dot_sum(a, b, c)

    np.testing.assert_array_equal(c, np.full((2, 2), 3, dtype=np.float32))


def test_fallback_different_dtype():
    a = np.ones((2, 2), dtype=np.float32)
    b = np.ones((2, 2), dtype=np.float64)
    c = np.ones((2, 2), dtype=np.float64)

    inplace_dot_sum(a, b, c)
    expected = np.ones_like(c) + a.astype(np.float64) @ b

    np.testing.assert_allclose(c, expected)


@pytest.mark.parametrize("order", ["C", "F"])
def test_c_and_f_contiguous(order):
    a = np.random.randn(3, 3).astype(np.float64, order=order)
    b = np.random.randn(3, 3).astype(np.float64, order=order)
    c = np.zeros((3, 3), dtype=np.float64, order=order)

    inplace_dot_sum(a, b, c)
    expected = a @ b

    np.testing.assert_allclose(c, expected)


def test_empty_matrices():
    a = np.empty((0, 3), dtype=np.float32)
    b = np.empty((3, 4), dtype=np.float32)
    c = np.empty((0, 4), dtype=np.float32)

    inplace_dot_sum(a, b, c)
    np.testing.assert_array_equal(c, np.empty((0, 4), dtype=np.float32))