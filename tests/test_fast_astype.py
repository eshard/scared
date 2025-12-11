import numpy as np


import pytest

from scared.utils.fast_astype import _data_order, fast_astype


@pytest.fixture
def sample_matrix():
    return np.arange(9, dtype=np.float64).reshape(3, 3)


# --- Tests for _data_order ------------------------------------------------

def test_data_order_c_contiguous(sample_matrix):
    """Should detect C-contiguous layout."""
    assert _data_order(sample_matrix) == "C"


def test_data_order_f_contiguous(sample_matrix):
    """Should detect F-contiguous layout."""
    fmat = np.asfortranarray(sample_matrix)
    assert _data_order(fmat) == "F"


def test_data_order_fc_both():
    """A 1x1 matrix is both C- and F-contiguous."""
    a = np.array([[1.0]])
    assert _data_order(a) == "FC"


# --- Tests for fast_astype ------------------------------------------------

def test_fast_astype_no_conversion_needed(sample_matrix):
    """If dtype and memory order are already correct, the same object should be returned."""
    res = fast_astype(sample_matrix, dtype="float64", order="C")
    assert res is sample_matrix


def test_fast_astype_type_conversion(sample_matrix):
    """Type conversion from float64 → float32."""
    res = fast_astype(sample_matrix, dtype="float32", order="C")
    assert res.dtype == np.float32
    np.testing.assert_allclose(res, sample_matrix.astype(np.float32))
    assert not (res is sample_matrix)


def test_fast_astype_order_conversion(sample_matrix):
    """Memory order conversion from C → F."""
    fmat = fast_astype(sample_matrix, dtype="float64", order="F")
    assert fmat.flags.f_contiguous
    assert not fmat.flags.c_contiguous
    np.testing.assert_array_equal(fmat, sample_matrix)


def test_fast_astype_type_and_order_conversion(sample_matrix):
    """Simultaneous dtype and memory order conversion."""
    res = fast_astype(sample_matrix, dtype="float32", order="F")
    assert res.dtype == np.float32
    assert res.flags.f_contiguous
    np.testing.assert_allclose(res, sample_matrix.astype(np.float32))


def test_fast_astype_invalid_order(sample_matrix):
    """Invalid memory order should raise ValueError."""
    with pytest.raises(ValueError):
        fast_astype(sample_matrix, order="Z")


def test_fast_astype_non_2d_array():
    """Non-2D arrays should fall back to NumPy’s standard astype."""
    a = np.arange(5, dtype=np.float32)
    res = fast_astype(a, dtype="float64", order="C")
    np.testing.assert_array_equal(res, a.astype(np.float64))


def test_fast_astype_empty_matrix():
    """Edge case: empty matrix."""
    a = np.empty((0, 3), dtype=np.float32)
    res = fast_astype(a, dtype="float64", order="C")
    assert res.shape == (0, 3)
    assert res.dtype == np.float64


def test_fast_astype_raises_if_wrong_type():
    with pytest.raises(TypeError, match='data to cast must be a ndarray'):
        fast_astype('foo')
