import numpy as np


import pytest

from scared.utils.fast_astype import _data_order, fast_astype


@pytest.fixture
def sample_matrix_c():
    return np.arange(9, dtype=np.float64).reshape(3, 3)


@pytest.fixture
def sample_matrix_f():
    return np.arange(9, dtype=np.float64).reshape(3, 3).T


@pytest.fixture
def sample_matrix_none():
    return np.arange(6 * 6).reshape((6, 6))[::2, ::2]


# --- Tests for _data_order ------------------------------------------------

def test_data_order_c_contiguous(sample_matrix_c):
    """Should detect C-contiguous layout."""
    assert _data_order(sample_matrix_c) == "C"


def test_data_order_f_contiguous(sample_matrix_f):
    """Should detect F-contiguous layout."""
    assert _data_order(sample_matrix_f) == "F"


def test_data_order_fc_both():
    """A 1x1 matrix is both C- and F-contiguous."""
    a = np.array([[1.0]])
    assert _data_order(a) == "FC"


def test_data_order_not_contiguous(sample_matrix_none):
    assert _data_order(sample_matrix_none) == ""


# --- Tests for fast_astype ------------------------------------------------

@pytest.mark.parametrize('sample_matrix', ['sample_matrix_c', 'sample_matrix_f'])
def test_fast_astype_no_conversion_needed(sample_matrix, request):
    """If dtype and memory order are already correct, the same object should be returned."""
    sample_matrix = request.getfixturevalue(sample_matrix)
    res = fast_astype(sample_matrix, dtype=sample_matrix.dtype, order=_data_order(sample_matrix))
    assert res is sample_matrix


@pytest.mark.parametrize('sample_matrix', ['sample_matrix_c', 'sample_matrix_f'])
def test_fast_astype_type_conversion(sample_matrix, request):
    """Type conversion from float64 → float32."""
    sample_matrix = request.getfixturevalue(sample_matrix)
    res = fast_astype(sample_matrix, dtype="float32", order="C")
    assert res.dtype == np.float32
    np.testing.assert_allclose(res, sample_matrix.astype(np.float32))
    assert not (res is sample_matrix)


def test_fast_astype_order_conversion(sample_matrix_c):
    """Memory order conversion from C → F."""
    fmat = fast_astype(sample_matrix_c, dtype="float64", order="F")
    assert fmat.flags.f_contiguous
    assert not fmat.flags.c_contiguous
    np.testing.assert_array_equal(fmat, sample_matrix_c)


def test_fast_astype_type_and_order_conversion(sample_matrix_c):
    """Simultaneous dtype and memory order conversion."""
    res = fast_astype(sample_matrix_c, dtype="float32", order="F")
    assert res.dtype == np.float32
    assert res.flags.f_contiguous
    np.testing.assert_allclose(res, sample_matrix_c.astype(np.float32))


def test_fast_astype_invalid_order(sample_matrix_c):
    """Invalid memory order should raise ValueError."""
    with pytest.raises(ValueError, match='Invalid order'):
        fast_astype(sample_matrix_c, order="Z")


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


@pytest.mark.parametrize('sample_matrix', ['sample_matrix_c', 'sample_matrix_f', 'sample_matrix_none'])
@pytest.mark.parametrize('out_order', ['F', 'C'])
@pytest.mark.parametrize('out_dtype', ['float32', 'float64', 'uint8'])
def test_fast_as_type_combinations(sample_matrix, out_dtype, out_order, request):
    res = fast_astype(request.getfixturevalue(sample_matrix), out_dtype, out_order)
    assert res.dtype == np.dtype(out_dtype)
    assert _data_order(res) == out_order
