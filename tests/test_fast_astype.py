import numpy as np
import pytest
from scared.utils.fast_astype import fast_astype, _data_order

# --- Fixtures -------------------------------------------------------------

@pytest.fixture
def sample_matrix():
    return np.arange(9, dtype=np.float64).reshape(3, 3)


# --- Tests sur _data_order ------------------------------------------------

def test_data_order_c_contiguous(sample_matrix):
    assert _data_order(sample_matrix) == "C"

def test_data_order_f_contiguous(sample_matrix):
    fmat = np.asfortranarray(sample_matrix)
    assert _data_order(fmat) == "F"

def test_data_order_fc_both():
    a = np.array([[1.0]])  # 1x1 is both C and F contiguous
    assert _data_order(a) == "FC"


# --- Tests sur fast_astype ------------------------------------------------

def test_fast_astype_no_conversion_needed(sample_matrix):
    """Si le dtype et l'ordre sont déjà bons, on doit retourner le même objet."""
    res = fast_astype(sample_matrix, dtype="float64", order="C")
    assert res is sample_matrix


def test_fast_astype_type_conversion(sample_matrix):
    """Conversion du dtype float64 -> float32"""
    res = fast_astype(sample_matrix, dtype="float32", order="C")
    assert res.dtype == np.float32
    np.testing.assert_allclose(res, sample_matrix.astype(np.float32))
    assert not (res is sample_matrix)


def test_fast_astype_order_conversion(sample_matrix):
    """Conversion d'ordre mémoire C -> F"""
    fmat = fast_astype(sample_matrix, dtype="float64", order="F")
    assert fmat.flags.f_contiguous
    assert not fmat.flags.c_contiguous
    np.testing.assert_array_equal(fmat, sample_matrix)


def test_fast_astype_type_and_order_conversion(sample_matrix):
    """Conversion simultanée de type et d’ordre"""
    res = fast_astype(sample_matrix, dtype="float32", order="F")
    assert res.dtype == np.float32
    assert res.flags.f_contiguous
    np.testing.assert_allclose(res, sample_matrix.astype(np.float32))


def test_fast_astype_invalid_order(sample_matrix):
    """Ordre invalide → ValueError"""
    with pytest.raises(ValueError):
        fast_astype(sample_matrix, order="Z")


def test_fast_astype_non_2d_array():
    """Les tableaux non 2D tombent sur np.astype standard"""
    a = np.arange(5, dtype=np.float32)
    res = fast_astype(a, dtype="float64", order="C")
    np.testing.assert_array_equal(res, a.astype(np.float64))


def test_fast_astype_empty_matrix():
    """Cas limite : matrice vide"""
    a = np.empty((0, 3), dtype=np.float32)
    res = fast_astype(a, dtype="float64", order="C")
    assert res.shape == (0, 3)
    assert res.dtype == np.float64
