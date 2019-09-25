import pytest
import numpy as np

from ..context import scared  # noqa: F401


def test_pad_fills_array_with_the_given_shape():
    array = np.array([[10, 11, 12], [13, 14, 15]])
    result = scared.signal_processing.pad(array, target_shape=(5, 5))
    expected = np.array([[10, 11, 12, 0, 0],
                         [13, 14, 15, 0, 0],
                         [0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0]])
    assert np.array_equal(result, expected)


def test_pad_fills_array_correctly_with_offset():
    array = np.array([[10, 11, 12], [13, 14, 15]])
    result = scared.signal_processing.pad(array, target_shape=(5, 5), offsets=[1, 2])
    expected = np.array([[0, 0, 0, 0, 0],
                         [0, 0, 10, 11, 12],
                         [0, 0, 13, 14, 15],
                         [0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0]])
    assert np.array_equal(result, expected)


def test_pad_fills_array_correctly_with_custom_value():
    array = np.array([[10, 11, 12], [13, 14, 15]])
    result = scared.signal_processing.pad(array, target_shape=(5, 5), pad_with=-2.1)
    expected = np.array([[10, 11, 12, -2.1, -2.1],
                         [13, 14, 15, -2.1, -2.1],
                         [-2.1, -2.1, -2.1, -2.1, -2.1],
                         [-2.1, -2.1, -2.1, -2.1, -2.1],
                         [-2.1, -2.1, -2.1, -2.1, -2.1]])
    assert np.array_equal(result, expected)


def test_pad_raises_exception_with_array_not_being_numpy_array():
    with pytest.raises(TypeError):
        scared.signal_processing.pad([1, 2, 3], target_shape=(5, 5))


def test_pad_raises_exception_with_target_shape_of_incorrect_type():
    array = np.array([[10, 11, 12], [13, 14, 15]])
    with pytest.raises(TypeError):
        scared.signal_processing.pad(array, target_shape=5)


def test_pad_raises_exception_with_offsets_of_incorrect_type():
    array = np.array([[10, 11, 12], [13, 14, 15]])
    with pytest.raises(TypeError):
        scared.signal_processing.pad(array, target_shape=(5, 5), offsets=5)


def test_pad_raises_exception_with_target_shape_having_wrong_number_of_elements():
    array = np.array([[10, 11, 12], [13, 14, 15]])
    with pytest.raises(ValueError):
        scared.signal_processing.pad(array, target_shape=(5,))
    with pytest.raises(ValueError):
        scared.signal_processing.pad(array, target_shape=(5, 5, 5))


def test_pad_raises_exception_with_offsets_having_wrong_number_of_elements():
    array = np.array([[10, 11, 12], [13, 14, 15]])
    with pytest.raises(ValueError):
        scared.signal_processing.pad(array, target_shape=(5, 5), offsets=(1,))
    with pytest.raises(ValueError):
        scared.signal_processing.pad(array, target_shape=(5, 5), offsets=(1, 2, 3))


def test_pad_raises_exception_with_target_shape_having_elements_of_too_small_size():
    array = np.array([[10, 11, 12], [13, 14, 15]])
    with pytest.raises(ValueError):
        scared.signal_processing.pad(array, target_shape=(2, 2))
    with pytest.raises(ValueError):
        scared.signal_processing.pad(array, target_shape=(5, 5), offsets=(3, 3))


def test_cast_array_casts_to_the_given_dtype():
    array = np.array([[10, 11, 12], [13, 14, 15]], dtype=np.float32)
    result_array = scared.signal_processing.cast_array(array, dtype="float64")
    assert result_array.dtype == np.float64


def test_cast_array_does_not_change_dtype_if_it_is_already_correct():
    array = np.array([[10, 11, 12], [13, 14, 15]], dtype=np.float32)
    initial_dtype = array.dtype
    result_array = scared.signal_processing.cast_array(array, dtype="float32")
    assert result_array.dtype is initial_dtype


def test_cast_array_raises_exception_with_array_no_being_numpy_ndarray():
    with pytest.raises(TypeError):
        scared.signal_processing.cast_array([1, 2, 3], dtype="float32")


def test_cast_array_raises_exception_with_dtype_being_wrong_str():
    array = np.array([[10, 11, 12], [13, 14, 15]])
    with pytest.raises(TypeError):
        scared.signal_processing.cast_array(array, dtype="foo")
    with pytest.raises(TypeError):
        scared.signal_processing.cast_array(array, dtype="float54")
