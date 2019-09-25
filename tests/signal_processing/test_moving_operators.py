import numpy as np
import pytest
from scipy.stats import skew, kurtosis

from ..context import scared  # noqa: F401


@pytest.fixture
def data_1d():
    return np.random.randint(0, 256, (100))


@pytest.fixture
def data_uint8():
    return np.random.randint(0, 256, (100), dtype="uint8")


@pytest.fixture
def data_2d():
    return np.random.randint(0, 256, (100, 100))


def max_diff_percent(a, b):
    return np.max(np.abs(a - b) / np.minimum(a, b) * 100)


def test_moving_sum_returns_correct_array_with_1d_data(data_1d):
    moved_data = scared.signal_processing.moving_sum(data_1d, 10)
    reference_data = []
    for i in range(data_1d.shape[-1] - 10 + 1):
        reference_data.append(data_1d[i:i + 10].sum())
    reference_data = np.array(reference_data)
    assert np.array_equal(moved_data, reference_data)


def test_moving_sum_returns_correct_array_with_1d_uint8_data(data_uint8):
    moved_data = scared.signal_processing.moving_sum(data_uint8, 10)
    reference_data = []
    for i in range(data_uint8.shape[-1] - 10 + 1):
        reference_data.append(data_uint8[i:i + 10].sum())
    reference_data = np.array(reference_data)
    assert np.array_equal(moved_data, reference_data)


def test_moving_sum_returns_correct_array_with_2d_data_on_axis_1(data_2d):
    moved_data = scared.signal_processing.moving_sum(data_2d, 10, axis=1)
    reference_data = []
    for data in data_2d:
        reference_data.append(scared.signal_processing.moving_sum(data, 10))
    reference_data = np.array(reference_data)
    assert np.array_equal(moved_data, reference_data)


def test_moving_sum_returns_correct_array_with_2d_data_on_axis_0(data_2d):
    moved_data = scared.signal_processing.moving_sum(data_2d, 10, axis=0)
    reference_data = scared.signal_processing.moving_sum(data_2d.T, 10, axis=1).T
    assert np.array_equal(moved_data, reference_data)


def test_moving_sum_raises_exxception_with_data_not_being_a_numpy_ndarray():
    with pytest.raises(TypeError):
        scared.signal_processing.moving_sum([1, 2, 3], 10)


def test_moving_sum_raises_exception_with_window_size_of_wrong_type(data_1d):
    with pytest.raises(TypeError):
        scared.signal_processing.moving_sum(data_1d, "foo")
    with pytest.raises(TypeError):
        scared.signal_processing.moving_sum(data_1d, 10.1)


def test_moving_sum_raises_exception_with_window_size_being_negative_or_0(data_1d):
    with pytest.raises(ValueError):
        scared.signal_processing.moving_sum(data_1d, -5)
    with pytest.raises(ValueError):
        scared.signal_processing.moving_sum(data_1d, 0)


def test_moving_sum_raises_exception_with_axis_type_of_wrong_type(data_1d):
    with pytest.raises(TypeError):
        scared.signal_processing.moving_sum(data_1d, 10, axis="foo")


def test_moving_sum_raises_exception_with_axis_having_wrong_values(data_1d):
    with pytest.raises(ValueError):
        scared.signal_processing.moving_sum(data_1d, 10, axis=1)
    with pytest.raises(ValueError):
        scared.signal_processing.moving_sum(data_1d, 10, axis=-2)


def test_moving_sum_raises_exception_with_window_size_having_incorrect_value_regarding_to_data_shape(data_1d):
    with pytest.raises(ValueError):
        scared.signal_processing.moving_sum(data_1d, 150)


def test_moving_mean_returns_correct_array_with_1d_data(data_1d):
    moved_data = scared.signal_processing.moving_mean(data_1d, 10)
    reference_data = []
    for i in range(data_1d.shape[-1] - 10 + 1):
        reference_data.append(data_1d[i:i + 10].mean())
    reference_data = np.array(reference_data)
    assert np.array_equal(moved_data, reference_data)


def test_moving_mean_returns_correct_array_with_1d_uint8_data(data_uint8):
    moved_data = scared.signal_processing.moving_mean(data_uint8, 10)
    reference_data = []
    for i in range(data_uint8.shape[-1] - 10 + 1):
        reference_data.append(data_uint8[i:i + 10].mean())
    reference_data = np.array(reference_data)
    assert max_diff_percent(moved_data, reference_data) < 1e-6


def test_moving_mean_raises_exxception_with_data_not_being_a_numpy_ndarray():
    with pytest.raises(TypeError):
        scared.signal_processing.moving_mean([1, 2, 3], 10)


def test_moving_mean_raises_exception_with_window_size_of_wrong_type(data_1d):
    with pytest.raises(TypeError):
        scared.signal_processing.moving_mean(data_1d, "foo")
    with pytest.raises(TypeError):
        scared.signal_processing.moving_mean(data_1d, 10.1)


def test_moving_mean_raises_exception_with_window_size_being_negative_or_0(data_1d):
    with pytest.raises(ValueError):
        scared.signal_processing.moving_mean(data_1d, -5)
    with pytest.raises(ValueError):
        scared.signal_processing.moving_mean(data_1d, 0)


def test_moving_mean_raises_exception_with_axis_type_of_wrong_type(data_1d):
    with pytest.raises(TypeError):
        scared.signal_processing.moving_mean(data_1d, 10, axis="foo")


def test_moving_mean_raises_exception_with_axis_having_wrong_values(data_1d):
    with pytest.raises(ValueError):
        scared.signal_processing.moving_mean(data_1d, 10, axis=1)
    with pytest.raises(ValueError):
        scared.signal_processing.moving_mean(data_1d, 10, axis=-2)


def test_moving_mean_raises_exception_with_window_size_having_incorrect_value_regarding_to_data_shape(data_1d):
    with pytest.raises(ValueError):
        scared.signal_processing.moving_mean(data_1d, 150)


def test_moving_var_returns_correct_array_with_1d_data(data_1d):
    moved_data = scared.signal_processing.moving_var(data_1d, 10)
    reference_data = []
    for i in range(data_1d.shape[-1] - 10 + 1):
        reference_data.append(data_1d[i:i + 10].var())
    reference_data = np.array(reference_data)
    assert max_diff_percent(moved_data, reference_data) < 1e-6


def test_moving_var_returns_correct_array_with_1d_uint8_data(data_uint8):
    moved_data = scared.signal_processing.moving_var(data_uint8, 10)
    reference_data = []
    for i in range(data_uint8.shape[-1] - 10 + 1):
        reference_data.append(data_uint8[i:i + 10].var())
    reference_data = np.array(reference_data)
    assert max_diff_percent(moved_data, reference_data) < 1e-6


def test_moving_var_raises_exxception_with_data_not_being_a_numpy_ndarray():
    with pytest.raises(TypeError):
        scared.signal_processing.moving_var([1, 2, 3], 10)


def test_moving_var_raises_exception_with_window_size_of_wrong_type(data_1d):
    with pytest.raises(TypeError):
        scared.signal_processing.moving_var(data_1d, "foo")
    with pytest.raises(TypeError):
        scared.signal_processing.moving_var(data_1d, 10.1)


def test_moving_var_raises_exception_with_window_size_being_negative_or_0(data_1d):
    with pytest.raises(ValueError):
        scared.signal_processing.moving_var(data_1d, -5)
    with pytest.raises(ValueError):
        scared.signal_processing.moving_var(data_1d, 0)


def test_moving_var_raises_exception_with_axis_type_of_wrong_type(data_1d):
    with pytest.raises(TypeError):
        scared.signal_processing.moving_var(data_1d, 10, axis="foo")


def test_moving_var_raises_exception_with_axis_having_wrong_values(data_1d):
    with pytest.raises(ValueError):
        scared.signal_processing.moving_var(data_1d, 10, axis=1)
    with pytest.raises(ValueError):
        scared.signal_processing.moving_var(data_1d, 10, axis=-2)


def test_moving_var_raises_exception_with_window_size_having_incorrect_value_regarding_to_data_shape(data_1d):
    with pytest.raises(ValueError):
        scared.signal_processing.moving_var(data_1d, 150)


def test_moving_std_returns_correct_array_with_1d_data(data_1d):
    moved_data = scared.signal_processing.moving_std(data_1d, 10)
    reference_data = []
    for i in range(data_1d.shape[-1] - 10 + 1):
        reference_data.append(data_1d[i:i + 10].std())
    reference_data = np.array(reference_data)
    assert max_diff_percent(moved_data, reference_data) < 1e-6


def test_moving_std_returns_correct_array_with_1d_uint8_data(data_uint8):
    moved_data = scared.signal_processing.moving_std(data_uint8, 10)
    reference_data = []
    for i in range(data_uint8.shape[-1] - 10 + 1):
        reference_data.append(data_uint8[i:i + 10].std())
    reference_data = np.array(reference_data)
    assert max_diff_percent(moved_data, reference_data) < 1e-6


def test_moving_std_raises_exxception_with_data_not_being_a_numpy_ndarray():
    with pytest.raises(TypeError):
        scared.signal_processing.moving_std([1, 2, 3], 10)


def test_moving_std_raises_exception_with_window_size_of_wrong_type(data_1d):
    with pytest.raises(TypeError):
        scared.signal_processing.moving_std(data_1d, "foo")
    with pytest.raises(TypeError):
        scared.signal_processing.moving_std(data_1d, 10.1)


def test_moving_std_raises_exception_with_window_size_being_negative_or_0(data_1d):
    with pytest.raises(ValueError):
        scared.signal_processing.moving_std(data_1d, -5)
    with pytest.raises(ValueError):
        scared.signal_processing.moving_std(data_1d, 0)


def test_moving_std_raises_exception_with_axis_type_of_wrong_type(data_1d):
    with pytest.raises(TypeError):
        scared.signal_processing.moving_std(data_1d, 10, axis="foo")


def test_moving_std_raises_exception_with_axis_having_wrong_values(data_1d):
    with pytest.raises(ValueError):
        scared.signal_processing.moving_std(data_1d, 10, axis=1)
    with pytest.raises(ValueError):
        scared.signal_processing.moving_std(data_1d, 10, axis=-2)


def test_moving_std_raises_exception_with_window_size_having_incorrect_value_regarding_to_data_shape(data_1d):
    with pytest.raises(ValueError):
        scared.signal_processing.moving_std(data_1d, 150)


def test_moving_skew_returns_correct_array_with_1d_data(data_1d):
    moved_data = scared.signal_processing.moving_skew(data_1d, 10)
    reference_data = []
    for i in range(data_1d.shape[-1] - 10 + 1):
        reference_data.append(skew(data_1d[i:i + 10]))
    reference_data = np.array(reference_data)
    assert max_diff_percent(moved_data, reference_data) < 1e-6


def test_moving_skew_returns_correct_array_with_1d_uint8_data(data_uint8):
    moved_data = scared.signal_processing.moving_skew(data_uint8, 10)
    reference_data = []
    for i in range(data_uint8.shape[-1] - 10 + 1):
        reference_data.append(skew(data_uint8[i:i + 10]))
    reference_data = np.array(reference_data)
    assert max_diff_percent(moved_data, reference_data) < 1e-6


def test_moving_skew_raises_exxception_with_data_not_being_a_numpy_ndarray():
    with pytest.raises(TypeError):
        scared.signal_processing.moving_skew([1, 2, 3], 10)


def test_moving_skew_raises_exception_with_window_size_of_wrong_type(data_1d):
    with pytest.raises(TypeError):
        scared.signal_processing.moving_skew(data_1d, "foo")
    with pytest.raises(TypeError):
        scared.signal_processing.moving_skew(data_1d, 10.1)


def test_moving_skew_raises_exception_with_window_size_being_negative_or_0(data_1d):
    with pytest.raises(ValueError):
        scared.signal_processing.moving_skew(data_1d, -5)
    with pytest.raises(ValueError):
        scared.signal_processing.moving_skew(data_1d, 0)


def test_moving_skew_raises_exception_with_axis_type_of_wrong_type(data_1d):
    with pytest.raises(TypeError):
        scared.signal_processing.moving_skew(data_1d, 10, axis="foo")


def test_moving_skew_raises_exception_with_axis_having_wrong_values(data_1d):
    with pytest.raises(ValueError):
        scared.signal_processing.moving_skew(data_1d, 10, axis=1)
    with pytest.raises(ValueError):
        scared.signal_processing.moving_skew(data_1d, 10, axis=-2)


def test_moving_skew_raises_exception_with_window_size_having_incorrect_value_regarding_to_data_shape(data_1d):
    with pytest.raises(ValueError):
        scared.signal_processing.moving_skew(data_1d, 150)


def test_moving_kurtosis_returns_correct_array_with_1d_data(data_1d):
    moved_data = scared.signal_processing.moving_kurtosis(data_1d, 10)
    reference_data = []
    for i in range(data_1d.shape[-1] - 10 + 1):
        reference_data.append(kurtosis(data_1d[i:i + 10]))
    reference_data = np.array(reference_data)
    assert max_diff_percent(moved_data, reference_data) < 1e-6


def test_moving_kurtosis_returns_correct_array_with_1d_uint8_data(data_uint8):
    moved_data = scared.signal_processing.moving_kurtosis(data_uint8, 10)
    reference_data = []
    for i in range(data_uint8.shape[-1] - 10 + 1):
        reference_data.append(kurtosis(data_uint8[i:i + 10]))
    reference_data = np.array(reference_data)
    assert max_diff_percent(moved_data, reference_data) < 1e-6


def test_moving_kurtosis_raises_exxception_with_data_not_being_a_numpy_ndarray():
    with pytest.raises(TypeError):
        scared.signal_processing.moving_kurtosis([1, 2, 3], 10)


def test_moving_kurtosis_raises_exception_with_window_size_of_wrong_type(data_1d):
    with pytest.raises(TypeError):
        scared.signal_processing.moving_kurtosis(data_1d, "foo")
    with pytest.raises(TypeError):
        scared.signal_processing.moving_kurtosis(data_1d, 10.1)


def test_moving_kurtosis_raises_exception_with_window_size_being_negative_or_0(data_1d):
    with pytest.raises(ValueError):
        scared.signal_processing.moving_kurtosis(data_1d, -5)
    with pytest.raises(ValueError):
        scared.signal_processing.moving_kurtosis(data_1d, 0)


def test_moving_kurtosis_raises_exception_with_axis_type_of_wrong_type(data_1d):
    with pytest.raises(TypeError):
        scared.signal_processing.moving_kurtosis(data_1d, 10, axis="foo")


def test_moving_kurtosis_raises_exception_with_axis_having_wrong_values(data_1d):
    with pytest.raises(ValueError):
        scared.signal_processing.moving_kurtosis(data_1d, 10, axis=1)
    with pytest.raises(ValueError):
        scared.signal_processing.moving_kurtosis(data_1d, 10, axis=-2)


def test_moving_kurtosis_raises_exception_with_window_size_having_incorrect_value_regarding_to_data_shape(data_1d):
    with pytest.raises(ValueError):
        scared.signal_processing.moving_kurtosis(data_1d, 150)
