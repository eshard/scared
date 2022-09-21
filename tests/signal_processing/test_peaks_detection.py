import warnings

import numpy as np
import pytest

from ..context import scared  # noqa: F401


def test_find_peaks_returns_correct_array():
    data = np.random.randint(0, 10, (100))
    indexes = np.array([20, 40, 60])
    data[indexes] = 20

    peaks = scared.signal_processing.find_peaks(data, 0, 20)
    assert np.array_equal(peaks, indexes)
    peaks = scared.signal_processing.find_peaks(data, 10, 20)
    assert np.array_equal(peaks, indexes)
    peaks = scared.signal_processing.find_peaks(data, 20, 20)
    assert np.array_equal(peaks, indexes)
    peaks = scared.signal_processing.find_peaks(data, 30, 20)
    assert np.array_equal(peaks, indexes[[0, -1]])


def test_find_peaks_returns_correct_array_with_negative_inf_threshold():
    data = np.zeros(100)
    peaks = scared.signal_processing.find_peaks(data, 5, -np.inf)
    assert np.array_equal(peaks, np.arange(0, 100, 5))


def test_find_peaks_returns_correct_array_with_positive_inf_threshold():
    data = np.zeros(100)
    peaks = scared.signal_processing.find_peaks(data, 5, np.inf)
    assert peaks.size == 0


def test_find_peaks_returns_correct_array_for_flat_area():
    data = np.zeros(100)
    peaks = scared.signal_processing.find_peaks(data, 20, 0)
    assert np.array_equal(peaks, np.array([0, 20, 40, 60, 80]))


def test_find_peaks_returns_correct_array_for_null_min_distance():
    data = np.ones(100)
    peaks = scared.signal_processing.find_peaks(data, 0, 0)
    assert np.array_equal(peaks, np.arange(100))


def test_find_peaks_returns_correct_array_for_double_peaks_and_bigger_distance():
    data = np.random.randint(0, 10, (100))
    indexes = np.array([20, 40, 60])
    data[indexes] = 20
    data[indexes + 1] = 20
    peaks = scared.signal_processing.find_peaks(data, 3, 20)
    assert np.array_equal(peaks, indexes)


def test_find_peaks_returns_correct_array_with_negative_values_in_data_and_positive_peaks():
    data = np.ones(100) - 10
    indexes = np.arange(0, 100, 10)
    data[indexes] = 5
    peaks = scared.signal_processing.find_peaks(data, 0, 0)
    assert np.array_equal(peaks, indexes)


def test_find_peaks_returns_correct_array_with_negative_values_in_data():
    data = np.ones(100) - 10
    peaks = scared.signal_processing.find_peaks(data, 0, -20)
    assert np.array_equal(peaks, np.arange(100))


def test_find_peaks_returns_correct_array_with_negative_values_in_data_andnegative_peaks():
    data = data = np.ones(100) - 50
    indexes = np.arange(0, 100, 10)
    data[indexes] = -10
    peaks = scared.signal_processing.find_peaks(data, 0, -20)
    assert np.array_equal(peaks, indexes)


def test_find_peaks_returns_correct_array_with_float_values_in_data():
    data = np.random.rand(100) / 2
    indexes = np.arange(0, 100, 10)
    data[indexes] += 0.6
    peaks = scared.signal_processing.find_peaks(data, 0, 0.5)
    assert np.array_equal(peaks, indexes)


def test_find_peaks_raises_exception_with_data_not_being_a_numpy_ndarray():
    with pytest.raises(TypeError):
        scared.signal_processing.find_peaks([1, 2, 3], 1, 1)


def test_find_peaks_raises_exception_with_data_not_having_correct_dimension():
    data = np.random.randint(0, 10, (100, 100))
    with pytest.raises(ValueError):
        scared.signal_processing.find_peaks(data, 1, 1)


def test_find_peaks_raises_exception_with_min_peak_distance_not_being_int_type():
    data = np.random.randint(0, 10, 100)
    with pytest.raises(TypeError):
        scared.signal_processing.find_peaks(data, "foo", 1)


def test_find_peaks_raises_exception_with_min_peak_distance_being_negative():
    data = np.random.randint(0, 10, 100)
    with pytest.raises(ValueError):
        scared.signal_processing.find_peaks(data, -1, 1)


def test_find_peaks_raises_exception_with_min_peak_height_not_being_int_type():
    data = np.random.randint(0, 10, 100)
    with pytest.raises(TypeError):
        scared.signal_processing.find_peaks(data, 1, "foo")
    with pytest.raises(TypeError):
        scared.signal_processing.find_peaks(data, 1, None)
    with pytest.raises(TypeError):
        scared.signal_processing.find_peaks(data, 1, {'foo': 'bar'})


@pytest.fixture
def data_width():
    data = np.zeros(100)
    data[[0, 1, 2, 3, 4, 5, 6]] = 1
    data[range(15, 25)] = 1
    data[range(50, 70)] = 1
    data[[-2, -1]] = 1
    return data


def test_find_width_returns_correct_array_with_positive_direction(data_width):
    result = scared.signal_processing.find_width(data_width, scared.signal_processing.Direction.POSITIVE, 0.5, 10)
    assert np.array_equal(result, [[15, 25], [50, 70]])


def test_find_width_returns_correct_array_with_negative_direction(data_width):
    result = scared.signal_processing.find_width(data_width, scared.signal_processing.Direction.NEGATIVE, 0.5, 10)
    assert np.array_equal(result, [[25, 50], [70, 98]])


def test_find_width_returns_correct_array_with_max_width(data_width):
    result = scared.signal_processing.find_width(data_width, scared.signal_processing.Direction.POSITIVE, 0.5, 10, max_width=10)
    assert np.array_equal(result, [[15, 25]])


def test_find_width_returns_correct_array_with_delta(data_width):
    result = scared.signal_processing.find_width(data_width, scared.signal_processing.Direction.POSITIVE, 0.5, 10, delta=2)
    assert np.array_equal(result, [[15, 25]])
    result = scared.signal_processing.find_width(data_width, scared.signal_processing.Direction.POSITIVE, 0.5, 15, delta=10)
    assert np.array_equal(result, [[15, 25], [50, 70]])


def test_find_width_returns_empty_array_with_too_big_min_width(data_width):
    result = scared.signal_processing.find_width(data_width, scared.signal_processing.Direction.POSITIVE, 0.5, 50)
    assert np.array_equal(result, np.empty((0, 2)))


def test_find_width_returns_empty_array_with_too_big_threshold(data_width):
    result = scared.signal_processing.find_width(data_width, scared.signal_processing.Direction.POSITIVE, 2, 10)
    assert np.array_equal(result, np.empty((0, 2)))
    result = scared.signal_processing.find_width(data_width, scared.signal_processing.Direction.NEGATIVE, 2, 10)
    assert np.array_equal(result, np.empty((0, 2)))


def test_find_width_returns_correct_array_with_max_width_and_delta_bein_ignored(data_width):
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        result = scared.signal_processing.find_width(data_width, scared.signal_processing.Direction.POSITIVE, 0.5, 10, max_width=10, delta=50)
    assert np.array_equal(result, [[15, 25]])


def test_find_width_raises_exception_with_data_not_being_a_numpy_ndarray():
    with pytest.raises(TypeError):
        scared.signal_processing.find_width([1, 2, 3], scared.signal_processing.Direction.POSITIVE, 1, 1)


def test_find_width_raises_exception_with_data_not_having_correct_dimension():
    data = np.random.randint(0, 10, (100, 100))
    with pytest.raises(ValueError):
        scared.signal_processing.find_width(data, scared.signal_processing.Direction.POSITIVE, 1, 1)


def test_find_width_raises_exception_with_direction_not_being_direction_type():
    data = np.random.randint(0, 10, 100)
    with pytest.raises(TypeError):
        scared.signal_processing.find_width(data, "foo", 1, 1)
    with pytest.raises(TypeError):
        scared.signal_processing.find_width(data, 1, 1, 1)


def test_find_width_raises_exception_with_threshold_not_being_int_or_float_type():
    data = np.random.randint(0, 10, 100)
    with pytest.raises(TypeError):
        scared.signal_processing.find_width(data, scared.signal_processing.Direction.POSITIVE, "foo", 1)


def test_find_width_raises_exception_with_min_width_not_being_int_type():
    data = np.random.randint(0, 10, 100)
    with pytest.raises(TypeError):
        scared.signal_processing.find_width(data, scared.signal_processing.Direction.POSITIVE, 1, "foo")


def test_find_width_raises_exception_with_min_width_being_negative_or_0():
    data = np.random.randint(0, 10, 100)
    with pytest.raises(ValueError):
        scared.signal_processing.find_width(data, scared.signal_processing.Direction.POSITIVE, 1, -1)


def test_find_width_raises_exception_with_max_width_not_being_int_type():
    data = np.random.randint(0, 10, 100)
    with pytest.raises(TypeError):
        scared.signal_processing.find_width(data, scared.signal_processing.Direction.POSITIVE, 1, 1, max_width="foo")


def test_find_width_raises_exception_with_max_width_being_negative_or_0():
    data = np.random.randint(0, 10, 100)
    with pytest.raises(ValueError):
        scared.signal_processing.find_width(data, scared.signal_processing.Direction.POSITIVE, 1, 1, max_width=-1)


def test_find_width_raises_exception_with_delta_not_being_int_type():
    data = np.random.randint(0, 10, 100)
    with pytest.raises(TypeError):
        scared.signal_processing.find_width(data, scared.signal_processing.Direction.POSITIVE, 1, 1, delta="foo")


def test_find_width_raises_exception_with_delta_being_negative_or_0():
    data = np.random.randint(0, 10, 100)
    with pytest.raises(ValueError):
        scared.signal_processing.find_width(data, scared.signal_processing.Direction.POSITIVE, 1, 1, delta=-1)


def test_find_width_raises_exception_with_delta_being_higher_than_min_width_and_no_max_width_specified():
    data = np.random.randint(0, 10, 100)
    with pytest.raises(ValueError):
        scared.signal_processing.find_width(data, scared.signal_processing.Direction.POSITIVE, 1, 4, delta=5)


def _extract_around_indexes_returns_correct_array_with_mode_stack(indexes, before, after, extended_indexes):
    data = np.random.rand(100)
    result = scared.signal_processing.extract_around_indexes(data, indexes, before, after, mode=scared.signal_processing.ExtractMode.STACK)
    expected = []
    for i in indexes:
        if extended_indexes:
            expected_data = data[np.arange(i - before, i + after + 1)]
        else:
            expected_data = data[i - before:i + after + 1]
        expected.append(expected_data)
    expected = np.array(expected)
    assert np.array_equal(result, expected)


def test_extract_around_indexes_returns_correct_array_with_mode_stack():
    _extract_around_indexes_returns_correct_array_with_mode_stack(np.array([20, 40, 60]), 5, 3, False)


def test_extract_around_indexes_returns_correct_array_with_mode_stack_and_limit_indexes():
    _extract_around_indexes_returns_correct_array_with_mode_stack(np.array([0, 96]), 5, 3, True)


def test_extract_around_indexes_returns_correct_array_with_mode_stack_and_middle_index():
    _extract_around_indexes_returns_correct_array_with_mode_stack(np.array([50]), 5, 3, True)


def test_extract_around_indexes_returns_correct_array_with_mode_stack_with_negative_before():
    _extract_around_indexes_returns_correct_array_with_mode_stack(np.array([20, 40, 60]), -1, 5, True)


def _extract_around_indexes_returns_correct_array_with_mode_concatenate(indexes, before, after, extended_indexes):
    data = np.random.rand(100)
    result = scared.signal_processing.extract_around_indexes(data, indexes, before, after, mode=scared.signal_processing.ExtractMode.CONCATENATE)
    expected = []
    for i in indexes:
        if extended_indexes:
            expected_data = data[np.arange(i - before, i + after + 1)]
        else:
            expected_data = data[i - before:i + after + 1]
        expected.append(expected_data)
    expected = np.concatenate(expected)
    assert np.array_equal(result, expected)


def test_extract_around_indexes_returns_correct_array_with_mode_concatenate():
    _extract_around_indexes_returns_correct_array_with_mode_concatenate(np.array([20, 40, 60]), 5, 3, False)


def test_extract_around_indexes_returns_correct_array_with_mode_concatenate_and_limit_indexes():
    _extract_around_indexes_returns_correct_array_with_mode_concatenate(np.array([0, 96]), 5, 3, True)


def test_extract_around_indexes_returns_correct_array_with_mode_concatenate_and_middle_index():
    _extract_around_indexes_returns_correct_array_with_mode_concatenate(np.array([50]), 5, 3, True)


def test_extract_around_indexes_returns_correct_array_with_mode_concatenate_with_negative_before():
    _extract_around_indexes_returns_correct_array_with_mode_concatenate(np.array([20, 40, 60]), -1, 5, True)


def _extract_around_indexes_returns_correct_array_with_mode_average(indexes, before, after, extended_indexes):
    data = np.random.rand(100)
    result = scared.signal_processing.extract_around_indexes(data, indexes, before, after, mode=scared.signal_processing.ExtractMode.AVERAGE)
    expected = []
    for i in indexes:
        if extended_indexes:
            expected_data = data[np.arange(i - before, i + after + 1)]
        else:
            expected_data = data[i - before:i + after + 1]
        expected.append(expected_data)
    expected = np.array(expected)
    expected = np.mean(expected, axis=0)
    assert np.array_equal(result, expected)


def test_extract_around_indexes_returns_correct_array_with_mode_average():
    _extract_around_indexes_returns_correct_array_with_mode_average(np.array([20, 40, 60]), 5, 3, False)


def test_extract_around_indexes_returns_correct_array_with_mode_average_and_limit_indexes():
    _extract_around_indexes_returns_correct_array_with_mode_average(np.array([0, 96]), 5, 3, True)


def test_extract_around_indexes_returns_correct_array_with_mode_average_and_middle_index():
    _extract_around_indexes_returns_correct_array_with_mode_average(np.array([50]), 5, 3, True)


def test_extract_around_indexes_returns_correct_array_with_mode_average_with_negative_before():
    _extract_around_indexes_returns_correct_array_with_mode_average(np.array([20, 40, 60]), -1, 5, True)


def test_extract_around_indexes_raises_exception_with_data_not_being_a_numpy_ndarray():
    with pytest.raises(TypeError):
        scared.signal_processing.extract_around_indexes([1, 2, 3], 1, 1, 1)


def test_extract_around_indexes_raises_exception_with_data_not_having_correct_dimension():
    data = np.random.randint(0, 10, (100, 100))
    with pytest.raises(ValueError):
        scared.signal_processing.extract_around_indexes(data, 1, 1, 1)


def test_extract_around_indexes_raises_exception_with_indexes_not_being_a_numpy_ndarray():
    data = np.random.randint(0, 10, 100)
    with pytest.raises(TypeError):
        scared.signal_processing.extract_around_indexes(data, [1, 2, 3], 1, 1)


def test_extract_around_indexes_raises_exception_with_indexes_not_having_correct_dimension():
    data = np.random.randint(0, 10, 100)
    indexes = np.random.randint(0, 10, (100, 100))
    with pytest.raises(ValueError):
        scared.signal_processing.extract_around_indexes(data, indexes, 1, 1)


def test_extract_around_indexes_raises_exception_with_indexes_not_being_an_integer_array():
    data = np.random.randint(0, 10, 100)
    indexes = np.random.rand(100)
    with pytest.raises(ValueError):
        scared.signal_processing.extract_around_indexes(data, indexes, 1, 1)


def test_extract_around_indexes_raises_exception_with_before_not_being_int_type():
    data = np.random.randint(0, 10, 100)
    with pytest.raises(TypeError):
        scared.signal_processing.extract_around_indexes(data, data, "foo", 1)


def test_extract_around_indexes_raises_exception_with_after_not_being_int_type():
    data = np.random.randint(0, 10, 100)
    with pytest.raises(TypeError):
        scared.signal_processing.extract_around_indexes(data, data, 1, "foo")


def test_extract_around_indexes_raises_exception_with_mode_not_being_extract_mode_type():
    data = np.random.randint(0, 10, 100)
    with pytest.raises(TypeError):
        scared.signal_processing.extract_around_indexes(data, data, 1, 1, mode="foo")
