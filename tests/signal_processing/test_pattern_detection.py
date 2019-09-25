import warnings

import numpy as np
import pytest
from scipy.stats import pearsonr
from scipy.spatial.distance import euclidean

from ..context import scared  # noqa: F401


def max_diff_percent(a, b):
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        tmp = np.abs(a - b) / np.minimum(a, b) * 100
    tmp[np.isinf(tmp)] = np.nan
    return np.nanmax(tmp)


def test_correlation_returns_correct_array():
    pattern = np.random.rand(10)
    trace = np.random.rand(50)
    trace[20:30] = pattern

    expected_result = []
    for i in range(50 - 10 + 1):
        tmp_trace = trace[i:i + 10]
        correlation, _ = pearsonr(tmp_trace, pattern)
        expected_result.append(correlation)
    expected_result = np.array(expected_result)

    result = scared.signal_processing.correlation(trace, pattern)

    assert len(expected_result) == len(result)
    assert max_diff_percent(expected_result, result) < 1e-6
    assert np.abs(result[20] - 1.0) < 1e-6


def test_correlation_returns_correct_array_for_big_pattern():
    pattern = np.random.rand(49)
    trace = np.random.rand(50)
    trace[1:] = pattern

    expected_result = []
    for i in range(50 - 49 + 1):
        tmp_trace = trace[i:i + 49]
        correlation, _ = pearsonr(tmp_trace, pattern)
        expected_result.append(correlation)
    expected_result = np.array(expected_result)

    result = scared.signal_processing.correlation(trace, pattern)

    assert len(expected_result) == len(result)
    assert max_diff_percent(expected_result, result) < 1e-6
    assert np.abs(result[-1] - 1.0) < 1e-6


def test_correlation_raises_exception_with_trace_not_being_a_numpy_ndarray():
    pattern = np.random.rand(50)
    with pytest.raises(TypeError):
        scared.signal_processing.correlation([1, 2, 3], pattern)


def test_correlation_raises_exception_with_pattern_not_being_a_numpy_ndarray():
    trace = np.random.rand(50)
    with pytest.raises(TypeError):
        scared.signal_processing.correlation(trace, [1, 2, 3])


def test_correlation_raises_exception_with_trace_not_having_correct_dimension():
    trace = np.random.rand(10, 10)
    pattern = np.random.rand(50)
    with pytest.raises(ValueError):
        scared.signal_processing.correlation(trace, pattern)


def test_correlation_raises_exception_with_pattern_not_having_correct_dimension():
    trace = np.random.rand(50)
    pattern = np.random.rand(10, 10)
    with pytest.raises(ValueError):
        scared.signal_processing.correlation(trace, pattern)


def test_correlation_raises_exception_with_trace_not_having_more_elements_than_pattern():
    with pytest.raises(ValueError):
        scared.signal_processing.correlation(np.random.rand(50), np.random.rand(50))
    with pytest.raises(ValueError):
        scared.signal_processing.correlation(trace=np.random.rand(49), pattern=np.random.rand(50))


def test_distance_returns_correct_array():
    pattern = np.random.rand(10)
    trace = np.random.rand(50)
    trace[20:30] = pattern

    expected_result = []
    for i in range(50 - 10 + 1):
        tmp_trace = trace[i:i + 10]
        distance = euclidean(tmp_trace, pattern)
        expected_result.append(distance)
    expected_result = np.array(expected_result)

    result = scared.signal_processing.distance(trace, pattern)

    assert len(expected_result) == len(result)
    assert max_diff_percent(expected_result, result) < 1e-6
    assert np.abs(result[20]) < 1e-6


def test_distance_returns_correct_array_for_big_pattern():
    pattern = np.random.rand(49)
    trace = np.random.rand(50)
    trace[1:] = pattern

    expected_result = []
    for i in range(50 - 49 + 1):
        tmp_trace = trace[i:i + 49]
        distance = euclidean(tmp_trace, pattern)
        expected_result.append(distance)
    expected_result = np.array(expected_result)

    result = scared.signal_processing.distance(trace, pattern)

    assert len(expected_result) == len(result)
    assert max_diff_percent(expected_result, result) < 1e-6
    assert np.abs(result[-1]) < 1e-6


def test_distance_raises_exception_with_trace_not_being_a_numpy_ndarray():
    pattern = np.random.rand(50)
    with pytest.raises(TypeError):
        scared.signal_processing.distance([1, 2, 3], pattern)


def test_distance_raises_exception_with_pattern_not_being_a_numpy_ndarray():
    trace = np.random.rand(50)
    with pytest.raises(TypeError):
        scared.signal_processing.distance(trace, [1, 2, 3])


def test_distance_raises_exception_with_trace_not_having_correct_dimension():
    trace = np.random.rand(10, 10)
    pattern = np.random.rand(50)
    with pytest.raises(ValueError):
        scared.signal_processing.distance(trace, pattern)


def test_distance_raises_exception_with_pattern_not_having_correct_dimension():
    trace = np.random.rand(50)
    pattern = np.random.rand(10, 10)
    with pytest.raises(ValueError):
        scared.signal_processing.distance(trace, pattern)


def test_distance_raises_exception_with_trace_not_having_more_elements_than_pattern():
    with pytest.raises(ValueError):
        scared.signal_processing.distance(np.random.rand(50), np.random.rand(50))
    with pytest.raises(ValueError):
        scared.signal_processing.distance(trace=np.random.rand(49), pattern=np.random.rand(50))


def bcdc(a, b):
    n = np.std(a - b)
    d = np.std(a + b)
    return n / d


def test_bcdc_returns_correct_array():
    pattern = np.random.rand(10)
    trace = np.random.rand(50)
    trace[20:30] = pattern

    expected_result = []
    for i in range(50 - 10 + 1):
        tmp_trace = trace[i:i + 10]
        bcdc_result = bcdc(tmp_trace, pattern)
        expected_result.append(bcdc_result)
    expected_result = np.array(expected_result)

    result = scared.signal_processing.bcdc(trace, pattern)

    assert len(expected_result) == len(result)
    assert max_diff_percent(expected_result, result) < 1e-6
    assert np.abs(result[20]) < 1e-6


def test_bcdc_returns_correct_array_for_big_pattern():
    pattern = np.random.rand(49)
    trace = np.random.rand(50)
    trace[1:] = pattern

    expected_result = []
    for i in range(50 - 49 + 1):
        tmp_trace = trace[i:i + 49]
        bcdc_result = bcdc(tmp_trace, pattern)
        expected_result.append(bcdc_result)
    expected_result = np.array(expected_result)

    result = scared.signal_processing.bcdc(trace, pattern)

    assert len(expected_result) == len(result)
    assert max_diff_percent(expected_result, result) < 1e-6
    assert np.abs(result[-1]) < 1e-6


def test_bcdc_raises_exception_with_trace_not_being_a_numpy_ndarray():
    pattern = np.random.rand(50)
    with pytest.raises(TypeError):
        scared.signal_processing.bcdc([1, 2, 3], pattern)


def test_bcdc_raises_exception_with_pattern_not_being_a_numpy_ndarray():
    trace = np.random.rand(50)
    with pytest.raises(TypeError):
        scared.signal_processing.bcdc(trace, [1, 2, 3])


def test_bcdc_raises_exception_with_trace_not_having_correct_dimension():
    trace = np.random.rand(10, 10)
    pattern = np.random.rand(50)
    with pytest.raises(ValueError):
        scared.signal_processing.bcdc(trace, pattern)


def test_bcdc_raises_exception_with_pattern_not_having_correct_dimension():
    trace = np.random.rand(50)
    pattern = np.random.rand(10, 10)
    with pytest.raises(ValueError):
        scared.signal_processing.bcdc(trace, pattern)


def test_bcdc_raises_exception_with_trace_not_having_more_elements_than_pattern():
    with pytest.raises(ValueError):
        scared.signal_processing.bcdc(np.random.rand(50), np.random.rand(50))
    with pytest.raises(ValueError):
        scared.signal_processing.bcdc(trace=np.random.rand(49), pattern=np.random.rand(50))
