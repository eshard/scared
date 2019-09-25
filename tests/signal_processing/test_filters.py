import warnings

import numpy as np
import pytest
from scipy import signal

from ..context import scared  # noqa: F401


@pytest.fixture
def trace():
    return np.sin(2 * np.pi * 1 * np.arange(1000) / (1000 // 100))


def max_diff_percent(a, b):
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        tmp = np.abs(a - b) / np.minimum(a, b) * 100
    tmp[np.isinf(tmp)] = np.nan
    return np.nanmax(tmp)


def test_butterworth_returns_correct_value_with_lowpass_filter_type_and_float32_precision(trace):
    b, a = signal.butter(3, 10e6 / (1e9 / 2), 'low')
    b = b.astype('float32')
    a = a.astype('float32')
    expected = signal.lfilter(b, a, trace)
    result = scared.signal_processing.butterworth(trace, 1e9, 10e6, filter_type=scared.signal_processing.FilterType.LOW_PASS)
    assert max_diff_percent(expected, result) < 1e-6


def test_butterworth_returns_correct_value_with_lowpass_filter_type_and_float64_precision(trace):
    b, a = signal.butter(3, 10e6 / (1e9 / 2), 'low')
    b = b.astype('float64', copy=False)
    a = a.astype('float64', copy=False)
    expected = signal.lfilter(b, a, trace)
    result = scared.signal_processing.butterworth(trace, 1e9, 10e6, precision="float64", filter_type=scared.signal_processing.FilterType.LOW_PASS)
    assert max_diff_percent(expected, result) < 1e-6


def test_butterworth_returns_correct_value_with_highpass_filter_type_and_float32_precision(trace):
    b, a = signal.butter(3, 10e6 / (1e9 / 2), 'highpass')
    b = b.astype('float32')
    a = a.astype('float32')
    expected = signal.lfilter(b, a, trace)
    result = scared.signal_processing.butterworth(trace, 1e9, 10e6, filter_type=scared.signal_processing.FilterType.HIGH_PASS)
    assert max_diff_percent(expected, result) < 1e-6


def test_butterworth_returns_correct_value_with_highpass_filter_type_and_float64_precision(trace):
    b, a = signal.butter(3, 10e6 / (1e9 / 2), 'highpass')
    b = b.astype('float64', copy=False)
    a = a.astype('float64', copy=False)
    expected = signal.lfilter(b, a, trace)
    result = scared.signal_processing.butterworth(trace, 1e9, 10e6, precision="float64", filter_type=scared.signal_processing.FilterType.HIGH_PASS)
    assert max_diff_percent(expected, result) < 1e-6


def test_butterworth_returns_correct_value_with_bandstop_filter_type_and_float32_precision(trace):
    b, a = signal.butter(3, [10e6 / (1e9 / 2), 100e6 / (1e9 / 2)], 'bandstop')
    b = b.astype('float32')
    a = a.astype('float32')
    expected = signal.lfilter(b, a, trace)
    result = scared.signal_processing.butterworth(trace, 1e9, [10e6, 100e6], filter_type=scared.signal_processing.FilterType.BAND_STOP)
    assert max_diff_percent(expected, result) < 1e-6


def test_butterworth_returns_correct_value_with_bandstop_filter_type_and_float64_precision(trace):
    b, a = signal.butter(3, [10e6 / (1e9 / 2), 100e6 / (1e9 / 2)], 'bandstop')
    b = b.astype('float64', copy=False)
    a = a.astype('float64', copy=False)
    expected = signal.lfilter(b, a, trace)
    result = scared.signal_processing.butterworth(trace, 1e9, [10e6, 100e6], precision="float64", filter_type=scared.signal_processing.FilterType.BAND_STOP)
    assert max_diff_percent(expected, result) < 1e-6


def test_butterworth_returns_correct_value_with_bandpass_filter_type_and_float32_precision(trace):
    b, a = signal.butter(3, [10e6 / (1e9 / 2), 100e6 / (1e9 / 2)], 'bandpass')
    b = b.astype('float32')
    a = a.astype('float32')
    expected = signal.lfilter(b, a, trace)
    result = scared.signal_processing.butterworth(trace, 1e9, [10e6, 100e6], filter_type=scared.signal_processing.FilterType.BAND_PASS)
    assert max_diff_percent(expected, result) < 1e-6


def test_butterworth_returns_correct_value_with_bandpass_filter_type_and_float64_precision(trace):
    b, a = signal.butter(3, [10e6 / (1e9 / 2), 100e6 / (1e9 / 2)], 'bandpass')
    b = b.astype('float64', copy=False)
    a = a.astype('float64', copy=False)
    expected = signal.lfilter(b, a, trace)
    result = scared.signal_processing.butterworth(trace, 1e9, [10e6, 100e6], precision="float64", filter_type=scared.signal_processing.FilterType.BAND_PASS)
    assert max_diff_percent(expected, result) < 1e-6


def test_butterworth_raises_exception_with_data_not_being_numpy_array():
    with pytest.raises(TypeError):
        scared.signal_processing.butterworth([1, 2, 3], 1e9, 10e6)


def test_butterworth_raises_exception_with_frequency_of_wrong_type(trace):
    with pytest.raises(TypeError):
        scared.signal_processing.butterworth(trace, "foo", 10e6)


def test_butterworth_raises_exception_with_frequency_being_negative_or_0(trace):
    with pytest.raises(ValueError):
        scared.signal_processing.butterworth(trace, -5, 10e6)
    with pytest.raises(ValueError):
        scared.signal_processing.butterworth(trace, -5.0, 10e6)
    with pytest.raises(ValueError):
        scared.signal_processing.butterworth(trace, 0, 10e6)


def test_butterworth_raises_exception_with_cutoff_of_wrong_type(trace):
    with pytest.raises(TypeError):
        scared.signal_processing.butterworth(trace, 1e9, "foo")
    with pytest.raises(TypeError):
        scared.signal_processing.butterworth(trace, 1e9, [2, "foo"])


def test_butterworth_raises_exception_with_cutoff_having_negative_or_0_values(trace):
    with pytest.raises(ValueError):
        scared.signal_processing.butterworth(trace, 1e9, -5)
    with pytest.raises(ValueError):
        scared.signal_processing.butterworth(trace, 1e9, [2, -5])


def test_butterworth_raises_exception_with_cutoff_having_more_than_2_values_or_less_than_one_value(trace):
    with pytest.raises(ValueError):
        scared.signal_processing.butterworth(trace, 1e9, [1, 2, 3])
    with pytest.raises(ValueError):
        scared.signal_processing.butterworth(trace, 1e9, [])


def test_butterworth_raises_exception_with_cutoff_having_one_value_and_filter_type_being_bandpass_or_bandstop(trace):
    with pytest.raises(ValueError):
        scared.signal_processing.butterworth(trace, 1e9, [1], filter_type=scared.signal_processing.FilterType.BAND_PASS)
    with pytest.raises(ValueError):
        scared.signal_processing.butterworth(trace, 1e9, [1], filter_type=scared.signal_processing.FilterType.BAND_STOP)


def test_butterworth_raises_exception_with_cutoff_having_two_values_with_the_second_not_bigger_than_the_first(trace):
    with pytest.raises(ValueError):
        scared.signal_processing.butterworth(trace, 1e9, [1, 0.5], filter_type=scared.signal_processing.FilterType.BAND_PASS)
    with pytest.raises(ValueError):
        scared.signal_processing.butterworth(trace, 1e9, [2, 1], filter_type=scared.signal_processing.FilterType.BAND_STOP)


def test_butterworth_raises_exception_with_cutoff_having_two_values_and_filter_type_being_lowpass_or_highpass(trace):
    with pytest.raises(ValueError):
        scared.signal_processing.butterworth(trace, 1e9, [1, 2], filter_type=scared.signal_processing.FilterType.LOW_PASS)
    with pytest.raises(ValueError):
        scared.signal_processing.butterworth(trace, 1e9, [1, 2], filter_type=scared.signal_processing.FilterType.HIGH_PASS)


def test_butterworth_raises_exception_with_cutoff_having_its_biggest_value_bigger_than_frequency_divided_by_2(trace):
    with pytest.raises(ValueError):
        scared.signal_processing.butterworth(trace, 15, [1, 10], filter_type=scared.signal_processing.FilterType.BAND_PASS)
    with pytest.raises(ValueError):
        scared.signal_processing.butterworth(trace, 15, 20, filter_type=scared.signal_processing.FilterType.HIGH_PASS)


def test_butterworth_raises_exception_with_filter_type_of_wrong_type(trace):
    with pytest.raises(TypeError):
        scared.signal_processing.butterworth(trace, 1e9, 10e6, filter_type="lowpass")


def test_butterworth_raises_exception_with_order_type_of_wrong_type(trace):
    with pytest.raises(TypeError):
        scared.signal_processing.butterworth(trace, 1e9, 10e6, order="foo")


def test_butterworth_raises_exception_with_order_being_negative_or_0(trace):
    with pytest.raises(ValueError):
        scared.signal_processing.butterworth(trace, 1e9, 10e6, order=0)
    with pytest.raises(ValueError):
        scared.signal_processing.butterworth(trace, 1e9, 10e6, order=-5)


def test_butterworth_raises_exception_with_axis_type_of_wrong_type(trace):
    with pytest.raises(TypeError):
        scared.signal_processing.butterworth(trace, 1e9, 10e6, axis="foo")


def test_butterworth_raises_exception_with_precision_having_wrong_representation(trace):
    with pytest.raises(TypeError):
        scared.signal_processing.butterworth(trace, 1e9, 10e6, precision="float54")
