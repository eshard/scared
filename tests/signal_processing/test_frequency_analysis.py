import numpy as np
import pytest

from ..context import scared  # noqa: F401


@pytest.fixture
def trace():
    return np.sin(2 * np.pi * 1 * np.arange(1000) / (1000 // 100))


def test_fft_returns_correct_arrays():
    data = np.sin(2 * np.pi * 1 * np.arange(20) / (1000 // 100))
    frequencies, magnitude, phase = scared.signal_processing.fft(data, 1e9)

    frequencies_expected = np.fft.rfftfreq(20) * 1e9
    fft = np.fft.rfft(data, axis=-1) / 20
    magnitude_expected = np.abs(fft)
    phase_ecpected = np.angle(fft) * 180 / np.pi

    assert np.allclose(frequencies, frequencies_expected)
    assert np.allclose(magnitude, magnitude_expected)
    assert np.allclose(phase, phase_ecpected)


def test_fft_with_2_dimension_array_returns_correct_arrays():
    data = np.sin(2 * np.pi * 1 * np.arange(10) / (1000 // 100))
    data_2 = np.vstack((data, data))
    frequencies, magnitude, phase = scared.signal_processing.fft(data_2, 1e9)

    frequencies_expected = np.fft.rfftfreq(10) * 1e9
    fft = np.fft.rfft(data, axis=-1) / 10
    magnitude_expected = np.abs(fft)
    phase_ecpected = np.angle(fft) * 180 / np.pi

    assert np.allclose(frequencies, frequencies_expected)
    assert np.allclose(magnitude, magnitude_expected)
    assert np.allclose(phase, phase_ecpected)


def test_fft_with_axis_given_returns_correct_arrays():
    data = np.sin(2 * np.pi * 1 * np.arange(10) / (1000 // 100))
    data_2 = np.vstack((data, data))
    frequencies, magnitude, phase = scared.signal_processing.fft(data_2, 1e9)
    frequencies_axis, magnitude_axis, phase_axis = scared.signal_processing.fft(data_2.T, 1e9, axis=0)
    assert np.array_equal(frequencies, frequencies_axis)
    assert np.array_equal(magnitude.T, magnitude_axis)
    assert np.array_equal(phase.T, phase_axis)


def test_fft_raises_exception_with_data_not_being_numpy_array():
    with pytest.raises(TypeError):
        scared.signal_processing.fft([1, 2, 3], 1e9)


def test_fft_raises_exception_with_frequency_of_wrong_type(trace):
    with pytest.raises(TypeError):
        scared.signal_processing.fft(trace, "foo")


def test_fft_raises_exception_with_frequency_being_negative_or_0(trace):
    with pytest.raises(ValueError):
        scared.signal_processing.fft(trace, -5)
    with pytest.raises(ValueError):
        scared.signal_processing.fft(trace, -5.0)
    with pytest.raises(ValueError):
        scared.signal_processing.fft(trace, 0)


def test_fft_raises_exception_with_axis_type_of_wrong_type(trace):
    with pytest.raises(TypeError):
        scared.signal_processing.fft(trace, 1e9, axis="foo")
