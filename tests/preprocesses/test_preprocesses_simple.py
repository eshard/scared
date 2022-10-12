from ..context import scared
import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose


@pytest.fixture(params=['int16', 'uint8', 'float32', 'float64'])
def traces(request):
    if request.param == 'int16':
        return np.random.randint(-10000, 10000, (500, 2001), dtype='int16')
    elif request.param == 'uint8':
        return np.random.randint(0, 256, (500, 2001), dtype='uint8')
    elif request.param == 'float32':
        return np.random.random((500, 2001)).astype('float32')
    elif request.param == 'float64':
        return np.random.random((500, 2001)).astype('float64')


def test_square_preprocess(traces):
    expected = np.square(traces, dtype=max(traces.dtype, 'float32'))
    result = scared.preprocesses.square(traces)
    assert result.dtype >= 'float32'
    assert_array_equal(expected, result)

    with pytest.raises(TypeError):
        scared.preprocesses.square('foo')


def test_serialize_bit_preprocess(traces):
    assert_array_equal(
        np.unpackbits(traces.astype('uint8'), axis=1),
        scared.preprocesses.serialize_bit(traces))

    with pytest.raises(TypeError):
        scared.preprocesses.serialize_bit('foo')


def test_fft_modulus_preprocess(traces):
    le = np.ceil(traces.shape[1] / 2).astype('uint32')
    expected = np.abs(np.fft.fft(traces))[:, :le]
    assert_array_equal(
        expected,
        scared.preprocesses.fft_modulus(traces))

    with pytest.raises(TypeError):
        scared.preprocesses.fft_modulus('foo')


def test_center(traces):
    expected = traces - np.mean(traces, axis=0, dtype=max(traces.dtype, 'float32'))
    result = scared.preprocesses.center(traces)
    assert result.dtype >= 'float32'
    assert_array_equal(expected, result)


def test_center_on_given_mean(traces):
    given_mean = np.mean(np.random.random((500, 2001)), axis=0)
    expected = traces - given_mean
    result = scared.preprocesses.CenterOn(mean=given_mean)(traces)
    assert result.dtype >= 'float32'
    assert_array_equal(expected, result)


def test_center_on_given_mean_works_if_mean_is_none(traces):
    given_mean = None
    expected = traces - np.nanmean(traces, axis=0, dtype=max(traces.dtype, 'float32'))
    result = scared.preprocesses.CenterOn(mean=given_mean)(traces)
    assert result.dtype >= 'float32'
    assert_array_equal(expected, result)


def test_center_on_is_a_preprocess():
    with pytest.raises(ValueError):
        scared.preprocesses.CenterOn(mean=np.random.random((500, 2000, 20)))(np.random.random((50, 2000, 20)))


def test_center_on_given_mean_raises_exception_if_incompatible_shapes(traces):
    wrong_mean = np.mean(traces, axis=1)

    with pytest.raises(scared.PreprocessError):
        scared.preprocesses.CenterOn(wrong_mean)(traces)


def test_center_on_given_precision(traces):
    for precision in ['float64', 'float128']:
        expected = traces - np.mean(traces, axis=0, dtype=precision)
        result = scared.preprocesses.CenterOn(precision=precision)(traces)
        assert result.dtype == precision
        assert_array_equal(expected, result)


def test_standardize(traces):
    expected = traces - np.mean(traces, axis=0, dtype=max(traces.dtype, 'float32'))
    expected /= np.std(traces, axis=0, dtype=max(traces.dtype, 'float32'))
    result = scared.preprocesses.standardize(traces)
    assert result.dtype >= 'float32'
    assert_array_equal(expected, result)


def test_standardize_on_given_mean(traces):
    given_mean = np.mean(np.random.random((500, 2001)), axis=0)
    expected = (traces - given_mean) / np.std(traces, axis=0, dtype=max(traces.dtype, 'float32'))
    result = scared.preprocesses.StandardizeOn(mean=given_mean)(traces)
    assert result.dtype >= 'float32'
    assert_array_equal(expected, result)


def test_standardize_on_given_std(traces):
    given_std = np.nanstd(np.random.random((500, 2001)), axis=0)
    expected = (traces - np.mean(traces, axis=0, dtype=max(traces.dtype, 'float32'))) / given_std
    result = scared.preprocesses.StandardizeOn(std=given_std)(traces)
    assert result.dtype >= 'float32'
    assert_array_equal(expected, result)


def test_standardize_on_given_std_and_mean(traces):
    given_mean = np.nanmean(np.random.random((500, 2001)), axis=0)
    given_std = np.nanstd(np.random.random((500, 2001)), axis=0)
    expected = (traces - given_mean) / given_std
    result = scared.preprocesses.StandardizeOn(std=given_std, mean=given_mean)(traces)
    assert result.dtype >= 'float32'
    assert_array_equal(expected, result)


def test_standardize_on_raises_exception_if_incompatible_shapes(traces):
    wrong_mean = np.mean(traces, axis=1)
    wrong_std = np.std(traces, axis=1)

    with pytest.raises(scared.PreprocessError):
        scared.preprocesses.StandardizeOn(mean=wrong_mean)(traces)

    with pytest.raises(scared.PreprocessError):
        scared.preprocesses.StandardizeOn(std=wrong_std)(traces)

    with pytest.raises(scared.PreprocessError):
        scared.preprocesses.StandardizeOn(std=wrong_std, mean=wrong_mean)(traces)


def test_standardize_on_given_precision(traces):
    for precision in ['float64', 'float128']:
        expected = traces - np.nanmean(traces, axis=0, dtype=precision)
        expected /= np.nanstd(traces, axis=0, dtype=precision)
        result = scared.preprocesses.StandardizeOn(precision=precision)(traces)
        assert result.dtype == precision
        assert_array_equal(expected, result)


def test_power_preprocess_raises_exception_if_power_not_a_number(traces):
    with pytest.raises(ValueError):
        scared.preprocesses.ToPower('foo')
    with pytest.raises(ValueError):
        scared.preprocesses.ToPower(power={"doo"})


@pytest.fixture(params=range(0, 10))
def powers(request):
    return request.param / 2


def test_power_preprocess(traces, powers):
    p = scared.preprocesses.ToPower(powers)
    assert p.power == powers
    assert isinstance(p, scared.Preprocess)
    expected = traces ** powers
    result = p(traces)
    assert result.dtype >= 'float32'
    assert_allclose(expected, result, rtol=1e-05, atol=1e-08)


def test_power_preprocess_default_value(traces):
    p = scared.preprocesses.ToPower()
    assert_array_equal(traces, p(traces))


def test_power_given_precision(traces, powers):
    for precision in ['float64', 'float128']:
        expected = np.power(traces, powers, dtype=precision)
        result = scared.preprocesses.ToPower(powers, precision=precision)(traces)
        assert result.dtype == precision
        assert_array_equal(expected, result)
