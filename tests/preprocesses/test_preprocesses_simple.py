from ..context import scared
import pytest
import numpy as np


@pytest.fixture(params=['int', 'uint', 'float'])
def traces(request):
    if request.param == 'int':
        return np.random.randint(-10000, 10000, (500, 2001), dtype='int16')
    elif request.param == 'uint':
        return np.random.randint(0, 256, (500, 2001), dtype='uint8')
    elif request.param == 'float':
        return np.random.random((500, 2001))


def test_square_preprocess(traces):
    assert np.array_equal(traces ** 2, scared.preprocesses.square(traces))

    with pytest.raises(TypeError):
        scared.preprocesses.square('foo')


def test_serialize_bit_preprocess(traces):
    assert np.array_equal(
        np.unpackbits(traces.astype('uint8'), axis=1),
        scared.preprocesses.serialize_bit(traces))

    with pytest.raises(TypeError):
        scared.preprocesses.serialize_bit('foo')


def test_fft_modulus_preprocess(traces):
    le = np.ceil(traces.shape[1] / 2).astype('uint32')
    expected = np.abs(np.fft.fft(traces))[:, :le]
    assert np.array_equal(
        expected,
        scared.preprocesses.fft_modulus(traces))

    with pytest.raises(TypeError):
        scared.preprocesses.fft_modulus('foo')


def test_center(traces):
    expected = traces - np.mean(traces, axis=0)
    result = scared.preprocesses.center(traces)
    assert np.array_equal(expected, result)


def test_center_on_given_mean(traces):
    given_mean = np.mean(np.random.random((500, 2001)), axis=0)
    expected = traces - given_mean
    result = scared.preprocesses.CenterOn(mean=given_mean)(traces)
    assert np.array_equal(expected, result)


def test_center_on_given_mean_works_if_mean_is_none(traces):
    given_mean = None
    expected = traces - np.nanmean(traces, axis=0)
    result = scared.preprocesses.CenterOn(mean=given_mean)(traces)
    assert np.array_equal(expected, result)


def test_center_on_is_a_preprocess():
    with pytest.raises(ValueError):
        scared.preprocesses.CenterOn(mean=np.random.random((500, 2000, 20)))(np.random.random((50, 2000, 20)))


def test_center_on_given_mean_raises_exception_if_incompatible_shapes(traces):
    wrong_mean = np.mean(traces, axis=1)

    with pytest.raises(scared.PreprocessError):
        scared.preprocesses.CenterOn(wrong_mean)(traces)


def test_standardize(traces):
    expected = traces - np.mean(traces, axis=0)
    expected /= np.std(traces, axis=0)
    result = scared.preprocesses.standardize(traces)
    assert np.array_equal(expected, result)


def test_standardize_on_given_mean(traces):
    given_mean = np.mean(np.random.random((500, 2001)), axis=0)
    expected = (traces - given_mean) / np.std(traces, axis=0)
    result = scared.preprocesses.StandardizeOn(mean=given_mean)(traces)
    assert np.array_equal(expected, result)


def test_standardize_on_given_std(traces):
    given_std = np.nanstd(np.random.random((500, 2001)), axis=0)
    expected = (traces - np.mean(traces, axis=0)) / given_std
    result = scared.preprocesses.StandardizeOn(std=given_std)(traces)
    assert np.array_equal(expected, result)


def test_standardize_on_given_std_and_mean(traces):
    given_mean = np.nanmean(np.random.random((500, 2001)), axis=0)
    given_std = np.nanstd(np.random.random((500, 2001)), axis=0)
    expected = (traces - given_mean) / given_std
    result = scared.preprocesses.StandardizeOn(std=given_std, mean=given_mean)(traces)
    assert np.array_equal(expected, result)


def test_standardize_on_raises_exception_if_incompatible_shapes(traces):
    wrong_mean = np.mean(traces, axis=1)
    wrong_std = np.std(traces, axis=1)

    with pytest.raises(scared.PreprocessError):
        scared.preprocesses.StandardizeOn(mean=wrong_mean)(traces)

    with pytest.raises(scared.PreprocessError):
        scared.preprocesses.StandardizeOn(std=wrong_std)(traces)

    with pytest.raises(scared.PreprocessError):
        scared.preprocesses.StandardizeOn(std=wrong_std, mean=wrong_mean)(traces)
