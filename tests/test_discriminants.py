from .context import scared
import numpy as np
import pytest


def test_discriminant_raises_exception_if_call_returns_data_with_incorrect_type():
    @scared.discriminant
    def d(data, axis):
        return "foo"
    with pytest.raises(ValueError):
        d(np.random.randint(0, 255, (500, 16), dtype='uint8'))

    @scared.discriminant
    def d(data, axis):
        return np.array(['foo'])
    with pytest.raises(ValueError):
        d(np.random.randint(0, 255, (500, 16), dtype='uint8'))


def test_discriminant_raises_exception_if_call_returns_data_with_incorrect_shape():
    @scared.discriminant
    def d(data, axis):
        return data

    with pytest.raises(ValueError):
        d(np.random.randint(0, 255, (500, 16), dtype='uint8'))


def test_discriminant_raises_exception_if_called_with_incorrect_data_type():
    @scared.discriminant
    def d(data, axis):
        return np.nanmax(data, axis=axis)

    with pytest.raises(TypeError):
        d("foo")
    with pytest.raises(TypeError):
        d(np.array(["foo"]))
    with pytest.raises(TypeError):
        d([1, 2, 3])


def test_max_discriminant():
    data = np.array([[1, 3, 5], [0, -3, 3]])
    assert [5, 3] == scared.nanmax(data).tolist()


def test_maxabs_discriminant():
    data = np.array([[1, 3, -5], [0, -3, 3]])
    assert [5, 3] == scared.maxabs(data).tolist()


def test_opposite_min_discriminant():
    data = np.array([[1, 3, 5], [0, 2, 3], [-4, 0, 2]])
    assert [-1, 0, 4] == scared.opposite_min(data).tolist()


def test_sum_discriminant():
    data = np.array([[1, 3, 5], [0, 2, 3], [-4, 0, 2]])
    assert [9, 5, -2] == scared.nansum(data).tolist()


def test_abs_sum_discriminant():
    data = np.array([[1, 3, 5], [0, 2, 3], [-4, 0, 2]])
    assert [9, 5, 6] == scared.abssum(data).tolist()


def test_discriminant_accept_axis_parameter():
    @scared.discriminant
    def d(data, axis):
        return np.sum(data, axis)

    data = np.random.randint(0, 255, (16, 500), dtype='uint8')
    assert (500, ) == d(data=data, axis=0).shape
