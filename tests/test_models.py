from .context import scared
import numpy as np
import pytest


@pytest.fixture(params=list(range(9)))
def bits(request):
    return request.param


@pytest.fixture(params=list(range(1, 18)))
def nb_words(request):
    return request.param


def test_subclassing_model_without_compute_function_raises_error():
    with pytest.raises(TypeError):
        class MyModel(scared.Model):

            @property
            def max_data_value(self):
                return 1
        MyModel()


def test_subclassing_model_without_max_data_value_raises_error():
    with pytest.raises(TypeError):
        class MyModel(scared.Model):
            def _compute(self, value, axis=-1):
                return value
        MyModel()


def test_model_subclass_raises_exception_if_compute_function_do_not_preserve_dimensions_correctly():
    class MyModel(scared.Model):

        def _compute(self, data, axis=-1):
            return np.min(data, axis=0)

        @property
        def max_data_value(self):
            return 1

    m = MyModel()
    with pytest.raises(ValueError):
        m(data=np.random.randint(0, 255, (500, 15), dtype='uint8'))


def test_value_model_can_be_init():
    vm = scared.Value()
    assert vm


def test_value_model_returns_identiy():
    vm = scared.Value()
    data = np.random.randint(0, 255, (16), dtype='uint8')
    res = vm(data)
    assert np.array_equal(res, data)


def test_value_model_raises_exception_when_given_non_numeric_array():
    vm = scared.Value()
    with pytest.raises(TypeError):
        vm(1)
    with pytest.raises(TypeError):
        vm("barr")
    with pytest.raises(ValueError):
        vm(np.array(["foo", "barr"]))


def test_value_model_preserves_first_dimension():
    vm = scared.Value()
    data = np.random.randint(0, 255, (500, 16), dtype='uint8')
    res = vm(data)
    assert data.shape[0] == res.shape[0]


def test_model_model_init_raises_exception_if_incorrect_bit_value_given():
    with pytest.raises(TypeError):
        scared.Monobit(bit="foo")
    with pytest.raises(TypeError):
        scared.Monobit(bit=2.3)
    with pytest.raises(ValueError):
        scared.Monobit(bit=-2)
    with pytest.raises(ValueError):
        scared.Monobit(bit=22)


def test_monobit_model_values_simple():
    vm = scared.Monobit(0)
    data = np.array([0, 1, 2, 3, 4], dtype='uint8')
    expected = np.array([0, 1, 0, 1, 0], dtype='uint8')
    res = vm(data)
    assert np.array_equal(expected, res)

    vm = scared.Monobit(1)
    expected = np.array([0, 0, 1, 1, 0], dtype='uint8')
    res = vm(data)
    assert np.array_equal(expected, res)


def test_monobit_model_raises_exception_when_given_non_numeric_array(bits):
    vm = scared.Monobit(bits)
    with pytest.raises(TypeError):
        vm(1)
    with pytest.raises(TypeError):
        vm("barr")
    with pytest.raises(ValueError):
        vm(np.array(["foo", "barr"]))


def test_monobit_model_shapes_and_type_output_validation(bits):
    vm = scared.Monobit(bits)
    data = np.random.randint(0, 510, (500, 16), dtype='uint16')
    res = vm(data)
    assert data.shape == res.shape
    assert res.dtype == 'uint8'


def test_hamming_weight_model_init_raises_exception_if_incorrect_nb_words_value_given():
    with pytest.raises(TypeError):
        scared.HammingWeight(nb_words="foo")
    with pytest.raises(TypeError):
        scared.HammingWeight(nb_words=2.3)
    with pytest.raises(ValueError):
        scared.HammingWeight(nb_words=-2)


def test_hamming_weight_model_values_simple():
    vm = scared.HammingWeight()
    data = np.array([0, 1, 3, 7, 15, 31, 63, 127, 255], dtype='uint8')
    expected = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8], dtype='uint32')
    res = vm(data)
    assert np.array_equal(expected, res)

    vm = scared.HammingWeight(2)
    expected = np.array([1, 5, 9, 13], dtype='uint32')
    res = vm(data)
    assert np.array_equal(expected, res)

    vm = scared.HammingWeight(3)
    expected = np.array([3, 12, 21], dtype='uint32')
    res = vm(data)
    assert np.array_equal(expected, res)

    data = np.array([
        [0, 1, 3, 7, 15, 31, 63, 127, 255],
        [0, 1, 3, 7, 15, 31, 63, 127, 255]], dtype='uint8')
    expected = np.array([
        [0, 1, 2, 3, 4, 5, 6, 7, 8],
        [0, 1, 2, 3, 4, 5, 6, 7, 8]], dtype='uint32')
    res = scared.HammingWeight()(data)
    assert np.array_equal(expected, res)

    expected = np.array([
        [1, 5, 9, 13],
        [1, 5, 9, 13]], dtype='uint32')
    res = scared.HammingWeight(2)(data)
    assert np.array_equal(expected, res)


def test_hamming_weight_model_raises_exception_when_given_non_bytes_array():
    vm = scared.HammingWeight()
    with pytest.raises(TypeError):
        vm(1)
    with pytest.raises(TypeError):
        vm("barr")
    with pytest.raises(ValueError):
        vm(np.array(["foo", "barr"]))
    with pytest.raises(ValueError):
        vm(np.array([564, 3.3]))


def test_hamming_weight_model_raises_exception_when_given_too_short_data():
    vm = scared.HammingWeight(nb_words=4)
    with pytest.raises(ValueError):
        vm(data=np.random.randint(0, 255, (500, 3), dtype='uint8'))


def test_hamming_weight_model_shapes_and_type_output_validation(nb_words):
    vm = scared.HammingWeight(nb_words)
    data = np.random.randint(0, 255, (500, 16), dtype='uint8')
    if nb_words > 16:
        with pytest.raises(ValueError):
            vm(data)
    else:
        res = vm(data)
        assert data.shape[:-1] == res.shape[:-1]
        assert data.shape[-1] // nb_words == res.shape[-1]
        assert res.dtype == 'uint32'


def test_monobit_model_provides_max_data_value_attribute(bits):
    monobit = scared.Monobit(bits)
    assert monobit.max_data_value == 1


def test_hamming_weight_model_provides_max_data_value_attribute(nb_words):
    hw = scared.HammingWeight(nb_words)
    assert hw.max_data_value == nb_words * 8


def test_value_model_provides_max_data_value_attribute():
    hw = scared.Value()
    assert hw.max_data_value == 256


def test_model_accepts_axis_parameters_with_default_value():
    val = scared.Value()
    data = np.random.randint(0, 255, (16, 500), dtype='uint8')
    res = val(data, axis=0)
    assert res.shape == data.shape

    m = scared.Monobit(5)
    res = m(data, axis=0)
    assert res.shape == data.shape

    hw = scared.HammingWeight()
    res = hw(data, axis=0)
    assert res.shape == data.shape

    data = np.array([
        [0, 1, 3, 7, 15, 31, 63, 127, 255],
        [0, 1, 3, 7, 15, 31, 63, 127, 255]], dtype='uint8').swapaxes(0, 1)
    expected = np.array([
        [0, 1, 2, 3, 4, 5, 6, 7, 8],
        [0, 1, 2, 3, 4, 5, 6, 7, 8]], dtype='uint32').swapaxes(0, 1)
    res = scared.HammingWeight()(data, axis=0)
    assert np.array_equal(expected, res)

    expected = np.array([
        [1, 5, 9, 13],
        [1, 5, 9, 13]], dtype='uint32').swapaxes(0, 1)
    res = scared.HammingWeight(2)(data, axis=0)
    assert np.array_equal(expected, res)
