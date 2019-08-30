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


def test_hamming_weight_uint8():
    a = np.array([[3, 28, 89, 100, 89, 158, 199],
                  [183, 46, 190, 184, 144, 163, 165],
                  [80, 122, 200, 112, 86, 135, 243]], dtype='uint8')
    b = np.array([[2, 3, 4, 3, 4, 5, 5],
                  [6, 4, 6, 4, 2, 4, 4],
                  [2, 5, 3, 3, 4, 4, 6]])
    hw = scared.HammingWeight()
    res = hw(a)
    assert np.array_equal(res, b)


def test_hamming_weight_uint16():
    a = np.array([[49492, 35012, 17375, 58517, 7925, 37294, 1960],
                  [12319, 21194, 3592, 62643, 7298, 52388, 17174],
                  [5624, 52639, 16319, 28373, 20979, 33310, 30228]], dtype='uint16')
    b = np.array([[6, 5, 10, 8, 10, 8, 6],
                  [7, 7, 4, 10, 5, 7, 6],
                  [8, 11, 13, 10, 9, 6, 7]])
    hw = scared.HammingWeight(expected_dtype='uint16')
    res = hw(a)
    assert np.array_equal(res, b)


def test_hamming_weight_uint32():
    a = np.array([[3011812195, 566284227, 3180952330, 3224490982, 1398648567, 2997601074, 3466990530],
                  [1745373437, 2305307499, 528675825, 3245007040, 3385056644, 1402421254, 890085974],
                  [3910116154, 3106231214, 503027565, 263222377, 3958549359, 3780560129, 2475275928]], dtype='uint32')
    b = np.array([[16, 14, 18, 15, 21, 16, 15],
                  [13, 15, 18, 12, 16, 12, 14],
                  [17, 17, 21, 15, 22, 13, 15]])
    hw = scared.HammingWeight(expected_dtype='uint32')
    res = hw(a)
    assert np.array_equal(res, b)


def test_hamming_weight_uint64():
    a = np.array([[1110681567975564723, 2603173294400796779, 8824234896931680599, 2581838972723594911,
                   6596729868911047227, 2074859436293001592, 11731185813118878444],
                  [16879640904508054626, 16622460785862736051, 15934166478821968156, 6382170377471969988,
                   14708844167980688964, 5124927357537709470, 8409390396293004342],
                  [2595719322730764045, 6531922580352310653, 3061678657169528223, 8881635347030906723,
                   4337229908681420868, 4353603953276958515, 6752076961829864544]], dtype='uint64')
    b = np.array([[33, 27, 40, 32, 35, 28, 31],
                  [30, 33, 34, 26, 25, 37, 25],
                  [31, 33, 31, 30, 31, 38, 30]])
    hw = scared.HammingWeight(expected_dtype='uint64')
    res = hw(a)
    assert np.array_equal(res, b)


def test_hamming_weight_wrong_expected_dtype():
    with pytest.raises(ValueError):
        scared.HammingWeight(expected_dtype='coucou')


def test_hamming_weight_uint16_with_wong_expected_dtype():
    a = np.array([[49492, 35012, 17375, 58517, 7925, 37294, 1960],
                  [12319, 21194, 3592, 62643, 7298, 52388, 17174],
                  [5624, 52639, 16319, 28373, 20979, 33310, 30228]], dtype='uint16')
    hw = scared.HammingWeight()
    with pytest.raises(ValueError):
        hw(a)
