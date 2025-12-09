from .context import scared  # noqa: F401
import pytest
import numpy as np
import numba as nb


def default_sf(guesses):
    return guesses


def test_selection_function_raises_exception_if_function_is_not_callable():
    with pytest.raises(TypeError):
        scared.selection_function(function='foo')


@pytest.mark.parametrize(
    "guesses, expected_error",
    [
        pytest.param(
            "foo", TypeError, id="string"
        ),
        pytest.param(
            16, TypeError, id="int"
        ),
        pytest.param(
            [], TypeError, id="list"
        ),
        pytest.param(
            [3, 4], TypeError, id="list_of_ints"
        ),
        pytest.param(
            3.14, TypeError, id="float"
        ),
        pytest.param(
            np.zeros((3), dtype=float), TypeError, id="float"
        ),
        pytest.param(
            np.zeros((3, 3), dtype=float), TypeError, id="float"
        )
    ]
)
def test_attack_selection_function_raises_exception_if_guesses_is_not_an_int_array(guesses, expected_error):
    with pytest.raises(expected_error):
        scared.attack_selection_function(function=default_sf, guesses=guesses)


def test_attack_selection_function_raises_exception_if_more_than_2d_ndarray_is_given():
    with pytest.raises(ValueError):
        scared.attack_selection_function(function=default_sf, guesses=np.ones((2, 2, 2), dtype=np.uint8))


def test_attack_selection_function_accept_range_guesses():
    sf = scared.attack_selection_function(function=default_sf, guesses=range(128))
    assert np.array_equal(sf.guesses, np.arange(128, dtype='uint8'))
    assert isinstance(str(sf), str)


def test_attack_selection_function_accept_guesses_object():
    sf = scared.attack_selection_function(function=default_sf, guesses=range(128))
    sf_2 = scared.attack_selection_function(function=default_sf, guesses=sf.guesses)
    assert np.all(sf.guesses == sf_2.guesses)
    assert sf.guesses is not sf_2.guesses


@pytest.mark.parametrize(
    "guesses, nb_words_guess",
    [
        pytest.param(range(256), 2),
        pytest.param(range(-127, 128), 2),
        pytest.param(range(10), 3)
    ]
)
def test_attack_selection_function_guesses_expand_to_multiple_words(guesses, nb_words_guess):
    sf = scared.attack_selection_function(function=default_sf, guesses=guesses, nb_words_guess=nb_words_guess)
    assert sf.guesses.ndim == 2
    assert len(sf.guesses) == len(guesses) ** nb_words_guess
    tiled_guesses = np.stack(np.meshgrid(*([guesses] * nb_words_guess), indexing='ij'), axis=-1).reshape(-1, nb_words_guess)
    assert np.all(sf.guesses == tiled_guesses)


def test_attack_selection_function_guesses_ignore_nb_words_if_2d_array_given():
    guesses = np.ones((5, 5), np.uint16)
    nb_words_guess = 5
    sf = scared.attack_selection_function(function=default_sf, guesses=guesses, nb_words_guess=nb_words_guess)
    assert sf.guesses.ndim == 2
    assert np.all(sf.guesses == guesses)


def test_attack_selection_function_guesses_accept_list_of_asymmetric_ndarrays():
    a = np.arange(128)
    b = np.arange(256)
    tiled_guesses = np.stack(np.meshgrid(a, b, indexing='ij'), axis=-1).reshape(-1, 2)
    sf = scared.attack_selection_function(function=default_sf, guesses=tiled_guesses)
    assert np.array_equal(sf.guesses, tiled_guesses)


def test_attack_selection_function_guesses_accept_list_of_asymmetric_signed_ndarrays():
    a = np.arange(3329 // 2 + 1)
    b = np.arange(-(3329 // 2), 3329 // 2 + 1)
    tiled_guesses = np.stack(np.meshgrid(a, b, indexing='ij'), axis=-1).reshape(-1, 2)
    sf = scared.attack_selection_function(function=default_sf, guesses=tiled_guesses)
    assert np.array_equal(sf.guesses, tiled_guesses)


def test_attack_selection_function_guesses_yield_correct_length():
    n = 256
    sf = scared.attack_selection_function(function=default_sf, guesses=range(n))
    assert len(sf.guesses) == n


def test_attack_selection_function_guesses_subscriptable_1d():
    sf = scared.attack_selection_function(function=default_sf, guesses=range(128))
    assert sf.guesses[3] == 3


def test_attack_selection_function_guesses_subscriptable_2d():
    sf = scared.attack_selection_function(function=default_sf, guesses=range(128), nb_words_guess=2)
    assert np.array_equal(sf.guesses[3], np.array([0, 3], dtype='uint8'))


def test_attack_selection_function_guesses_is_iterable():
    sf = scared.attack_selection_function(function=default_sf, guesses=range(3))

    # 1. Check that obj is an instance of an iterable
    from collections.abc import Iterable
    assert isinstance(sf.guesses, Iterable)

    # 2. Check that iteration works as expected
    assert list(sf.guesses) == [0, 1, 2]


def test_attack_selection_function_guesses_returns_iterator():
    sf = scared.attack_selection_function(function=default_sf, guesses=range(3))
    iterator = iter(sf.guesses)
    from collections.abc import Iterator
    assert isinstance(iterator, Iterator)


def test_attack_selection_function_guesses_not_from_zero():
    sf = scared.attack_selection_function(function=default_sf, guesses=range(128, 256))
    assert sf.guesses[0] == 128
    assert sf.guesses[127] == 255


@pytest.mark.parametrize(
    "guesses, nb_words_guess, expected_dtype, explicit_dtype",
    [
        pytest.param(range(128), 1, np.uint8, None, id="uint8_implicit"),
        pytest.param(range(3329), 1, np.uint16, None, id="uint16_implicit"),
        pytest.param(range(65537), 1, np.uint32, None, id="uint32u_implicit"),
        pytest.param(range(-(3329 // 2), (3329 // 2) + 1), 1, np.int16, None, id="int16_implicit"),
        pytest.param(range(-(3329 // 2), (3329 // 2) + 1), 2, np.int16, None, id="int16_implicit_2d"),
        pytest.param(range(128), 1, np.uint32, np.uint32, id="uint32_explicit"),
        pytest.param(np.arange(3329, dtype=np.uint16), 1, np.int16, np.int16, id="int16_explicit"),
        pytest.param(np.arange(3329, dtype=np.uint16), 2, np.int16, np.int16, id="int16_explicit_2d")
    ]
)
def test_attack_selection_function_guesses_dtypes(guesses, nb_words_guess, expected_dtype, explicit_dtype):
    sf = scared.attack_selection_function(function=default_sf, guesses=guesses, guesses_dtype=explicit_dtype)
    assert sf.guesses.dtype == expected_dtype


def test_attack_selection_function_guesses_default_value():
    sf = scared.attack_selection_function(function=default_sf)
    assert np.array_equal(sf.guesses, np.arange(256, dtype='uint8'))
    assert isinstance(str(sf), str)


def test_guesses_correct_type():
    sf = scared.attack_selection_function(function=default_sf, guesses=range(128), nb_words_guess=2)
    assert type(sf.guesses) is scared.selection_functions.guesses.Guesses


@nb.vectorize
def xor_nb_vectorized(a, b):
    return a ^ b


def xor_callable(guesses: np.ndarray, data: np.ndarray):
    ret_arr = np.empty((guesses.shape[0], data.shape[0], data.shape[1]), dtype=data.dtype)
    guesses = guesses.reshape((guesses.shape[0], -1))
    data = data.reshape((data.shape[0], data.shape[1] // guesses.shape[1], -1))

    # To simulate what a user without vectorization would do
    for i, guess in enumerate(guesses):
        for j, data_trace in enumerate(data):
            for k, word in enumerate(data_trace):
                ret_arr[i, j, k:k + guesses.shape[-1]] = word ^ guess
    return ret_arr.swapaxes(0, 1)


@pytest.mark.parametrize("op", [np.bitwise_xor, xor_nb_vectorized, xor_callable])
@pytest.mark.parametrize("guesses, nb_words_guess", [
    pytest.param(range(256), 1, id="guesses_1d"),
    pytest.param(range(32), 2, id="guesses_2d")
])
def test_guesses_expand(op, guesses, nb_words_guess):
    sf = scared.attack_selection_function(function=lambda x: x, guesses=guesses, nb_words_guess=nb_words_guess)
    data = np.random.randint(0, 256, (100, 10))
    assert sf.guesses.expand(data, op).ndim == 3
    assert sf.guesses.expand(data, op).shape[0:2] == (data.shape[0], len(sf.guesses))  # Last dimension is inferred


def test_guesses_expand_raises_exception_if_operation_is_not_callable():
    with pytest.raises(TypeError):
        sf = scared.attack_selection_function(function=lambda data, guesses: guesses.expand(data, 1))
        sf(data=np.array(1))


def test_guesses_expand_raises_exception_if_arr_is_not_ndarray():
    sf = scared.attack_selection_function(function=lambda x: x)
    with pytest.raises(TypeError):
        sf.guesses.expand(arr=1, op=np.bitwise_xor)


def test_guesses_expand_raises_exception_if_arr_is_more_than_2d():
    sf = scared.attack_selection_function(function=lambda x: x)
    with pytest.raises(ValueError):
        sf.guesses.expand(arr=np.zeros((2, 2, 2)), op=np.bitwise_xor)


def test_guesses_expand_raises_exception_if_guesses_words_do_not_divide_data_words():
    data = np.random.randint(0, 256, (100, 3))
    sf = scared.attack_selection_function(function=lambda data, guesses: guesses.expand(data, np.bitwise_xor), guesses=range(256), nb_words_guess=2)
    with pytest.raises(ValueError):
        sf(data=data)


def test_guesses_expand_raises_exception_if_op_does_not_return_ndarray():
    def op(guesses, arr):
        return 1
    data = np.random.randint(0, 256, (100, 3))
    sf = scared.attack_selection_function(function=lambda x: x)
    with pytest.raises(TypeError):
        sf.guesses.expand(data, op)


@pytest.mark.parametrize("op", [
    lambda guesses, arr: np.empty((2, 2, 2, 2), dtype=guesses.dtype),
    lambda guesses, arr: np.empty((2, 2, 2), dtype=guesses.dtype)
])
def test_guesses_expand_raises_exception_if_op_does_a_well_shaped_ndarray(op):
    data = np.random.randint(0, 256, (100, 3))
    sf = scared.attack_selection_function(function=lambda x: x)
    with pytest.raises(ValueError):
        sf.guesses.expand(data, op)


def test_attack_selection_function_raises_exception_if_function_is_not_callable():
    with pytest.raises(TypeError):
        scared.attack_selection_function(function='foo')


def test_reverse_selection_function_init_raises_exception_if_function_is_not_callable():
    with pytest.raises(TypeError):
        scared.reverse_selection_function(function='foo')


def test_selection_function_call_raises_exception_if_missing_kwargs():
    @scared.selection_function
    def sf(plain):
        return plain
    with pytest.raises(scared.SelectionFunctionError):
        sf(keys=np.random.randint(0, 255, (16), dtype='uint8'))
    assert isinstance(str(sf), str)

    @scared.attack_selection_function
    def sf(plain, guesses):
        return plain
    with pytest.raises(scared.SelectionFunctionError):
        sf(keys=np.random.randint(0, 255, (16), dtype='uint8'))
    assert isinstance(str(sf), str)

    @scared.reverse_selection_function
    def sf(plain, guesses):
        return plain
    with pytest.raises(scared.SelectionFunctionError):
        sf(keys=np.random.randint(0, 255, (16), dtype='uint8'))
    assert isinstance(str(sf), str)


def test_attack_selection_function_raises_exception_if_expected_key_function_is_not_a_callable():
    with pytest.raises(TypeError):
        scared.attack_selection_function(function=default_sf, expected_key_function='foo')
    with pytest.raises(TypeError):
        scared.attack_selection_function(function=default_sf, expected_key_function=12340)
    with pytest.raises(TypeError):
        scared.attack_selection_function(function=default_sf, expected_key_function={})


def test_attack_selection_function_with_expected_key_function():
    datas = {
        'plaintext': np.random.randint(0, 255, (500, 16), dtype='uint8'),
        'ciphertext': np.random.randint(0, 255, (500, 16), dtype='uint8'),
        'key': np.random.randint(0, 255, (500, 16), dtype='uint8')
    }

    def expected_key(key):
        return scared.aes.key_schedule(key)[0]

    @scared.attack_selection_function(expected_key_function=expected_key)
    def first_bytes(guesses, plaintext):
        out = np.empty(tuple([len(guesses), plaintext.shape[0], 16]), dtype='uint8')
        for guess in guesses:
            out[guess] = scared.aes.sub_bytes(np.bitwise_xor(plaintext, guess))  # TODO: Value of guesses, not index
        return out.swapaxes(0, 1)

    computed_key = first_bytes.compute_expected_key(**datas)
    assert np.array_equal(computed_key, scared.aes.key_schedule(datas['key'])[0])
    assert isinstance(str(first_bytes), str)


def test_attack_selection_function_returns_none_without_expected_key_function():
    datas = {
        'plaintext': np.random.randint(0, 255, (500, 16), dtype='uint8'),
        'ciphertext': np.random.randint(0, 255, (500, 16), dtype='uint8'),
        'key': np.random.randint(0, 255, (500, 16), dtype='uint8')
    }

    @scared.attack_selection_function
    def first_bytes(guesses, plaintext):
        out = np.empty(tuple([len(guesses), plaintext.shape[0], 16]), dtype='uint8')
        for guess in guesses:
            out[guess] = scared.aes.sub_bytes(np.bitwise_xor(plaintext, guess))  # TODO: Value of guesses, not their index
        return out.swapaxes(0, 1)

    assert first_bytes.compute_expected_key(**datas) is None
    assert isinstance(str(first_bytes), str)


def test_attack_selection_function_compute_expected_key_raises_exception_if_missing_expected_key_function_args():
    datas = {
        'plaintext': np.random.randint(0, 255, (500, 16), dtype='uint8'),
        'ciphertext': np.random.randint(0, 255, (500, 16), dtype='uint8'),
        'keys': np.random.randint(0, 255, (500, 16), dtype='uint8')
    }

    def expected_key(key):
        return scared.aes.key_schedule(key)[0]

    @scared.attack_selection_function(expected_key_function=expected_key)
    def first_bytes(guesses, plaintext):
        out = np.empty(tuple([len(guesses), plaintext.shape[0], 16]), dtype='uint8')
        for guess in guesses:
            out[guess] = scared.aes.sub_bytes(np.bitwise_xor(plaintext, guess))  # TODO: Value of guesses, not their index
        return out.swapaxes(0, 1)

    with pytest.raises(scared.SelectionFunctionError):
        first_bytes.compute_expected_key(**datas)
    assert isinstance(str(first_bytes), str)


def test_selection_function_computes_intermediate_datas():
    datas = {
        'plaintext': np.random.randint(0, 255, (500, 16), dtype='uint8'),
        'ciphertext': np.random.randint(0, 255, (500, 16), dtype='uint8'),
        'key': np.random.randint(0, 255, (500, 16), dtype='uint8')
    }

    @scared.attack_selection_function
    def sf(plaintext, guesses):
        out = np.empty(tuple([len(guesses), plaintext.shape[0], 16]), dtype='uint8')
        for guess in guesses:
            out[guess] = np.bitwise_xor(plaintext, guess)  # TODO: Value of guesses, not their index
        return out.swapaxes(0, 1)
    res = sf(**datas)
    assert res.shape[:2] == (datas['plaintext'].shape[0], 256)
    assert isinstance(str(sf), str)

    @scared.selection_function
    def sf(plaintext):
        out = np.empty(tuple([1, plaintext.shape[0], 16]), dtype='uint8')
        out[0] = np.bitwise_xor(plaintext, 0)
        return out.swapaxes(0, 1)
    res = sf(**datas)
    assert res.shape[:2] == (datas['plaintext'].shape[0], 1)
    assert isinstance(str(sf), str)

    @scared.reverse_selection_function
    def sf(plaintext):
        out = np.empty(tuple([1, plaintext.shape[0], 16]), dtype='uint8')
        out[0] = np.bitwise_xor(plaintext, 0)
        return out.swapaxes(0, 1)
    res = sf(**datas)
    assert res.shape[:2] == (datas['plaintext'].shape[0], 1)
    assert isinstance(str(sf), str)


def test_selection_function_raises_exceptions_if_intermediate_values_has_inconsistent_shape():
    datas = {
        'plaintext': np.random.randint(0, 255, (500, 16), dtype='uint8'),
        'ciphertext': np.random.randint(0, 255, (500, 16), dtype='uint8'),
        'key': np.random.randint(0, 255, (500, 16), dtype='uint8')
    }

    @scared.attack_selection_function
    def sf(plaintext, guesses):
        out = np.empty(tuple([len(guesses), plaintext.shape[0], 16]), dtype='uint8')
        for guess in guesses:
            out[guess] = np.bitwise_xor(plaintext, guess)  # TODO: Value of guesses, not their index
        return out[:, 0]

    with pytest.raises(scared.SelectionFunctionError):
        sf(**datas)
    assert isinstance(str(sf), str)

    @scared.selection_function
    def sf(plaintext):
        out = np.empty(tuple([1, plaintext.shape[0], 16]), dtype='uint8')
        out[0] = np.bitwise_xor(plaintext, 0)
        return out[:, 0]

    with pytest.raises(scared.SelectionFunctionError):
        sf(**datas)
    assert isinstance(str(sf), str)

    @scared.reverse_selection_function
    def sf(plaintext):
        out = np.empty(tuple([1, plaintext.shape[0], 16]), dtype='uint8')
        out[0] = np.bitwise_xor(plaintext, 0)
        return out[:, 0]

    with pytest.raises(scared.SelectionFunctionError):
        sf(**datas)
    assert isinstance(str(sf), str)


def test_selection_function_init_raises_exception_with_improper_words():
    with pytest.raises(TypeError):
        scared.attack_selection_function(function=default_sf, words="foo")
    with pytest.raises(TypeError):
        scared.attack_selection_function(function=default_sf, words=np.array([1.2, 3.0]))
    with pytest.raises(TypeError):
        scared.attack_selection_function(function=default_sf, words=12.0)

    with pytest.raises(TypeError):
        scared.selection_function(function=default_sf, words="foo")
    with pytest.raises(TypeError):
        scared.selection_function(function=default_sf, words=np.array([1.2, 3.0]))
    with pytest.raises(TypeError):
        scared.selection_function(function=default_sf, words=12.0)

    with pytest.raises(TypeError):
        scared.reverse_selection_function(function=default_sf, words="foo")
    with pytest.raises(TypeError):
        scared.reverse_selection_function(function=default_sf, words=np.array([1.2, 3.0]))
    with pytest.raises(TypeError):
        scared.reverse_selection_function(function=default_sf, words=12.0)


def test_selection_function_accepts_list_for_words():
    sf = scared.selection_function(function=default_sf, words=[1, 2, 4])
    assert isinstance(sf.words, np.ndarray)
    assert np.array_equal(sf.words, np.array([1, 2, 4], dtype='uint8'))
    assert isinstance(str(sf), str)


def test_selection_function_computes_intermediate_datas_with_words_selection():
    datas = {
        'plaintext': np.random.randint(0, 255, (500, 16), dtype='uint8'),
        'ciphertext': np.random.randint(0, 255, (500, 16), dtype='uint8'),
        'key': np.random.randint(0, 255, (500, 16), dtype='uint8')
    }

    @scared.attack_selection_function(words=np.arange(datas['plaintext'].shape[1] - 2))
    def sf(plaintext, guesses):
        out = np.empty(tuple([len(guesses), plaintext.shape[0], 16]), dtype='uint8')
        for guess in guesses:
            out[guess] = np.bitwise_xor(plaintext, guess)  # TODO: Value of guesses, not their index
        return out.swapaxes(0, 1)

    assert isinstance(str(sf), str)

    res = sf(**datas)
    assert res.shape == (datas['plaintext'].shape[0], 256, datas['plaintext'].shape[1] - 2)

    @scared.selection_function(words=np.arange(datas['plaintext'].shape[1] - 2))
    def sf(plaintext):
        out = np.empty(tuple([1, plaintext.shape[0], 16]), dtype='uint8')
        out[0] = np.bitwise_xor(plaintext, 0)
        return out.swapaxes(0, 1)

    assert isinstance(str(sf), str)
    res = sf(**datas)
    assert res.shape == (datas['plaintext'].shape[0], 1, datas['plaintext'].shape[1] - 2)

    @scared.reverse_selection_function(words=np.arange(datas['plaintext'].shape[1] - 2))
    def sf(plaintext):
        out = np.empty(tuple([1, plaintext.shape[0], 16]), dtype='uint8')
        out[0] = np.bitwise_xor(plaintext, 0)
        return out.swapaxes(0, 1)

    assert isinstance(str(sf), str)
    res = sf(**datas)
    assert res.shape == (datas['plaintext'].shape[0], 1, datas['plaintext'].shape[1] - 2)


def test_selection_function_compute_raises_exception_if_words_selection_is_inconsistent():
    @scared.attack_selection_function(words=18)
    def sf(plaintext, guesses):
        out = np.empty(tuple([len(guesses), plaintext.shape[0], 16]), dtype='uint8')
        for guess in guesses:
            out[guess] = np.bitwise_xor(plaintext, guess)  # TODO: Value of guesses, not their index
        return out.swapaxes(0, 1)
    with pytest.raises(scared.SelectionFunctionError):
        sf(plaintext=np.random.randint(0, 255, (200, 16), dtype='uint8'))
    assert isinstance(str(sf), str)

    @scared.selection_function(words=18)
    def sf(plaintext):
        out = np.empty(tuple([1, plaintext.shape[0], 16]), dtype='uint8')
        out[0] = np.bitwise_xor(plaintext, 0)
        return out.swapaxes(0, 1)
    with pytest.raises(scared.SelectionFunctionError):
        sf(plaintext=np.random.randint(0, 255, (200, 16), dtype='uint8'))
    assert isinstance(str(sf), str)

    @scared.reverse_selection_function(words=18)
    def sf(plaintext):
        out = np.empty(tuple([1, plaintext.shape[0], 16]), dtype='uint8')
        out[0] = np.bitwise_xor(plaintext, 0)
        return out.swapaxes(0, 1)
    with pytest.raises(scared.SelectionFunctionError):
        sf(plaintext=np.random.randint(0, 255, (200, 16), dtype='uint8'))
    assert isinstance(str(sf), str)
