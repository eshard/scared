from .context import scared  # noqa: F401
import pytest
import numpy as np


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
            [], ValueError, id="empty_list"
        ),
        pytest.param(
            [3, 4], TypeError, id="list_of_ints"
        ),
        pytest.param(
            3.14, TypeError, id="float"
        ),
        pytest.param(
            [[], []], TypeError, id="list_of_empty_lists"
        ),
        pytest.param(
            [["foo"], ["bar"]], TypeError, id="list_of_lists_of_string"
        ),
        pytest.param(
            [[3.14], [2.72]], TypeError, id="list_of_lists_of_float"
        )
    ]
)
def test_attack_selection_function_raises_exception_if_guesses_is_not_an_int_array(guesses, expected_error):
    with pytest.raises(expected_error):
        scared.attack_selection_function(function=default_sf, guesses=guesses)


def test_attack_selection_function_raises_exception_if_more_than_2d_ndarray_is_given():
    with pytest.raises(ValueError):
        scared.attack_selection_function(function=default_sf, guesses=np.ones((2, 2, 2), dtype=np.uint8))
    with pytest.raises(ValueError):
        scared.attack_selection_function(function=default_sf, guesses=[np.ones((2, 2), dtype=np.uint8), np.ones((2, 2), dtype=np.uint8)])


def test_attack_selection_function_accept_range_guesses():
    sf = scared.attack_selection_function(function=default_sf, guesses=range(128))
    assert np.array_equal(sf.guesses, np.arange(128, dtype='uint8'))
    assert isinstance(str(sf), str)


def test_attack_selection_function_accept_list_of_ranges():
    sf = scared.attack_selection_function(function=default_sf, guesses=[range(128), range(128)])
    ind_guess = np.array(range(128))
    tiled_guesses = np.stack(np.meshgrid(ind_guess, ind_guess, indexing='ij'), axis=-1).reshape(-1, 2)
    assert np.array_equal(sf.guesses, tiled_guesses)
    assert sf.guesses.ndim == 2
    assert sf.guesses.shape == (128 * 128, 2)


def test_attack_selection_function_accept_list_of_asymmetric_ndarrays():
    a = np.arange(128)
    b = np.arange(256)
    sf = scared.attack_selection_function(function=default_sf, guesses=[a, b])
    tiled_guesses = np.stack(np.meshgrid(a, b, indexing='ij'), axis=-1).reshape(-1, 2)
    assert np.array_equal(sf.guesses, tiled_guesses)
    assert sf.guesses.ndim == 2
    assert sf.guesses.shape == (128 * 256, 2)


def test_attack_selection_function_accept_list_of_asymmetric_signed_ndarrays():
    a = np.arange(3329 // 2 + 1)
    b = np.arange(-(3329 // 2), 3329 // 2 + 1)
    sf = scared.attack_selection_function(function=default_sf, guesses=[a, b])
    tiled_guesses = np.stack(np.meshgrid(a, b, indexing='ij'), axis=-1).reshape(-1, 2)
    assert np.array_equal(sf.guesses, tiled_guesses)
    assert sf.guesses.ndim == 2
    assert sf.guesses.shape == (len(a) * len(b), 2)


def test_attack_selection_function_guesses_yield_correct_length():
    n = 256
    sf = scared.attack_selection_function(function=default_sf, guesses=range(n))
    assert len(sf.guesses) == n
    sf = scared.attack_selection_function(function=default_sf, guesses=[range(n), range(n)])
    assert len(sf.guesses) == n * n


def test_attack_selection_function_guesses_subscriptable_1d():
    sf = scared.attack_selection_function(function=default_sf, guesses=range(128))
    assert sf.guesses[3] == 3


def test_attack_selection_function_guesses_subscriptable_2d():
    sf = scared.attack_selection_function(function=default_sf, guesses=[range(128), range(128)])
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
    "guesses, expected_dtype, explicit_dtype",
    [
        pytest.param(range(128), np.uint8, None, id="uint8_implicit"),
        pytest.param(range(3329), np.uint16, None, id="uint16_implicit"),
        pytest.param(range(65537), np.uint32, None, id="uint32_implicit"),
        pytest.param(range(-(3329 // 2), (3329 // 2) + 1), np.int16, None, id="int16_implicit"),
        pytest.param([range(3329 // 2 + 1), range(-(3329 // 2), 3329 // 2 + 1)], np.int16, None, id="int16_implicit_list"),
        pytest.param(range(128), np.uint32, np.uint32, id="uint32_explicit"),
        pytest.param(np.arange(3329, dtype=np.uint16), np.int16, np.int16, id="int16_explicit")
    ]
)
def test_attack_selection_function_guesses_dtypes(guesses, expected_dtype, explicit_dtype):
    sf = scared.attack_selection_function(function=default_sf, guesses=guesses, guesses_dtype=explicit_dtype)
    assert sf.guesses.dtype == expected_dtype


def test_attack_selection_function_guesses_default_value():
    sf = scared.attack_selection_function(function=default_sf)
    assert np.array_equal(sf.guesses, np.arange(256, dtype='uint8'))
    assert isinstance(str(sf), str)


def test_guesses_in_decorated_function():
    sf = scared.attack_selection_function(function=default_sf, guesses=[range(128), range(256)])
    assert type(sf.guesses) is scared.selection_functions.guesses.Guesses


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
