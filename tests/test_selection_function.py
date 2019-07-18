from .context import scared  # noqa: F401
import pytest
import numpy as np


def default_sf(guesses):
    return guesses


def test_selection_function_raises_exception_if_function_is_not_callable():
    with pytest.raises(TypeError):
        scared.selection_function(function='foo')


def test_attack_selection_function_raises_exception_if_guesses_is_not_a_bytes_array():
    with pytest.raises(TypeError):
        scared.attack_selection_function(function=default_sf, guesses="foo")
    with pytest.raises(TypeError):
        scared.attack_selection_function(function=default_sf, guesses=16)
    with pytest.raises(TypeError):
        scared.attack_selection_function(function=default_sf, guesses=[])


def test_attack_selection_function_accept_range_guesses():
    sf = scared.attack_selection_function(function=default_sf, guesses=range(128))
    assert np.array_equal(sf.guesses, np.arange(128, dtype='uint8'))
    assert isinstance(str(sf), str)


def test_attack_selection_function_guesses_default_value():
    sf = scared.attack_selection_function(function=default_sf)
    assert np.array_equal(sf.guesses, np.arange(256, dtype='uint8'))
    assert isinstance(str(sf), str)


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
            out[guess] = scared.aes.sub_bytes(np.bitwise_xor(plaintext, guess))
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
            out[guess] = scared.aes.sub_bytes(np.bitwise_xor(plaintext, guess))
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
            out[guess] = scared.aes.sub_bytes(np.bitwise_xor(plaintext, guess))
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
            out[guess] = np.bitwise_xor(plaintext, guess)
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
            out[guess] = np.bitwise_xor(plaintext, guess)
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
            out[guess] = np.bitwise_xor(plaintext, guess)
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
            out[guess] = np.bitwise_xor(plaintext, guess)
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
