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


def test_attack_selection_function_guesses_default_value():
    sf = scared.attack_selection_function(function=default_sf)
    assert np.array_equal(sf.guesses, np.arange(256, dtype='uint8'))


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
    with pytest.raises(TypeError):
        sf(keys=np.random.randint(0, 255, (16), dtype='uint8'))

    @scared.attack_selection_function
    def sf(plain, guesses):
        return plain
    with pytest.raises(TypeError):
        sf(keys=np.random.randint(0, 255, (16), dtype='uint8'))

    @scared.reverse_selection_function
    def sf(plain, guesses):
        return plain
    with pytest.raises(TypeError):
        sf(keys=np.random.randint(0, 255, (16), dtype='uint8'))


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

    @scared.selection_function
    def sf(plaintext):
        out = np.empty(tuple([1, plaintext.shape[0], 16]), dtype='uint8')
        out[0] = np.bitwise_xor(plaintext, 0)
        return out.swapaxes(0, 1)
    res = sf(**datas)
    assert res.shape[:2] == (datas['plaintext'].shape[0], 1)

    @scared.reverse_selection_function
    def sf(plaintext):
        out = np.empty(tuple([1, plaintext.shape[0], 16]), dtype='uint8')
        out[0] = np.bitwise_xor(plaintext, 0)
        return out.swapaxes(0, 1)
    res = sf(**datas)
    assert res.shape[:2] == (datas['plaintext'].shape[0], 1)


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

    with pytest.raises(ValueError):
        sf(**datas)

    @scared.selection_function
    def sf(plaintext):
        out = np.empty(tuple([1, plaintext.shape[0], 16]), dtype='uint8')
        out[0] = np.bitwise_xor(plaintext, 0)
        return out[:, 0]

    with pytest.raises(ValueError):
        sf(**datas)

    @scared.reverse_selection_function
    def sf(plaintext):
        out = np.empty(tuple([1, plaintext.shape[0], 16]), dtype='uint8')
        out[0] = np.bitwise_xor(plaintext, 0)
        return out[:, 0]

    with pytest.raises(ValueError):
        sf(**datas)


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

    res = sf(**datas)
    assert res.shape == (datas['plaintext'].shape[0], 256, datas['plaintext'].shape[1] - 2)

    @scared.selection_function(words=np.arange(datas['plaintext'].shape[1] - 2))
    def sf(plaintext):
        out = np.empty(tuple([1, plaintext.shape[0], 16]), dtype='uint8')
        out[0] = np.bitwise_xor(plaintext, 0)
        return out.swapaxes(0, 1)

    res = sf(**datas)
    assert res.shape == (datas['plaintext'].shape[0], 1, datas['plaintext'].shape[1] - 2)

    @scared.reverse_selection_function(words=np.arange(datas['plaintext'].shape[1] - 2))
    def sf(plaintext):
        out = np.empty(tuple([1, plaintext.shape[0], 16]), dtype='uint8')
        out[0] = np.bitwise_xor(plaintext, 0)
        return out.swapaxes(0, 1)

    res = sf(**datas)
    assert res.shape == (datas['plaintext'].shape[0], 1, datas['plaintext'].shape[1] - 2)


def test_selection_function_compute_raises_exception_if_words_selection_is_inconsistent():
    @scared.attack_selection_function(words=18)
    def sf(plaintext, guesses):
        out = np.empty(tuple([len(guesses), plaintext.shape[0], 16]), dtype='uint8')
        for guess in guesses:
            out[guess] = np.bitwise_xor(plaintext, guess)
        return out.swapaxes(0, 1)
    with pytest.raises(ValueError):
        sf(plaintext=np.random.randint(0, 255, (200, 16), dtype='uint8'))

    @scared.selection_function(words=18)
    def sf(plaintext):
        out = np.empty(tuple([1, plaintext.shape[0], 16]), dtype='uint8')
        out[0] = np.bitwise_xor(plaintext, 0)
        return out.swapaxes(0, 1)
    with pytest.raises(ValueError):
        sf(plaintext=np.random.randint(0, 255, (200, 16), dtype='uint8'))

    @scared.reverse_selection_function(words=18)
    def sf(plaintext):
        out = np.empty(tuple([1, plaintext.shape[0], 16]), dtype='uint8')
        out[0] = np.bitwise_xor(plaintext, 0)
        return out.swapaxes(0, 1)
    with pytest.raises(ValueError):
        sf(plaintext=np.random.randint(0, 255, (200, 16), dtype='uint8'))
