from .context import scared  # noqa: F401
from scared import aes, selection_functions
import numpy as np
import pytest


def test_aes_encrypt_first_round_key_with_default_arguments():
    sf = aes.selection_functions.encrypt.FirstAddRoundKey()
    assert sf.guesses.tolist() == list(range(256))
    assert sf.words is Ellipsis
    assert sf.target_tag == 'plaintext'
    assert sf.key_tag == 'key'
    assert isinstance(sf, selection_functions.SelectionFunction)
    data = np.random.randint(0, 255, (10, 16), dtype='uint8')
    expected = np.empty((10, 256, 16), dtype='uint8')
    for i in np.arange(256, dtype='uint8'):
        expected[:, i, :] = np.bitwise_xor(data, i)
    assert np.array_equal(expected, sf(plaintext=data))
    master_key = np.random.randint(0, 255, (16,), dtype='uint8')
    expected_key = aes.key_schedule(master_key)[0]
    assert np.array_equal(expected_key, sf.compute_expected_key(key=master_key))
    assert isinstance(str(sf), str)


def test_aes_encrypt_first_round_key_with_alternative_args():
    sf = aes.selection_functions.encrypt.FirstAddRoundKey(
        plaintext_tag='plain',
        words=6,
        guesses=_alt_guesses,
        key_tag='thekey'
    )
    assert sf.guesses.tolist() == _alt_guesses.tolist()
    assert sf.words == 6
    assert sf.target_tag == 'plain'
    assert sf.key_tag == 'thekey'
    assert isinstance(sf, selection_functions.SelectionFunction)
    data = np.random.randint(0, 255, (10, 16), dtype='uint8')
    expected = np.empty((10, len(_alt_guesses), 16), dtype='uint8')
    for i, guess in enumerate(_alt_guesses):
        expected[:, i, :] = np.bitwise_xor(data, guess)
    assert np.array_equal(expected[:, :, 6], sf(plain=data))
    master_key = np.random.randint(0, 255, (16,), dtype='uint8')
    expected_key = aes.key_schedule(master_key)[0]
    assert np.array_equal(expected_key, sf.compute_expected_key(thekey=master_key))
    assert isinstance(str(sf), str)


def test_aes_encrypt_last_round_key_with_default_arguments():
    sf = aes.selection_functions.encrypt.LastAddRoundKey()
    assert sf.guesses.tolist() == list(range(256))
    assert sf.words is Ellipsis
    assert sf.target_tag == 'ciphertext'
    assert sf.key_tag == 'key'
    assert isinstance(sf, selection_functions.SelectionFunction)
    data = np.random.randint(0, 255, (10, 16), dtype='uint8')
    expected = np.empty((10, 256, 16), dtype='uint8')
    for i in np.arange(256, dtype='uint8'):
        expected[:, i, :] = np.bitwise_xor(data, i)
    assert np.array_equal(expected, sf(ciphertext=data))
    master_key = np.random.randint(0, 255, (16,), dtype='uint8')
    expected_key = aes.key_schedule(master_key)[-1]
    assert np.array_equal(expected_key, sf.compute_expected_key(key=master_key))
    assert isinstance(str(sf), str)


def test_aes_encrypt_last_round_key_with_alternative_args():
    sf = aes.selection_functions.encrypt.LastAddRoundKey(
        ciphertext_tag='nop',
        words=6,
        guesses=_alt_guesses,
        key_tag='thekey'
    )
    assert sf.guesses.tolist() == _alt_guesses.tolist()
    assert sf.words == 6
    assert sf.target_tag == 'nop'
    assert sf.key_tag == 'thekey'
    assert isinstance(sf, selection_functions.SelectionFunction)
    data = np.random.randint(0, 255, (10, 16), dtype='uint8')
    expected = np.empty((10, len(_alt_guesses), 16), dtype='uint8')
    for i, guess in enumerate(_alt_guesses):
        expected[:, i, :] = np.bitwise_xor(data, guess)
    assert np.array_equal(expected[:, :, 6], sf(nop=data))
    master_key = np.random.randint(0, 255, (16,), dtype='uint8')
    expected_key = aes.key_schedule(master_key)[-1]
    assert np.array_equal(expected_key, sf.compute_expected_key(thekey=master_key))
    assert isinstance(str(sf), str)


def test_aes_encrypt_first_sub_bytes_with_default_arguments():
    sf = aes.selection_functions.encrypt.FirstSubBytes()
    assert sf.guesses.tolist() == list(range(256))
    assert sf.words is Ellipsis
    assert sf.target_tag == 'plaintext'
    assert sf.key_tag == 'key'
    assert isinstance(sf, selection_functions.SelectionFunction)
    data = np.random.randint(0, 255, (10, 16), dtype='uint8')
    expected = np.empty((10, 256, 16), dtype='uint8')
    for i in np.arange(256, dtype='uint8'):
        expected[:, i, :] = np.bitwise_xor(data, i)
    expected = aes.sub_bytes(expected)
    assert np.array_equal(expected, sf(plaintext=data))
    master_key = np.random.randint(0, 255, (16,), dtype='uint8')
    expected_key = aes.key_schedule(master_key)[0]
    assert np.array_equal(expected_key, sf.compute_expected_key(key=master_key))
    assert isinstance(str(sf), str)


def test_aes_encrypt_first_sub_bytes_with_alternative_args():
    sf = aes.selection_functions.encrypt.FirstSubBytes(
        plaintext_tag='foo',
        words=slice(2, 8),
        guesses=_alt_guesses,
        key_tag='thekey'
    )
    assert sf.guesses.tolist() == _alt_guesses.tolist()
    assert sf.words == slice(2, 8, None)
    assert sf.target_tag == 'foo'
    assert sf.key_tag == 'thekey'
    assert isinstance(sf, selection_functions.SelectionFunction)
    data = np.random.randint(0, 255, (10, 16), dtype='uint8')
    expected = np.empty((10, len(_alt_guesses), 16), dtype='uint8')
    for i, guess in enumerate(_alt_guesses):
        expected[:, i, :] = np.bitwise_xor(data, guess)
    expected = aes.sub_bytes(expected)
    assert np.array_equal(expected[:, :, slice(2, 8)], sf(foo=data))
    master_key = np.random.randint(0, 255, (16,), dtype='uint8')
    expected_key = aes.key_schedule(master_key)[0]
    assert np.array_equal(expected_key, sf.compute_expected_key(thekey=master_key))
    assert isinstance(str(sf), str)


def test_aes_encrypt_last_sub_bytes_with_default_arguments():
    sf = aes.selection_functions.encrypt.LastSubBytes()
    assert sf.guesses.tolist() == list(range(256))
    assert sf.words is Ellipsis
    assert sf.target_tag == 'ciphertext'
    assert sf.key_tag == 'key'
    assert isinstance(sf, selection_functions.SelectionFunction)
    data = np.random.randint(0, 255, (10, 16), dtype='uint8')
    expected = np.empty((10, 256, 16), dtype='uint8')
    for i in np.arange(256, dtype='uint8'):
        expected[:, i, :] = np.bitwise_xor(data, i)
    expected = aes.inv_sub_bytes(expected)
    assert np.array_equal(expected, sf(ciphertext=data))
    master_key = np.random.randint(0, 255, (16,), dtype='uint8')
    expected_key = aes.key_schedule(master_key)[-1]
    assert np.array_equal(expected_key, sf.compute_expected_key(key=master_key))
    assert isinstance(str(sf), str)


def test_aes_encrypt_last_sub_bytes_with_alternative_args():
    sf = aes.selection_functions.encrypt.LastSubBytes(
        ciphertext_tag='foo',
        words=slice(2, 8),
        guesses=_alt_guesses,
        key_tag='thekey'
    )
    assert sf.guesses.tolist() == _alt_guesses.tolist()
    assert sf.words == slice(2, 8, None)
    assert sf.target_tag == 'foo'
    assert sf.key_tag == 'thekey'
    assert isinstance(sf, selection_functions.SelectionFunction)
    data = np.random.randint(0, 255, (10, 16), dtype='uint8')
    expected = np.empty((10, len(_alt_guesses), 16), dtype='uint8')
    for i, guess in enumerate(_alt_guesses):
        expected[:, i, :] = np.bitwise_xor(data, guess)
    expected = aes.inv_sub_bytes(expected)
    assert np.array_equal(expected[:, :, slice(2, 8)], sf(foo=data))
    master_key = np.random.randint(0, 255, (16,), dtype='uint8')
    expected_key = aes.key_schedule(master_key)[-1]
    assert np.array_equal(expected_key, sf.compute_expected_key(thekey=master_key))
    assert isinstance(str(sf), str)


def test_aes_encrypt_delta_r_last_rounds_with_default_arguments():
    sf = aes.selection_functions.encrypt.DeltaRLastRounds()
    assert sf.guesses.tolist() == list(range(256))
    assert sf.words is Ellipsis
    assert sf.target_tag == 'ciphertext'
    assert sf.key_tag == 'key'
    assert isinstance(sf, selection_functions.SelectionFunction)
    data = np.random.randint(0, 255, (10, 16), dtype='uint8')
    expected = np.empty((10, 256, 16), dtype='uint8')
    for i in np.arange(256, dtype='uint8'):
        expected[:, i, :] = np.bitwise_xor(data, i)
    s = aes.inv_sub_bytes(state=expected)
    expected = np.bitwise_xor(aes.shift_rows(data), s.swapaxes(0, 1)).swapaxes(0, 1)
    assert np.array_equal(expected, sf(ciphertext=data))
    master_key = np.random.randint(0, 255, (16,), dtype='uint8')
    expected_key = aes.key_schedule(master_key)[-1]
    assert np.array_equal(expected_key, sf.compute_expected_key(key=master_key))
    assert isinstance(str(sf), str)


_alt_guesses = np.array([0, 2, 3, 6], dtype='uint8')


def test_aes_encrypt_delta_r_last_rounds_with_alternative_args():
    sf = aes.selection_functions.encrypt.DeltaRLastRounds(
        ciphertext_tag='foo',
        words=slice(2, 8),
        guesses=_alt_guesses,
        key_tag='thekey'
    )
    assert sf.guesses.tolist() == _alt_guesses.tolist()
    assert sf.words == slice(2, 8, None)
    assert sf.target_tag == 'foo'
    assert isinstance(sf, selection_functions.SelectionFunction)
    data = np.random.randint(0, 255, (10, 16), dtype='uint8')
    expected = np.empty((10, len(_alt_guesses), 16), dtype='uint8')
    for i, guess in enumerate(_alt_guesses):
        expected[:, i, :] = np.bitwise_xor(data, guess)
    s = aes.inv_sub_bytes(state=expected)
    expected = np.bitwise_xor(aes.shift_rows(data), s.swapaxes(0, 1)).swapaxes(0, 1)
    assert np.array_equal(expected[:, :, slice(2, 8)], sf(foo=data))
    master_key = np.random.randint(0, 255, (16,), dtype='uint8')
    expected_key = aes.key_schedule(master_key)[-1]
    assert np.array_equal(expected_key, sf.compute_expected_key(thekey=master_key))
    assert isinstance(str(sf), str)


def test_aes_decrypt_first_round_key_with_default_arguments():
    sf = aes.selection_functions.decrypt.FirstAddRoundKey()
    assert sf.guesses.tolist() == list(range(256))
    assert sf.words is Ellipsis
    assert sf.target_tag == 'ciphertext'
    assert sf.key_tag == 'key'
    assert isinstance(sf, selection_functions.SelectionFunction)
    data = np.random.randint(0, 255, (10, 16), dtype='uint8')
    expected = np.empty((10, 256, 16), dtype='uint8')
    for i in np.arange(256, dtype='uint8'):
        expected[:, i, :] = np.bitwise_xor(data, i)
    assert np.array_equal(expected, sf(ciphertext=data))
    master_key = np.random.randint(0, 255, (16,), dtype='uint8')
    expected_key = aes.key_schedule(master_key)[-1]
    assert np.array_equal(expected_key, sf.compute_expected_key(key=master_key))
    assert isinstance(str(sf), str)


def test_aes_decrypt_first_round_key_with_alternative_args():
    sf = aes.selection_functions.decrypt.FirstAddRoundKey(
        ciphertext_tag='cif',
        words=6,
        guesses=_alt_guesses,
        key_tag='thekey'
    )
    assert sf.guesses.tolist() == _alt_guesses.tolist()
    assert sf.words == 6
    assert sf.target_tag == 'cif'
    assert sf.key_tag == 'thekey'
    assert isinstance(sf, selection_functions.SelectionFunction)
    data = np.random.randint(0, 255, (10, 16), dtype='uint8')
    expected = np.empty((10, len(_alt_guesses), 16), dtype='uint8')
    for i, guess in enumerate(_alt_guesses):
        expected[:, i, :] = np.bitwise_xor(data, guess)
    assert np.array_equal(expected[:, :, 6], sf(cif=data))
    master_key = np.random.randint(0, 255, (16,), dtype='uint8')
    expected_key = aes.key_schedule(master_key)[-1]
    assert np.array_equal(expected_key, sf.compute_expected_key(thekey=master_key))
    assert isinstance(str(sf), str)


def test_aes_decrypt_last_round_key_with_default_arguments():
    sf = aes.selection_functions.decrypt.LastAddRoundKey()
    assert sf.guesses.tolist() == list(range(256))
    assert sf.words is Ellipsis
    assert sf.target_tag == 'plaintext'
    assert isinstance(sf, selection_functions.SelectionFunction)
    data = np.random.randint(0, 255, (10, 16), dtype='uint8')
    expected = np.empty((10, 256, 16), dtype='uint8')
    for i in np.arange(256, dtype='uint8'):
        expected[:, i, :] = np.bitwise_xor(data, i)
    assert np.array_equal(expected, sf(plaintext=data))
    master_key = np.random.randint(0, 255, (16,), dtype='uint8')
    expected_key = aes.key_schedule(master_key)[0]
    assert np.array_equal(expected_key, sf.compute_expected_key(key=master_key))
    assert sf.key_tag == 'key'
    assert isinstance(str(sf), str)


def test_aes_decrypt_last_round_key_with_alternative_args():
    sf = aes.selection_functions.decrypt.LastAddRoundKey(
        plaintext_tag='nop',
        words=6,
        guesses=_alt_guesses,
        key_tag='thekey'
    )
    assert sf.guesses.tolist() == _alt_guesses.tolist()
    assert sf.words == 6
    assert sf.target_tag == 'nop'
    assert isinstance(sf, selection_functions.SelectionFunction)
    data = np.random.randint(0, 255, (10, 16), dtype='uint8')
    expected = np.empty((10, len(_alt_guesses), 16), dtype='uint8')
    for i, guess in enumerate(_alt_guesses):
        expected[:, i, :] = np.bitwise_xor(data, guess)
    assert np.array_equal(expected[:, :, 6], sf(nop=data))
    master_key = np.random.randint(0, 255, (16,), dtype='uint8')
    expected_key = aes.key_schedule(master_key)[0]
    assert np.array_equal(expected_key, sf.compute_expected_key(thekey=master_key))
    assert sf.key_tag == 'thekey'
    assert isinstance(str(sf), str)


def test_aes_decrypt_first_sub_bytes_with_default_arguments():
    sf = aes.selection_functions.decrypt.FirstSubBytes()
    assert sf.guesses.tolist() == list(range(256))
    assert sf.words is Ellipsis
    assert sf.target_tag == 'ciphertext'
    assert isinstance(sf, selection_functions.SelectionFunction)
    data = np.random.randint(0, 255, (10, 16), dtype='uint8')
    expected = np.empty((10, 256, 16), dtype='uint8')
    for i in np.arange(256, dtype='uint8'):
        expected[:, i, :] = np.bitwise_xor(data, i)
    expected = aes.inv_sub_bytes(expected)
    assert np.array_equal(expected, sf(ciphertext=data))
    master_key = np.random.randint(0, 255, (16,), dtype='uint8')
    expected_key = aes.key_schedule(master_key)[-1]
    assert np.array_equal(expected_key, sf.compute_expected_key(key=master_key))
    assert sf.key_tag == 'key'
    assert isinstance(str(sf), str)


def test_aes_decrypt_first_sub_bytes_with_alternative_args():
    sf = aes.selection_functions.decrypt.FirstSubBytes(
        ciphertext_tag='foo',
        words=slice(2, 8),
        guesses=_alt_guesses,
        key_tag='thekey'
    )
    assert sf.guesses.tolist() == _alt_guesses.tolist()
    assert sf.words == slice(2, 8, None)
    assert sf.target_tag == 'foo'
    assert isinstance(sf, selection_functions.SelectionFunction)
    data = np.random.randint(0, 255, (10, 16), dtype='uint8')
    expected = np.empty((10, len(_alt_guesses), 16), dtype='uint8')
    for i, guess in enumerate(_alt_guesses):
        expected[:, i, :] = np.bitwise_xor(data, guess)
    expected = aes.inv_sub_bytes(expected)
    assert np.array_equal(expected[:, :, slice(2, 8)], sf(foo=data))
    master_key = np.random.randint(0, 255, (16,), dtype='uint8')
    expected_key = aes.key_schedule(master_key)[-1]
    assert np.array_equal(expected_key, sf.compute_expected_key(thekey=master_key))
    assert sf.key_tag == 'thekey'
    assert isinstance(str(sf), str)


def test_aes_decrypt_last_sub_bytes_with_default_arguments():
    sf = aes.selection_functions.decrypt.LastSubBytes()
    assert sf.guesses.tolist() == list(range(256))
    assert sf.words is Ellipsis
    assert sf.target_tag == 'plaintext'
    assert isinstance(sf, selection_functions.SelectionFunction)
    data = np.random.randint(0, 255, (10, 16), dtype='uint8')
    expected = np.empty((10, 256, 16), dtype='uint8')
    for i in np.arange(256, dtype='uint8'):
        expected[:, i, :] = np.bitwise_xor(data, i)
    expected = aes.sub_bytes(expected)
    assert np.array_equal(expected, sf(plaintext=data))
    master_key = np.random.randint(0, 255, (16,), dtype='uint8')
    expected_key = aes.key_schedule(master_key)[0]
    assert np.array_equal(expected_key, sf.compute_expected_key(key=master_key))
    assert sf.key_tag == 'key'
    assert isinstance(str(sf), str)


def test_aes_decrypt_last_sub_bytes_with_alternative_args():
    sf = aes.selection_functions.decrypt.LastSubBytes(
        plaintext_tag='foo',
        words=slice(2, 8),
        guesses=_alt_guesses,
        key_tag='thekey'
    )
    assert sf.guesses.tolist() == _alt_guesses.tolist()
    assert sf.words == slice(2, 8, None)
    assert sf.target_tag == 'foo'
    assert isinstance(sf, selection_functions.SelectionFunction)
    data = np.random.randint(0, 255, (10, 16), dtype='uint8')
    expected = np.empty((10, len(_alt_guesses), 16), dtype='uint8')
    for i, guess in enumerate(_alt_guesses):
        expected[:, i, :] = np.bitwise_xor(data, guess)
    expected = aes.sub_bytes(expected)
    assert np.array_equal(expected[:, :, slice(2, 8)], sf(foo=data))
    master_key = np.random.randint(0, 255, (16,), dtype='uint8')
    expected_key = aes.key_schedule(master_key)[0]
    assert np.array_equal(expected_key, sf.compute_expected_key(thekey=master_key))
    assert sf.key_tag == 'thekey'
    assert isinstance(str(sf), str)


def test_aes_decrypt_delta_r_first_rounds_with_default_arguments():
    sf = aes.selection_functions.decrypt.DeltaRFirstRounds()
    assert sf.guesses.tolist() == list(range(256))
    assert sf.words is Ellipsis
    assert sf.target_tag == 'ciphertext'
    assert isinstance(sf, selection_functions.SelectionFunction)
    data = np.random.randint(0, 255, (10, 16), dtype='uint8')
    expected = np.empty((10, 256, 16), dtype='uint8')
    for i in np.arange(256, dtype='uint8'):
        expected[:, i, :] = np.bitwise_xor(data, i)
    s = aes.inv_sub_bytes(state=expected)
    expected = np.bitwise_xor(aes.shift_rows(data), s.swapaxes(0, 1)).swapaxes(0, 1)
    assert np.array_equal(expected, sf(ciphertext=data))
    master_key = np.random.randint(0, 255, (16,), dtype='uint8')
    expected_key = aes.key_schedule(master_key)[-1]
    assert np.array_equal(expected_key, sf.compute_expected_key(key=master_key))
    assert sf.key_tag == 'key'
    assert isinstance(str(sf), str)


def test_aes_decrypt_delta_r_first_rounds_with_alternative_args():
    sf = aes.selection_functions.decrypt.DeltaRFirstRounds(
        ciphertext_tag='foo',
        words=slice(2, 8),
        guesses=_alt_guesses,
        key_tag='thekey'
    )
    assert sf.guesses.tolist() == _alt_guesses.tolist()
    assert sf.words == slice(2, 8, None)
    assert sf.target_tag == 'foo'
    assert isinstance(sf, selection_functions.SelectionFunction)
    data = np.random.randint(0, 255, (10, 16), dtype='uint8')
    expected = np.empty((10, len(_alt_guesses), 16), dtype='uint8')
    for i, guess in enumerate(_alt_guesses):
        expected[:, i, :] = np.bitwise_xor(data, guess)
    s = aes.inv_sub_bytes(state=expected)
    expected = np.bitwise_xor(aes.shift_rows(data), s.swapaxes(0, 1)).swapaxes(0, 1)
    assert np.array_equal(expected[:, :, slice(2, 8)], sf(foo=data))
    master_key = np.random.randint(0, 255, (16,), dtype='uint8')
    expected_key = aes.key_schedule(master_key)[-1]
    assert np.array_equal(expected_key, sf.compute_expected_key(thekey=master_key))
    assert sf.key_tag == 'thekey'
    assert isinstance(str(sf), str)


@pytest.mark.parametrize('sf', [scared.aes.selection_functions.encrypt.DeltaRLastRounds,
                                scared.aes.selection_functions.encrypt.FirstAddRoundKey,
                                scared.aes.selection_functions.encrypt.FirstSubBytes,
                                scared.aes.selection_functions.encrypt.LastAddRoundKey,
                                scared.aes.selection_functions.encrypt.LastSubBytes])
# see https://gitlab.com/eshard/scared/-/issues/81
def test_aes_encrypt_output_dtype_even_if_input_not_uint8(sf):
    data = np.random.randint(0, 256, (5, 16), dtype='int64')
    output = sf()(plaintext=data, ciphertext=data)
    assert output.dtype == np.uint8


@pytest.mark.parametrize('sf', [scared.aes.selection_functions.decrypt.DeltaRFirstRounds,
                                scared.aes.selection_functions.decrypt.FirstAddRoundKey,
                                scared.aes.selection_functions.decrypt.FirstSubBytes,
                                scared.aes.selection_functions.decrypt.LastAddRoundKey,
                                scared.aes.selection_functions.decrypt.LastSubBytes])
# see https://gitlab.com/eshard/scared/-/issues/81
def test_aes_decrypt_output_dtype_even_if_input_not_uint8(sf):
    data = np.random.randint(0, 256, (5, 16), dtype='int64')
    output = sf()(plaintext=data, ciphertext=data)
    assert output.dtype == np.uint8


# see https://gitlab.com/eshard/scared/-/issues/82
def test_aes_encrypt_docstring_and_name():
    klass = aes.selection_functions.encrypt.FirstAddRoundKey
    assert klass.__name__ == 'FirstAddRoundKey'
    assert 'encrypt' in klass.__doc__.split('\n')[0] and 'first' in klass.__doc__.split('\n')[0]
    klass = aes.selection_functions.encrypt.LastAddRoundKey
    assert klass.__name__ == 'LastAddRoundKey'
    assert 'encrypt' in klass.__doc__.split('\n')[0] and 'last' in klass.__doc__.split('\n')[0]
    klass = aes.selection_functions.encrypt.FirstSubBytes
    assert klass.__name__ == 'FirstSubBytes'
    assert 'encrypt' in klass.__doc__.split('\n')[0] and 'first' in klass.__doc__.split('\n')[0]
    klass = aes.selection_functions.encrypt.LastSubBytes
    assert klass.__name__ == 'LastSubBytes'
    assert 'encrypt' in klass.__doc__.split('\n')[0] and 'last' in klass.__doc__.split('\n')[0]
    klass = aes.selection_functions.encrypt.DeltaRLastRounds
    assert klass.__name__ == 'DeltaRLastRounds'
    assert 'encrypt' in klass.__doc__.split('\n')[0] and 'last' in klass.__doc__.split('\n')[0]


# see https://gitlab.com/eshard/scared/-/issues/82
def test_aes_decrypt_docstring_and_name():
    klass = aes.selection_functions.decrypt.FirstAddRoundKey
    assert klass.__name__ == 'FirstAddRoundKey'
    assert 'decrypt' in klass.__doc__.split('\n')[0] and 'first' in klass.__doc__.split('\n')[0]
    klass = aes.selection_functions.decrypt.LastAddRoundKey
    assert klass.__name__ == 'LastAddRoundKey'
    assert 'decrypt' in klass.__doc__.split('\n')[0] and 'last' in klass.__doc__.split('\n')[0]
    klass = aes.selection_functions.decrypt.FirstSubBytes
    assert klass.__name__ == 'FirstSubBytes'
    assert 'decrypt' in klass.__doc__.split('\n')[0] and 'first' in klass.__doc__.split('\n')[0]
    klass = aes.selection_functions.decrypt.LastSubBytes
    assert klass.__name__ == 'LastSubBytes'
    assert 'decrypt' in klass.__doc__.split('\n')[0] and 'last' in klass.__doc__.split('\n')[0]
    klass = aes.selection_functions.decrypt.DeltaRFirstRounds
    assert klass.__name__ == 'DeltaRFirstRounds'
    assert 'decrypt' in klass.__doc__.split('\n')[0] and 'first' in klass.__doc__.split('\n')[0]
