from .context import scared  # noqa: F401
from scared import selection_functions, aes
import numpy as np


def test_aes_encrypt_first_round_key_with_default_arguments():
    sf = scared.selection_functions.aes.encrypt.first_add_round_key()
    assert sf.guesses.tolist() == list(range(256))
    assert sf.words is Ellipsis
    assert sf.target_tag == 'plaintext'
    assert isinstance(sf, selection_functions.SelectionFunction)
    data = np.random.randint(0, 255, (10, 16), dtype='uint8')
    expected = np.empty((10, 256, 16), dtype='uint8')
    for i in np.arange(256, dtype='uint8'):
        expected[:, i, :] = np.bitwise_xor(data, i)
    assert np.array_equal(expected, sf(plaintext=data))


def test_aes_encrypt_first_round_key_with_alternative_args():
    sf = scared.selection_functions.aes.encrypt.first_add_round_key(
        plaintext_tag='plain',
        words=6,
        guesses=np.arange(16, dtype='uint8')
    )
    assert sf.guesses.tolist() == list(range(16))
    assert sf.words == 6
    assert sf.target_tag == 'plain'
    assert isinstance(sf, selection_functions.SelectionFunction)
    data = np.random.randint(0, 255, (10, 16), dtype='uint8')
    expected = np.empty((10, 16, 16), dtype='uint8')
    for i in np.arange(16, dtype='uint8'):
        expected[:, i, :] = np.bitwise_xor(data, i)
    assert np.array_equal(expected[:, :, 6], sf(plain=data))


def test_aes_encrypt_last_round_key_with_default_arguments():
    sf = scared.selection_functions.aes.encrypt.last_add_round_key()
    assert sf.guesses.tolist() == list(range(256))
    assert sf.words is Ellipsis
    assert sf.target_tag == 'ciphertext'
    assert isinstance(sf, selection_functions.SelectionFunction)
    data = np.random.randint(0, 255, (10, 16), dtype='uint8')
    expected = np.empty((10, 256, 16), dtype='uint8')
    for i in np.arange(256, dtype='uint8'):
        expected[:, i, :] = np.bitwise_xor(data, i)
    assert np.array_equal(expected, sf(ciphertext=data))


def test_aes_encrypt_last_round_key_with_alternative_args():
    sf = scared.selection_functions.aes.encrypt.last_add_round_key(
        ciphertext_tag='nop',
        words=6,
        guesses=np.arange(16, dtype='uint8')
    )
    assert sf.guesses.tolist() == list(range(16))
    assert sf.words == 6
    assert sf.target_tag == 'nop'
    assert isinstance(sf, selection_functions.SelectionFunction)
    data = np.random.randint(0, 255, (10, 16), dtype='uint8')
    expected = np.empty((10, 16, 16), dtype='uint8')
    for i in np.arange(16, dtype='uint8'):
        expected[:, i, :] = np.bitwise_xor(data, i)
    assert np.array_equal(expected[:, :, 6], sf(nop=data))


def test_aes_encrypt_first_sub_bytes_with_default_arguments():
    sf = scared.selection_functions.aes.encrypt.first_sub_bytes()
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


def test_aes_encrypt_first_sub_bytes_with_alternative_args():
    sf = scared.selection_functions.aes.encrypt.first_sub_bytes(
        plaintext_tag='foo',
        words=slice(2, 8),
        guesses=np.arange(16, dtype='uint8')
    )
    assert sf.guesses.tolist() == list(range(16))
    assert sf.words == slice(2, 8, None)
    assert sf.target_tag == 'foo'
    assert isinstance(sf, selection_functions.SelectionFunction)
    data = np.random.randint(0, 255, (10, 16), dtype='uint8')
    expected = np.empty((10, 16, 16), dtype='uint8')
    for i in np.arange(16, dtype='uint8'):
        expected[:, i, :] = np.bitwise_xor(data, i)
    expected = aes.sub_bytes(expected)
    assert np.array_equal(expected[:, :, slice(2, 8)], sf(foo=data))


def test_aes_encrypt_last_sub_bytes_with_default_arguments():
    sf = scared.selection_functions.aes.encrypt.last_sub_bytes()
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


def test_aes_encrypt_last_sub_bytes_with_alternative_args():
    sf = scared.selection_functions.aes.encrypt.last_sub_bytes(
        ciphertext_tag='foo',
        words=slice(2, 8),
        guesses=np.arange(16, dtype='uint8')
    )
    assert sf.guesses.tolist() == list(range(16))
    assert sf.words == slice(2, 8, None)
    assert sf.target_tag == 'foo'
    assert isinstance(sf, selection_functions.SelectionFunction)
    data = np.random.randint(0, 255, (10, 16), dtype='uint8')
    expected = np.empty((10, 16, 16), dtype='uint8')
    for i in np.arange(16, dtype='uint8'):
        expected[:, i, :] = np.bitwise_xor(data, i)
    expected = aes.inv_sub_bytes(expected)
    assert np.array_equal(expected[:, :, slice(2, 8)], sf(foo=data))


def test_aes_encrypt_delta_r_last_rounds_with_default_arguments():
    sf = scared.selection_functions.aes.encrypt.delta_r_last_rounds()
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


def test_aes_encrypt_delta_r_last_rounds_with_alternative_args():
    sf = scared.selection_functions.aes.encrypt.delta_r_last_rounds(
        ciphertext_tag='foo',
        words=slice(2, 8),
        guesses=np.arange(16, dtype='uint8')
    )
    assert sf.guesses.tolist() == list(range(16))
    assert sf.words == slice(2, 8, None)
    assert sf.target_tag == 'foo'
    assert isinstance(sf, selection_functions.SelectionFunction)
    data = np.random.randint(0, 255, (10, 16), dtype='uint8')
    expected = np.empty((10, 16, 16), dtype='uint8')
    for i in np.arange(16, dtype='uint8'):
        expected[:, i, :] = np.bitwise_xor(data, i)
    s = aes.inv_sub_bytes(state=expected)
    expected = np.bitwise_xor(aes.shift_rows(data), s.swapaxes(0, 1)).swapaxes(0, 1)
    assert np.array_equal(expected[:, :, slice(2, 8)], sf(foo=data))


def test_aes_decrypt_first_round_key_with_default_arguments():
    sf = scared.selection_functions.aes.decrypt.first_add_round_key()
    assert sf.guesses.tolist() == list(range(256))
    assert sf.words is Ellipsis
    assert sf.target_tag == 'ciphertext'
    assert isinstance(sf, selection_functions.SelectionFunction)
    data = np.random.randint(0, 255, (10, 16), dtype='uint8')
    expected = np.empty((10, 256, 16), dtype='uint8')
    for i in np.arange(256, dtype='uint8'):
        expected[:, i, :] = np.bitwise_xor(data, i)
    assert np.array_equal(expected, sf(ciphertext=data))


def test_aes_decrypt_first_round_key_with_alternative_args():
    sf = scared.selection_functions.aes.decrypt.first_add_round_key(
        ciphertext_tag='cif',
        words=6,
        guesses=np.arange(16, dtype='uint8')
    )
    assert sf.guesses.tolist() == list(range(16))
    assert sf.words == 6
    assert sf.target_tag == 'cif'
    assert isinstance(sf, selection_functions.SelectionFunction)
    data = np.random.randint(0, 255, (10, 16), dtype='uint8')
    expected = np.empty((10, 16, 16), dtype='uint8')
    for i in np.arange(16, dtype='uint8'):
        expected[:, i, :] = np.bitwise_xor(data, i)
    assert np.array_equal(expected[:, :, 6], sf(cif=data))


def test_aes_decrypt_last_round_key_with_default_arguments():
    sf = scared.selection_functions.aes.decrypt.last_add_round_key()
    assert sf.guesses.tolist() == list(range(256))
    assert sf.words is Ellipsis
    assert sf.target_tag == 'plaintext'
    assert isinstance(sf, selection_functions.SelectionFunction)
    data = np.random.randint(0, 255, (10, 16), dtype='uint8')
    expected = np.empty((10, 256, 16), dtype='uint8')
    for i in np.arange(256, dtype='uint8'):
        expected[:, i, :] = np.bitwise_xor(data, i)
    assert np.array_equal(expected, sf(plaintext=data))


def test_aes_decrypt_last_round_key_with_alternative_args():
    sf = scared.selection_functions.aes.decrypt.last_add_round_key(
        plaintext_tag='nop',
        words=6,
        guesses=np.arange(16, dtype='uint8')
    )
    assert sf.guesses.tolist() == list(range(16))
    assert sf.words == 6
    assert sf.target_tag == 'nop'
    assert isinstance(sf, selection_functions.SelectionFunction)
    data = np.random.randint(0, 255, (10, 16), dtype='uint8')
    expected = np.empty((10, 16, 16), dtype='uint8')
    for i in np.arange(16, dtype='uint8'):
        expected[:, i, :] = np.bitwise_xor(data, i)
    assert np.array_equal(expected[:, :, 6], sf(nop=data))


def test_aes_decrypt_first_sub_bytes_with_default_arguments():
    sf = scared.selection_functions.aes.decrypt.first_sub_bytes()
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


def test_aes_decrypt_first_sub_bytes_with_alternative_args():
    sf = scared.selection_functions.aes.decrypt.first_sub_bytes(
        ciphertext_tag='foo',
        words=slice(2, 8),
        guesses=np.arange(16, dtype='uint8')
    )
    assert sf.guesses.tolist() == list(range(16))
    assert sf.words == slice(2, 8, None)
    assert sf.target_tag == 'foo'
    assert isinstance(sf, selection_functions.SelectionFunction)
    data = np.random.randint(0, 255, (10, 16), dtype='uint8')
    expected = np.empty((10, 16, 16), dtype='uint8')
    for i in np.arange(16, dtype='uint8'):
        expected[:, i, :] = np.bitwise_xor(data, i)
    expected = aes.inv_sub_bytes(expected)
    assert np.array_equal(expected[:, :, slice(2, 8)], sf(foo=data))


def test_aes_decrypt_last_sub_bytes_with_default_arguments():
    sf = scared.selection_functions.aes.decrypt.last_sub_bytes()
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


def test_aes_decrypt_last_sub_bytes_with_alternative_args():
    sf = scared.selection_functions.aes.decrypt.last_sub_bytes(
        plaintext_tag='foo',
        words=slice(2, 8),
        guesses=np.arange(16, dtype='uint8')
    )
    assert sf.guesses.tolist() == list(range(16))
    assert sf.words == slice(2, 8, None)
    assert sf.target_tag == 'foo'
    assert isinstance(sf, selection_functions.SelectionFunction)
    data = np.random.randint(0, 255, (10, 16), dtype='uint8')
    expected = np.empty((10, 16, 16), dtype='uint8')
    for i in np.arange(16, dtype='uint8'):
        expected[:, i, :] = np.bitwise_xor(data, i)
    expected = aes.sub_bytes(expected)
    assert np.array_equal(expected[:, :, slice(2, 8)], sf(foo=data))


def test_aes_decrypt_delta_r_first_rounds_with_default_arguments():
    sf = scared.selection_functions.aes.decrypt.delta_r_first_rounds()
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


def test_aes_decrypt_delta_r_first_rounds_with_alternative_args():
    sf = scared.selection_functions.aes.decrypt.delta_r_first_rounds(
        ciphertext_tag='foo',
        words=slice(2, 8),
        guesses=np.arange(16, dtype='uint8')
    )
    assert sf.guesses.tolist() == list(range(16))
    assert sf.words == slice(2, 8, None)
    assert sf.target_tag == 'foo'
    assert isinstance(sf, selection_functions.SelectionFunction)
    data = np.random.randint(0, 255, (10, 16), dtype='uint8')
    expected = np.empty((10, 16, 16), dtype='uint8')
    for i in np.arange(16, dtype='uint8'):
        expected[:, i, :] = np.bitwise_xor(data, i)
    s = aes.inv_sub_bytes(state=expected)
    expected = np.bitwise_xor(aes.shift_rows(data), s.swapaxes(0, 1)).swapaxes(0, 1)
    assert np.array_equal(expected[:, :, slice(2, 8)], sf(foo=data))
