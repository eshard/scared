from .context import scared  # noqa: F401
from scared import des, selection_functions
import numpy as np


def test_des_encrypt_first_round_key_with_default_arguments():
    sf = des.selection_functions.encrypt.FirstAddRoundKey()
    assert sf.guesses.tolist() == list(range(64))
    assert sf.words is Ellipsis
    assert sf.target_tag == 'plaintext'
    assert sf.key_tag == 'key'
    assert isinstance(sf, selection_functions.SelectionFunction)
    data = np.random.randint(0, 255, (10, 8), dtype='uint8')
    expected = np.empty((10, 64, 8), dtype='uint8')
    for i in np.arange(64, dtype='uint8'):
        expanded_guess = np.bitwise_xor(np.zeros((128), dtype=np.uint8), i)
        expected[:, i, :] = des.encrypt(data, expanded_guess, at_round=0, after_step=des.Steps.ADD_ROUND_KEY)
    assert np.array_equal(expected, sf(plaintext=data))
    master_key = np.random.randint(0, 255, (8,), dtype='uint8')
    expected_key = des.key_schedule(master_key)[0]
    assert np.array_equal(expected_key, sf.compute_expected_key(key=master_key))
    assert isinstance(str(sf), str)


_alt_guesses = np.array([2, 4, 5, 9], dtype='uint8')


def test_des_encrypt_first_round_key_with_alternative_args():
    sf = des.selection_functions.encrypt.FirstAddRoundKey(
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
    data = np.random.randint(0, 255, (10, 8), dtype='uint8')
    expected = np.empty((10, len(_alt_guesses), 8), dtype='uint8')
    for i, guess in enumerate(_alt_guesses):
        expanded_guess = np.bitwise_xor(np.zeros((128), dtype=np.uint8), guess)
        expected[:, i, :] = des.encrypt(data, expanded_guess, at_round=0, after_step=des.Steps.ADD_ROUND_KEY)
    assert np.array_equal(expected[:, :, 6], sf(plain=data))
    master_key = np.random.randint(0, 255, (8,), dtype='uint8')
    expected_key = des.key_schedule(master_key)[0]
    assert np.array_equal(expected_key, sf.compute_expected_key(thekey=master_key))
    assert isinstance(str(sf), str)


def test_des_encrypt_last_round_key_with_default_arguments():
    sf = des.selection_functions.encrypt.LastAddRoundKey()
    assert sf.guesses.tolist() == list(range(64))
    assert sf.words is Ellipsis
    assert sf.target_tag == 'ciphertext'
    assert sf.key_tag == 'key'
    assert isinstance(sf, selection_functions.SelectionFunction)
    data = np.random.randint(0, 255, (10, 8), dtype='uint8')
    expected = np.empty((10, 64, 8), dtype='uint8')
    for i in np.arange(64, dtype='uint8'):
        expanded_guess = np.bitwise_xor(np.zeros((128), dtype=np.uint8), i)
        expected[:, i, :] = des.encrypt(data, expanded_guess, at_round=0, after_step=des.Steps.ADD_ROUND_KEY)
    assert np.array_equal(expected, sf(ciphertext=data))
    master_key = np.random.randint(0, 255, (8,), dtype='uint8')
    expected_key = des.key_schedule(master_key)[-1]
    assert np.array_equal(expected_key, sf.compute_expected_key(key=master_key))
    assert isinstance(str(sf), str)


def test_des_encrypt_last_round_key_with_alternative_args():
    sf = des.selection_functions.encrypt.LastAddRoundKey(
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
    data = np.random.randint(0, 255, (10, 8), dtype='uint8')
    expected = np.empty((10, len(_alt_guesses), 8), dtype='uint8')
    for i, guess in enumerate(_alt_guesses):
        expanded_guess = np.bitwise_xor(np.zeros((128), dtype=np.uint8), guess)
        expected[:, i, :] = des.encrypt(data, expanded_guess, at_round=0, after_step=des.Steps.ADD_ROUND_KEY)
    assert np.array_equal(expected[:, :, 6], sf(nop=data))
    master_key = np.random.randint(0, 255, (8,), dtype='uint8')
    expected_key = des.key_schedule(master_key)[-1]
    assert np.array_equal(expected_key, sf.compute_expected_key(thekey=master_key))
    assert isinstance(str(sf), str)


def test_des_encrypt_first_sboxes_with_default_arguments():
    sf = des.selection_functions.encrypt.FirstSboxes()
    assert sf.guesses.tolist() == list(range(64))
    assert sf.words is Ellipsis
    assert sf.target_tag == 'plaintext'
    assert sf.key_tag == 'key'
    assert isinstance(sf, selection_functions.SelectionFunction)
    data = np.random.randint(0, 255, (10, 8), dtype='uint8')
    expected = np.empty((10, 64, 8), dtype='uint8')
    for i in np.arange(64, dtype='uint8'):
        expanded_guess = np.bitwise_xor(np.zeros((128), dtype=np.uint8), i)
        expected[:, i, :] = des.encrypt(data, expanded_guess, at_round=0, after_step=des.Steps.SBOXES)
    assert np.array_equal(expected, sf(plaintext=data))
    master_key = np.random.randint(0, 255, (8,), dtype='uint8')
    expected_key = des.key_schedule(master_key)[0]
    assert np.array_equal(expected_key, sf.compute_expected_key(key=master_key))
    assert isinstance(str(sf), str)


def test_des_encrypt_first_sboxes_with_alternative_args():
    sf = des.selection_functions.encrypt.FirstSboxes(
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
    data = np.random.randint(0, 255, (10, 8), dtype='uint8')
    expected = np.empty((10, len(_alt_guesses), 8), dtype='uint8')
    for i, guess in enumerate(_alt_guesses):
        expanded_guess = np.bitwise_xor(np.zeros((128), dtype=np.uint8), guess)
        expected[:, i, :] = des.encrypt(data, expanded_guess, at_round=0, after_step=des.Steps.SBOXES)
    assert np.array_equal(expected[:, :, slice(2, 8)], sf(foo=data))
    master_key = np.random.randint(0, 255, (8,), dtype='uint8')
    expected_key = des.key_schedule(master_key)[0]
    assert np.array_equal(expected_key, sf.compute_expected_key(thekey=master_key))
    assert isinstance(str(sf), str)


def test_des_encrypt_last_sboxes_with_default_arguments():
    sf = des.selection_functions.encrypt.LastSboxes()
    assert sf.guesses.tolist() == list(range(64))
    assert sf.words is Ellipsis
    assert sf.target_tag == 'ciphertext'
    assert sf.key_tag == 'key'
    assert isinstance(sf, selection_functions.SelectionFunction)
    data = np.random.randint(0, 255, (10, 8), dtype='uint8')
    expected = np.empty((10, 64, 8), dtype='uint8')
    for i in np.arange(64, dtype='uint8'):
        expanded_guess = np.bitwise_xor(np.zeros((128), dtype=np.uint8), i)
        expected[:, i, :] = des.encrypt(data, expanded_guess, at_round=0, after_step=des.Steps.SBOXES)
    assert np.array_equal(expected, sf(ciphertext=data))
    master_key = np.random.randint(0, 255, (8,), dtype='uint8')
    expected_key = des.key_schedule(master_key)[-1]
    assert np.array_equal(expected_key, sf.compute_expected_key(key=master_key))
    assert isinstance(str(sf), str)


def test_des_encrypt_last_sboxes_with_alternative_args():
    sf = des.selection_functions.encrypt.LastSboxes(
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
    data = np.random.randint(0, 255, (10, 8), dtype='uint8')
    expected = np.empty((10, len(_alt_guesses), 8), dtype='uint8')
    for i, guess in enumerate(_alt_guesses):
        expanded_guess = np.bitwise_xor(np.zeros((128), dtype=np.uint8), guess)
        expected[:, i, :] = des.encrypt(data, expanded_guess, at_round=0, after_step=des.Steps.SBOXES)
    assert np.array_equal(expected[:, :, slice(2, 8)], sf(foo=data))
    master_key = np.random.randint(0, 255, (8,), dtype='uint8')
    expected_key = des.key_schedule(master_key)[-1]
    assert np.array_equal(expected_key, sf.compute_expected_key(thekey=master_key))
    assert isinstance(str(sf), str)


def test_des_encrypt_feistelr_first_rounds_with_default_arguments():
    sf = des.selection_functions.encrypt.FeistelRFirstRounds()
    assert sf.guesses.tolist() == list(range(64))
    assert sf.words is Ellipsis
    assert sf.target_tag == 'plaintext'
    assert sf.key_tag == 'key'
    assert isinstance(sf, selection_functions.SelectionFunction)
    data = np.random.randint(0, 255, (10, 8), dtype='uint8')
    expected = np.empty((10, 64, 8), dtype='uint8')
    for i in np.arange(64, dtype='uint8'):
        expanded_guess = np.bitwise_xor(np.zeros((128), dtype=np.uint8), i)
        expected[:, i, :] = des.encrypt(data, expanded_guess, at_round=0, after_step=des.Steps.INV_PERMUTATION_P_RIGHT)
    assert np.array_equal(expected, sf(plaintext=data))
    master_key = np.random.randint(0, 255, (8,), dtype='uint8')
    expected_key = des.key_schedule(master_key)[0]
    assert np.array_equal(expected_key, sf.compute_expected_key(key=master_key))
    assert isinstance(str(sf), str)


def test_des_encrypt_feistelr_first_rounds_with_alternative_args():
    sf = des.selection_functions.encrypt.FeistelRFirstRounds(
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
    data = np.random.randint(0, 255, (10, 8), dtype='uint8')
    expected = np.empty((10, len(_alt_guesses), 8), dtype='uint8')
    for i, guess in enumerate(_alt_guesses):
        expanded_guess = np.bitwise_xor(np.zeros((128), dtype=np.uint8), guess)
        expected[:, i, :] = des.encrypt(data, expanded_guess, at_round=0, after_step=des.Steps.INV_PERMUTATION_P_RIGHT)
    assert np.array_equal(expected[:, :, slice(2, 8)], sf(foo=data))
    master_key = np.random.randint(0, 255, (8,), dtype='uint8')
    expected_key = des.key_schedule(master_key)[0]
    assert np.array_equal(expected_key, sf.compute_expected_key(thekey=master_key))
    assert isinstance(str(sf), str)


def test_des_encrypt_feistelr_last_rounds_with_default_arguments():
    sf = des.selection_functions.encrypt.FeistelRLastRounds()
    assert sf.guesses.tolist() == list(range(64))
    assert sf.words is Ellipsis
    assert sf.target_tag == 'ciphertext'
    assert sf.key_tag == 'key'
    assert isinstance(sf, selection_functions.SelectionFunction)
    data = np.random.randint(0, 255, (10, 8), dtype='uint8')
    expected = np.empty((10, 64, 8), dtype='uint8')
    for i in np.arange(64, dtype='uint8'):
        expanded_guess = np.bitwise_xor(np.zeros((128), dtype=np.uint8), i)
        expected[:, i, :] = des.encrypt(data, expanded_guess, at_round=0, after_step=des.Steps.INV_PERMUTATION_P_RIGHT)
    assert np.array_equal(expected, sf(ciphertext=data))
    master_key = np.random.randint(0, 255, (8,), dtype='uint8')
    expected_key = des.key_schedule(master_key)[-1]
    assert np.array_equal(expected_key, sf.compute_expected_key(key=master_key))
    assert isinstance(str(sf), str)


def test_des_encrypt_feistelr_last_rounds_with_alternative_args():
    sf = des.selection_functions.encrypt.FeistelRLastRounds(
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
    data = np.random.randint(0, 255, (10, 8), dtype='uint8')
    expected = np.empty((10, len(_alt_guesses), 8), dtype='uint8')
    for i, guess in enumerate(_alt_guesses):
        expanded_guess = np.bitwise_xor(np.zeros((128), dtype=np.uint8), guess)
        expected[:, i, :] = des.encrypt(data, expanded_guess, at_round=0, after_step=des.Steps.INV_PERMUTATION_P_RIGHT)
    assert np.array_equal(expected[:, :, slice(2, 8)], sf(foo=data))
    master_key = np.random.randint(0, 255, (8,), dtype='uint8')
    expected_key = des.key_schedule(master_key)[-1]
    assert np.array_equal(expected_key, sf.compute_expected_key(thekey=master_key))
    assert isinstance(str(sf), str)


def test_des_encrypt_delta_r_first_rounds_with_default_arguments():
    sf = des.selection_functions.encrypt.DeltaRFirstRounds()
    assert sf.guesses.tolist() == list(range(64))
    assert sf.words is Ellipsis
    assert sf.target_tag == 'plaintext'
    assert sf.key_tag == 'key'
    assert isinstance(sf, selection_functions.SelectionFunction)
    data = np.random.randint(0, 255, (10, 8), dtype='uint8')
    expected = np.empty((10, 64, 8), dtype='uint8')
    for i in np.arange(64, dtype='uint8'):
        expanded_guess = np.bitwise_xor(np.zeros((128), dtype=np.uint8), i)
        expected[:, i, :] = des.encrypt(data, expanded_guess, at_round=0, after_step=des.Steps.INV_PERMUTATION_P_DELTA_RIGHT)
    assert np.array_equal(expected, sf(plaintext=data))
    master_key = np.random.randint(0, 255, (8,), dtype='uint8')
    expected_key = des.key_schedule(master_key)[0]
    assert np.array_equal(expected_key, sf.compute_expected_key(key=master_key))
    assert isinstance(str(sf), str)


def test_des_encrypt_delta_r_first_rounds_with_alternative_args():
    sf = des.selection_functions.encrypt.DeltaRFirstRounds(
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
    data = np.random.randint(0, 255, (10, 8), dtype='uint8')
    expected = np.empty((10, len(_alt_guesses), 8), dtype='uint8')
    for i, guess in enumerate(_alt_guesses):
        expanded_guess = np.bitwise_xor(np.zeros((128), dtype=np.uint8), guess)
        expected[:, i, :] = des.encrypt(data, expanded_guess, at_round=0, after_step=des.Steps.INV_PERMUTATION_P_DELTA_RIGHT)
    assert np.array_equal(expected[:, :, slice(2, 8)], sf(foo=data))
    master_key = np.random.randint(0, 255, (8,), dtype='uint8')
    expected_key = des.key_schedule(master_key)[0]
    assert np.array_equal(expected_key, sf.compute_expected_key(thekey=master_key))
    assert isinstance(str(sf), str)


def test_des_encrypt_delta_r_last_rounds_with_default_arguments():
    sf = des.selection_functions.encrypt.DeltaRLastRounds()
    assert sf.guesses.tolist() == list(range(64))
    assert sf.words is Ellipsis
    assert sf.target_tag == 'ciphertext'
    assert sf.key_tag == 'key'
    assert isinstance(sf, selection_functions.SelectionFunction)
    data = np.random.randint(0, 255, (10, 8), dtype='uint8')
    expected = np.empty((10, 64, 8), dtype='uint8')
    for i in np.arange(64, dtype='uint8'):
        expanded_guess = np.bitwise_xor(np.zeros((128), dtype=np.uint8), i)
        expected[:, i, :] = des.encrypt(data, expanded_guess, at_round=0, after_step=des.Steps.INV_PERMUTATION_P_DELTA_RIGHT)
    assert np.array_equal(expected, sf(ciphertext=data))
    master_key = np.random.randint(0, 255, (8,), dtype='uint8')
    expected_key = des.key_schedule(master_key)[-1]
    assert np.array_equal(expected_key, sf.compute_expected_key(key=master_key))
    assert isinstance(str(sf), str)


def test_des_encrypt_delta_r_last_rounds_with_alternative_args():
    sf = des.selection_functions.encrypt.DeltaRLastRounds(
        ciphertext_tag='foo',
        words=slice(2, 8),
        guesses=_alt_guesses,
        key_tag='thekey'
    )
    assert sf.guesses.tolist() == _alt_guesses.tolist()
    assert sf.words == slice(2, 8, None)
    assert sf.target_tag == 'foo'
    assert isinstance(sf, selection_functions.SelectionFunction)
    data = np.random.randint(0, 255, (10, 8), dtype='uint8')
    expected = np.empty((10, len(_alt_guesses), 8), dtype='uint8')
    for i, guess in enumerate(_alt_guesses):
        expanded_guess = np.bitwise_xor(np.zeros((128), dtype=np.uint8), guess)
        expected[:, i, :] = des.encrypt(data, expanded_guess, at_round=0, after_step=des.Steps.INV_PERMUTATION_P_DELTA_RIGHT)
    assert np.array_equal(expected[:, :, slice(2, 8)], sf(foo=data))
    master_key = np.random.randint(0, 255, (8,), dtype='uint8')
    expected_key = des.key_schedule(master_key)[-1]
    assert np.array_equal(expected_key, sf.compute_expected_key(thekey=master_key))
    assert isinstance(str(sf), str)


def test_des_decrypt_first_round_key_with_default_arguments():
    sf = des.selection_functions.decrypt.FirstAddRoundKey()
    assert sf.guesses.tolist() == list(range(64))
    assert sf.words is Ellipsis
    assert sf.target_tag == 'ciphertext'
    assert sf.key_tag == 'key'
    assert isinstance(sf, selection_functions.SelectionFunction)
    data = np.random.randint(0, 255, (10, 8), dtype='uint8')
    expected = np.empty((10, 64, 8), dtype='uint8')
    for i in np.arange(64, dtype='uint8'):
        expanded_guess = np.bitwise_xor(np.zeros((128), dtype=np.uint8), i)
        expected[:, i, :] = des.encrypt(data, expanded_guess, at_round=0, after_step=des.Steps.ADD_ROUND_KEY)
    assert np.array_equal(expected, sf(ciphertext=data))
    master_key = np.random.randint(0, 255, (8,), dtype='uint8')
    expected_key = des.key_schedule(master_key)[-1]
    assert np.array_equal(expected_key, sf.compute_expected_key(key=master_key))
    assert isinstance(str(sf), str)


def test_des_decrypt_first_round_key_with_alternative_args():
    sf = des.selection_functions.decrypt.FirstAddRoundKey(
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
    data = np.random.randint(0, 255, (10, 8), dtype='uint8')
    expected = np.empty((10, len(_alt_guesses), 8), dtype='uint8')
    for i, guess in enumerate(_alt_guesses):
        expanded_guess = np.bitwise_xor(np.zeros((128), dtype=np.uint8), guess)
        expected[:, i, :] = des.encrypt(data, expanded_guess, at_round=0, after_step=des.Steps.ADD_ROUND_KEY)
    assert np.array_equal(expected[:, :, 6], sf(cif=data))
    master_key = np.random.randint(0, 255, (8,), dtype='uint8')
    expected_key = des.key_schedule(master_key)[-1]
    assert np.array_equal(expected_key, sf.compute_expected_key(thekey=master_key))
    assert isinstance(str(sf), str)


def test_des_decrypt_last_round_key_with_default_arguments():
    sf = des.selection_functions.decrypt.LastAddRoundKey()
    assert sf.guesses.tolist() == list(range(64))
    assert sf.words is Ellipsis
    assert sf.target_tag == 'plaintext'
    assert isinstance(sf, selection_functions.SelectionFunction)
    data = np.random.randint(0, 255, (10, 8), dtype='uint8')
    expected = np.empty((10, 64, 8), dtype='uint8')
    for i in np.arange(64, dtype='uint8'):
        expanded_guess = np.bitwise_xor(np.zeros((128), dtype=np.uint8), i)
        expected[:, i, :] = des.encrypt(data, expanded_guess, at_round=0, after_step=des.Steps.ADD_ROUND_KEY)
    assert np.array_equal(expected, sf(plaintext=data))
    master_key = np.random.randint(0, 255, (8,), dtype='uint8')
    expected_key = des.key_schedule(master_key)[0]
    assert np.array_equal(expected_key, sf.compute_expected_key(key=master_key))
    assert sf.key_tag == 'key'
    assert isinstance(str(sf), str)


def test_des_decrypt_last_round_key_with_alternative_args():
    sf = des.selection_functions.decrypt.LastAddRoundKey(
        plaintext_tag='nop',
        words=6,
        guesses=_alt_guesses,
        key_tag='thekey'
    )
    assert sf.guesses.tolist() == _alt_guesses.tolist()
    assert sf.words == 6
    assert sf.target_tag == 'nop'
    assert isinstance(sf, selection_functions.SelectionFunction)
    data = np.random.randint(0, 255, (10, 8), dtype='uint8')
    expected = np.empty((10, len(_alt_guesses), 8), dtype='uint8')
    for i, guess in enumerate(_alt_guesses):
        expanded_guess = np.bitwise_xor(np.zeros((128), dtype=np.uint8), guess)
        expected[:, i, :] = des.encrypt(data, expanded_guess, at_round=0, after_step=des.Steps.ADD_ROUND_KEY)
    assert np.array_equal(expected[:, :, 6], sf(nop=data))
    master_key = np.random.randint(0, 255, (8,), dtype='uint8')
    expected_key = des.key_schedule(master_key)[0]
    assert np.array_equal(expected_key, sf.compute_expected_key(thekey=master_key))
    assert sf.key_tag == 'thekey'
    assert isinstance(str(sf), str)


def test_des_decrypt_first_sboxes_with_default_arguments():
    sf = des.selection_functions.decrypt.FirstSboxes()
    assert sf.guesses.tolist() == list(range(64))
    assert sf.words is Ellipsis
    assert sf.target_tag == 'ciphertext'
    assert isinstance(sf, selection_functions.SelectionFunction)
    data = np.random.randint(0, 255, (10, 8), dtype='uint8')
    expected = np.empty((10, 64, 8), dtype='uint8')
    for i in np.arange(64, dtype='uint8'):
        expanded_guess = np.bitwise_xor(np.zeros((128), dtype=np.uint8), i)
        expected[:, i, :] = des.encrypt(data, expanded_guess, at_round=0, after_step=des.Steps.SBOXES)
    assert np.array_equal(expected, sf(ciphertext=data))
    master_key = np.random.randint(0, 255, (8,), dtype='uint8')
    expected_key = des.key_schedule(master_key)[-1]
    assert np.array_equal(expected_key, sf.compute_expected_key(key=master_key))
    assert sf.key_tag == 'key'
    assert isinstance(str(sf), str)


def test_des_decrypt_first_sboxes_with_alternative_args():
    sf = des.selection_functions.decrypt.FirstSboxes(
        ciphertext_tag='foo',
        words=slice(2, 8),
        guesses=_alt_guesses,
        key_tag='thekey'
    )
    assert sf.guesses.tolist() == _alt_guesses.tolist()
    assert sf.words == slice(2, 8, None)
    assert sf.target_tag == 'foo'
    assert isinstance(sf, selection_functions.SelectionFunction)
    data = np.random.randint(0, 255, (10, 8), dtype='uint8')
    expected = np.empty((10, len(_alt_guesses), 8), dtype='uint8')
    for i, guess in enumerate(_alt_guesses):
        expanded_guess = np.bitwise_xor(np.zeros((128), dtype=np.uint8), guess)
        expected[:, i, :] = des.encrypt(data, expanded_guess, at_round=0, after_step=des.Steps.SBOXES)
    assert np.array_equal(expected[:, :, slice(2, 8)], sf(foo=data))
    master_key = np.random.randint(0, 255, (8,), dtype='uint8')
    expected_key = des.key_schedule(master_key)[-1]
    assert np.array_equal(expected_key, sf.compute_expected_key(thekey=master_key))
    assert sf.key_tag == 'thekey'
    assert isinstance(str(sf), str)


def test_des_decrypt_last_sboxes_with_default_arguments():
    sf = des.selection_functions.decrypt.LastSboxes()
    assert sf.guesses.tolist() == list(range(64))
    assert sf.words is Ellipsis
    assert sf.target_tag == 'plaintext'
    assert isinstance(sf, selection_functions.SelectionFunction)
    data = np.random.randint(0, 255, (10, 8), dtype='uint8')
    expected = np.empty((10, 64, 8), dtype='uint8')
    for i in np.arange(64, dtype='uint8'):
        expanded_guess = np.bitwise_xor(np.zeros((128), dtype=np.uint8), i)
        expected[:, i, :] = des.encrypt(data, expanded_guess, at_round=0, after_step=des.Steps.SBOXES)
    assert np.array_equal(expected, sf(plaintext=data))
    master_key = np.random.randint(0, 255, (8,), dtype='uint8')
    expected_key = des.key_schedule(master_key)[0]
    assert np.array_equal(expected_key, sf.compute_expected_key(key=master_key))
    assert sf.key_tag == 'key'
    assert isinstance(str(sf), str)


def test_des_decrypt_last_sboxes_with_alternative_args():
    sf = des.selection_functions.decrypt.LastSboxes(
        plaintext_tag='foo',
        words=slice(2, 8),
        guesses=_alt_guesses,
        key_tag='thekey'
    )
    assert sf.guesses.tolist() == _alt_guesses.tolist()
    assert sf.words == slice(2, 8, None)
    assert sf.target_tag == 'foo'
    assert isinstance(sf, selection_functions.SelectionFunction)
    data = np.random.randint(0, 255, (10, 8), dtype='uint8')
    expected = np.empty((10, len(_alt_guesses), 8), dtype='uint8')
    for i, guess in enumerate(_alt_guesses):
        expanded_guess = np.bitwise_xor(np.zeros((128), dtype=np.uint8), guess)
        expected[:, i, :] = des.encrypt(data, expanded_guess, at_round=0, after_step=des.Steps.SBOXES)
    assert np.array_equal(expected[:, :, slice(2, 8)], sf(foo=data))
    master_key = np.random.randint(0, 255, (8,), dtype='uint8')
    expected_key = des.key_schedule(master_key)[0]
    assert np.array_equal(expected_key, sf.compute_expected_key(thekey=master_key))
    assert sf.key_tag == 'thekey'
    assert isinstance(str(sf), str)


def test_des_decrypt_first_feistelr_first_rounds_with_default_arguments():
    sf = des.selection_functions.decrypt.FeistelRFirstRounds()
    assert sf.guesses.tolist() == list(range(64))
    assert sf.words is Ellipsis
    assert sf.target_tag == 'ciphertext'
    assert isinstance(sf, selection_functions.SelectionFunction)
    data = np.random.randint(0, 255, (10, 8), dtype='uint8')
    expected = np.empty((10, 64, 8), dtype='uint8')
    for i in np.arange(64, dtype='uint8'):
        expanded_guess = np.bitwise_xor(np.zeros((128), dtype=np.uint8), i)
        expected[:, i, :] = des.encrypt(data, expanded_guess, at_round=0, after_step=des.Steps.INV_PERMUTATION_P_RIGHT)
    assert np.array_equal(expected, sf(ciphertext=data))
    master_key = np.random.randint(0, 255, (8,), dtype='uint8')
    expected_key = des.key_schedule(master_key)[-1]
    assert np.array_equal(expected_key, sf.compute_expected_key(key=master_key))
    assert sf.key_tag == 'key'
    assert isinstance(str(sf), str)


def test_des_decrypt_feistelr_first_rounds_with_alternative_args():
    sf = des.selection_functions.decrypt.FeistelRFirstRounds(
        ciphertext_tag='foo',
        words=slice(2, 8),
        guesses=_alt_guesses,
        key_tag='thekey'
    )
    assert sf.guesses.tolist() == _alt_guesses.tolist()
    assert sf.words == slice(2, 8, None)
    assert sf.target_tag == 'foo'
    assert isinstance(sf, selection_functions.SelectionFunction)
    data = np.random.randint(0, 255, (10, 8), dtype='uint8')
    expected = np.empty((10, len(_alt_guesses), 8), dtype='uint8')
    for i, guess in enumerate(_alt_guesses):
        expanded_guess = np.bitwise_xor(np.zeros((128), dtype=np.uint8), guess)
        expected[:, i, :] = des.encrypt(data, expanded_guess, at_round=0, after_step=des.Steps.INV_PERMUTATION_P_RIGHT)
    assert np.array_equal(expected[:, :, slice(2, 8)], sf(foo=data))
    master_key = np.random.randint(0, 255, (8,), dtype='uint8')
    expected_key = des.key_schedule(master_key)[-1]
    assert np.array_equal(expected_key, sf.compute_expected_key(thekey=master_key))
    assert sf.key_tag == 'thekey'
    assert isinstance(str(sf), str)


def test_des_decrypt_feistelr_last_rounds_with_default_arguments():
    sf = des.selection_functions.decrypt.FeistelRLastRounds()
    assert sf.guesses.tolist() == list(range(64))
    assert sf.words is Ellipsis
    assert sf.target_tag == 'plaintext'
    assert isinstance(sf, selection_functions.SelectionFunction)
    data = np.random.randint(0, 255, (10, 8), dtype='uint8')
    expected = np.empty((10, 64, 8), dtype='uint8')
    for i in np.arange(64, dtype='uint8'):
        expanded_guess = np.bitwise_xor(np.zeros((128), dtype=np.uint8), i)
        expected[:, i, :] = des.encrypt(data, expanded_guess, at_round=0, after_step=des.Steps.INV_PERMUTATION_P_RIGHT)
    assert np.array_equal(expected, sf(plaintext=data))
    master_key = np.random.randint(0, 255, (8,), dtype='uint8')
    expected_key = des.key_schedule(master_key)[0]
    assert np.array_equal(expected_key, sf.compute_expected_key(key=master_key))
    assert sf.key_tag == 'key'
    assert isinstance(str(sf), str)


def test_des_decrypt_feistelr_last_rounds_with_alternative_args():
    sf = des.selection_functions.decrypt.FeistelRLastRounds(
        plaintext_tag='foo',
        words=slice(2, 8),
        guesses=_alt_guesses,
        key_tag='thekey'
    )
    assert sf.guesses.tolist() == _alt_guesses.tolist()
    assert sf.words == slice(2, 8, None)
    assert sf.target_tag == 'foo'
    assert isinstance(sf, selection_functions.SelectionFunction)
    data = np.random.randint(0, 255, (10, 8), dtype='uint8')
    expected = np.empty((10, len(_alt_guesses), 8), dtype='uint8')
    for i, guess in enumerate(_alt_guesses):
        expanded_guess = np.bitwise_xor(np.zeros((128), dtype=np.uint8), guess)
        expected[:, i, :] = des.encrypt(data, expanded_guess, at_round=0, after_step=des.Steps.INV_PERMUTATION_P_RIGHT)
    assert np.array_equal(expected[:, :, slice(2, 8)], sf(foo=data))
    master_key = np.random.randint(0, 255, (8,), dtype='uint8')
    expected_key = des.key_schedule(master_key)[0]
    assert np.array_equal(expected_key, sf.compute_expected_key(thekey=master_key))
    assert sf.key_tag == 'thekey'
    assert isinstance(str(sf), str)


def test_des_decrypt_delta_r_first_rounds_with_default_arguments():
    sf = des.selection_functions.decrypt.DeltaRFirstRounds()
    assert sf.guesses.tolist() == list(range(64))
    assert sf.words is Ellipsis
    assert sf.target_tag == 'ciphertext'
    assert isinstance(sf, selection_functions.SelectionFunction)
    data = np.random.randint(0, 255, (10, 8), dtype='uint8')
    expected = np.empty((10, 64, 8), dtype='uint8')
    for i in np.arange(64, dtype='uint8'):
        expanded_guess = np.bitwise_xor(np.zeros((128), dtype=np.uint8), i)
        expected[:, i, :] = des.encrypt(data, expanded_guess, at_round=0, after_step=des.Steps.INV_PERMUTATION_P_DELTA_RIGHT)
    assert np.array_equal(expected, sf(ciphertext=data))
    master_key = np.random.randint(0, 255, (8,), dtype='uint8')
    expected_key = des.key_schedule(master_key)[-1]
    assert np.array_equal(expected_key, sf.compute_expected_key(key=master_key))
    assert sf.key_tag == 'key'
    assert isinstance(str(sf), str)


def test_des_decrypt_delta_r_first_rounds_with_alternative_args():
    sf = des.selection_functions.decrypt.DeltaRFirstRounds(
        ciphertext_tag='foo',
        words=slice(2, 8),
        guesses=_alt_guesses,
        key_tag='thekey'
    )
    assert sf.guesses.tolist() == _alt_guesses.tolist()
    assert sf.words == slice(2, 8, None)
    assert sf.target_tag == 'foo'
    assert isinstance(sf, selection_functions.SelectionFunction)
    data = np.random.randint(0, 255, (10, 8), dtype='uint8')
    expected = np.empty((10, len(_alt_guesses), 8), dtype='uint8')
    for i, guess in enumerate(_alt_guesses):
        expanded_guess = np.bitwise_xor(np.zeros((128), dtype=np.uint8), guess)
        expected[:, i, :] = des.encrypt(data, expanded_guess, at_round=0, after_step=des.Steps.INV_PERMUTATION_P_DELTA_RIGHT)
    assert np.array_equal(expected[:, :, slice(2, 8)], sf(foo=data))
    master_key = np.random.randint(0, 255, (8,), dtype='uint8')
    expected_key = des.key_schedule(master_key)[-1]
    assert np.array_equal(expected_key, sf.compute_expected_key(thekey=master_key))
    assert sf.key_tag == 'thekey'
    assert isinstance(str(sf), str)


def test_des_decrypt_delta_r_last_rounds_with_default_arguments():
    sf = des.selection_functions.decrypt.DeltaRLastRounds()
    assert sf.guesses.tolist() == list(range(64))
    assert sf.words is Ellipsis
    assert sf.target_tag == 'plaintext'
    assert isinstance(sf, selection_functions.SelectionFunction)
    data = np.random.randint(0, 255, (10, 8), dtype='uint8')
    expected = np.empty((10, 64, 8), dtype='uint8')
    for i in np.arange(64, dtype='uint8'):
        expanded_guess = np.bitwise_xor(np.zeros((128), dtype=np.uint8), i)
        expected[:, i, :] = des.encrypt(data, expanded_guess, at_round=0, after_step=des.Steps.INV_PERMUTATION_P_DELTA_RIGHT)
    assert np.array_equal(expected, sf(plaintext=data))
    master_key = np.random.randint(0, 255, (8,), dtype='uint8')
    expected_key = des.key_schedule(master_key)[0]
    assert np.array_equal(expected_key, sf.compute_expected_key(key=master_key))
    assert sf.key_tag == 'key'
    assert isinstance(str(sf), str)


def test_des_decrypt_delta_r_last_rounds_with_alternative_args():
    sf = des.selection_functions.decrypt.DeltaRLastRounds(
        plaintext_tag='foo',
        words=slice(2, 8),
        guesses=_alt_guesses,
        key_tag='thekey'
    )
    assert sf.guesses.tolist() == _alt_guesses.tolist()
    assert sf.words == slice(2, 8, None)
    assert sf.target_tag == 'foo'
    assert isinstance(sf, selection_functions.SelectionFunction)
    data = np.random.randint(0, 255, (10, 8), dtype='uint8')
    expected = np.empty((10, len(_alt_guesses), 8), dtype='uint8')
    for i, guess in enumerate(_alt_guesses):
        expanded_guess = np.bitwise_xor(np.zeros((128), dtype=np.uint8), guess)
        expected[:, i, :] = des.encrypt(data, expanded_guess, at_round=0, after_step=des.Steps.INV_PERMUTATION_P_DELTA_RIGHT)
    assert np.array_equal(expected[:, :, slice(2, 8)], sf(foo=data))
    master_key = np.random.randint(0, 255, (8,), dtype='uint8')
    expected_key = des.key_schedule(master_key)[0]
    assert np.array_equal(expected_key, sf.compute_expected_key(thekey=master_key))
    assert sf.key_tag == 'thekey'
    assert isinstance(str(sf), str)


# see https://gitlab.com/eshard/scared/-/issues/82
def test_des_encrypt_docstring_and_name():
    klass = des.selection_functions.encrypt.FirstAddRoundKey
    assert klass.__name__ == 'FirstAddRoundKey'
    assert 'encrypt' in klass.__doc__.split('\n')[0] and 'first' in klass.__doc__.split('\n')[0]
    klass = des.selection_functions.encrypt.LastAddRoundKey
    assert klass.__name__ == 'LastAddRoundKey'
    assert 'encrypt' in klass.__doc__.split('\n')[0] and 'last' in klass.__doc__.split('\n')[0]
    klass = des.selection_functions.encrypt.FirstSboxes
    assert klass.__name__ == 'FirstSboxes'
    assert 'encrypt' in klass.__doc__.split('\n')[0] and 'first' in klass.__doc__.split('\n')[0]
    klass = des.selection_functions.encrypt.LastSboxes
    assert klass.__name__ == 'LastSboxes'
    assert 'encrypt' in klass.__doc__.split('\n')[0] and 'last' in klass.__doc__.split('\n')[0]

    klass = des.selection_functions.encrypt.FeistelRFirstRounds
    assert klass.__name__ == 'FeistelRFirstRounds'
    assert 'encrypt' in klass.__doc__.split('\n')[0] and 'first' in klass.__doc__.split('\n')[0]
    klass = des.selection_functions.encrypt.FeistelRLastRounds
    assert klass.__name__ == 'FeistelRLastRounds'
    assert 'encrypt' in klass.__doc__.split('\n')[0] and 'last' in klass.__doc__.split('\n')[0]
    klass = des.selection_functions.encrypt.DeltaRFirstRounds
    assert klass.__name__ == 'DeltaRFirstRounds'
    assert 'encrypt' in klass.__doc__.split('\n')[0] and 'first' in klass.__doc__.split('\n')[0]
    klass = des.selection_functions.encrypt.DeltaRLastRounds
    assert klass.__name__ == 'DeltaRLastRounds'
    assert 'encrypt' in klass.__doc__.split('\n')[0] and 'last' in klass.__doc__.split('\n')[0]


# see https://gitlab.com/eshard/scared/-/issues/82
def test_des_decrypt_docstring_and_name():
    klass = des.selection_functions.decrypt.FirstAddRoundKey
    assert klass.__name__ == 'FirstAddRoundKey'
    assert 'decrypt' in klass.__doc__.split('\n')[0] and 'first' in klass.__doc__.split('\n')[0]
    klass = des.selection_functions.decrypt.LastAddRoundKey
    assert klass.__name__ == 'LastAddRoundKey'
    assert 'decrypt' in klass.__doc__.split('\n')[0] and 'last' in klass.__doc__.split('\n')[0]
    klass = des.selection_functions.decrypt.FirstSboxes
    assert klass.__name__ == 'FirstSboxes'
    assert 'decrypt' in klass.__doc__.split('\n')[0] and 'first' in klass.__doc__.split('\n')[0]
    klass = des.selection_functions.decrypt.LastSboxes
    assert klass.__name__ == 'LastSboxes'
    assert 'decrypt' in klass.__doc__.split('\n')[0] and 'last' in klass.__doc__.split('\n')[0]

    klass = des.selection_functions.decrypt.FeistelRFirstRounds
    assert klass.__name__ == 'FeistelRFirstRounds'
    assert 'decrypt' in klass.__doc__.split('\n')[0] and 'first' in klass.__doc__.split('\n')[0]
    klass = des.selection_functions.decrypt.FeistelRLastRounds
    assert klass.__name__ == 'FeistelRLastRounds'
    assert 'decrypt' in klass.__doc__.split('\n')[0] and 'last' in klass.__doc__.split('\n')[0]
    klass = des.selection_functions.decrypt.DeltaRFirstRounds
    assert klass.__name__ == 'DeltaRFirstRounds'
    assert 'decrypt' in klass.__doc__.split('\n')[0] and 'first' in klass.__doc__.split('\n')[0]
    klass = des.selection_functions.decrypt.DeltaRLastRounds
    assert klass.__name__ == 'DeltaRLastRounds'
    assert 'decrypt' in klass.__doc__.split('\n')[0] and 'last' in klass.__doc__.split('\n')[0]
