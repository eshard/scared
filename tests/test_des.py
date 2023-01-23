from .context import scared  # noqa: F401
from scared import des
import pytest
import numpy as np
from itertools import product
from Crypto.Cipher import DES as crypto_des  # noqa: N811
from Crypto.Cipher import DES3 as crypto_des3  # noqa: N811


_key_sizes = (8, 16, 24)
_multiple_keys = _multiple_states = (True, False)
_cases = list(product(_key_sizes, _multiple_keys, _multiple_states))
number_of_keys = 20


@pytest.fixture(params=_cases)
def encrypt_cases(request):
    key_size, mult_keys, mult_state = request.param
    return _cases_cipher(key_size, mult_keys, mult_state)


@pytest.fixture(params=_cases)
def decrypt_cases(request):
    key_size, mult_keys, mult_state = request.param
    return _cases_cipher(key_size, mult_keys, mult_state, mode='decrypt')


def _cases_cipher(key_size, mult_keys, mult_state, mode='encrypt'):  # noqa
    if mult_keys:
        keys = np.random.randint(0, 255, (number_of_keys, key_size), dtype='uint8')
        if mult_state:
            state = np.random.randint(0, 255, (number_of_keys, 8), dtype='uint8')
            expected = np.empty((number_of_keys, 8), dtype='uint8')
            for i, key in enumerate(keys):
                if key_size == 8:
                    ciphertext = crypto_des.new(key.tobytes(), crypto_des.MODE_ECB)
                elif key_size == 16 or key_size == 24:
                    ciphertext = crypto_des3.new(key.tobytes(), crypto_des3.MODE_ECB)
                expected[i] = np.frombuffer(getattr(ciphertext, mode)(state[i].tobytes()), dtype='uint8')
        else:
            state = np.random.randint(0, 255, (8), dtype='uint8')
            expected = np.empty((number_of_keys, 8), dtype='uint8')
            for i, key in enumerate(keys):
                if key_size == 8:
                    ciphertext = crypto_des.new(key.tobytes(), crypto_des.MODE_ECB)
                elif key_size == 16 or key_size == 24:
                    ciphertext = crypto_des3.new(key.tobytes(), crypto_des3.MODE_ECB)
                expected[i] = np.frombuffer(getattr(ciphertext, mode)(state.tobytes()), dtype='uint8')
    else:
        keys = np.random.randint(0, 255, (key_size), dtype='uint8')
        if key_size == 8:
            ciphertext = crypto_des.new(keys.tobytes(), crypto_des.MODE_ECB)
        elif key_size == 16 or key_size == 24:
            ciphertext = crypto_des3.new(keys.tobytes(), crypto_des3.MODE_ECB)
        if mult_state:
            state = np.random.randint(0, 255, (number_of_keys, 8), dtype='uint8')
            expected = np.empty((number_of_keys, 8), dtype='uint8')
            for i, s in enumerate(state):
                expected[i] = np.frombuffer(getattr(ciphertext, mode)(s.tobytes()), dtype='uint8')
        else:
            state = np.random.randint(0, 255, (8), dtype='uint8')
            expected = np.frombuffer(getattr(ciphertext, mode)(state.tobytes()), dtype='uint8')
    return {'keys': keys, 'state': state, 'key_size': key_size, 'expected': expected}


@pytest.fixture
def des_data():
    return np.load('tests/samples/des_data_tests.npz')


# key_schedule


def test_key_schedule_raises_exception_if_key_isnt_array():
    with pytest.raises(TypeError):
        des.key_schedule(key='foo')


def test_key_schedule_raises_exception_if_interrupt_after_round_isnt_int():
    with pytest.raises(TypeError):
        key = np.random.randint(0, 255, (8, ), dtype='uint8')
        des.key_schedule(key=key, interrupt_after_round='foo')


def test_key_schedule_raises_exception_if_key_isnt_bytes_array():
    with pytest.raises(ValueError):
        key = np.random.randint(0, 255, size=(8,)).astype('float32')
        des.key_schedule(key=key)


def test_key_schedule_raises_exception_if_key_hasnt_a_valid_size():
    with pytest.raises(ValueError):
        key = np.random.randint(0, 255, size=(16,), dtype='uint8')
        des.key_schedule(key=key)


def test_key_schedule_returns_appropriate_key_for_des_key(des_data):
    output = des.key_schedule(des_data['des_key'])
    assert np.array_equal(output, des_data['des_key_schedule_output'])
    assert output.shape == (16, 8)


def test_key_schedule_returns_correct_shape_with_several_keys():
    output = des.key_schedule(np.random.randint(0, 255, (15, 8), dtype='uint8'))
    assert output.shape == (15, 16, 8)


def test_key_schedule_returns_array_of_6bits_numbers():
    keys = np.random.randint(0, 255, (100, 8), dtype='uint8')
    expected_keys = des.key_schedule(keys)
    for element in np.nditer(expected_keys):
        assert element <= 63


# Reverse key schedule


def test_get_master_key_raises_exception_if_round_key_isnt_array():
    array = np.random.randint(256, size=8, dtype=np.uint8)
    with pytest.raises(TypeError):
        des.get_master_key('foo', 0, array, array)


def test_get_master_key_raises_exception_if_round_key_isnt_bytes_array():
    array = np.random.randint(0, 256, (8, ), dtype=np.uint8)
    round_key = np.random.randint(0, 64, (8, )).astype('float32')
    with pytest.raises(ValueError):
        des.get_master_key(round_key, 0, array, array)


def test_get_master_key_raises_exception_if_round_key_hasnt_a_valid_size_array():
    array = np.random.randint(256, size=8, dtype=np.uint8)
    round_key = np.random.randint(64, size=10, dtype=np.uint8)
    with pytest.raises(ValueError):
        des.get_master_key(round_key, 0, array, array)


def test_get_master_key_raises_exception_if_round_key_isnt_8x6bit_array():
    array = np.random.randint(256, size=8, dtype=np.uint8)
    round_key = np.random.randint(64, 255, size=8, dtype=np.uint8)
    with pytest.raises(ValueError):
        des.get_master_key(round_key, 0, array, array)


def test_get_master_key_raises_exception_if_nb_round_isnt_integer():
    array = np.random.randint(64, size=8, dtype=np.uint8)
    with pytest.raises(TypeError):
        des.get_master_key(array, 'foo', array, array)


def test_get_master_key_raises_exception_if_nb_round_has_incorrect_values():
    array = np.random.randint(64, size=8, dtype=np.uint8)
    with pytest.raises(ValueError):
        des.get_master_key(array, -1, array, array)
    with pytest.raises(ValueError):
        des.get_master_key(array, 16, array, array)


def test_get_master_key_raises_exception_if_plaintext_isnt_array():
    array = np.random.randint(256, size=8, dtype=np.uint8)
    with pytest.raises(TypeError):
        des.get_master_key(array, 0, 'foo', array)


def test_get_master_key_raises_exception_if_plaintext_isnt_bytes_array():
    array = np.random.randint(0, 64, (8, ), dtype=np.uint8)
    plaintext = np.random.randint(0, 256, (8, )).astype('float32')
    with pytest.raises(ValueError):
        des.get_master_key(array, 0, plaintext, array)


def test_get_master_key_raises_exception_if_plaintext_hasnt_a_valid_size_array():
    array = np.random.randint(256, size=8, dtype=np.uint8)
    plaintext = np.random.randint(256, size=10, dtype=np.uint8)
    with pytest.raises(ValueError):
        des.get_master_key(array, 0, plaintext, array)


def test_get_master_key_raises_exception_if_ciphertext_isnt_array():
    array = np.random.randint(256, size=8, dtype=np.uint8)
    with pytest.raises(TypeError):
        des.get_master_key(array, 0, array, 'foo')


def test_get_master_key_raises_exception_if_ciphertext_isnt_bytes_array():
    array = np.random.randint(64, size=8, dtype=np.uint8)
    ciphertext = np.random.randint(0, 256, (8, )).astype('float32')
    with pytest.raises(ValueError):
        des.get_master_key(array, 0, array, ciphertext)


def test_get_master_key_raises_exception_if_ciphertext_hasnt_a_valid_size_array():
    array = np.random.randint(256, size=8, dtype=np.uint8)
    ciphertext = np.random.randint(256, size=10, dtype=np.uint8)
    with pytest.raises(ValueError):
        des.get_master_key(array, 0, array, ciphertext)


def test_get_master_key_retrieves_des_master_key_able_to_decrypt_the_ciphertext():
    des_key = np.random.randint(256, size=8, dtype=np.uint8)
    plaintext = np.random.randint(256, size=8, dtype=np.uint8)
    expected_ciphertext = des.encrypt(plaintext, des_key)
    des_key_schedule = des.key_schedule(des_key)

    for round_index, round_key in enumerate(des_key_schedule):
        found_master_key = des.get_master_key(
            round_key, round_index, plaintext, expected_ciphertext)
        assert found_master_key is not None
        plaintext_from_found_key = np.frombuffer(crypto_des.new(found_master_key.tobytes(),
                                                                crypto_des.MODE_ECB)
                                                 .decrypt(expected_ciphertext.tobytes()
                                                          ), dtype='uint8')
        assert np.array_equal(plaintext_from_found_key, plaintext)


def test_initial_permutation_raises_exception_if_state_is_not_array():
    with pytest.raises(TypeError):
        des.initial_permutation(state='foo')
    with pytest.raises(TypeError):
        des.initial_permutation(state=12)


def test_initial_permutation_raises_exception_if_state_is_not_a_correct_array():
    with pytest.raises(ValueError):
        des.initial_permutation(state=np.random.randint(0, 255, (12, 16), dtype='uint8'))
    with pytest.raises(ValueError):
        des.initial_permutation(state=np.random.randint(0, 255, (12, 8)).astype('float32'))


def test_initial_permutation_returns_correct_array(des_data):
    state = des_data['input_state']
    expected = des_data['expected_initial_permutation']
    assert np.array_equal(
        expected,
        des.initial_permutation(state=state)
    )
    assert expected.shape == state.shape


def test_expansive_permutation_raises_exception_if_state_is_not_array():
    with pytest.raises(TypeError):
        des.expansive_permutation(state='foo')
    with pytest.raises(TypeError):
        des.expansive_permutation(state=12)


def test_expansive_permutation_raises_exception_if_state_is_not_a_correct_array():
    with pytest.raises(ValueError):
        des.expansive_permutation(state=np.random.randint(0, 255, (12, 16), dtype='uint8'))
    with pytest.raises(ValueError):
        des.expansive_permutation(state=np.random.randint(0, 255, (12, 8), dtype='uint8'))
    with pytest.raises(ValueError):
        des.expansive_permutation(state=np.random.randint(0, 255, (12, 4)).astype('float32'))


def test_expansive_permutation_returns_correct_array(des_data):
    state = des_data['expected_initial_permutation'][:, 4:8]
    expected = des_data['expected_expansive_permutation']
    assert np.array_equal(
        expected,
        des.expansive_permutation(state=state)
    )
    assert expected.shape == (15, 8)
    assert state.shape == (15, 4)


def test_expansive_permutation_returns_array_of_6bits_numbers():
    state = np.random.randint(0, 255, (1000, 4), dtype='uint8')
    expected = des.expansive_permutation(state)
    for element in np.nditer(expected):
        assert element <= 63


def test_add_round_key_returns_correct_result_array_batch_of_n_keys_and_n_rows_state(des_data):
    state = des_data['expected_expansive_permutation']
    key = des_data['des_key_schedule_output'][:-1]
    expected = des_data['expected_add_round_key']
    assert np.array_equal(
        expected,
        des.add_round_key(state=state, keys=key)
    )
    assert expected.shape == state.shape == (15, 8)


def test_add_round_key_returns_correct_result_array_batch_of_1_key_and_n_rows_state(des_data):
    state = des_data['expected_expansive_permutation']
    key = des_data['des_key_schedule_output'][0]
    expected = des_data['expected_add_round_key_1_key']
    assert np.array_equal(
        expected,
        des.add_round_key(state=state, keys=key)
    )
    assert expected.shape == state.shape == (15, 8)


def test_add_round_key_returns_correct_result_array_batch_of_n_key_and_1_row_state(des_data):
    state = des_data['expected_expansive_permutation'][0]
    key = des_data['des_key_schedule_output']
    expected = des_data['expected_add_round_key_1_state']
    assert np.array_equal(
        expected,
        des.add_round_key(state=state, keys=key)
    )
    assert expected.shape == key.shape == (16, 8)


def test_add_round_key_raises_exception_if_state_is_not_array():
    with pytest.raises(TypeError):
        des.add_round_key(state='foo', keys=np.random.randint(0, 63, (8), dtype='uint8'))
    with pytest.raises(TypeError):
        des.add_round_key(state=12, keys=np.random.randint(0, 63, (8), dtype='uint8'))


def test_add_round_key_raises_exception_if_state_is_not_a_correct_array():
    with pytest.raises(ValueError):
        des.add_round_key(state=np.random.randint(0, 63, (12, 12), dtype='uint8'), keys=np.random.randint(0, 63, (8), dtype='uint8'))
    with pytest.raises(ValueError):
        des.add_round_key(state=np.random.randint(0, 63, (12, 8)).astype('float32'), keys=np.random.randint(0, 63, (8), dtype='uint8'))


def test_add_round_key_raises_exception_if_key_is_not_array():
    with pytest.raises(TypeError):
        des.add_round_key(keys='foo', state=np.random.randint(0, 63, (8), dtype='uint8'))
    with pytest.raises(TypeError):
        des.add_round_key(keys=12, state=np.random.randint(0, 63, (8), dtype='uint8'))


def test_add_round_key_raises_exception_if_key_is_not_a_correct_array():
    with pytest.raises(ValueError):
        des.add_round_key(keys=np.random.randint(0, 63, (12, 12), dtype='uint8'), state=np.random.randint(0, 63, (8), dtype='uint8'))
    with pytest.raises(ValueError):
        des.add_round_key(keys=np.random.randint(0, 63, (12, 8)).astype('float32'), state=np.random.randint(0, 63, (8), dtype='uint8'))


def test_add_round_key_raises_exception_if_key_and_state_dims_are_incompatible():
    with pytest.raises(ValueError):
        des.add_round_key(keys=np.random.randint(0, 63, (12, 8), dtype='uint8'), state=np.random.randint(0, 63, (10, 8), dtype='uint8'))


def test_add_round_key_returns_array_of_6bits_numbers():
    state = np.random.randint(0, 63, (1000, 8), dtype='uint8')
    key = np.random.randint(0, 63, (1000, 8), dtype='uint8')
    expected = des.add_round_key(state, key)
    for element in np.nditer(expected):
        assert element <= 63


def test_sboxes_raises_exception_if_state_is_not_array():
    with pytest.raises(TypeError):
        des.sboxes(state='foo')
    with pytest.raises(TypeError):
        des.sboxes(state=12)


def test_sboxes_raises_exception_if_state_is_not_a_correct_array():
    with pytest.raises(ValueError):
        des.sboxes(state=np.random.randint(0, 63, (12, 16), dtype='uint8'))
    with pytest.raises(ValueError):
        des.sboxes(state=np.random.randint(0, 63, (12, 8)).astype('float32'))


def test_sboxes_returns_correct_array(des_data):
    state = des_data['expected_add_round_key']
    expected = des_data['expected_sboxes']
    assert np.array_equal(
        expected,
        des.sboxes(state=state)
    )
    assert expected.shape == state.shape


def test_sboxes_returns_array_of_4bits_numbers():
    state = np.random.randint(0, 63, (1000, 8), dtype='uint8')
    expected = des.sboxes(state)
    for element in np.nditer(expected):
        assert element <= 15


def test_permutation_p_raises_exception_if_state_is_not_array():
    with pytest.raises(TypeError):
        des.permutation_p(state='foo')
    with pytest.raises(TypeError):
        des.permutation_p(state=12)


def test_permutation_p_raises_exception_if_state_is_not_a_correct_array():
    with pytest.raises(ValueError):
        des.permutation_p(state=np.random.randint(0, 15, (12, 16), dtype='uint8'))
    with pytest.raises(ValueError):
        des.permutation_p(state=np.random.randint(0, 15, (12, 8)).astype('float32'))


def test_permutation_p_returns_correct_array(des_data):
    state = des_data['expected_sboxes']
    expected = des_data['expected_permutation_p']
    assert np.array_equal(
        expected,
        des.permutation_p(state=state)
    )
    assert expected.shape == (15, 4)
    assert state.shape == (15, 8)


def test_inv_permutation_p_raises_exception_if_state_is_not_array():
    with pytest.raises(TypeError):
        des.inv_permutation_p(state='foo')
    with pytest.raises(TypeError):
        des.inv_permutation_p(state=12)


def test_inv_permutation_p_raises_exception_if_state_is_not_a_correct_array():
    with pytest.raises(ValueError):
        des.inv_permutation_p(state=np.random.randint(0, 255, (12, 16), dtype='uint8'))
    with pytest.raises(ValueError):
        des.inv_permutation_p(state=np.random.randint(0, 255, (12, 8), dtype='uint8'))
    with pytest.raises(ValueError):
        des.inv_permutation_p(state=np.random.randint(0, 255, (12, 4)).astype('float32'))


def test_inv_permutation_p_returns_correct_array(des_data):
    state = des_data['expected_permutation_p']
    expected = des_data['expected_sboxes']
    assert np.array_equal(
        expected,
        des.inv_permutation_p(state=state)
    )
    assert expected.shape == (15, 8)
    assert state.shape == (15, 4)


def test_inv_permutation_p_returns_array_of_4bits_numbers():
    state = np.random.randint(0, 255, (1000, 4), dtype='uint8')
    expected = des.inv_permutation_p(state)
    for element in np.nditer(expected):
        assert element <= 15


def test_final_permutation_raises_exception_if_state_is_not_array():
    with pytest.raises(TypeError):
        des.final_permutation(state='foo')
    with pytest.raises(TypeError):
        des.final_permutation(state=12)


def test_final_permutation_raises_exception_if_state_is_not_a_correct_array():
    with pytest.raises(ValueError):
        des.final_permutation(state=np.random.randint(0, 255, (12, 16), dtype='uint8'))
    with pytest.raises(ValueError):
        des.final_permutation(state=np.random.randint(0, 255, (12, 8)).astype('float32'))


def test_final_permutation_returns_correct_array(des_data):
    state = des_data['input_state']
    expected = des_data['expected_final_permutation']
    assert np.array_equal(
        expected,
        des.final_permutation(state=state)
    )
    assert expected.shape == state.shape


# Encryption / decryption


def test_encrypt_raises_exception_if_plaintext_or_key_is_not_a_byte_array_of_appropriate_length():
    with pytest.raises(TypeError):
        des.encrypt(plaintext='foo', key=np.random.randint(0, 255, (8), dtype='uint8'))
    with pytest.raises(TypeError):
        des.encrypt(plaintext=12, key=np.random.randint(0, 255, (8), dtype='uint8'))
    with pytest.raises(ValueError):
        des.encrypt(plaintext=np.random.randint(0, 255, (12), dtype='uint8'), key=np.random.randint(0, 255, (8), dtype='uint8'))
    with pytest.raises(ValueError):
        des.encrypt(plaintext=np.random.randint(0, 255, (8)).astype('float32'), key=np.random.randint(0, 255, (8), dtype='uint8'))

    with pytest.raises(TypeError):
        des.encrypt(key='foo', plaintext=np.random.randint(0, 255, (8), dtype='uint8'))
    with pytest.raises(TypeError):
        des.encrypt(key=12, plaintext=np.random.randint(0, 255, (8), dtype='uint8'))
    with pytest.raises(ValueError):
        des.encrypt(key=np.random.randint(0, 255, (12), dtype='uint8'), plaintext=np.random.randint(0, 255, (8), dtype='uint8'))
    with pytest.raises(ValueError):
        des.encrypt(key=np.random.randint(0, 255, (8)).astype('float32'), plaintext=np.random.randint(0, 255, (8), dtype='uint8'))


def test_encrypt_raises_exception_if_plaintext_and_keys_multiple_are_incompatible():
    with pytest.raises(ValueError):
        des.encrypt(
            plaintext=np.random.randint(0, 255, (10, 8), dtype='uint8'),
            key=np.random.randint(0, 255, (9, 8), dtype='uint8')
        )

    with pytest.raises(ValueError):
        des.encrypt(
            plaintext=np.random.randint(0, 255, (2, 10, 8), dtype='uint8'),
            key=np.random.randint(0, 255, (10, 8), dtype='uint8')
        )


def test_full_encrypt(encrypt_cases):
    assert np.array_equal(
        encrypt_cases['expected'],
        des.encrypt(plaintext=encrypt_cases['state'], key=encrypt_cases['keys'])
    )


def test_full_encrypt_with_int32_data(encrypt_cases):
    assert np.array_equal(
        encrypt_cases['expected'],
        des.encrypt(plaintext=encrypt_cases['state'].astype('int32'),
                    key=encrypt_cases['keys'].astype('int32')))


def test_full_encrypt_with_int8_data():
    data = np.random.randint(0, 128, (10, 8), dtype='uint8')
    key = np.random.randint(0, 128, (10, 8), dtype='uint8')
    expected = des.encrypt(data, key)
    assert np.array_equal(des.encrypt(data.astype('int8'), key.astype('int8')), expected)


def test_full_encrypt_with_expanded_keys(encrypt_cases):
    if encrypt_cases['keys'].shape[-1] >= 8:
        # we expand only first 8 bytes of the key
        if encrypt_cases['keys'].ndim == 2:
            expanded_key1 = des.key_schedule(encrypt_cases['keys'][:, 0:8])
            expanded_key = expanded_key1.reshape(expanded_key1.shape[:-2] + (-1,))
        else:
            expanded_key1 = des.key_schedule(encrypt_cases['keys'][0:8])
            expanded_key = expanded_key1.reshape(expanded_key1.shape[:-2] + (-1,))
    if encrypt_cases['keys'].shape[-1] >= 16:
        # we expand second 8 bytes of the key
        if encrypt_cases['keys'].ndim == 2:
            expanded_key2 = des.key_schedule(encrypt_cases['keys'][:, 8:16])
            expanded_key2 = expanded_key2.reshape(expanded_key2.shape[:-2] + (-1,))
            expanded_key = np.append(expanded_key, expanded_key2, axis=1)
        else:
            expanded_key2 = des.key_schedule(encrypt_cases['keys'][8:16])
            expanded_key2 = expanded_key2.reshape(expanded_key2.shape[:-2] + (-1,))
            expanded_key = np.append(expanded_key, expanded_key2, axis=0)
    if encrypt_cases['keys'].shape[-1] == 24:
        # we expand third 8 bytes of the key
        if encrypt_cases['keys'].ndim == 2:
            expanded_key3 = des.key_schedule(encrypt_cases['keys'][:, 16:24])
            expanded_key3 = expanded_key3.reshape(expanded_key3.shape[:-2] + (-1,))
            expanded_key = np.append(expanded_key, expanded_key3, axis=1)
        else:
            expanded_key3 = des.key_schedule(encrypt_cases['keys'][16:24])
            expanded_key3 = expanded_key3.reshape(expanded_key3.shape[:-2] + (-1,))
            expanded_key = np.append(expanded_key, expanded_key3, axis=0)
    assert np.array_equal(
        encrypt_cases['expected'],
        des.encrypt(plaintext=encrypt_cases['state'], key=expanded_key)
    )


def test_full_decrypt(decrypt_cases):
    assert np.array_equal(
        decrypt_cases['expected'],
        des.decrypt(ciphertext=decrypt_cases['state'], key=decrypt_cases['keys'])
    )


def test_full_decrypt_with_expanded_keys(decrypt_cases):
    if decrypt_cases['keys'].shape[-1] >= 8:
        if decrypt_cases['keys'].ndim == 2:
            expanded_key1 = des.key_schedule(decrypt_cases['keys'][:, 0:8])
            expanded_key = expanded_key1.reshape(expanded_key1.shape[:-2] + (-1,))
        else:
            expanded_key1 = des.key_schedule(decrypt_cases['keys'][0:8])
            expanded_key = expanded_key1.reshape(expanded_key1.shape[:-2] + (-1,))
    if decrypt_cases['keys'].shape[-1] >= 16:
        if decrypt_cases['keys'].ndim == 2:
            expanded_key2 = des.key_schedule(decrypt_cases['keys'][:, 8:16])
            expanded_key2 = expanded_key2.reshape(expanded_key2.shape[:-2] + (-1,))
            expanded_key = np.append(expanded_key, expanded_key2, axis=1)
        else:
            expanded_key2 = des.key_schedule(decrypt_cases['keys'][8:16])
            expanded_key2 = expanded_key2.reshape(expanded_key2.shape[:-2] + (-1,))
            expanded_key = np.append(expanded_key, expanded_key2, axis=0)
    if decrypt_cases['keys'].shape[-1] == 24:
        if decrypt_cases['keys'].ndim == 2:
            expanded_key3 = des.key_schedule(decrypt_cases['keys'][:, 16:24])
            expanded_key3 = expanded_key3.reshape(expanded_key3.shape[:-2] + (-1,))
            expanded_key = np.append(expanded_key, expanded_key3, axis=1)
        else:
            expanded_key3 = des.key_schedule(decrypt_cases['keys'][16:24])
            expanded_key3 = expanded_key3.reshape(expanded_key3.shape[:-2] + (-1,))
            expanded_key = np.append(expanded_key, expanded_key3, axis=0)
    assert np.array_equal(
        decrypt_cases['expected'],
        des.decrypt(ciphertext=decrypt_cases['state'], key=expanded_key)
    )


def test_simple_encrypt_with_des_key(des_data):
    key = des_data['des_key']
    plain = des_data['plaintext']
    expected_cipher = des_data['des_ciphertext']
    cipher = des.encrypt(key=key, plaintext=plain)
    assert np.array_equal(expected_cipher, cipher)


def test_decrypt_raises_exception_if_ciphertext_or_key_is_not_a_byte_array_of_appropriate_length():
    with pytest.raises(TypeError):
        des.decrypt(ciphertext='foo', key=np.random.randint(0, 255, (8), dtype='uint8'))
    with pytest.raises(TypeError):
        des.decrypt(ciphertext=12, key=np.random.randint(0, 255, (8), dtype='uint8'))
    with pytest.raises(ValueError):
        des.decrypt(ciphertext=np.random.randint(0, 255, (12), dtype='uint8'), key=np.random.randint(0, 255, (8), dtype='uint8'))
    with pytest.raises(ValueError):
        des.decrypt(ciphertext=np.random.randint(0, 255, (8)).astype('float32'), key=np.random.randint(0, 255, (8), dtype='uint8'))

    with pytest.raises(TypeError):
        des.decrypt(key='foo', ciphertext=np.random.randint(0, 255, (8), dtype='uint8'))
    with pytest.raises(TypeError):
        des.decrypt(key=12, ciphertext=np.random.randint(0, 255, (8), dtype='uint8'))
    with pytest.raises(ValueError):
        des.decrypt(key=np.random.randint(0, 255, (12), dtype='uint8'), ciphertext=np.random.randint(0, 255, (8), dtype='uint8'))
    with pytest.raises(ValueError):
        des.decrypt(key=np.random.randint(0, 255, (8)).astype('float32'), ciphertext=np.random.randint(0, 255, (8), dtype='uint8'))


def test_decrypt_raises_exception_if_ciphertext_and_keys_multiple_are_incompatible():
    with pytest.raises(ValueError):
        des.decrypt(
            ciphertext=np.random.randint(0, 255, (10, 8), dtype='uint8'),
            key=np.random.randint(0, 255, (9, 8), dtype='uint8')
        )

    with pytest.raises(ValueError):
        des.decrypt(
            ciphertext=np.random.randint(0, 255, (2, 10, 8), dtype='uint8'),
            key=np.random.randint(0, 255, (10, 8), dtype='uint8')
        )


def test_simple_decrypt_with_des_key(des_data):
    key = des_data['des_key']
    cipher = des_data['des_ciphertext']
    expected_plain = des_data['plaintext']
    plain = des.decrypt(key=key, ciphertext=cipher)
    assert np.array_equal(expected_plain, plain)


def _generate_test_data_for_intermediate_values(key, plain, des_number, output_name):
    # this function is not a test but a helper to update expected test results
    result = np.zeros((des_number, 16, 10, 8), dtype=np.uint8)
    for des_number in range(des_number):
        for round_number in range(16):
            for step_number in range(10):
                result[des_number][round_number][step_number] = des.encrypt(plain, key, at_des=des_number, at_round=round_number, after_step=step_number)
    d_des = dict(np.load('tests/samples/des_data_tests.npz'))
    d_des[output_name] = result
    np.savez('tests/samples/des_data_tests.npz', **d_des)


def test_encrypt_stop_at_intermediate_value_with_des(des_data):
    int_values = des_data['des_encrypt_intermediate_outputs']
    key = des_data['des_key']
    plain = des_data['plaintext']
    for des_number, des_value in enumerate(int_values):
        for round_number, round_value in enumerate(des_value):
            for step_number, step_value in enumerate(round_value):
                value = des.encrypt(plaintext=plain, key=key, at_des=des_number, at_round=round_number, after_step=step_number)
                assert np.array_equal(step_value, value)

    def _generate_values_for_test_encrypt_stop_at_intermediate_value_with_des(des_data):
        _generate_test_data_for_intermediate_values(des_data['des_key'], des_data['plaintext'], 1, 'des_encrypt_intermediate_outputs')


def test_encrypt_stop_at_intermediate_value_with_tdes3(des_data):
    int_values = des_data['tdes3_encrypt_intermediate_outputs']
    key = des_data['tdes3_key']
    plain = des_data['plaintext']
    for des_number, des_value in enumerate(int_values):
        for round_number, round_value in enumerate(des_value):
            for step_number, step_value in enumerate(round_value):
                value = des.encrypt(plaintext=plain, key=key, at_des=des_number, at_round=round_number, after_step=step_number)
                assert np.array_equal(step_value, value)

    def _generate_values_for_test_encrypt_stop_at_intermediate_value_with_des(des_data):
        _generate_test_data_for_intermediate_values(des_data['tdes3_key'], des_data['plaintext'], 3, 'tdes3_encrypt_intermediate_outputs')


def test_decrypt_stop_at_intermediate_value_with_des(des_data):
    int_values = des_data['des_decrypt_intermediate_outputs']
    key = des_data['des_key']
    cipher = des_data['des_ciphertext']
    for des_number, des_value in enumerate(int_values):
        for round_number, round_value in enumerate(des_value):
            for step_number, step_value in enumerate(round_value):
                value = des.decrypt(ciphertext=cipher, key=key, at_des=des_number, at_round=round_number, after_step=step_number)
                assert np.array_equal(step_value, value)

    def _generate_values_for_test_encrypt_stop_at_intermediate_value_with_des(des_data):
        _generate_test_data_for_intermediate_values(des_data['des_key'], des_data['des_ciphertext'], 1, 'des_decrypt_intermediate_outputs')


def test_decrypt_stop_at_intermediate_value_with_tdes3(des_data):
    int_values = des_data['tdes3_decrypt_intermediate_outputs']
    key = des_data['tdes3_key']
    cipher = des_data['des_ciphertext']
    for des_number, des_value in enumerate(int_values):
        for round_number, round_value in enumerate(des_value):
            for step_number, step_value in enumerate(round_value):
                value = des.decrypt(ciphertext=cipher, key=key, at_des=des_number, at_round=round_number, after_step=step_number)
                assert np.array_equal(step_value, value)

    def _generate_values_for_test_encrypt_stop_at_intermediate_value_with_des(des_data):
        _generate_test_data_for_intermediate_values(des_data['tdes3_key'], des_data['des_ciphertext'], 3, 'tdes3_decrypt_intermediate_outputs')


def test_encrypt_raises_exception_if_improper_round_or_step_or_des_type(des_data):
    key = des_data['des_key']
    plain = des_data['plaintext']

    with pytest.raises(TypeError):
        des.encrypt(plaintext=plain, key=key, at_round='foo')
    with pytest.raises(TypeError):
        des.encrypt(plaintext=plain, key=key, after_step='foo')
    with pytest.raises(TypeError):
        des.encrypt(plaintext=plain, key=key, at_des='foo')


def test_decrypt_raises_exception_if_improper_round_or_step_or_des_type(des_data):
    key = des_data['des_key']
    plain = des_data['plaintext']

    with pytest.raises(TypeError):
        des.decrypt(plain, key=key, at_round='foo')
    with pytest.raises(TypeError):
        des.decrypt(plain, key=key, after_step='foo')
    with pytest.raises(TypeError):
        des.decrypt(plain, key=key, at_des='foo')


def test_encrypt_raises_exception_if_round_is_negative_or_too_high(des_data):
    des_key = des_data['des_key']
    tdes2_key = des_data['tdes2_key']
    tdes3_key = des_data['tdes3_key']
    plain = des_data['plaintext']

    with pytest.raises(ValueError):
        des.encrypt(plaintext=plain, key=des_key, at_round=-1)
    with pytest.raises(ValueError):
        des.encrypt(plaintext=plain, key=des_key, at_round=16)
    with pytest.raises(ValueError):
        des.encrypt(plaintext=plain, key=tdes2_key, at_round=-1)
    with pytest.raises(ValueError):
        des.encrypt(plaintext=plain, key=tdes2_key, at_round=16)
    with pytest.raises(ValueError):
        des.encrypt(plaintext=plain, key=tdes3_key, at_round=-1)
    with pytest.raises(ValueError):
        des.encrypt(plaintext=plain, key=tdes3_key, at_round=16)


def test_decrypt_raises_exception_if_round_is_negative_or_too_high(des_data):
    des_key = des_data['des_key']
    tdes2_key = des_data['tdes2_key']
    tdes3_key = des_data['tdes3_key']
    cipher = des_data['des_ciphertext']

    with pytest.raises(ValueError):
        des.decrypt(cipher, key=des_key, at_round=-1)
    with pytest.raises(ValueError):
        des.decrypt(cipher, key=des_key, at_round=16)
    with pytest.raises(ValueError):
        des.decrypt(cipher, key=tdes2_key, at_round=-1)
    with pytest.raises(ValueError):
        des.decrypt(cipher, key=tdes2_key, at_round=16)
    with pytest.raises(ValueError):
        des.decrypt(cipher, key=tdes3_key, at_round=-1)
    with pytest.raises(ValueError):
        des.decrypt(cipher, key=tdes3_key, at_round=16)


def test_encrypt_raises_exception_if_step_is_incorrect(des_data):
    des_key = des_data['des_key']
    plain = des_data['plaintext']

    with pytest.raises(ValueError):
        des.encrypt(plaintext=plain, key=des_key, after_step=-1)
    with pytest.raises(ValueError):
        des.encrypt(plaintext=plain, key=des_key, after_step=10)


def test_decrypt_raises_exception_if_step_is_incorrect(des_data):
    des_key = des_data['des_key']
    plain = des_data['plaintext']

    with pytest.raises(ValueError):
        des.decrypt(plain, key=des_key, after_step=-1)
    with pytest.raises(ValueError):
        des.decrypt(plain, key=des_key, after_step=10)


def test_encrypt_raises_exception_if_at_des_is_negative_or_too_high(des_data):
    des_key = des_data['des_key']
    tdes2_key = des_data['tdes2_key']
    tdes3_key = des_data['tdes3_key']
    plain = des_data['plaintext']

    with pytest.raises(ValueError):
        des.encrypt(plaintext=plain, key=des_key, at_des=-1)
    with pytest.raises(ValueError):
        des.encrypt(plaintext=plain, key=des_key, at_des=1)
    with pytest.raises(ValueError):
        des.encrypt(plaintext=plain, key=tdes2_key, at_des=-1)
    with pytest.raises(ValueError):
        des.encrypt(plaintext=plain, key=tdes2_key, at_des=3)
    with pytest.raises(ValueError):
        des.encrypt(plaintext=plain, key=tdes3_key, at_des=-1)
    with pytest.raises(ValueError):
        des.encrypt(plaintext=plain, key=tdes3_key, at_des=3)


def test_decrypt_raises_exception_if_at_des_is_negative_or_too_high(des_data):
    des_key = des_data['des_key']
    tdes2_key = des_data['tdes2_key']
    tdes3_key = des_data['tdes3_key']
    cipher = des_data['des_ciphertext']

    with pytest.raises(ValueError):
        des.decrypt(cipher, key=des_key, at_des=-1)
    with pytest.raises(ValueError):
        des.decrypt(cipher, key=des_key, at_des=1)
    with pytest.raises(ValueError):
        des.decrypt(cipher, key=tdes2_key, at_des=-1)
    with pytest.raises(ValueError):
        des.decrypt(cipher, key=tdes2_key, at_des=3)
    with pytest.raises(ValueError):
        des.decrypt(cipher, key=tdes3_key, at_des=-1)
    with pytest.raises(ValueError):
        des.decrypt(cipher, key=tdes3_key, at_des=3)
