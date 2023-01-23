from .context import scared  # noqa: F401
from scared import aes
import pytest
import numpy as np
from itertools import product
from Crypto.Cipher import AES as crypto_aes  # noqa: N811


_key_sizes = (16, 24, 32)
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


def _cases_cipher(key_size, mult_keys, mult_state, mode='encrypt'):
    if mult_keys:
        keys = np.random.randint(0, 255, (number_of_keys, key_size), dtype='uint8')
        if mult_state:
            state = np.random.randint(0, 255, (number_of_keys, 16), dtype='uint8')
            expected = np.empty((number_of_keys, 16), dtype='uint8')
            for i, key in enumerate(keys):
                a = crypto_aes.new(key.tobytes(), crypto_aes.MODE_ECB)
                expected[i] = np.frombuffer(getattr(a, mode)(state[i].tobytes()), dtype='uint8')
        else:
            state = np.random.randint(0, 255, (16), dtype='uint8')
            expected = np.empty((number_of_keys, 16), dtype='uint8')
            for i, key in enumerate(keys):
                a = crypto_aes.new(key.tobytes(), crypto_aes.MODE_ECB)
                expected[i] = np.frombuffer(getattr(a, mode)(state.tobytes()), dtype='uint8')
    else:
        keys = np.random.randint(0, 255, (key_size), dtype='uint8')
        a = crypto_aes.new(keys.tobytes(), crypto_aes.MODE_ECB)
        if mult_state:
            state = np.random.randint(0, 255, (number_of_keys, 16), dtype='uint8')
            expected = np.empty((number_of_keys, 16), dtype='uint8')
            for i, s in enumerate(state):
                expected[i] = np.frombuffer(getattr(a, mode)(s.tobytes()), dtype='uint8')
        else:
            state = np.random.randint(0, 255, (16), dtype='uint8')
            expected = np.frombuffer(getattr(a, mode)(state.tobytes()), dtype='uint8')
    return {'keys': keys, 'state': state, 'expected': expected}


@pytest.fixture
def aes_data():
    return np.load('tests/samples/aes_data_tests.npz')


def test_key_expansion_raise_exception_if_key_isnt_array():
    with pytest.raises(TypeError):
        aes.key_expansion(key_cols='foo')
    with pytest.raises(TypeError):
        aes.key_expansion(key_cols=123456465)
    with pytest.raises(TypeError):
        aes.key_expansion(key_cols={'shape': 12})


def test_key_expansion_raise_exception_if_key_isnt_bytes_array():
    with pytest.raises(ValueError):
        key = np.random.random_sample((16, ))
        aes.key_expansion(key_cols=key)


def test_key_expansion_raise_exception_if_key_hasnt_a_valid_size():
    with pytest.raises(ValueError):
        key = np.random.randint(0, 255, size=(8,), dtype='uint8')
        aes.key_expansion(key_cols=key)


def test_key_expansion_raise_exception_if_col_in_or_col_out_is_not_int():
    with pytest.raises(TypeError):
        key = np.random.randint(0, 255, (16, ), dtype='uint8')
        aes.key_expansion(key_cols=key, col_in='foo')
    with pytest.raises(TypeError):
        key = np.random.randint(0, 255, (24, ), dtype='uint8')
        aes.key_expansion(key_cols=key, col_in=0, col_out='foo')


def test_key_expansion_raise_exception_if_col_in_or_col_out_is_negative():
    with pytest.raises(ValueError):
        key = np.random.randint(0, 255, (16, ), dtype='uint8')
        aes.key_expansion(key_cols=key, col_in=-2, col_out=40)
    with pytest.raises(ValueError):
        key = np.random.randint(0, 255, (24, ), dtype='uint8')
        aes.key_expansion(key_cols=key, col_in=0, col_out=-12)


def test_key_expansion_raise_exception_if_col_out_is_greater_than_max_cols():
    with pytest.raises(ValueError):
        key = np.random.randint(0, 255, (16, ), dtype='uint8')
        aes.key_expansion(key_cols=key, col_in=0, col_out=50)
    with pytest.raises(ValueError):
        key = np.random.randint(0, 255, (24, ), dtype='uint8')
        aes.key_expansion(key_cols=key, col_in=0, col_out=60)
    with pytest.raises(ValueError):
        key = np.random.randint(0, 255, (32, ), dtype='uint8')
        aes.key_expansion(key_cols=key, col_in=0, col_out=66)


def test_key_expansion_bwd_128(aes_data):
    round_keys = aes_data['128_key_schedule_output'].reshape(11 * 16)
    for i in range(0, 11):
        key_cols = round_keys[i * 16: (i + 1) * 16]
        master = aes.key_expansion(key_cols=key_cols, col_in=i * 4, col_out=0)
        assert np.array_equal(master[0], round_keys[:(i + 1) * 16])


def test_key_expansion_bwd_192(aes_data):
    round_keys = aes_data['192_key_schedule_output'].reshape(13 * 16)
    for i in range(8, 0, -1):
        key_cols = round_keys[16 + (i - 1) * 24: 16 + i * 24]
        master = aes.key_expansion(key_cols=key_cols, col_in=int((16 + (i - 1) * 24) / 4), col_out=0)
        assert np.array_equal(master[0], round_keys[:16 + i * 24])


def test_key_expansion_bwd_256(aes_data):
    master = np.random.randint(0, 255, (32,), dtype='uint8')
    round_keys = aes.key_expansion(key_cols=master, col_in=0, col_out=60)[0]
    for i in range(7, 0, -1):
        key_cols = round_keys[(i - 1) * 32: i * 32]
        master = aes.key_expansion(key_cols=key_cols, col_in=int(((i - 1) * 32) / 4), col_out=0)
        res = master[0].reshape((-1, 4))
        exp = round_keys[:i * 32].reshape((-1, 4))
        for i, w in enumerate(res):
            assert w.tolist() == exp[i].tolist()
    round_keys = aes_data['256_key_schedule_output'].reshape(15 * 16)
    for i in range(7, 0, -1):
        key_cols = round_keys[(i - 1) * 32: i * 32]
        master = aes.key_expansion(key_cols=key_cols, col_in=int(((i - 1) * 32) / 4), col_out=0)
        assert np.array_equal(master[0], round_keys[:i * 32])


@pytest.fixture(params=[(4, 16, 11), (6, 24, 12), (8, 32, 14)])
def key_expansion_params(request):
    return {
        'cols_size': request.param[0],
        'key': np.random.randint(0, 255, (request.param[1],), dtype='uint8'),
        'rounds': range(0, request.param[2])
    }


def test_key_expansion_fwd_starting_from_any_intermediate_state(key_expansion_params):
    key = key_expansion_params['key']
    rounds = key_expansion_params['rounds']
    cols_size = key_expansion_params['cols_size']
    full_schedule = aes.key_schedule(key).reshape((-1, 4))
    for intermediate_round in rounds:
        schedule_part_1 = aes.key_expansion(
            key,
            col_out=cols_size + intermediate_round * 4
        ).reshape((-1, 4))

        assert np.array_equal(
            schedule_part_1,
            full_schedule[:cols_size + intermediate_round * 4]
        )
        sched_part_2 = aes.key_expansion(
            schedule_part_1[len(schedule_part_1) - cols_size:].reshape(-1),
            col_in=intermediate_round * 4,
        ).reshape((-1, 4))
        expected = full_schedule[intermediate_round * 4:]

        for i, v in enumerate(sched_part_2):
            assert np.array_equal(
                v,
                expected[i]
            )


def test_key_expansion_bwd_starting_from_any_intermediate_state(key_expansion_params):
    key = key_expansion_params['key']
    cols_size = key_expansion_params['cols_size']
    rounds = range(key_expansion_params['rounds'].stop - 1, key_expansion_params['rounds'].start - 1, -1)
    full_schedule = aes.key_schedule(key).reshape((-1, 4))
    _in = len(full_schedule) - cols_size
    key_base = full_schedule[_in:].reshape((cols_size * 4,))
    for intermediate_round in rounds:
        _out = _in - intermediate_round * 4
        schedule_part_1 = aes.key_expansion(
            key_base,
            col_in=_in,
            col_out=_out
        ).reshape((-1, 4))
        assert np.array_equal(
            schedule_part_1,
            full_schedule[_out:]
        )
        sched_part_2 = aes.key_expansion(
            schedule_part_1[:cols_size].reshape(-1),
            col_in=_out,
            col_out=0
        ).reshape((-1, 4))
        expected = full_schedule[:_out + cols_size]
        for i, v in enumerate(sched_part_2):
            assert np.array_equal(
                v,
                expected[i]
            )


def test_key_expansion_fwd_128(aes_data):
    master = aes_data['128_key']
    keys = aes.key_expansion(master, col_in=0)
    assert np.array_equal(aes_data['128_key_schedule_output'].reshape(176), keys[0])


def test_key_expansion_fwd_192(aes_data):
    master = aes_data['192_key']
    keys = aes.key_expansion(master, col_in=0)
    assert np.array_equal(aes_data['192_key_schedule_output'].reshape(13 * 16), keys[0])


def test_key_expansion_fwd_256(aes_data):
    master = aes_data['256_key']
    keys = aes.key_expansion(master, col_in=0)
    assert np.array_equal(aes_data['256_key_schedule_output'].reshape(15 * 16), keys[0])


def test_key_schedule_returns_appropriate_keys_for_128_key(aes_data):
    output = aes.key_schedule(aes_data['128_key'])
    assert np.array_equal(
        output,
        aes_data['128_key_schedule_output']
    )
    assert output.shape == (11, 16)
    output = aes.key_schedule(np.random.randint(0, 255, (15, 16), dtype='uint8'))
    assert output.shape == (15, 11, 16)


def test_inv_key_schedule_returns_appropriate_keys(aes_data):
    round_keys = aes_data['128_key_schedule_output']
    for i, round_key in enumerate(round_keys):
        res = aes.inv_key_schedule(round_key, i)
        assert np.array_equal(round_keys, res[0])


def test_inv_key_schedule_with_multiple_keys_returns_appropriate_keys():
    master_keys = np.random.randint(0, 255, (10, 16), dtype='uint8')
    round_keys = aes.key_schedule(master_keys)
    for i in range(round_keys.shape[1]):
        res = aes.inv_key_schedule(round_keys[:, i, :], i)
        assert np.array_equal(round_keys, res)


def test_key_schedule_returns_appropriate_keys_for_192_key(aes_data):
    output = aes.key_schedule(aes_data['192_key'])
    assert np.array_equal(
        output,
        aes_data['192_key_schedule_output']
    )
    assert output.shape == (13, 16)
    output = aes.key_schedule(np.random.randint(0, 255, (15, 24), dtype='uint8'))
    assert output.shape == (15, 13, 16)


def test_key_schedule_returns_appropriate_keys_for_256_key(aes_data):
    output = aes.key_schedule(aes_data['256_key'])
    assert np.array_equal(
        output,
        aes_data['256_key_schedule_output']
    )
    assert output.shape == (15, 16)
    output = aes.key_schedule(np.random.randint(0, 255, (15, 32), dtype='uint8'))
    assert output.shape == (15, 15, 16)


def test_key_schedule_raise_exception_if_key_isnt_array():
    with pytest.raises(TypeError):
        aes.key_schedule(key='foo')
    with pytest.raises(TypeError):
        aes.key_schedule(key=123456465)
    with pytest.raises(TypeError):
        aes.key_schedule(key={'shape': 12})


def test_key_schedule_raise_exception_if_key_isnt_bytes_array():
    with pytest.raises(ValueError):
        key = np.random.random_sample((16, ))
        aes.key_schedule(key=key)


def test_key_schedule_raise_exception_if_key_hasnt_a_valid_size():
    with pytest.raises(ValueError):
        key = np.random.randint(0, 255, size=(8,))
        aes.key_schedule(key=key)


def test_sub_bytes_raises_exception_if_state_is_not_array():
    with pytest.raises(TypeError):
        aes.sub_bytes(state='foo')
    with pytest.raises(TypeError):
        aes.sub_bytes(state=12)


def test_sub_bytes_raises_exception_if_state_is_not_a_correct_array():
    with pytest.raises(ValueError):
        aes.sub_bytes(state=np.random.randint(0, 255, (12, 12), dtype='uint8'))
    with pytest.raises(ValueError):
        aes.sub_bytes(state=np.random.randint(0, 255, (12, 16)).astype('float32'))


def test_sub_bytes_returns_correct_array(aes_data):
    state = aes_data['input_state']
    expected = aes_data['expected_sub_bytes']
    assert np.array_equal(
        expected,
        aes.sub_bytes(state=state)
    )
    assert expected.shape == state.shape


def test_inv_sub_bytes_raises_exception_if_state_is_not_array():
    with pytest.raises(TypeError):
        aes.inv_sub_bytes(state='foo')
    with pytest.raises(TypeError):
        aes.inv_sub_bytes(state=12)


def test_inv_sub_bytes_raises_exception_if_state_is_not_a_correct_array():
    with pytest.raises(ValueError):
        aes.inv_sub_bytes(state=np.random.randint(0, 255, (12, 12), dtype='uint8'))
    with pytest.raises(ValueError):
        aes.inv_sub_bytes(state=np.random.randint(0, 255, (12, 16)).astype('float32'))


def test_inv_sub_bytes_returns_correct_array(aes_data):
    state = aes_data['input_state']
    expected = aes_data['expected_inv_sub_bytes']
    assert np.array_equal(
        expected,
        aes.inv_sub_bytes(state=state)
    )
    assert expected.shape == state.shape


def test_shift_rows_returns_correct_array(aes_data):
    state = aes_data['input_state']
    expected = aes_data['expected_shift_rows']
    assert np.array_equal(
        expected,
        aes.shift_rows(state=state)
    )
    assert expected.shape == state.shape


def test_shift_rows_raises_exception_if_state_is_not_array():
    with pytest.raises(TypeError):
        aes.shift_rows(state='foo')
    with pytest.raises(TypeError):
        aes.shift_rows(state=12)


def test_shift_rows_raises_exception_if_state_is_not_a_correct_array():
    with pytest.raises(ValueError):
        aes.shift_rows(state=np.random.randint(0, 255, (12, 12), dtype='uint8'))
    with pytest.raises(ValueError):
        aes.shift_rows(state=np.random.randint(0, 255, (12, 16)).astype('float32'))


def test_inv_shift_rows_returns_correct_array(aes_data):
    state = aes_data['input_state']
    expected = aes_data['expected_inv_shift_rows']
    assert np.array_equal(
        expected,
        aes.inv_shift_rows(state=state)
    )
    assert expected.shape == state.shape


def test_inv_shift_rows_raises_exception_if_state_is_not_array():
    with pytest.raises(TypeError):
        aes.inv_shift_rows(state='foo')
    with pytest.raises(TypeError):
        aes.inv_shift_rows(state=12)


def test_inv_shift_rows_raises_exception_if_state_is_not_a_correct_array():
    with pytest.raises(ValueError):
        aes.inv_shift_rows(state=np.random.randint(0, 255, (12, 12), dtype='uint8'))
    with pytest.raises(ValueError):
        aes.inv_shift_rows(state=np.random.randint(0, 255, (12, 16)).astype('float32'))


def test_mix_columns_returns_correct_array(aes_data):
    state = aes_data['input_state']
    expected = aes_data['expected_mix_columns']
    assert np.array_equal(
        expected,
        aes.mix_columns(state=state)
    )
    assert expected.shape == state.shape


def test_mix_columns_raises_exception_if_state_is_not_array():
    with pytest.raises(TypeError):
        aes.mix_columns(state='foo')
    with pytest.raises(TypeError):
        aes.mix_columns(state=12)


def test_mix_columns_raises_exception_if_state_is_not_a_correct_array():
    with pytest.raises(ValueError):
        aes.mix_columns(state=np.random.randint(0, 255, (12, 12), dtype='uint8'))
    with pytest.raises(ValueError):
        aes.mix_columns(state=np.random.randint(0, 255, (12, 16)).astype('float32'))


def test_mix_column_returns_correct_column_vector(aes_data):
    state = aes_data['input_vectors']
    expected = aes_data['expected_mix_column']
    assert np.array_equal(
        expected,
        aes.mix_column(state)
    )
    assert expected.shape == state.shape


def test_mix_column_raises_exception_if_state_is_not_array():
    with pytest.raises(TypeError):
        aes.mix_column(vectors='foo')
    with pytest.raises(TypeError):
        aes.mix_column(vectors=12)


def test_mix_column_raises_exception_if_state_is_not_a_correct_array():
    with pytest.raises(ValueError):
        aes.mix_column(vectors=np.random.randint(0, 255, (12, 12), dtype='uint8'))
    with pytest.raises(ValueError):
        aes.mix_column(vectors=np.random.randint(0, 255, (12, 4)).astype('float32'))


def test_inv_mix_column_returns_correct_column_vector(aes_data):
    state = aes_data['input_vectors']
    expected = aes_data['expected_inv_mix_column']
    assert np.array_equal(
        expected,
        aes.inv_mix_column(state)
    )
    assert expected.shape == state.shape


def test_inv_mix_column_raises_exception_if_state_is_not_array():
    with pytest.raises(TypeError):
        aes.inv_mix_column(vectors='foo')
    with pytest.raises(TypeError):
        aes.inv_mix_column(vectors=12)


def test_inv_mix_column_raises_exception_if_state_is_not_a_correct_array():
    with pytest.raises(ValueError):
        aes.inv_mix_column(vectors=np.random.randint(0, 255, (12, 12), dtype='uint8'))
    with pytest.raises(ValueError):
        aes.inv_mix_column(vectors=np.random.randint(0, 255, (12, 4)).astype('float32'))


def test_inv_mix_columns_returns_correct_column_vector(aes_data):
    state = aes_data['input_state']
    expected = aes_data['expected_inv_mix_columns']
    assert np.array_equal(
        expected,
        aes.inv_mix_columns(state)
    )
    assert expected.shape == state.shape


def test_inv_mix_columns_raises_exception_if_state_is_not_array():
    with pytest.raises(TypeError):
        aes.inv_mix_columns(state='foo')
    with pytest.raises(TypeError):
        aes.inv_mix_columns(state=12)


def test_inv_mix_columns_raises_exception_if_state_is_not_a_correct_array():
    with pytest.raises(ValueError):
        aes.inv_mix_columns(state=np.random.randint(0, 255, (12, 12), dtype='uint8'))
    with pytest.raises(ValueError):
        aes.inv_mix_columns(state=np.random.randint(0, 255, (12, 16)).astype('float32'))


def test_add_round_key_returns_correct_result_array_batch_of_n_keys_and_n_rows_state(aes_data):
    state = aes_data['input_state']
    key = aes_data['input_round_key']
    expected = aes_data['expected_add_round_key']
    assert np.array_equal(
        expected,
        aes.add_round_key(state=state, keys=key)
    )
    assert expected.shape == state.shape


def test_add_round_key_returns_correct_result_array_batch_of_1_key_and_n_rows_state(aes_data):
    state = aes_data['input_state']
    key = aes_data['input_round_key'][0]
    expected = aes_data['expected_add_round_key_1_key']
    assert np.array_equal(
        expected,
        aes.add_round_key(state=state, keys=key)
    )
    assert expected.shape == state.shape


def test_add_round_key_returns_correct_result_array_batch_of_n_key_and_1_row_state(aes_data):
    state = aes_data['input_state'][0]
    key = aes_data['input_round_key']
    expected = aes_data['expected_add_round_key_1_state']
    assert np.array_equal(
        expected,
        aes.add_round_key(state=state, keys=key)
    )


def test_add_round_key_raises_exception_if_state_is_not_array():
    with pytest.raises(TypeError):
        aes.add_round_key(state='foo', keys=np.random.randint(0, 255, (16), dtype='uint8'))
    with pytest.raises(TypeError):
        aes.add_round_key(state=12, keys=np.random.randint(0, 255, (16), dtype='uint8'))


def test_add_round_key_raises_exception_if_state_is_not_a_correct_array():
    with pytest.raises(ValueError):
        aes.add_round_key(state=np.random.randint(0, 255, (12, 12), dtype='uint8'), keys=np.random.randint(0, 255, (16), dtype='uint8'))
    with pytest.raises(ValueError):
        aes.add_round_key(state=np.random.randint(0, 255, (12, 16)).astype('float32'), keys=np.random.randint(0, 255, (16), dtype='uint8'))


def test_add_round_key_raises_exception_if_key_is_not_array():
    with pytest.raises(TypeError):
        aes.add_round_key(keys='foo', state=np.random.randint(0, 255, (16), dtype='uint8'))
    with pytest.raises(TypeError):
        aes.add_round_key(keys=12, state=np.random.randint(0, 255, (16), dtype='uint8'))


def test_add_round_key_raises_exception_if_key_is_not_a_correct_array():
    with pytest.raises(ValueError):
        aes.add_round_key(keys=np.random.randint(0, 255, (12, 12), dtype='uint8'), state=np.random.randint(0, 255, (16), dtype='uint8'))
    with pytest.raises(ValueError):
        aes.add_round_key(keys=np.random.randint(0, 255, (12, 16)).astype('float32'), state=np.random.randint(0, 255, (16), dtype='uint8'))


def test_add_round_key_raises_exception_if_key_and_state_dims_are_incompatible():
    with pytest.raises(ValueError):
        aes.add_round_key(keys=np.random.randint(0, 255, (12, 16), dtype='uint8'), state=np.random.randint(0, 255, (10, 16), dtype='uint8'))


def test_inv_add_round_key_is_the_same_function_than_add_round_key():
    assert aes.inv_add_round_key == aes.add_round_key


def test_encrypt_raises_exception_if_plaintext_or_key_is_not_a_byte_array_of_appropriate_length():
    with pytest.raises(TypeError):
        aes.encrypt(plaintext='foo', key=np.random.randint(0, 255, (16), dtype='uint8'))
    with pytest.raises(TypeError):
        aes.encrypt(plaintext=12, key=np.random.randint(0, 255, (16), dtype='uint8'))
    with pytest.raises(ValueError):
        aes.encrypt(plaintext=np.random.randint(0, 255, (12), dtype='uint8'), key=np.random.randint(0, 255, (16), dtype='uint8'))
    with pytest.raises(ValueError):
        aes.encrypt(plaintext=np.random.randint(0, 255, (16)).astype('float32'), key=np.random.randint(0, 255, (16), dtype='uint8'))

    with pytest.raises(TypeError):
        aes.encrypt(key='foo', plaintext=np.random.randint(0, 255, (16), dtype='uint8'))
    with pytest.raises(TypeError):
        aes.encrypt(key=12, plaintext=np.random.randint(0, 255, (16), dtype='uint8'))
    with pytest.raises(ValueError):
        aes.encrypt(key=np.random.randint(0, 255, (12), dtype='uint8'), plaintext=np.random.randint(0, 255, (16), dtype='uint8'))
    with pytest.raises(ValueError):
        aes.encrypt(key=np.random.randint(0, 255, (16)).astype('float32'), plaintext=np.random.randint(0, 255, (16), dtype='uint8'))


def test_encrypt_raises_exception_if_plaintext_and_keys_multiple_are_incompatible():
    with pytest.raises(ValueError):
        aes.encrypt(
            plaintext=np.random.randint(0, 255, (10, 16), dtype='uint8'),
            key=np.random.randint(0, 255, (9, 16), dtype='uint8')
        )

    with pytest.raises(ValueError):
        aes.encrypt(
            plaintext=np.random.randint(0, 255, (2, 10, 16), dtype='uint8'),
            key=np.random.randint(0, 255, (16), dtype='uint8')
        )


def test_full_encrypt(encrypt_cases):
    assert np.array_equal(
        encrypt_cases['expected'],
        aes.encrypt(plaintext=encrypt_cases['state'], key=encrypt_cases['keys'])
    )


def test_full_decrypt(decrypt_cases):
    assert np.array_equal(
        decrypt_cases['expected'],
        aes.decrypt(ciphertext=decrypt_cases['state'], key=decrypt_cases['keys'])
    )


def test_simple_encrypt_with_128_key(aes_data):
    key = aes_data['128_key']
    plain = aes_data['plaintext']
    expected_cipher = aes_data['128_ciphertext']
    cipher = aes.encrypt(key=key, plaintext=plain)
    assert np.array_equal(expected_cipher, cipher)


def test_simple_encrypt_with_int32_data(aes_data):
    key = aes_data['128_key'].astype('int32')
    plain = aes_data['plaintext'].astype('int32')
    expected_cipher = aes_data['128_ciphertext']
    cipher = aes.encrypt(key=key, plaintext=plain)
    assert np.array_equal(expected_cipher, cipher)


def test_encrypt_with_int8_data():
    data = np.random.randint(0, 128, (10, 16), dtype='uint8')
    key = np.random.randint(0, 128, (10, 16), dtype='uint8')
    expected = aes.encrypt(data, key)
    assert np.array_equal(aes.encrypt(data.astype('int8'), key.astype('int8')), expected)


def test_decrypt_raises_exception_if_ciphertext_or_key_is_not_a_byte_array_of_appropriate_length():
    with pytest.raises(TypeError):
        aes.decrypt(ciphertext='foo', key=np.random.randint(0, 255, (16), dtype='uint8'))
    with pytest.raises(TypeError):
        aes.decrypt(ciphertext=12, key=np.random.randint(0, 255, (16), dtype='uint8'))
    with pytest.raises(ValueError):
        aes.decrypt(ciphertext=np.random.randint(0, 255, (12), dtype='uint8'), key=np.random.randint(0, 255, (16), dtype='uint8'))
    with pytest.raises(ValueError):
        aes.decrypt(ciphertext=np.random.randint(0, 255, (16)).astype('float32'), key=np.random.randint(0, 255, (16), dtype='uint8'))

    with pytest.raises(TypeError):
        aes.decrypt(key='foo', ciphertext=np.random.randint(0, 255, (16), dtype='uint8'))
    with pytest.raises(TypeError):
        aes.decrypt(key=12, ciphertext=np.random.randint(0, 255, (16), dtype='uint8'))
    with pytest.raises(ValueError):
        aes.decrypt(key=np.random.randint(0, 255, (12), dtype='uint8'), ciphertext=np.random.randint(0, 255, (16), dtype='uint8'))
    with pytest.raises(ValueError):
        aes.decrypt(key=np.random.randint(0, 255, (16)).astype('float32'), ciphertext=np.random.randint(0, 255, (16), dtype='uint8'))


def test_decrypt_raises_exception_if_ciphertext_and_keys_multiple_are_incompatible():
    with pytest.raises(ValueError):
        aes.decrypt(
            ciphertext=np.random.randint(0, 255, (10, 16), dtype='uint8'),
            key=np.random.randint(0, 255, (9, 16), dtype='uint8')
        )

    with pytest.raises(ValueError):
        aes.decrypt(
            ciphertext=np.random.randint(0, 255, (2, 10, 16), dtype='uint8'),
            key=np.random.randint(0, 255, (16), dtype='uint8')
        )


def test_simple_decrypt_with_128_key(aes_data):
    key = aes_data['128_key']
    cipher = aes_data['128_ciphertext']
    expected_plain = aes_data['plaintext']
    plain = aes.decrypt(key=key, ciphertext=cipher)
    assert np.array_equal(expected_plain, plain)


def test_encrypt_stop_at_intermediate_value(aes_data):
    int_values = aes_data['128_encrypt_intermediate_outputs']
    key = aes_data['128_key']
    plain = aes_data['plaintext']
    for _round, vals in enumerate(int_values):
        for step, expected in enumerate(vals):
            value = aes.encrypt(plaintext=plain, key=key, at_round=_round, after_step=step)
            assert np.array_equal(expected, value)


def test_decrypt_stop_at_intermediate_value(aes_data):
    int_values = aes_data['128_decrypt_intermediate_outputs']
    key = aes_data['128_key']
    cipher = aes_data['128_ciphertext']
    for _round, vals in enumerate(int_values):
        for step, expected in enumerate(vals):
            value = aes.decrypt(ciphertext=cipher, key=key, at_round=_round, after_step=step)
            assert np.array_equal(expected, value)


def test_encrypt_raises_exception_if_improper_round_or_step_type(aes_data):
    key = aes_data['128_key']
    plain = aes_data['plaintext']

    with pytest.raises(TypeError):
        aes.encrypt(plaintext=plain, key=key, at_round='foo')
    with pytest.raises(TypeError):
        aes.encrypt(plaintext=plain, key=key, after_step='foo')


def test_decrypt_raises_exception_if_improper_round_or_step_type(aes_data):
    key = aes_data['128_key']
    plain = aes_data['plaintext']

    with pytest.raises(TypeError):
        aes.decrypt(plain, key=key, at_round='foo')
    with pytest.raises(TypeError):
        aes.decrypt(plain, key=key, after_step='foo')


def test_encrypt_raises_exception_if_round_is_negative_or_too_high(aes_data):
    key_128 = aes_data['128_key']
    key_192 = aes_data['192_key']
    key_256 = aes_data['256_key']
    plain = aes_data['plaintext']

    with pytest.raises(ValueError):
        aes.encrypt(plaintext=plain, key=key_128, at_round=-1)
    with pytest.raises(ValueError):
        aes.encrypt(plaintext=plain, key=key_128, at_round=12)
    with pytest.raises(ValueError):
        aes.encrypt(plaintext=plain, key=key_192, at_round=14)
    with pytest.raises(ValueError):
        aes.encrypt(plaintext=plain, key=key_256, at_round=17)


def test_decrypt_raises_exception_if_round_is_negative_or_too_high(aes_data):
    key_128 = aes_data['128_key']
    key_192 = aes_data['192_key']
    key_256 = aes_data['256_key']
    plain = aes_data['plaintext']

    with pytest.raises(ValueError):
        aes.decrypt(plain, key=key_128, at_round=-1)
    with pytest.raises(ValueError):
        aes.decrypt(plain, key=key_128, at_round=12)
    with pytest.raises(ValueError):
        aes.decrypt(plain, key=key_192, at_round=14)
    with pytest.raises(ValueError):
        aes.decrypt(plain, key=key_256, at_round=17)


def test_encrypt_raises_exception_if_step_is_incorrect(aes_data):
    key_128 = aes_data['128_key']
    plain = aes_data['plaintext']

    with pytest.raises(ValueError):
        aes.encrypt(plaintext=plain, key=key_128, after_step=-1)
    with pytest.raises(ValueError):
        aes.encrypt(plaintext=plain, key=key_128, after_step=4)


def test_decrypt_raises_exception_if_step_is_incorrect(aes_data):
    key_128 = aes_data['128_key']
    plain = aes_data['plaintext']

    with pytest.raises(ValueError):
        aes.decrypt(plain, key=key_128, after_step=-1)
    with pytest.raises(ValueError):
        aes.decrypt(plain, key=key_128, after_step=4)
