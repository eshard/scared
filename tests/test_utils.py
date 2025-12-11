from .context import scared  # noqa: F401
from scared.utils.misc import _is_bytes_array
import pytest
import numpy as np
import time


@pytest.mark.parametrize('dtype', ['uint8', 'int8', int, 'uint16', 'int16', 'uint32', 'int32', 'uint64', 'int64'])
def test_is_bytes_array_return_true_for_all_dtypes(dtype):
    assert _is_bytes_array(np.random.randint(0, 128, (42, ), dtype=dtype))


def test_is_bytes_array_raises_if_not_array():
    with pytest.raises(TypeError, match='array should be a Numpy ndarray instance, not'):
        _is_bytes_array('foo')
    with pytest.raises(TypeError, match='array should be a Numpy ndarray instance, not'):
        _is_bytes_array([1, 2, 3])


def test_is_bytes_array_raises_if_negative_values():
    with pytest.raises(ValueError, match='but lowest value'):
        _is_bytes_array(np.zeros(42, dtype='int32') - 1)


def test_is_bytes_array_raises_if_too_high_values():
    with pytest.raises(ValueError, match='but highest value'):
        _is_bytes_array(np.zeros(42, dtype='int32') + 256)

# Take care of performance of functions that massively use these function, like AES or DES.
#  see https://gitlab.com/eshard/scared/-/merge_requests/66


def test_function_much_faster_if_input_is_uint8():
    data_u8 = np.zeros((10_000, 10_000), dtype='uint8')
    data_u32 = np.zeros((10_000, 10_000), dtype='uint32')
    t0 = time.process_time()
    _is_bytes_array(data_u8)
    pt_u8 = time.process_time() - t0
    t0 = time.process_time()
    _is_bytes_array(data_u32)
    pt_u32 = time.process_time() - t0
    assert pt_u32 > (10 * pt_u8)


def test_aes_not_too_slowed_down():
    n = 100_000
    data_u8 = np.zeros((n, 16), dtype='uint8')
    data_u32 = np.zeros((n, 16), dtype='uint32')
    key = np.zeros(16, dtype='uint8')

    t0 = time.process_time()
    scared.aes.encrypt(data_u8, key)
    pt_u8 = time.process_time() - t0
    t0 = time.process_time()
    scared.aes.encrypt(data_u32, key)
    pt_u32 = time.process_time() - t0
    assert pt_u32 < (1.5 * pt_u8)


def test_aes_keyschedule_not_too_slowed_down():
    n = 100_000
    key_u8 = np.zeros((n, 16), dtype='uint8')
    key_u32 = np.zeros((n, 16), dtype='uint32')

    t0 = time.process_time()
    scared.aes.key_schedule(key_u8)
    pt_u8 = time.process_time() - t0
    t0 = time.process_time()
    scared.aes.key_schedule(key_u32)
    pt_u32 = time.process_time() - t0
    assert pt_u32 < (1.5 * pt_u8)


def test_aes_sboxes_not_too_slowed_down():
    n = 1_000_000
    data_u8 = np.zeros((n, 16), dtype='uint8')
    data_u32 = np.zeros((n, 16), dtype='uint32')

    t0 = time.process_time()
    scared.aes.sub_bytes(data_u8)
    pt_u8 = time.process_time() - t0
    t0 = time.process_time()
    scared.aes.sub_bytes(data_u32)
    pt_u32 = time.process_time() - t0
    assert pt_u32 < (2.2 * pt_u8)


def test_des_not_too_slowed_down():
    n = 10_000
    data_u8 = np.zeros((n, 8), dtype='uint8')
    data_u32 = np.zeros((n, 8), dtype='uint32')
    key = np.zeros(16, dtype='uint8')

    t0 = time.process_time()
    scared.des.encrypt(data_u8, key)
    pt_u8 = time.process_time() - t0
    t0 = time.process_time()
    scared.des.encrypt(data_u32, key)
    pt_u32 = time.process_time() - t0
    assert pt_u32 < (1.5 * pt_u8)


def test_des_keyschedule_not_too_slowed_down():
    n = 10_000
    key_u8 = np.zeros((n, 8), dtype='uint8')
    key_u32 = np.zeros((n, 8), dtype='uint32')

    t0 = time.process_time()
    scared.des.key_schedule(key_u8)
    pt_u8 = time.process_time() - t0
    t0 = time.process_time()
    scared.des.key_schedule(key_u32)
    pt_u32 = time.process_time() - t0
    assert pt_u32 < (1.5 * pt_u8)


def test_utils_legacy_import():
    from scared import _utils  # noqa
    print(_utils._is_bytes_array)
