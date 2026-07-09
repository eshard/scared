"""Tests for the SCALib-backed CPA attack wrapper."""

import pickle

import numpy as np
import pytest

from scared import aes, maxabs, HammingWeight, Value
from scared import DistinguisherError
from scared.scalib import CPAAttackSCALib
from scared.selection_functions.base import _AttackSelectionFunctionWrapped

pytestmark = pytest.mark.scalib

_LEAK_SAMPLE = 5
_SECRET_KEY_BYTE = 0x2B


class _Batch:
    """Minimal traces batch exposing samples and metadatas."""

    def __init__(self, samples, metadatas):
        self.samples = samples
        self.metadatas = metadatas


def _hamming_weight_sbox_table():
    """Return HW(Sbox[j]) for j in [0, 256), using scared's own primitives."""
    reference = aes.selection_functions.encrypt.FirstSubBytes(guesses=np.array([0], dtype='uint8'))
    classes = np.tile(np.arange(256, dtype='uint8')[:, None], (1, 16))
    return HammingWeight()(reference(plaintext=classes))[:, 0, 0]


@pytest.fixture
def leaking_traces():
    """Build synthetic 16-byte traces leaking HW(Sbox[plaintext[:, 0] ^ key]) at one sample."""
    rng = np.random.default_rng(0)
    n_traces, n_samples = 4000, 10
    plaintext = rng.integers(0, 256, size=(n_traces, 16), dtype='uint8')
    sf = aes.selection_functions.encrypt.FirstSubBytes(guesses=np.array([_SECRET_KEY_BYTE], dtype='uint8'))
    intermediate = HammingWeight()(sf(plaintext=plaintext))[:, 0, 0]
    traces = rng.normal(0.0, 1.0, size=(n_traces, n_samples)).astype('float32')
    traces[:, _LEAK_SAMPLE] += 4.0 * intermediate.astype('float32')
    return traces, {'plaintext': plaintext}


def _run(attack, samples, metadatas):
    attack.process(_Batch(samples, metadatas))
    attack.compute_results()


def test_model_table_matches_hamming_weight_of_sbox(leaking_traces):
    traces, metadatas = leaking_traces
    attack = CPAAttackSCALib(selection_function=aes.selection_functions.encrypt.FirstSubBytes(), model=HammingWeight(), discriminant=maxabs)
    _run(attack, traces, metadatas)
    assert np.allclose(attack._model_matrix[0], _hamming_weight_sbox_table())


def test_recovers_secret_key(leaking_traces):
    traces, metadatas = leaking_traces
    attack = CPAAttackSCALib(selection_function=aes.selection_functions.encrypt.FirstSubBytes(), model=HammingWeight(), discriminant=maxabs)
    _run(attack, traces, metadatas)
    assert attack.scores.argmax(axis=0)[0] == _SECRET_KEY_BYTE


def test_results_shape_and_dtype(leaking_traces):
    traces, metadatas = leaking_traces
    sf = aes.selection_functions.encrypt.FirstSubBytes()
    attack = CPAAttackSCALib(selection_function=sf, model=HammingWeight(), discriminant=maxabs, precision='float64')
    _run(attack, traces, metadatas)
    assert attack.results.shape == (256, 16, traces.shape[1])
    assert attack.results.dtype == np.float64


def test_labels_are_raw_target_bytes(leaking_traces):
    traces, metadatas = leaking_traces
    attack = CPAAttackSCALib(selection_function=aes.selection_functions.encrypt.FirstSubBytes(), model=HammingWeight(), discriminant=maxabs)
    labels = attack.compute_intermediate_values(metadatas)
    assert labels.dtype == np.uint16
    assert np.array_equal(labels, metadatas['plaintext'].astype('uint16'))


def test_value_model_is_supported(leaking_traces):
    traces, metadatas = leaking_traces
    attack = CPAAttackSCALib(selection_function=aes.selection_functions.encrypt.FirstAddRoundKey(), model=Value(), discriminant=maxabs)
    _run(attack, traces, metadatas)
    assert attack.scores.shape == (256, 16)


def test_non_xor_selection_function_is_rejected():
    rng = np.random.default_rng(1)
    ciphertext = rng.integers(0, 256, size=(500, 16), dtype='uint8')
    attack = CPAAttackSCALib(selection_function=aes.selection_functions.encrypt.DeltaRLastRounds(), model=HammingWeight(), discriminant=maxabs)
    with pytest.raises(DistinguisherError):
        attack.compute_intermediate_values({'ciphertext': ciphertext})


def _multiplicative_body(data, guesses):
    values = np.empty((len(guesses),) + data.shape, dtype='uint8')
    scaled = data.astype('uint16')
    for index, guess in enumerate(guesses):
        values[index] = ((scaled * int(guess)) % 256).astype('uint8')
    return values.swapaxes(0, 1)


def test_non_xor_custom_selection_function_with_valid_guesses_is_rejected():
    rng = np.random.default_rng(4)
    plaintext = rng.integers(0, 256, size=(3000, 16), dtype='uint8')
    sf = _AttackSelectionFunctionWrapped(function=_multiplicative_body, guesses=np.arange(256, dtype='uint8'),
                                         words=None, target_tag='plaintext', target_name='data')
    attack = CPAAttackSCALib(selection_function=sf, model=Value(), discriminant=maxabs)
    with pytest.raises(DistinguisherError):
        attack.compute_intermediate_values({'plaintext': plaintext})


@pytest.mark.parametrize('bad_guesses', [
    np.array([300, 400], dtype='uint16'),
    np.array([-5, 5], dtype='int64'),
])
def test_out_of_range_guess_space_is_rejected(bad_guesses):
    rng = np.random.default_rng(3)
    plaintext = rng.integers(0, 256, size=(100, 16), dtype='uint8')
    sf = aes.selection_functions.encrypt.FirstSubBytes(guesses=bad_guesses)
    attack = CPAAttackSCALib(selection_function=sf, model=HammingWeight(), discriminant=maxabs)
    with pytest.raises(DistinguisherError):
        attack.compute_intermediate_values({'plaintext': plaintext})


def test_reduced_guess_space_recovers_key(leaking_traces):
    traces, metadatas = leaking_traces
    guesses = np.array([0x20, _SECRET_KEY_BYTE, 0x40], dtype='uint8')
    sf = aes.selection_functions.encrypt.FirstSubBytes(guesses=guesses)
    attack = CPAAttackSCALib(selection_function=sf, model=HammingWeight(), discriminant=maxabs)
    _run(attack, traces, metadatas)
    assert attack.results.shape == (3, 16, traces.shape[1])
    assert attack.scores.argmax(axis=0)[0] == 1


def test_degenerate_partition_raises_on_compute():
    rng = np.random.default_rng(2)
    plaintext = np.zeros((50, 16), dtype='uint8')
    traces = rng.normal(size=(50, 6)).astype('float32')
    sf = aes.selection_functions.encrypt.FirstSubBytes(guesses=np.array([0, 1, 2], dtype='uint8'))
    attack = CPAAttackSCALib(selection_function=sf, model=HammingWeight(), discriminant=maxabs)
    attack.process(_Batch(traces, {'plaintext': plaintext}))
    with pytest.raises(DistinguisherError):
        attack.compute_results()


def test_model_cache_is_reused_and_dropped_on_pickle(leaking_traces):
    traces, metadatas = leaking_traces
    attack = CPAAttackSCALib(selection_function=aes.selection_functions.encrypt.FirstSubBytes(), model=HammingWeight(), discriminant=maxabs)
    _run(attack, traces, metadatas)
    assert attack._models is not None
    assert attack._models.shape == (16, 256, traces.shape[1])
    cached = attack._models
    attack.compute_results()
    assert attack._models is cached
    restored = pickle.loads(pickle.dumps(attack))
    assert restored._models is None


def test_full_guess_range_matches_explicit_gather(leaking_traces):
    traces, metadatas = leaking_traces
    sf = aes.selection_functions.encrypt.FirstSubBytes()
    attack = CPAAttackSCALib(selection_function=sf, model=HammingWeight(), discriminant=maxabs, precision='float64')
    _run(attack, traces, metadatas)
    models = np.ascontiguousarray(np.broadcast_to(attack._model_matrix[:, :, None], (16, 256, traces.shape[1])), dtype=np.float64)
    reference = attack._cpa.get_correlation(models).transpose(1, 0, 2)[np.arange(256)]
    assert np.allclose(attack.results, reference)


def test_pickling_drops_engine_and_keeps_results(leaking_traces):
    traces, metadatas = leaking_traces
    attack = CPAAttackSCALib(selection_function=aes.selection_functions.encrypt.FirstSubBytes(), model=HammingWeight(), discriminant=maxabs)
    _run(attack, traces, metadatas)

    restored = pickle.loads(pickle.dumps(attack))
    assert restored._cpa is None
    assert restored.scores.argmax(axis=0)[0] == _SECRET_KEY_BYTE
    with pytest.raises(RuntimeError):
        restored._update(traces, metadatas['plaintext'].astype('uint16'))
