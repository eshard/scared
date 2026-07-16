"""Tests for the partitioned CPA attack (scared-only, no SCALib dependency)."""

from .context import scared  # noqa: F401

import pickle

import numpy as np
import pytest

from scared import aes, maxabs, HammingWeight, Value, CPAAttack, CPAPartitionedAttack
from scared import DistinguisherError
from scared.selection_functions.base import _AttackSelectionFunctionWrapped

_LEAK_SAMPLE = 5
_SECRET_KEY_BYTE = 0x2B


class _Batch:
    """Minimal traces batch exposing samples and metadatas."""

    def __init__(self, samples, metadatas):
        self.samples = samples
        self.metadatas = metadatas


def _run(attack, samples, metadatas):
    attack.process(_Batch(samples, metadatas))
    attack.compute_results()


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


def test_model_table_matches_hamming_weight_of_sbox(leaking_traces):
    traces, metadatas = leaking_traces
    attack = CPAPartitionedAttack(selection_function=aes.selection_functions.encrypt.FirstSubBytes(), model=HammingWeight(), discriminant=maxabs)
    _run(attack, traces, metadatas)
    assert np.allclose(attack._models[:, 0, 0], _hamming_weight_sbox_table())


def test_recovers_secret_key(leaking_traces):
    traces, metadatas = leaking_traces
    attack = CPAPartitionedAttack(selection_function=aes.selection_functions.encrypt.FirstSubBytes(), model=HammingWeight(), discriminant=maxabs)
    _run(attack, traces, metadatas)
    assert attack.scores.argmax(axis=0)[0] == _SECRET_KEY_BYTE


def test_results_shape_and_dtype(leaking_traces):
    traces, metadatas = leaking_traces
    sf = aes.selection_functions.encrypt.FirstSubBytes()
    attack = CPAPartitionedAttack(selection_function=sf, model=HammingWeight(), discriminant=maxabs, precision='float64')
    _run(attack, traces, metadatas)
    assert attack.results.shape == (256, 16, traces.shape[1])
    assert attack.results.dtype == np.float64


def test_labels_are_raw_target_bytes(leaking_traces):
    traces, metadatas = leaking_traces
    attack = CPAPartitionedAttack(selection_function=aes.selection_functions.encrypt.FirstSubBytes(), model=HammingWeight(), discriminant=maxabs)
    labels = attack.compute_intermediate_values(metadatas)
    assert np.array_equal(labels, metadatas['plaintext'])


def test_value_model_is_supported(leaking_traces):
    traces, metadatas = leaking_traces
    attack = CPAPartitionedAttack(selection_function=aes.selection_functions.encrypt.FirstAddRoundKey(), model=Value(), discriminant=maxabs)
    _run(attack, traces, metadatas)
    assert attack.scores.shape == (256, 16)


def test_reduced_guess_space_recovers_key(leaking_traces):
    traces, metadatas = leaking_traces
    guesses = np.array([0x20, _SECRET_KEY_BYTE, 0x40], dtype='uint8')
    sf = aes.selection_functions.encrypt.FirstSubBytes(guesses=guesses)
    attack = CPAPartitionedAttack(selection_function=sf, model=HammingWeight(), discriminant=maxabs)
    _run(attack, traces, metadatas)
    assert attack.results.shape == (3, 16, traces.shape[1])
    assert attack.scores.argmax(axis=0)[0] == 1


def test_correlation_matches_builtin_cpa_xor(leaking_traces):
    traces, metadatas = leaking_traces
    sf_builtin = aes.selection_functions.encrypt.FirstSubBytes()
    builtin = CPAAttack(selection_function=sf_builtin, model=HammingWeight(), discriminant=maxabs, precision='float64')
    _run(builtin, traces, metadatas)

    sf_partitioned = aes.selection_functions.encrypt.FirstSubBytes()
    attack = CPAPartitionedAttack(selection_function=sf_partitioned, model=HammingWeight(), discriminant=maxabs, precision='float64')
    _run(attack, traces, metadatas)

    assert np.allclose(attack.results, builtin.results, atol=1e-9, equal_nan=True)


##########################################
#   Explicit partitions                   #
##########################################


def test_explicit_partitions_control_class_set(leaking_traces):
    traces, metadatas = leaking_traces
    partitions = np.arange(256, dtype='int32')
    attack = CPAPartitionedAttack(partitions=partitions, selection_function=aes.selection_functions.encrypt.FirstSubBytes(),
                                  model=HammingWeight(), discriminant=maxabs)
    _run(attack, traces, metadatas)
    assert attack.scores.argmax(axis=0)[0] == _SECRET_KEY_BYTE
    assert attack.counters.shape == (16, 256)


def test_auto_partitions_derived_from_data(leaking_traces):
    traces, metadatas = leaking_traces
    sf = aes.selection_functions.encrypt.FirstSubBytes()
    attack = CPAPartitionedAttack(selection_function=sf, model=HammingWeight(), discriminant=maxabs)
    attack.compute_intermediate_values(metadatas)
    assert np.array_equal(attack.partitions, np.arange(int(metadatas['plaintext'].max()) + 1))


def test_negative_target_without_partitions_raises():
    sf = aes.selection_functions.encrypt.FirstSubBytes()
    attack = CPAPartitionedAttack(selection_function=sf, model=HammingWeight(), discriminant=maxabs)
    with pytest.raises(DistinguisherError):
        attack.compute_intermediate_values({'plaintext': np.array([[-1, 0]], dtype='int64')})


def test_accumulators_are_exposed_after_processing(leaking_traces):
    traces, metadatas = leaking_traces
    attack = CPAPartitionedAttack(selection_function=aes.selection_functions.encrypt.FirstSubBytes(), model=HammingWeight(), discriminant=maxabs)
    _run(attack, traces, metadatas)
    assert attack.sum.shape == (traces.shape[1], 16, 256)
    assert attack.sum_square.shape == (traces.shape[1], 16, 256)
    assert attack.counters.shape == (16, 256)
    assert attack.counters[0].sum() == traces.shape[0]


##########################################
#   Arbitrary combinations (no group)     #
##########################################


def _multiplicative_body(data, guesses):
    values = np.empty((len(guesses),) + data.shape, dtype='uint8')
    scaled = data.astype('uint16')
    for index, guess in enumerate(guesses):
        values[index] = ((scaled * int(guess)) % 256).astype('uint8')
    return values.swapaxes(0, 1)


def test_multiplicative_selection_function_matches_builtin_cpa():
    rng = np.random.default_rng(4)
    plaintext = rng.integers(0, 256, size=(3000, 16), dtype='uint8')
    traces = rng.normal(size=(3000, 8)).astype('float32')
    metadatas = {'plaintext': plaintext}

    def _sf():
        return _AttackSelectionFunctionWrapped(function=_multiplicative_body, guesses=np.arange(1, 256, dtype='uint8'),
                                               words=None, target_tag='plaintext', target_name='data')

    builtin = CPAAttack(selection_function=_sf(), model=Value(), discriminant=maxabs, precision='float64')
    _run(builtin, traces, metadatas)
    attack = CPAPartitionedAttack(selection_function=_sf(), model=Value(), discriminant=maxabs, precision='float64')
    _run(attack, traces, metadatas)

    assert np.allclose(attack.results, builtin.results, atol=1e-9, equal_nan=True)


def test_joint_guessing_is_rejected():
    plaintext = np.zeros((100, 16), dtype='uint8')

    def _body(data, guesses):
        return np.zeros((data.shape[0], len(guesses), 1), dtype='uint8')

    sf = _AttackSelectionFunctionWrapped(function=_body, guesses=np.arange(4, dtype='uint8'), words=None,
                                         target_tag='plaintext', target_name='data', nb_words_guess=2)
    attack = CPAPartitionedAttack(selection_function=sf, model=HammingWeight(), discriminant=maxabs)
    with pytest.raises(DistinguisherError):
        attack.compute_intermediate_values({'plaintext': plaintext})


def test_structure_check_rejects_inconsistent_model(leaking_traces):
    traces, metadatas = leaking_traces
    attack = CPAPartitionedAttack(selection_function=aes.selection_functions.encrypt.FirstSubBytes(), model=HammingWeight(), discriminant=maxabs)
    attack.compute_intermediate_values(metadatas)
    attack._models = np.zeros_like(attack._models)
    selected = attack._select_words(np.asarray(metadatas['plaintext']))
    with pytest.raises(DistinguisherError):
        attack._verify_structure(np.asarray(metadatas['plaintext']), selected)


def test_degenerate_partition_raises_on_compute():
    rng = np.random.default_rng(2)
    plaintext = np.zeros((50, 16), dtype='uint8')
    traces = rng.normal(size=(50, 6)).astype('float32')
    sf = aes.selection_functions.encrypt.FirstSubBytes(guesses=np.array([0, 1, 2], dtype='uint8'))
    attack = CPAPartitionedAttack(selection_function=sf, model=HammingWeight(), discriminant=maxabs, partitions=np.arange(256, dtype='int32'))
    attack.process(_Batch(traces, {'plaintext': plaintext}))
    with pytest.raises(DistinguisherError):
        attack.compute_results()


##########################################
#   Modular addition (arbitrary modulus)  #
##########################################

_ADD_MODULUS = 251
_ADD_KEY = 37


def _make_additive_sf(q, guesses=None):
    """Build a single-word additive selection function ``sf(data, guess) = (data + guess) mod q``."""
    dtype = 'uint8' if q <= 256 else 'uint16'

    def additive_body(data, guesses):
        scaled = data.astype('int64')
        out = np.empty((len(guesses),) + data.shape, dtype=dtype)
        for index, guess in enumerate(guesses):
            out[index] = (scaled + int(guess)) % q
        return out.swapaxes(0, 1)

    if guesses is None:
        guesses = np.arange(q, dtype='uint16')
    return _AttackSelectionFunctionWrapped(function=additive_body, guesses=guesses, words=None,
                                           target_tag='data', target_name='data')


@pytest.fixture
def additive_leaking_traces():
    """Synthetic single-word traces leaking HW((data + key) mod q) at one sample, q non-power-of-two."""
    rng = np.random.default_rng(0)
    n_traces, n_samples = 4000, 10
    data = rng.integers(0, _ADD_MODULUS, size=(n_traces, 1), dtype='int64')
    intermediate = (data[:, 0] + _ADD_KEY) % _ADD_MODULUS
    leak = HammingWeight()(intermediate.astype('uint8'))
    traces = rng.normal(0.0, 1.0, size=(n_traces, n_samples)).astype('float32')
    traces[:, _LEAK_SAMPLE] += 4.0 * leak.astype('float32')
    return traces, {'data': data}


def test_add_recovers_secret_key(additive_leaking_traces):
    traces, metadatas = additive_leaking_traces
    attack = CPAPartitionedAttack(selection_function=_make_additive_sf(_ADD_MODULUS), model=HammingWeight(),
                                  discriminant=maxabs, partitions=np.arange(_ADD_MODULUS, dtype='int32'))
    _run(attack, traces, metadatas)
    assert attack.results.shape == (_ADD_MODULUS, 1, traces.shape[1])
    assert attack.scores.argmax(axis=0)[0] == _ADD_KEY


def test_add_correlation_matches_builtin_cpa(additive_leaking_traces):
    traces, metadatas = additive_leaking_traces
    builtin = CPAAttack(selection_function=_make_additive_sf(_ADD_MODULUS), model=HammingWeight(), discriminant=maxabs, precision='float64')
    _run(builtin, traces, metadatas)

    attack = CPAPartitionedAttack(selection_function=_make_additive_sf(_ADD_MODULUS), model=HammingWeight(),
                                  discriminant=maxabs, partitions=np.arange(_ADD_MODULUS, dtype='int32'), precision='float64')
    _run(attack, traces, metadatas)

    assert np.allclose(attack.results, builtin.results, atol=1e-9, equal_nan=True)


def test_signed_targets_supported_with_explicit_partitions():
    q = 6001
    rng = np.random.default_rng(7)
    data = rng.integers(-3000, 3001, size=(2000, 1), dtype='int64')
    traces = rng.normal(size=(2000, 6)).astype('float32')
    partitions = np.arange(-3000, 3001, dtype='int32')

    builtin = CPAAttack(selection_function=_make_additive_sf(q, guesses=np.array([0, 1234], dtype='uint16')),
                        model=Value(), discriminant=maxabs, precision='float64')
    _run(builtin, traces, {'data': data})
    attack = CPAPartitionedAttack(selection_function=_make_additive_sf(q, guesses=np.array([0, 1234], dtype='uint16')),
                                  model=Value(), discriminant=maxabs, partitions=partitions, precision='float64')
    _run(attack, traces, {'data': data})

    assert np.allclose(attack.results, builtin.results, atol=1e-6, equal_nan=True)


def test_pickling_preserves_results(leaking_traces):
    traces, metadatas = leaking_traces
    attack = CPAPartitionedAttack(selection_function=aes.selection_functions.encrypt.FirstSubBytes(), model=HammingWeight(), discriminant=maxabs)
    _run(attack, traces, metadatas)

    restored = pickle.loads(pickle.dumps(attack))
    assert restored.scores.argmax(axis=0)[0] == _SECRET_KEY_BYTE
    assert np.allclose(restored.results, attack.results, equal_nan=True)
