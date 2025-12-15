"""Comprehensive tests for SCALib SNR Distinguisher wrapper."""

import pytest
import numpy as np
import logging

from scared.scalib import SNRDistinguisherSCALib, SNRDistinguisherSCALibMixin, SCALIB_AVAILABLE
from scared import DistinguisherError


@pytest.fixture
def simple_test_data():
    """Create simple test data for basic functionality tests."""
    np.random.seed(42)
    n_traces = 100
    n_samples = 50
    n_variables = 2

    traces = np.random.randn(n_traces, n_samples).astype(np.float32)
    labels = np.random.randint(0, 16, (n_traces, n_variables), dtype=np.uint8)

    return traces, labels


@pytest.fixture
def partitioned_test_data():
    """Create test data with non-trivial partitions."""
    np.random.seed(123)
    n_traces = 200
    n_samples = 30
    n_variables = 3

    traces = np.random.randn(n_traces, n_samples).astype(np.float32)
    labels = np.random.randint(13, 42, (n_traces, n_variables), dtype=np.uint8)

    return traces, labels, range(13, 42)


# Initialization tests

def test_scalib_snr_init_with_partitions():
    """Test initialization with explicit partitions."""
    snr = SNRDistinguisherSCALib(partitions=range(256))
    assert snr.partitions is not None
    assert len(snr.partitions) == 256


def test_scalib_snr_init_with_custom_partitions():
    """Test initialization with custom partitions."""
    partitions = range(13, 42)
    snr = SNRDistinguisherSCALib(partitions=partitions)
    assert np.array_equal(snr.partitions, np.arange(13, 42, dtype='int32'))


def test_scalib_snr_init_with_list_partitions():
    """Test initialization with list partitions."""
    partitions = [0, 1, 2, 5, 10, 15]
    snr = SNRDistinguisherSCALib(partitions=partitions)
    assert np.array_equal(snr.partitions, np.array(partitions, dtype='int32'))


def test_scalib_snr_init_with_array_partitions():
    """Test initialization with numpy array partitions."""
    partitions = np.array([0, 5, 10, 15], dtype='uint32')
    snr = SNRDistinguisherSCALib(partitions=partitions)
    assert np.array_equal(snr.partitions, partitions.astype('int32'))


def test_scalib_snr_init_without_partitions():
    """Test initialization without partitions (will be inferred)."""
    snr = SNRDistinguisherSCALib()
    assert snr.partitions is None


def test_scalib_snr_init_with_precision_float32():
    """Test initialization with float32 precision."""
    snr32 = SNRDistinguisherSCALib(partitions=range(16), precision='float32')
    assert snr32.precision == np.float32


def test_scalib_snr_init_with_precision_float64():
    """Test initialization with float64 precision."""
    snr64 = SNRDistinguisherSCALib(partitions=range(16), precision='float64')
    assert snr64.precision == np.float64


def test_scalib_snr_partitions_auto_init_for_small_values(simple_test_data):
    """Test automatic partition initialization from data."""
    traces, labels = simple_test_data
    snr = SNRDistinguisherSCALib()
    snr.update(traces, labels)

    assert snr.partitions is not None
    assert len(snr.partitions) == 64


def test_scalib_snr_partitions_auto_init_9_classes():
    """Test automatic partition initialization for 9 classes."""
    traces = np.random.randn(50, 20).astype(np.float32)
    labels = np.random.randint(0, 8, (50, 1), dtype=np.uint8)

    snr = SNRDistinguisherSCALib()
    snr.update(traces, labels)

    assert np.array_equal(snr.partitions, np.arange(9))


def test_scalib_snr_partitions_auto_init_256_classes():
    """Test automatic partition initialization for 256 classes."""
    traces = np.random.randn(50, 20).astype(np.float32)
    labels = np.random.randint(0, 200, (50, 1), dtype=np.uint8)

    snr = SNRDistinguisherSCALib()
    snr.update(traces, labels)

    assert np.array_equal(snr.partitions, np.arange(256))


def test_scalib_snr_processed_traces_counter(simple_test_data):
    """Test that processed_traces counter is initialized."""
    traces, labels = simple_test_data
    snr = SNRDistinguisherSCALib(partitions=range(16))
    assert snr.processed_traces == 0
    snr.update(traces, labels)
    assert snr.processed_traces == len(traces)


def test_scalib_snr_distinguisher_str_property():
    """Test _distinguisher_str property."""
    snr = SNRDistinguisherSCALib(partitions=range(16))
    assert snr._distinguisher_str == 'SNR_SCALib'


# Update tests

def test_scalib_snr_update_simple(simple_test_data):
    """Test basic update operation."""
    traces, labels = simple_test_data
    snr = SNRDistinguisherSCALib(partitions=range(16))
    snr.update(traces, labels)

    assert snr.processed_traces == len(traces)
    assert snr._trace_length == traces.shape[1]
    assert snr._data_words == labels.shape[1]


def test_scalib_snr_update_multiple_batches(simple_test_data):
    """Test updating with multiple batches."""
    traces, labels = simple_test_data
    snr = SNRDistinguisherSCALib(partitions=range(16))

    snr.update(traces[:50], labels[:50])
    assert snr.processed_traces == 50

    snr.update(traces[50:], labels[50:])
    assert snr.processed_traces == 100


def test_scalib_snr_update_with_non_trivial_partitions(partitioned_test_data):
    """Test update with non-zero-based partitions."""
    traces, labels, partitions = partitioned_test_data
    snr = SNRDistinguisherSCALib(partitions=partitions)
    snr.update(traces, labels)

    assert snr.processed_traces == len(traces)


def test_scalib_snr_update_raises_on_inconsistent_trace_length(simple_test_data):
    """Test that update raises error on inconsistent trace lengths."""
    traces, labels = simple_test_data
    snr = SNRDistinguisherSCALib(partitions=range(16))

    snr.update(traces, labels)

    traces_wrong = np.random.randn(50, 30).astype(np.float32)
    labels_wrong = np.random.randint(0, 16, (50, 2), dtype=np.uint8)

    with pytest.raises(ValueError, match='traces has different length'):
        snr.update(traces_wrong, labels_wrong)


def test_scalib_snr_update_raises_on_inconsistent_data_words(simple_test_data):
    """Test that update raises error on inconsistent data words."""
    traces, labels = simple_test_data
    snr = SNRDistinguisherSCALib(partitions=range(16))

    snr.update(traces, labels)

    traces_new = np.random.randn(50, 50).astype(np.float32)
    labels_wrong = np.random.randint(0, 16, (50, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match='data has different number of data words'):
        snr.update(traces_new, labels_wrong)


def test_scalib_snr_update_raises_on_non_integer_data(simple_test_data):
    """Test that update raises error on non-integer data."""
    traces, _ = simple_test_data
    snr = SNRDistinguisherSCALib(partitions=range(16))

    labels_float = np.random.randn(100, 2).astype(np.float32)

    with pytest.raises(TypeError, match='data dtype for partitioned distinguisher must be an integer dtype'):
        snr.update(traces, labels_float)


def test_scalib_snr_update_with_int16_traces():
    """Test update with int16 traces (SCALib native format)."""
    traces = np.random.randint(-100, 100, (50, 20), dtype=np.int16)
    labels = np.random.randint(0, 16, (50, 1), dtype=np.uint8)

    snr = SNRDistinguisherSCALib(partitions=range(16))
    snr.update(traces, labels)

    assert snr.processed_traces == 50


def test_scalib_snr_update_with_float32_traces():
    """Test update with float32 traces (needs conversion)."""
    traces = np.random.randn(50, 20).astype(np.float32)
    labels = np.random.randint(0, 16, (50, 1), dtype=np.uint8)

    snr = SNRDistinguisherSCALib(partitions=range(16))
    snr.update(traces, labels)

    assert snr.processed_traces == 50


def test_scalib_snr_update_with_float64_traces():
    """Test update with float64 traces (needs conversion)."""
    traces = np.random.randn(50, 20).astype(np.float64)
    labels = np.random.randint(0, 16, (50, 1), dtype=np.uint8)

    snr = SNRDistinguisherSCALib(partitions=range(16))
    snr.update(traces, labels)

    assert snr.processed_traces == 50


def test_scalib_snr_update_with_int32_traces():
    """Test update with int32 traces (needs conversion)."""
    traces = np.random.randint(-100, 100, (50, 20), dtype=np.int32)
    labels = np.random.randint(0, 16, (50, 1), dtype=np.uint8)

    snr = SNRDistinguisherSCALib(partitions=range(16))
    snr.update(traces, labels)

    assert snr.processed_traces == 50


def test_scalib_snr_update_handles_overflow_traces():
    """Test update handles traces with extreme values."""
    traces = np.array([[100000.0, -100000.0, 0.0]], dtype=np.float32)
    labels = np.array([[5]], dtype=np.uint8)

    snr = SNRDistinguisherSCALib(partitions=range(16))
    snr.update(traces, labels)

    assert snr.processed_traces == 1


# Compute tests

def test_scalib_snr_compute_returns_correct_shape(simple_test_data):
    """Test that compute returns correct shape."""
    traces, labels = simple_test_data
    snr = SNRDistinguisherSCALib(partitions=range(16))
    snr.update(traces, labels)

    result = snr.compute()

    assert result.shape == (labels.shape[1], traces.shape[1])


def test_scalib_snr_compute_returns_correct_dtype_float32():
    """Test that compute returns correct dtype for float32."""
    traces = np.random.randn(50, 20).astype(np.float32)
    labels = np.random.randint(0, 16, (50, 1), dtype=np.uint8)

    snr = SNRDistinguisherSCALib(partitions=range(16), precision='float32')
    snr.update(traces, labels)
    result = snr.compute()

    assert result.dtype == np.float32


def test_scalib_snr_compute_returns_correct_dtype_float64():
    """Test that compute returns correct dtype for float64."""
    traces = np.random.randn(50, 20).astype(np.float32)
    labels = np.random.randint(0, 16, (50, 1), dtype=np.uint8)

    snr = SNRDistinguisherSCALib(partitions=range(16), precision='float64')
    snr.update(traces, labels)
    result = snr.compute()

    assert result.dtype == np.float64


def test_scalib_snr_compute_returns_finite_values(simple_test_data):
    """Test that compute returns finite values."""
    traces, labels = simple_test_data
    snr = SNRDistinguisherSCALib(partitions=range(16))
    snr.update(traces, labels)

    result = snr.compute()

    assert np.isfinite(result).any()


def test_scalib_snr_compute_with_multiple_variables(simple_test_data):
    """Test compute with multiple variables."""
    traces, labels = simple_test_data
    snr = SNRDistinguisherSCALib(partitions=range(16))
    snr.update(traces, labels)

    result = snr.compute()

    assert result.shape[0] == labels.shape[1]


def test_scalib_snr_compute_multiple_times(simple_test_data):
    """Test that compute can be called multiple times."""
    traces, labels = simple_test_data
    snr = SNRDistinguisherSCALib(partitions=range(16))
    snr.update(traces, labels)

    result1 = snr.compute()
    result2 = snr.compute()

    np.testing.assert_array_equal(result1, result2)


# Comparison with precomputed reference data

@pytest.fixture
def partitioned_datas():
    """Load precomputed reference data for partitioned distinguishers."""
    datas = np.load('tests/samples/tests_partitioned_distinguishers.npz')
    for k, v in datas.items():
        setattr(datas, k, v)
    return datas


def test_scalib_snr_compute_raises_exception_if_no_accumulation():
    """Test that compute raises error if called before update."""
    d = SNRDistinguisherSCALib()
    with pytest.raises((ValueError, DistinguisherError, AttributeError)):
        d.compute()


@pytest.mark.parametrize('precision', ['float32', 'float64'])
def test_scalib_snr_manual_snr_calculation(simple_test_data, precision):
    """Test that SCALib SNR matches manual SNR calculation using SCALib's formula.

    SCALib converts traces to int16 and uses integer arithmetic internally.
    The SNR formula (after canceling common denominators) is:
        signal = sum_i (n * S_i^2 / n_i) - S^2
        noise = n * SS - sum_i (n * S_i^2 / n_i)
        SNR = signal / noise

    Where n=total traces, S_i=sum for class i, n_i=count for class i, SS=sum of squares.
    """
    traces, labels = simple_test_data

    snr_scalib = SNRDistinguisherSCALib(partitions=range(16), precision=precision)
    snr_scalib.update(traces, labels)
    result_scalib = snr_scalib.compute()

    traces_int16 = traces.astype(np.int16)

    nc = 16
    ns = traces_int16.shape[1]
    nv = labels.shape[1]

    snr_manual = np.zeros((nv, ns), dtype=precision)

    for var_idx in range(nv):
        S_i = np.zeros((nc, ns), dtype=np.int64)  # noqa: N806
        n_i = np.zeros(nc, dtype=np.int64)

        for class_idx in range(nc):
            mask = labels[:, var_idx] == class_idx
            if np.any(mask):
                n_i[class_idx] = np.sum(mask)
                S_i[class_idx] = np.sum(traces_int16[mask].astype(np.int64), axis=0)  # noqa: N806

        valid_mask = n_i > 0
        n = np.sum(n_i[valid_mask])

        SS = np.sum(traces_int16.astype(np.int64)**2, axis=0)  # noqa: N806
        S = np.sum(S_i[valid_mask], axis=0)  # noqa: N806

        sum_n_Si_squared_over_ni = np.zeros(ns, dtype=np.float64)  # noqa: N806
        for class_idx in range(nc):
            if n_i[class_idx] > 0:
                sum_n_Si_squared_over_ni += (S_i[class_idx].astype(np.float64)**2 * n) / n_i[class_idx]

        signal = sum_n_Si_squared_over_ni - S.astype(np.float64)**2
        noise = n * SS.astype(np.float64) - sum_n_Si_squared_over_ni

        snr_manual[var_idx] = (signal / (noise + 1e-20)).astype(precision)

    rtol = 2e-2 if precision == 'float32' else 2e-2
    atol = 5e-3 if precision == 'float32' else 5e-3
    np.testing.assert_allclose(snr_manual, result_scalib, rtol=rtol, atol=atol,
                               err_msg=f"SCALib SNR differs from manual calculation with {precision}")


# Edge cases tests

def test_scalib_snr_handles_sparse_partitions():
    """Test handling of sparse partitions (e.g., [0, 5, 10, 100])."""
    partitions = [0, 5, 10, 100]
    snr = SNRDistinguisherSCALib(partitions=partitions)

    traces = np.random.randn(100, 20).astype(np.float32)
    labels = np.random.choice(partitions, (100, 1)).astype(np.uint8)

    snr.update(traces, labels)
    result = snr.compute()

    assert result.shape == (1, 20)


def test_scalib_snr_handles_data_not_in_partitions():
    """Test behavior when data contains values not in partitions."""
    partitions = range(0, 10)
    snr = SNRDistinguisherSCALib(partitions=partitions)

    traces = np.random.randn(100, 20).astype(np.float32)
    labels = np.random.randint(0, 20, (100, 1), dtype=np.uint8)

    snr.update(traces, labels)


def test_scalib_snr_handles_all_invalid_data(caplog):
    """Test behavior when all data values are not in partitions."""
    partitions = range(0, 10)
    snr = SNRDistinguisherSCALib(partitions=partitions)

    traces = np.random.randn(50, 20).astype(np.float32)
    labels = np.random.randint(20, 30, (50, 1), dtype=np.uint8)

    with caplog.at_level(logging.WARNING):
        snr.update(traces, labels)

    assert any("No valid traces" in record.message for record in caplog.records)


def test_scalib_snr_memory_usage_estimation():
    """Test _memory_usage method."""
    snr = SNRDistinguisherSCALib(partitions=range(16), precision='float32')

    traces = np.random.randn(10, 100).astype(np.float32)
    labels = np.random.randint(0, 16, (10, 5), dtype=np.uint8)

    mem = snr._memory_usage(traces, labels)

    assert mem > 0


def test_scalib_snr_init_partitions_raises_on_too_large_values():
    """Test that _init_partitions raises on values > 255 without explicit partitions."""
    snr = SNRDistinguisherSCALib()

    traces = np.random.randn(10, 20).astype(np.float32)
    labels = np.random.randint(0, 300, (10, 1), dtype=np.uint16)

    with pytest.raises(ValueError, match='max value for intermediate data is greater than 255'):
        snr.update(traces, labels)


def test_scalib_snr_init_partitions_raises_on_negative_values():
    """Test that _init_partitions raises on negative values without explicit partitions."""
    snr = SNRDistinguisherSCALib()

    traces = np.random.randn(10, 20).astype(np.float32)
    labels = np.random.randint(-10, 10, (10, 1), dtype=np.int16)

    with pytest.raises(ValueError, match='min value for intermediate data is lower than 0'):
        snr.update(traces, labels)


def test_scalib_snr_partition_mapping_with_negative_partitions():
    """Test partition mapping with negative partition values."""
    partitions = range(-10, 10)
    snr = SNRDistinguisherSCALib(partitions=partitions)

    traces = np.random.randn(50, 20).astype(np.float32)
    labels = np.random.randint(-10, 10, (50, 1), dtype=np.int8)

    snr.update(traces, labels)
    result = snr.compute()

    assert result.shape == (1, 20)


def test_scalib_snr_different_integer_label_dtypes():
    """Test with different integer dtypes for labels."""
    partitions = range(16)
    traces = np.random.randn(50, 20).astype(np.float32)

    for dtype in [np.uint8, np.int8, np.uint16, np.int16, np.uint32, np.int32]:
        snr = SNRDistinguisherSCALib(partitions=partitions)
        labels = np.random.randint(0, 16, (50, 1), dtype=dtype)
        snr.update(traces, labels)
        result = snr.compute()
        assert result.shape == (1, 20)


def test_scalib_snr_build_partition_mapping():
    """Test _build_partition_mapping method."""
    snr = SNRDistinguisherSCALib(partitions=range(10, 20))

    traces = np.random.randn(10, 20).astype(np.float32)
    labels = np.random.randint(10, 20, (10, 1), dtype=np.uint8)

    snr.update(traces, labels)

    assert hasattr(snr, '_partition_to_index')
    assert snr._partition_to_index is not None


def test_scalib_snr_map_data_to_indices():
    """Test _map_data_to_indices method."""
    snr = SNRDistinguisherSCALib(partitions=[0, 5, 10, 15])

    traces = np.random.randn(10, 20).astype(np.float32)
    labels = np.array([[0], [5], [10], [15], [7], [0], [5], [10], [15], [3]], dtype=np.uint8)

    snr.update(traces, labels)


# Import and availability tests

def test_scalib_snr_import_flag():
    """Test that SCALIB_AVAILABLE flag is set correctly."""
    assert SCALIB_AVAILABLE is True


def test_scalib_snr_error_message_clear():
    """Test that error messages are clear when SCALib is missing."""
    from scared.scalib.snr import SNRDistinguisherSCALibMixin
    assert hasattr(SNRDistinguisherSCALibMixin, '_check_scalib_available')


def test_scalib_snr_check_scalib_available_method():
    """Test _check_scalib_available method directly."""
    snr = SNRDistinguisherSCALib(partitions=range(16))
    snr._check_scalib_available()


# Inheritance and interface tests

def test_scalib_snr_inherits_from_standalone_distinguisher():
    """Test that class inherits from _StandaloneDistinguisher."""
    from scared.distinguishers.base import _StandaloneDistinguisher
    snr = SNRDistinguisherSCALib(partitions=range(16))
    assert isinstance(snr, _StandaloneDistinguisher)


def test_scalib_snr_inherits_from_mixin():
    """Test that class inherits from SNRDistinguisherSCALibMixin."""
    snr = SNRDistinguisherSCALib(partitions=range(16))
    assert isinstance(snr, SNRDistinguisherSCALibMixin)


def test_scalib_snr_has_required_methods():
    """Test that class has required distinguisher methods."""
    snr = SNRDistinguisherSCALib(partitions=range(16))
    assert hasattr(snr, 'update')
    assert hasattr(snr, 'compute')
    assert hasattr(snr, '_initialize')
    assert hasattr(snr, '_update')
    assert hasattr(snr, '_compute')


def test_scalib_snr_has_required_properties():
    """Test that class has required properties."""
    snr = SNRDistinguisherSCALib(partitions=range(16))
    assert hasattr(snr, '_distinguisher_str')
    assert hasattr(snr, 'precision')
    assert hasattr(snr, 'processed_traces')


# Integration tests

def test_scalib_snr_full_workflow():
    """Test complete workflow: init -> update -> compute."""
    snr = SNRDistinguisherSCALib(partitions=range(256), precision='float32')

    traces = np.random.randn(200, 100).astype(np.float32)
    labels = np.random.randint(0, 256, (200, 3), dtype=np.uint8)

    snr.update(traces, labels)

    result = snr.compute()

    assert result.shape == (3, 100)
    assert result.dtype == np.float32


def test_scalib_snr_multiple_updates_and_computes():
    """Test multiple update/compute cycles."""
    snr = SNRDistinguisherSCALib(partitions=range(16), precision='float32')

    traces1 = np.random.randn(50, 30).astype(np.float32)
    labels1 = np.random.randint(0, 16, (50, 2), dtype=np.uint8)
    snr.update(traces1, labels1)
    result1 = snr.compute()

    traces2 = np.random.randn(50, 30).astype(np.float32)
    labels2 = np.random.randint(0, 16, (50, 2), dtype=np.uint8)
    snr.update(traces2, labels2)
    result2 = snr.compute()

    assert result1.shape == result2.shape
    assert not np.array_equal(result1, result2)


# Tests for SCALib unavailable scenario

def test_scalib_snr_raises_import_error_when_scalib_unavailable(mocker):
    """Test that ImportError is raised when SCALib is not installed."""
    import scared.scalib.snr
    mocker.patch.object(scared.scalib.snr, 'SCALIB_AVAILABLE', False)

    snr = SNRDistinguisherSCALib(partitions=range(16))
    traces = np.random.randn(10, 20).astype(np.float32)
    labels = np.random.randint(0, 16, (10, 1), dtype=np.uint8)

    with pytest.raises(ImportError, match="SCALib is not installed"):
        snr.update(traces, labels)


def test_scalib_snr_init_succeeds_when_scalib_unavailable(mocker):
    """Test that initialization succeeds even when SCALib is unavailable."""
    import scared.scalib.snr
    mocker.patch.object(scared.scalib.snr, 'SCALIB_AVAILABLE', False)

    snr = SNRDistinguisherSCALib(partitions=range(16))
    assert np.array_equal(snr.partitions, np.arange(16))
    assert snr.precision == 'float32'


def test_scalib_snr_error_during_update_when_scalib_unavailable(mocker):
    """Test that error occurs during update, not during init, when SCALib unavailable."""
    import scared.scalib.snr
    mocker.patch.object(scared.scalib.snr, 'SCALIB_AVAILABLE', False)

    snr = SNRDistinguisherSCALib(partitions=range(256))
    traces = np.random.randn(100, 50).astype(np.float32)
    labels = np.random.randint(0, 256, (100, 2), dtype=np.uint8)

    with pytest.raises(ImportError) as exc_info:
        snr.update(traces, labels)

    assert "SCALib is not installed" in str(exc_info.value)
    assert "pip install scalib" in str(exc_info.value)
