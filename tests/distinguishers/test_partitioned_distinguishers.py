from ..context import scared
import pytest
import numpy as np


@pytest.fixture
def partitioned_datas():
    datas = np.load('tests/samples/tests_partitioned_distinguishers.npz')
    for k, v in datas.items():
        setattr(datas, k, v)
    return datas


class DumbPartDistinguisher(scared.PartitionedDistinguisher):
    def _compute(self):
        pass

    @property
    def _distinguisher_str(self):
        return 'DumbPart'


def test_partitioned_distinguishers_analyses_raises_exception_if_incorrect_partition():

    with pytest.raises(TypeError):
        DumbPartDistinguisher(partitions='foo')
    with pytest.raises(TypeError):
        DumbPartDistinguisher(partitions={})
    with pytest.raises(TypeError):
        DumbPartDistinguisher(partitions=[1, 23])
    with pytest.raises(ValueError):
        DumbPartDistinguisher(partitions=np.array([1.2, 3]))


def test_partitioned_analyses_raises_exception_at_init_if_partitions_is_none_and_value_gt_than_255():
    d = DumbPartDistinguisher()

    with pytest.raises(ValueError):
        d.update(traces=np.random.randint(0, 255, (500, 200), dtype='int16'), data=np.random.randint(0, 3000, (500, 16), dtype='uint16'))


def test_partitioned_analyses_init_initialize_partitions_if_none():
    d = DumbPartDistinguisher()

    d.update(
        traces=np.random.randint(0, 255, (500, 200), dtype='int16'),
        data=np.random.randint(0, 9, (500, 16), dtype='uint8')
    )

    assert d.partitions == range(9)

    d = DumbPartDistinguisher()

    d.update(
        traces=np.random.randint(0, 255, (500, 200), dtype='int16'),
        data=np.random.randint(0, 64, (500, 16), dtype='uint8')
    )

    assert d.partitions == range(64)

    d = DumbPartDistinguisher()

    d.update(
        traces=np.random.randint(0, 255, (500, 200), dtype='int16'),
        data=np.random.randint(0, 255, (500, 16), dtype='uint8')
    )

    assert d.partitions == range(256)


def test_partitioned_analyses_init_initialize_accumulators():
    d = DumbPartDistinguisher()

    d.update(
        traces=np.random.randint(0, 255, (500, 200), dtype='int16'),
        data=np.random.randint(0, 9, (500, 4096), dtype='uint8')
    )

    assert d.partitions == range(9)
    assert d.sum.shape == (200, 4096, 9)
    assert d.sum_square.shape == (200, 4096, 9)
    assert d.counters.shape == (4096, 9)


def test_partitioned_analyses_init_raises_error_if_accumulators_are_too_large_for_memory():
    d = DumbPartDistinguisher()
    traces = np.random.randint(0, 255, (500, 200000), dtype='uint8')
    data = np.random.randint(0, 255, (500, 40096), dtype='uint8')

    with pytest.raises(MemoryError):
        d.update(traces=traces, data=data)


def test_partitioned_analyses_update_raises_error_if_shapes_are_inconsistent():
    d = DumbPartDistinguisher()
    traces = np.random.randint(0, 255, (10, 200), dtype='uint8')
    data = np.random.randint(0, 8, (50, 64), dtype='uint8')

    with pytest.raises(ValueError):
        d.update(traces=traces, data=data)

    traces = np.random.randint(0, 255, (50, 200), dtype='uint8')
    data = np.random.randint(0, 8, (50, 64), dtype='uint8')

    d.update(traces=traces, data=data)

    with pytest.raises(ValueError):
        d.update(
            traces=np.random.randint(0, 255, (10, 20), dtype='uint8'),
            data=np.random.randint(0, 255, (10, 64), dtype='uint8')
        )

    with pytest.raises(ValueError):
        d.update(
            traces=np.random.randint(0, 255, (10, 200), dtype='uint8'),
            data=np.random.randint(0, 255, (10, 62), dtype='uint8')
        )


def test_partitioned_analyses_update_accumulators(partitioned_datas):
    d = scared.ANOVADistinguisher()

    d.update(traces=partitioned_datas.traces_1, data=partitioned_datas.data_1)
    assert partitioned_datas.processed_traces_1 == d.processed_traces
    assert np.array_equal(partitioned_datas.counters_1, d.counters)
    assert np.array_equal(partitioned_datas.e1_1.swapaxes(0, 1), d.sum)
    assert np.array_equal(partitioned_datas.e2_1.swapaxes(0, 1), d.sum_square)
    d.compute()
    d.update(traces=partitioned_datas.traces_2, data=partitioned_datas.data_2)
    assert partitioned_datas.processed_traces_2 == d.processed_traces
    assert np.array_equal(partitioned_datas.counters_2, d.counters)
    assert np.array_equal(partitioned_datas.e1_2.swapaxes(0, 1), d.sum)
    assert np.array_equal(partitioned_datas.e2_2.swapaxes(0, 1), d.sum_square)


def test_anova_compute_raises_exception_if_no_accumulation(partitioned_datas):
    d = scared.ANOVADistinguisher()

    with pytest.raises(scared.DistinguisherError):
        d.compute()


def test_anova_compute(partitioned_datas):
    d = scared.ANOVADistinguisher()

    d.update(traces=partitioned_datas.traces_1, data=partitioned_datas.data_1)
    d.update(traces=partitioned_datas.traces_2, data=partitioned_datas.data_2)

    results = d.compute()
    assert np.array_equal(partitioned_datas.result_anova, results)


def test_nicv_compute_raises_exception_if_no_accumulation(partitioned_datas):
    d = scared.NICVDistinguisher()

    with pytest.raises(scared.DistinguisherError):
        d.compute()


def test_nicv_compute(partitioned_datas):
    d = scared.NICVDistinguisher()

    d.update(traces=partitioned_datas.traces_1, data=partitioned_datas.data_1)
    d.update(traces=partitioned_datas.traces_2, data=partitioned_datas.data_2)

    results = d.compute()
    assert np.array_equal(partitioned_datas.result_nicv, results)


def test_snr_compute_raises_exception_if_no_accumulation(partitioned_datas):
    d = scared.SNRDistinguisher()

    with pytest.raises(scared.DistinguisherError):
        d.compute()


def test_snr_compute(partitioned_datas):
    d = scared.SNRDistinguisher()

    d.update(traces=partitioned_datas.traces_1, data=partitioned_datas.data_1)
    d.update(traces=partitioned_datas.traces_2, data=partitioned_datas.data_2)

    results = d.compute()
    assert np.array_equal(partitioned_datas.result_snr, results)


def test_mia_distinguisher_raises_exceptions_if_partitions_is_incorrect():
    with pytest.raises(TypeError):
        scared.MIADistinguisher(partitions='foo')
    with pytest.raises(TypeError):
        scared.MIADistinguisher(partitions={})
    with pytest.raises(TypeError):
        scared.MIADistinguisher(partitions=[1, 23])
    with pytest.raises(ValueError):
        scared.MIADistinguisher(partitions=np.array([1.2, 3]))


def test_mia_distinguisher_raises_exceptions_if_bins_number_is_incorrect():
    with pytest.raises(TypeError):
        scared.MIADistinguisher(bins_number='foo')
    with pytest.raises(TypeError):
        scared.MIADistinguisher(bins_number={})
    with pytest.raises(TypeError):
        scared.MIADistinguisher(bins_number=[1, 23])
    with pytest.raises(TypeError):
        scared.MIADistinguisher(bins_number=np.array([1.2, 3]))
    with pytest.raises(TypeError):
        scared.MIADistinguisher(bins_number=38.45)


def test_mia_analyses_raises_exception_at_init_if_partitions_is_none_and_value_gt_than_255():
    d = scared.MIADistinguisher()

    with pytest.raises(ValueError):
        d.update(traces=np.random.randint(0, 255, (500, 200), dtype='int16'), data=np.random.randint(0, 3000, (500, 16), dtype='uint16'))


def test_mia_analyses_init_initialize_partitions_if_none():
    d = scared.MIADistinguisher()

    d.update(
        traces=np.random.randint(0, 255, (10, 20), dtype='uint8'),
        data=np.random.randint(0, 9, (10, 16), dtype='uint8')
    )

    assert d.partitions == range(9)

    d = scared.MIADistinguisher()

    d.update(
        traces=np.random.randint(0, 255, (10, 20), dtype='uint8'),
        data=np.random.randint(0, 64, (10, 16), dtype='uint8')
    )

    assert d.partitions == range(64)

    d = scared.MIADistinguisher()

    d.update(
        traces=np.random.randint(0, 255, (10, 20), dtype='uint8'),
        data=np.random.randint(0, 255, (10, 16), dtype='uint8')
    )

    assert d.partitions == range(256)


def test_mia_analyses_first_update_initialize_parameters():
    d = scared.MIADistinguisher()

    assert d.bin_edges is None
    assert d.y_window is None

    traces = np.random.randint(0, 255, (50, 20), dtype='int16')
    d.update(
        traces=traces,
        data=np.random.randint(0, 9, (50, 16), dtype='uint8')
    )

    assert d.partitions == range(9)
    assert d.bins_number == 128
    assert d.bin_edges is not None
    assert d.y_window is not None
    assert d.y_window == (np.min(traces), np.max(traces))
    assert np.array_equal(d.bin_edges, np.linspace(np.min(traces), np.max(traces), 129))
    assert d.accumulators.shape == (20, 128, 9, 16)


def test_mia_analyses_init_raises_error_if_accumulators_are_too_large_for_memory():
    d = scared.MIADistinguisher()
    traces = np.random.randint(0, 255, (500, 200000), dtype='uint8')
    data = np.random.randint(0, 255, (500, 40096), dtype='uint8')

    with pytest.raises(MemoryError):
        d.update(traces=traces, data=data)


def test_mia_analyses_update_raises_error_if_shapes_are_inconsistent():
    d = scared.MIADistinguisher()
    traces = np.random.randint(0, 255, (10, 200), dtype='uint8')
    data = np.random.randint(0, 8, (50, 64), dtype='uint8')

    with pytest.raises(ValueError):
        d.update(traces=traces, data=data)

    traces = np.random.randint(0, 255, (50, 200), dtype='uint8')
    data = np.random.randint(0, 8, (50, 64), dtype='uint8')

    d.update(traces=traces, data=data)

    with pytest.raises(ValueError):
        d.update(
            traces=np.random.randint(0, 255, (10, 20), dtype='uint8'),
            data=np.random.randint(0, 255, (10, 64), dtype='uint8')
        )

    with pytest.raises(ValueError):
        d.update(
            traces=np.random.randint(0, 255, (10, 200), dtype='uint8'),
            data=np.random.randint(0, 255, (10, 62), dtype='uint8')
        )


def test_mia_analyses_update_accumulators(partitioned_datas):
    d = scared.MIADistinguisher(bins_number=256)

    d.y_window = 1
    d.bin_edges = partitioned_datas.mia_bin_edges

    d.update(traces=partitioned_datas.traces_1, data=partitioned_datas.data_1)

    assert partitioned_datas.processed_traces_1 == d.processed_traces
    expected = partitioned_datas.mia_first_update_accu.swapaxes(0, 1).swapaxes(1, -1)
    assert np.array_equal(expected, d.accumulators)

    d.compute()
    d.update(traces=partitioned_datas.traces_2, data=partitioned_datas.data_2)
    assert partitioned_datas.processed_traces_2 == d.processed_traces
    expected = partitioned_datas.mia_second_update_accu.swapaxes(0, 1).swapaxes(1, -1)
    assert np.array_equal(expected, d.accumulators)


def test_mia_analyses_update_accumulators_int16_traces(partitioned_datas):
    d = scared.MIADistinguisher()

    d.y_window = 1
    d.bin_edges = partitioned_datas.mia_int16_bin_edges

    d.update(
        traces=partitioned_datas.mia_int16_traces_1,
        data=partitioned_datas.mia_int16_data_1
    )

    expected = partitioned_datas.mia_int16_update_1.swapaxes(0, -1).swapaxes(1, 2).swapaxes(2, 3)
    assert np.array_equal(expected, d.accumulators)

    d.compute()
    d.update(traces=partitioned_datas.mia_int16_traces_2, data=partitioned_datas.mia_int16_data_2)
    expected = partitioned_datas.mia_int16_update_2.swapaxes(0, -1).swapaxes(1, 2).swapaxes(2, 3)
    assert np.array_equal(expected, d.accumulators)


def test_mia_compute_int_16_traces(partitioned_datas):
    d = scared.MIADistinguisher(bin_edges=partitioned_datas.mia_int16_bin_edges)

    d.y_window = 1

    d.update(traces=partitioned_datas.mia_int16_traces_1, data=partitioned_datas.mia_int16_data_1)
    d.update(traces=partitioned_datas.mia_int16_traces_2, data=partitioned_datas.mia_int16_data_2)

    expected = partitioned_datas.mia_int16_result
    res = d.compute()
    assert np.allclose(expected, res, rtol=1e-05)


def test_mia_compute_int_8_traces(partitioned_datas):
    d = scared.MIADistinguisher()

    d.y_window = 1
    d.bin_edges = partitioned_datas.mia_int8_bin_edges

    d.update(traces=partitioned_datas.traces_1, data=partitioned_datas.data_1)
    d.update(traces=partitioned_datas.traces_2, data=partitioned_datas.data_2)

    expected = partitioned_datas.mia_int8_result
    res = d.compute()
    assert np.allclose(expected, res, rtol=1e-05)


_bin_edges_fail = {
    'bin number': 12,
    'np method': 'auto',
    'No edges': None
}


@pytest.fixture(params=_bin_edges_fail.keys())
def bin_edges_fail_key(request):
    return request.param


def test_mia_with_invalid_bin_edges_raises_exception(bin_edges_fail_key):
    bin_edges = _bin_edges_fail[bin_edges_fail_key]
    with pytest.raises(TypeError):
        d = scared.MIADistinguisher()
        d.bin_edges = bin_edges

    with pytest.raises(TypeError):
        d = scared.MIADistinguisher(bin_edges=bin_edges)
        d.bin_edges = bin_edges
