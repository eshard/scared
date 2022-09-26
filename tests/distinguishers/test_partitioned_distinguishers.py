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


def test_partitions_well_set():
    dpd = DumbPartDistinguisher(partitions=np.array([1, 2, 3], dtype='uint32'))
    assert np.array_equal(dpd.partitions, np.array([1, 2, 3]))
    dpd = DumbPartDistinguisher(partitions=np.array([1, 2, 3], dtype='int32'))
    assert np.array_equal(dpd.partitions, np.array([1, 2, 3]))
    dpd = DumbPartDistinguisher(partitions=[1, 2, 3])
    assert np.array_equal(dpd.partitions, np.array([1, 2, 3]))
    dpd = DumbPartDistinguisher(partitions=range(1, 4))
    assert np.array_equal(dpd.partitions, np.array([1, 2, 3]))


def test_partitioned_distinguishers_analyses_raises_exception_if_incorrect_partition():
    with pytest.raises(TypeError):
        DumbPartDistinguisher(partitions='foo')
    with pytest.raises(TypeError):
        DumbPartDistinguisher(partitions={})
    with pytest.raises(ValueError):
        DumbPartDistinguisher(partitions=np.array([1.2, 3]))


def test_partitioned_analyses_raises_exception_at_init_if_partitions_is_none_and_value_gt_than_255():
    d = DumbPartDistinguisher()
    with pytest.raises(ValueError, match='max value for intermediate data is greater than 255'):
        d.update(traces=np.random.randint(0, 255, (500, 200), dtype='int16'), data=np.random.randint(0, 3000, (500, 16), dtype='uint16'))


def test_partitioned_analyses_raises_exception_at_init_if_partitions_is_none_and_value_lt_than_0():
    d = DumbPartDistinguisher()
    with pytest.raises(ValueError, match='min value for intermediate data is lower than 0'):
        d.update(traces=np.random.randint(0, 255, (500, 200), dtype='int16'), data=np.random.randint(-3000, 1, (500, 16), dtype='int16'))


def test_partitioned_analyses_init_initialize_partitions_if_none():
    d = DumbPartDistinguisher()
    d.update(traces=np.random.randint(0, 255, (500, 200), dtype='int16'),
             data=np.random.randint(0, 9, (500, 16), dtype='uint8'))
    assert np.array_equal(d.partitions, np.arange(9))

    d = DumbPartDistinguisher()
    d.update(traces=np.random.randint(0, 255, (500, 200), dtype='int16'),
             data=np.random.randint(0, 64, (500, 16), dtype='uint8'))
    assert np.array_equal(d.partitions, np.arange(64))

    d = DumbPartDistinguisher()
    d.update(traces=np.random.randint(0, 255, (500, 200), dtype='int16'),
             data=np.random.randint(0, 255, (500, 16), dtype='uint8'))
    assert np.array_equal(d.partitions, np.arange(256))


def test_partitioned_analyses_init_initialize_accumulators():
    d = DumbPartDistinguisher()
    d.update(traces=np.random.randint(0, 255, (500, 200), dtype='int16'),
             data=np.random.randint(0, 9, (500, 4096), dtype='uint8'))

    assert np.array_equal(d.partitions, np.arange(9))
    assert d.sum.shape == (200, 4096, 9)
    assert d.sum_square.shape == (200, 4096, 9)
    assert d.counters.shape == (4096, 9)


def test_data_to_partition_index_function():
    traces = np.random.randint(0, 256, (1, 3), dtype='uint8')
    data = np.random.randint(0, 18, (1, 256), dtype='uint8')
    partitions = range(0, 18, 2)
    d = DumbPartDistinguisher(partitions=partitions)
    d._initialize(traces, data)
    new_data = d._data_to_partition_index(data)

    assert np.array_equal(d.partitions, np.arange(0, 18, 2))
    assert new_data.shape == data.shape

    for d, nd in zip(data.flatten(), new_data.flatten()):
        idx = np.where(d == partitions)[0]
        idx = -1 if len(idx) == 0 else idx[0]
        assert nd == idx


def test_partitioned_analyses_init_initialize_accumulators_with_sparse_partitions():
    d = DumbPartDistinguisher(partitions=range(0, 18, 2))
    d.update(traces=np.random.randint(0, 255, (500, 200), dtype='int16'),
             data=2 * np.random.randint(0, 9, (500, 4096), dtype='uint8'))

    assert np.array_equal(d.partitions, np.arange(0, 18, 2))
    assert d.sum.shape == (200, 4096, 9)
    assert d.sum_square.shape == (200, 4096, 9)
    assert d.counters.shape == (4096, 9)


def test_partitioned_analyses_init_raises_error_if_accumulators_are_too_large_for_memory():
    d = DumbPartDistinguisher()
    traces = np.random.randint(0, 255, (500, 2000000), dtype='uint8')
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


def test_partitioned_analyses_update_accumulators_method_1(partitioned_datas):
    d = scared.ANOVADistinguisher()
    d._accumulate_core_2 = d._accumulate_core_1
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


def test_partitioned_analyses_update_accumulators_method_2(partitioned_datas):
    d = scared.ANOVADistinguisher()
    d._accumulate_core_1 = d._accumulate_core_2
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


def test_partitioned_analyses_update_accumulators_method_selection(partitioned_datas):
    d = scared.ANOVADistinguisher(partitions=range(32))
    d.update(traces=partitioned_datas.traces_1, data=partitioned_datas.data_1)
    assert not hasattr(d, '_timings')

    d = scared.ANOVADistinguisher(partitions=range(9))
    d.update(traces=partitioned_datas.traces_1, data=partitioned_datas.data_1)
    assert hasattr(d, '_timings')
    assert d._timings[0] > 0
    d.update(traces=partitioned_datas.traces_1, data=partitioned_datas.data_1)
    assert d._timings[1] > 0


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
