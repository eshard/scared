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


def test_partitioned_analyses_raises_if_max_partitions_gte_2p16():
    with pytest.raises(ValueError, match='partition values must be in '):
        DumbPartDistinguisher(partitions=range(2**16 - 10, 2**16 + 10))


def test_partitioned_analyses_raises_if_max_partitions_lte_2p16():
    with pytest.raises(ValueError, match='partition values must be in '):
        DumbPartDistinguisher(partitions=range(-2**16 - 10, -2**16 + 10))


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


def test_data_to_partition_index_function_with_negative_partitions():
    traces = np.random.randint(0, 256, (1, 3), dtype='uint8')
    data = np.random.randint(-10, 18, (1, 256), dtype='int8')
    partitions = range(-10, 18, 2)
    d = DumbPartDistinguisher(partitions=partitions)
    d._initialize(traces, data)
    new_data = d._data_to_partition_index(data)

    assert np.array_equal(d.partitions, np.arange(-10, 18, 2))
    assert new_data.shape == data.shape

    for d, nd in zip(data.flatten(), new_data.flatten()):
        idx = np.where(d == partitions)[0]
        idx = -1 if len(idx) == 0 else idx[0]
        assert nd == idx


@pytest.mark.parametrize('dtype', ['complex', 'float32', 'bool'])
def test_data_to_partition_index_raises_if_not_integer(dtype):
    traces = np.random.randint(0, 256, (1, 3), dtype='uint8')
    data = np.random.randint(0, 2, (1, 256), dtype='uint8')
    data = data.astype(dtype)
    partitions = range(0, 18, 2)
    d = DumbPartDistinguisher(partitions=partitions)
    with pytest.raises(TypeError, match='data dtype for partitioned distinguisher, including MIA and Template, must be an integer dtype, not'):
        d.update(traces=traces, data=data)


@pytest.mark.parametrize('dtype', ['uint8', 'int8', 'uint16', 'int16', 'uint32', 'int32', 'uint64', 'int64', int])
def test_data_to_partition_index_works_with_all_integer_dtypes(dtype):
    traces = np.random.randint(0, 256, (1, 3), dtype='uint8')
    data = np.random.randint(0, 2, (1, 256), dtype='uint8')
    data = data.astype(dtype)
    partitions = range(0, 18, 2)
    d = DumbPartDistinguisher(partitions=partitions)
    d.update(traces=traces, data=data)


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

    with pytest.raises(scared.distinguishers.DistinguisherError, match='This analysis will probably need more than 90'):
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


def test_anova_compute_raises_exception_if_no_accumulation():
    d = scared.ANOVADistinguisher()
    with pytest.raises(scared.DistinguisherError):
        d.compute()


def test_anova_compute(partitioned_datas):
    d = scared.ANOVADistinguisher()

    d.update(traces=partitioned_datas.traces_1, data=partitioned_datas.data_1)
    d.update(traces=partitioned_datas.traces_2, data=partitioned_datas.data_2)

    results = d.compute()
    assert np.allclose(partitioned_datas.result_anova, results)  # equal changed in allclose since move to numpy 2, diff is <5e-74


def test_nicv_compute_raises_exception_if_no_accumulation():
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


@pytest.fixture(params=[1, 2], scope='module')
def test_vectors_partitioned_sum_precision(request):
    """Compute the same Partitioned update in C/F order, float32/64 precision, and both accumulate_core implementations."""
    dataset = np.load('tests/samples/dataset_for_precision_errors.npz')

    distinguishers = {}
    for order in ['C', 'F']:
        for precision in ['float32', 'float64']:
            traces = np.asarray(dataset['samples'], order=order, dtype=precision)
            data = np.asarray(scared.Monobit(0)(dataset['data'][:, 2:3]), order=order)

            anova = distinguishers[order, precision] = scared.distinguishers.ANOVADistinguisher(partitions=range(2), precision=precision)

            # Surcharge implementation
            if request.param == 1:
                anova._accumulate_core_2 = anova._accumulate_core_1
            elif request.param == 2:
                anova._accumulate_core_1 = anova._accumulate_core_2

            for _ in range(5):  # Process 5 times the same batch
                anova.update(traces, data)
    return distinguishers


@pytest.mark.parametrize('precision', ['float32', 'float64'])
def test_partitioned_update_order_independent(precision, test_vectors_partitioned_sum_precision):
    """See issue https://gitlab.com/eshard/scared/-/issues/65 for details."""
    distinguishers = test_vectors_partitioned_sum_precision
    for attribute in ['sum', 'sum_square']:
        np.testing.assert_array_equal(getattr(distinguishers['F', precision], attribute),
                                      getattr(distinguishers['C', precision], attribute),
                                      err_msg=f'Difference between F/C inputs, for attribute {attribute} in precision {precision}')


@pytest.mark.parametrize('order', ['C', 'F'])
def test_partitioned_update_acceptable_precision(order, test_vectors_partitioned_sum_precision):
    """See issue https://gitlab.com/eshard/scared/-/issues/65 for details."""
    distinguishers = test_vectors_partitioned_sum_precision
    for attribute in ['sum', 'sum_square']:
        np.testing.assert_allclose(getattr(distinguishers[order, 'float32'], attribute),
                                   getattr(distinguishers[order, 'float64'], attribute),
                                   atol=200,
                                   rtol=1e-6,
                                   err_msg=f'Float32 accumulators too far from Float64 reference for attribute {attribute}')
