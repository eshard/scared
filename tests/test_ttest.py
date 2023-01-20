from .context import scared  # noqa: F401
import pytest
import numpy as np


@pytest.fixture
def ths_1():
    shape = (2000, 1000)
    sample = np.random.randint(0, 256, (shape[1],), dtype='uint8')
    plain = np.random.randint(0, 256, (16), dtype='uint8')
    samples = np.array([sample for i in range(shape[0])], dtype='uint8')
    plaintext = np.array([plain for i in range(shape[0])], dtype='uint8')
    return scared.traces.formats.read_ths_from_ram(samples=samples, plaintext=plaintext)


@pytest.fixture
def ths_2():
    shape = (2000, 1000)
    samples = np.random.randint(0, 256, shape, dtype='uint8')
    plaintext = np.random.randint(0, 256, (shape[0], 16), dtype='uint8')
    return scared.traces.formats.read_ths_from_ram(samples=samples, plaintext=plaintext)


def test_ttest_analysis_raises_exception_if_invalid_precision_is_passed():
    with pytest.raises(TypeError):
        scared.TTestAnalysis(precision='foo')
    with pytest.raises(ValueError):
        scared.TTestAnalysis(precision='int8')


def test_ttest_analysis_initialize_two_ttest_accumulator(ths_1, ths_2):
    cont = scared.TTestContainer(ths_1, ths_2)
    analysis = scared.TTestAnalysis()
    analysis.run(cont)
    assert analysis.accumulators is not None
    assert len(analysis.accumulators) == 2
    assert isinstance(analysis.accumulators[0], scared.TTestThreadAccumulator)
    assert analysis.accumulators[0].precision == 'float32'
    assert isinstance(analysis.accumulators[1], scared.TTestThreadAccumulator)
    assert analysis.accumulators[1].precision == 'float32'


def test_accumulator_update_raises_exception_if_invalid_traces():
    with pytest.raises(TypeError):
        accu = scared.TTestThreadAccumulator(precision='float32')
        accu.update('foo')


def test_ttest_accumulator_update_its_accumulator(ths_1):
    accu = scared.TTestThreadAccumulator(precision='float32')
    accu.update(ths_1.samples[:100, :])
    traces = ths_1.samples[:100, :].astype('float32')
    assert np.array_equal(accu.sum, np.sum(traces, axis=0))
    assert np.array_equal(accu.sum_squared, np.sum(traces ** 2, axis=0))
    assert accu.processed_traces == 100
    accu.update(ths_1.samples[100:151, :])
    traces = ths_1.samples[:151, :].astype('float32')
    assert np.array_equal(accu.sum, np.sum(traces, axis=0))
    assert np.array_equal(accu.sum_squared, np.sum(traces ** 2, axis=0))
    assert accu.processed_traces == 151


def test_ttest_accumulator_raises_is_run_with_not_container():
    d = scared.TTestThreadAccumulator('float32')
    with pytest.raises(ValueError, match='Please give a Container'):
        d.run('foo')


def test_ttest_accumulator_raises_is_start_with_not_container():
    d = scared.TTestThreadAccumulator('float32')
    with pytest.raises(ValueError, match='Please give a Container'):
        d.start('foo')
        d.join()


def test_ttest_accumulator_compute_raise_exception_if_no_trace_are_processed_or_state_not_initialized():
    d = scared.TTestThreadAccumulator('float32')
    with pytest.raises(scared.TTestError):
        d.compute()


def test_ttest_accumulator_run(ths_1):
    accu = scared.TTestThreadAccumulator(precision='float32')
    cont = scared.Container(ths_1[:100])
    accu.run(cont)
    traces = ths_1.samples[:100, :].astype('float32')
    assert np.array_equal(accu.sum, np.sum(traces, axis=0))
    assert np.array_equal(accu.sum_squared, np.sum(traces ** 2, axis=0))
    assert accu.processed_traces == 100


def test_ttest_accumulator_start(ths_1):
    accu = scared.TTestThreadAccumulator(precision='float32')
    cont = scared.Container(ths_1[:100])
    accu.start(cont)
    traces = ths_1.samples[:100, :].astype('float32')
    accu.join()
    assert np.array_equal(accu.sum, np.sum(traces, axis=0))
    assert np.array_equal(accu.sum_squared, np.sum(traces ** 2, axis=0))
    assert accu.processed_traces == 100


def test_ttest_accumulator_raises_if_start_and_already_alive(ths_1):
    accu = scared.TTestThreadAccumulator(precision='float32')
    cont = scared.Container(ths_1)
    accu.start(cont)
    with pytest.raises(RuntimeError, match='Thread is already running. Use the'):
        accu.start(cont)


def test_ttest_accumulator_compute(ths_1, ths_2):
    accu = scared.TTestThreadAccumulator(precision='float32')
    accu.update(ths_1.samples[:100, :])
    traces = ths_1.samples[:100, :].astype('float32')
    assert np.array_equal(accu.sum, np.sum(traces, axis=0))
    assert np.array_equal(accu.sum_squared, np.sum(traces ** 2, axis=0))
    assert accu.processed_traces == 100

    accu.compute()
    assert np.array_equal(accu.mean, np.sum(traces, axis=0) / 100)
    assert np.array_equal(accu.var, np.sum(traces ** 2, axis=0) / 100 - (np.sum(traces, axis=0) / 100) ** 2)


def test_ttest_analysis_run(ths_1, ths_2):
    cont = scared.TTestContainer(ths_1, ths_2)
    analysis = scared.TTestAnalysis(precision='float64')
    analysis.run(cont)

    t_1 = ths_1.samples[:].astype('float64')
    t_2 = ths_2.samples[:].astype('float64')

    mean_1 = np.sum(t_1, axis=0) / len(ths_1)
    mean_2 = np.sum(t_2, axis=0) / len(ths_2)
    var_1 = (np.sum(t_1 ** 2, axis=0) / len(ths_1) - mean_1 ** 2) / len(ths_1)
    var_2 = (np.sum(t_2 ** 2, axis=0) / len(ths_2) - mean_2 ** 2) / len(ths_2)
    expected = (mean_1 - mean_2) / np.sqrt(var_1 + var_2)

    assert analysis.result is not None
    assert np.array_equal(expected, analysis.result)


def test_ttest_analysis_no_threads_after_run(ths_1, ths_2):
    cont = scared.TTestContainer(ths_1, ths_2)
    analysis = scared.TTestAnalysis(precision='float64')
    analysis.run(cont)

    assert analysis.accumulators[0]._thread is None
    assert analysis.accumulators[1]._thread is None


def test_ttest_analysis_run_raises_exception_if_container_not_ttest_container():
    analysis = scared.TTestAnalysis(precision='float64')
    with pytest.raises(TypeError):
        analysis.run('foo')


def test_ttest_analysis_run_with_frame(ths_1, ths_2):
    cont = scared.TTestContainer(ths_1, ths_2, frame=slice(0, 100))
    analysis = scared.TTestAnalysis(precision='float64')
    analysis.run(cont)

    t_1 = ths_1.samples[:, :100].astype('float64')
    t_2 = ths_2.samples[:, :100].astype('float64')

    mean_1 = np.sum(t_1, axis=0) / len(ths_1)
    mean_2 = np.sum(t_2, axis=0) / len(ths_2)
    var_1 = (np.sum(t_1 ** 2, axis=0) / len(ths_1) - mean_1 ** 2) / len(ths_1)
    var_2 = (np.sum(t_2 ** 2, axis=0) / len(ths_2) - mean_2 ** 2) / len(ths_2)
    expected = (mean_1 - mean_2) / np.sqrt(var_1 + var_2)

    assert analysis.result is not None
    assert np.array_equal(expected, analysis.result)


def test_ttest_analysis_run_with_preprocesses(ths_1, ths_2):
    cont = scared.TTestContainer(ths_1, ths_2, preprocesses=[scared.preprocesses.square])
    analysis = scared.TTestAnalysis(precision='float64')
    analysis.run(cont)

    t_1 = scared.preprocesses.square(ths_1.samples[:]).astype('float64')
    t_2 = scared.preprocesses.square(ths_2.samples[:]).astype('float64')

    mean_1 = np.sum(t_1, axis=0) / len(ths_1)
    mean_2 = np.sum(t_2, axis=0) / len(ths_2)
    var_1 = (np.sum(t_1 ** 2, axis=0) / len(ths_1) - mean_1 ** 2) / len(ths_1)
    var_2 = (np.sum(t_2 ** 2, axis=0) / len(ths_2) - mean_2 ** 2) / len(ths_2)
    expected = (mean_1 - mean_2) / np.sqrt(var_1 + var_2)

    assert analysis.result is not None
    assert np.array_equal(expected, analysis.result)


def test_ttest_analysis_run_twice(ths_1, ths_2):
    ths_1 = ths_1[:100]
    ths_2 = ths_2[:100]
    cont = scared.TTestContainer(ths_1, ths_2)
    analysis = scared.TTestAnalysis()
    analysis.run(cont)
    analysis.run(cont)

    assert analysis.accumulators[0].processed_traces == 2 * len(ths_1)
    assert analysis.accumulators[1].processed_traces == 2 * len(ths_2)

    t_1 = ths_1.samples[:]
    t_1 = np.vstack([t_1, t_1]).astype('float32')
    t_2 = ths_2.samples[:]
    t_2 = np.vstack([t_2, t_2]).astype('float32')

    sum_1 = np.sum(t_1, axis=0)
    sum_2 = np.sum(t_2, axis=0)
    mean_1 = sum_1 / len(t_1)
    mean_2 = sum_2 / len(t_2)
    sum_square_1 = np.sum(t_1 ** 2, axis=0)
    sum_square_2 = np.sum(t_2 ** 2, axis=0)
    var_1 = (sum_square_1 / len(t_1) - mean_1 ** 2) / len(t_1)
    var_2 = (sum_square_2 / len(t_2) - mean_2 ** 2) / len(t_2)
    expected = (mean_1 - mean_2) / np.sqrt(var_1 + var_2)

    assert np.array_equal(sum_1, analysis.accumulators[0].sum)
    assert np.array_equal(sum_2, analysis.accumulators[1].sum)

    assert np.array_equal(sum_square_1, analysis.accumulators[0].sum_squared)
    assert np.array_equal(sum_square_2, analysis.accumulators[1].sum_squared)

    assert analysis.result is not None
    assert np.array_equal(expected, analysis.result)


def test_exception_well_passed_to_analysis_level(ths_1, ths_2):

    class Batch():
        def __init__(self, data):
            self.data = data

        @property
        def samples(self):
            return self.data

        def __len__(self):
            return len(self.data)

    class DummyTTestContainer(scared.TTestContainer):
        def __init__(self, cont1, cont2):
            self.containers = [cont1, cont2]

    class DummyContainer(scared.Container):
        def __init__(self, batches, ths):
            self.inner_batches = batches
            self._ths = ths
            self.frame = None
            self.preprocesses = []

        def batches(self):
            return self.inner_batches

    cont1 = DummyContainer([Batch(np.random.rand(100, 1000)),
                            Batch(np.random.rand(100, 1000)),
                            Batch(['sdlkjfhskdjfhkjsdhf', 'smdkljfhsdkjfhkjfsdhfkjsh'])], ths_1)

    cont2 = DummyContainer([Batch(np.random.rand(100, 1000)),
                            Batch(np.random.rand(100, 1000)),
                            Batch(['dfyrthfghfghfghfh', 'dsfwvcxcgftutyujghjdsqs'])], ths_2)

    cont = DummyTTestContainer(cont1, cont2)
    analysis = scared.TTestAnalysis()
    with pytest.raises(TypeError, match="traces must be numpy ndarray, not <class 'list'>"):
        analysis.run(cont)


@pytest.fixture(scope='module')
def test_vectors_ttest_sum_precision():
    """Compute the same t-test accumulator in C/F order and float32/64 precision."""
    dataset = np.load('tests/samples/dataset_for_precision_errors.npz')

    ttests = {}
    for order in ['C', 'F']:
        for precision in ['float32', 'float64']:
            traces = np.asarray(dataset['samples'], order=order, dtype=precision)

            ttest = ttests[order, precision] = scared.ttest.TTestThreadAccumulator(precision)

            for _ in range(5):  # Process 5 times the same batch
                ttest.update(traces)
    return ttests


@pytest.mark.parametrize('precision', ['float32', 'float64'])
def test_ttest_update_order_independent(precision, test_vectors_ttest_sum_precision):
    """See issue https://gitlab.com/eshard/scared/-/issues/65 for details."""
    ttests = test_vectors_ttest_sum_precision
    for attribute in ['sum', 'sum_squared']:
        np.testing.assert_array_equal(getattr(ttests['F', precision], attribute),
                                      getattr(ttests['C', precision], attribute),
                                      err_msg=f'Difference between F/C inputs, for attribute {attribute} in precision {precision}')


@pytest.mark.parametrize('order', ['C', 'F'])
def test_cpa_update_acceptable_precision(order, test_vectors_ttest_sum_precision):
    """See issue https://gitlab.com/eshard/scared/-/issues/65 for details."""
    ttests = test_vectors_ttest_sum_precision
    for attribute in ['sum', 'sum_squared']:
        np.testing.assert_allclose(getattr(ttests[order, 'float32'], attribute),
                                   getattr(ttests[order, 'float64'], attribute),
                                   atol=0,
                                   rtol=1e-6,
                                   err_msg=f'Float32 accumulators too far from Float64 reference for attribute {attribute}')
