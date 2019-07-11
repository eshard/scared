from .context import scared  # noqa: F401
import pytest
import numpy as np


@pytest.fixture
def ths_1():
    shape = (2000, 1000)
    sample = np.random.randint(0, 255, (1000,), dtype='uint8')
    plain = np.random.randint(0, 255, (16), dtype='uint8')
    samples = np.array([sample for i in range(shape[0])], dtype='uint8')
    plaintext = np.array([plain for i in range(shape[0])], dtype='uint8')
    return scared.traces.formats.read_ths_from_ram(samples=samples, plaintext=plaintext)


@pytest.fixture
def ths_2():
    shape = (2000, 1000)
    samples = np.random.randint(0, 255, shape, dtype='uint8')
    plaintext = np.random.randint(0, 255, (shape[0], 16), dtype='uint8')
    return scared.traces.formats.read_ths_from_ram(samples=samples, plaintext=plaintext)


def test_ttest_container_raises_exception_if_incorrect_ths_provided(ths_1, ths_2):
    with pytest.raises(TypeError):
        scared.TTestContainer(ths_1='foo', ths_2='bar')
    with pytest.raises(TypeError):
        scared.TTestContainer(ths_1=ths_1, ths_2='bar')
    with pytest.raises(TypeError):
        scared.TTestContainer(ths_1='foo', ths_2=ths_2)


def test_ttest_container_raises_exception_if_incorrect_frame_provided(ths_1, ths_2):
    with pytest.raises(TypeError):
        scared.TTestContainer(ths_1=ths_1, ths_2=ths_2, frame='foo')
    with pytest.raises(TypeError):
        scared.TTestContainer(ths_1, ths_2, frame=2121.1)
    with pytest.raises(TypeError):
        scared.TTestContainer(ths_1, ths_2, frame={})


def test_container_raises_error_if_bad_preprocesses(ths_1, ths_2):
    with pytest.raises(TypeError):
        scared.TTestContainer(ths_1, ths_2, preprocesses='foo')
    with pytest.raises(TypeError):
        scared.TTestContainer(ths_1, ths_2, preprocesses=['foo', 123])
    with pytest.raises(TypeError):
        scared.TTestContainer(ths_1, ths_2, preprocesses=134)


def test_ttest_analysis_raises_exception_if_invalid_precision_is_passed():
    with pytest.raises(TypeError):
        scared.TTestAnalysis(precision='foo')
    with pytest.raises(ValueError):
        scared.TTestAnalysis(precision='int8')


def test_ttest_analysis_initialize_two_ttest_accumulator(ths_1, ths_2):
    analysis = scared.TTestAnalysis()
    assert analysis.accumulators is not None
    assert len(analysis.accumulators) == 2
    assert isinstance(analysis.accumulators[0], scared.TTestAccumulator)
    assert analysis.accumulators[0].precision == 'float32'
    assert isinstance(analysis.accumulators[1], scared.TTestAccumulator)
    assert analysis.accumulators[1].precision == 'float32'


def test_accumulator_update_raises_exception_if_invalid_traces():
    with pytest.raises(TypeError):
        accu = scared.TTestAccumulator(precision='float32')
        accu.update('foo')


def test_ttest_accumulator_update_its_accumulator(ths_1):
    accu = scared.TTestAccumulator(precision='float32')
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


def test_ttest_compute_raise_exception_if_no_trace_are_processed_or_state_not_initialized():
    d = scared.TTestAccumulator('float32')
    with pytest.raises(scared.TTestError):
        d.compute()


def test_ttest_compute(ths_1, ths_2):
    accu = scared.TTestAccumulator(precision='float32')
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


def test_ttest_analysis_run_raises_exception_if_container_not_ttest_container():
    analysis = scared.TTestAnalysis(precision='float64')
    with pytest.raises(TypeError):
        analysis.run('foo')
