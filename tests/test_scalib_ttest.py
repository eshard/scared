"""Comprehensive tests for SCALib TTest wrapper.

These tests verify that TTestAnalysisSCALib provides correct results matching
the scared TTestAnalysis implementation while using SCALib's optimized backend.
"""

from .context import scared  # noqa: F401
import pytest
import numpy as np


@pytest.fixture
def ths_1():
    """First trace set with fixed samples."""
    shape = (2000, 1000)
    sample = np.random.randint(0, 256, (shape[1],), dtype='uint8')
    plain = np.random.randint(0, 256, (16), dtype='uint8')
    samples = np.array([sample for i in range(shape[0])], dtype='uint8')
    plaintext = np.array([plain for i in range(shape[0])], dtype='uint8')
    return scared.traces.formats.read_ths_from_ram(samples=samples, plaintext=plaintext)


@pytest.fixture
def ths_2():
    """Second trace set with random samples."""
    shape = (2000, 1000)
    samples = np.random.randint(0, 256, shape, dtype='uint8')
    plaintext = np.random.randint(0, 256, (shape[0], 16), dtype='uint8')
    return scared.traces.formats.read_ths_from_ram(samples=samples, plaintext=plaintext)


def test_scalib_ttest_raises_if_scalib_not_available(mocker):
    """Test that ImportError is raised when SCALib is not available."""
    mocker.patch('scared.scalib.ttest.SCALIB_AVAILABLE', False)
    with pytest.raises(ImportError, match='SCALib is not installed'):
        from scared.scalib.ttest import TTestAnalysisSCALib
        TTestAnalysisSCALib()


def test_scalib_ttest_init_default_parameters():
    """Test initialization with default parameters."""
    from scared.scalib import TTestAnalysisSCALib
    ttest = TTestAnalysisSCALib()
    assert ttest.order == 1


def test_scalib_ttest_init_with_order_2():
    """Test initialization with second-order t-test."""
    from scared.scalib import TTestAnalysisSCALib
    ttest = TTestAnalysisSCALib(order=2)
    assert ttest.order == 2


def test_scalib_ttest_init_with_order_3():
    """Test initialization with third-order t-test."""
    from scared.scalib import TTestAnalysisSCALib
    ttest = TTestAnalysisSCALib(order=3)
    assert ttest.order == 3


def test_scalib_ttest_init_raises_on_invalid_order():
    """Test that invalid order raises appropriate errors."""
    from scared.scalib import TTestAnalysisSCALib
    with pytest.raises(TypeError):
        TTestAnalysisSCALib(order='foo')
    with pytest.raises(ValueError):
        TTestAnalysisSCALib(order=0)
    with pytest.raises(ValueError):
        TTestAnalysisSCALib(order=-1)


def test_scalib_ttest_run_raises_on_invalid_container():
    """Test that run raises error with invalid container."""
    from scared.scalib import TTestAnalysisSCALib
    ttest = TTestAnalysisSCALib()
    with pytest.raises(TypeError, match='ttest_container should be a TTestContainer'):
        ttest.run('not a container')


def test_scalib_ttest_str_representation():
    """Test string representation of the analysis."""
    from scared.scalib import TTestAnalysisSCALib
    ttest = TTestAnalysisSCALib(order=1)
    assert str(ttest) == 't-Test analysis (SCALib, order=1)'
    ttest = TTestAnalysisSCALib(order=2)
    assert str(ttest) == 't-Test analysis (SCALib, order=2)'


def test_scalib_ttest_basic_run(ths_1, ths_2):
    """Test basic t-test computation."""
    from scared.scalib import TTestAnalysisSCALib
    container = scared.TTestContainer(ths_1, ths_2)
    ttest = TTestAnalysisSCALib()
    ttest.run(container)

    assert ttest.result is not None
    assert ttest.result.shape == (1000,)
    assert ttest.result.dtype == np.float64


def test_scalib_ttest_results_match_manual_calculation(ths_1, ths_2):
    """Test that SCALib t-test matches manual Welch's t-test calculation."""
    from scared.scalib import TTestAnalysisSCALib

    container = scared.TTestContainer(ths_1, ths_2)
    ttest = TTestAnalysisSCALib()
    ttest.run(container)

    traces_1 = ths_1.samples[:].astype(np.int16)
    traces_2 = ths_2.samples[:].astype(np.int16)

    mean_1 = np.mean(traces_1, axis=0, dtype=np.float64)
    mean_2 = np.mean(traces_2, axis=0, dtype=np.float64)

    var_1 = np.var(traces_1, axis=0, dtype=np.float64, ddof=0)
    var_2 = np.var(traces_2, axis=0, dtype=np.float64, ddof=0)

    n_1 = traces_1.shape[0]
    n_2 = traces_2.shape[0]

    t_manual = (mean_1 - mean_2) / np.sqrt(var_1 / n_1 + var_2 / n_2)

    np.testing.assert_allclose(ttest.result, t_manual, rtol=1e-10, atol=1e-10)


def test_scalib_ttest_results_match_scared_ttest(ths_1, ths_2):
    """Test that SCALib t-test matches scared TTestAnalysis results."""
    from scared.scalib import TTestAnalysisSCALib

    container = scared.TTestContainer(ths_1, ths_2)

    ttest_scalib = TTestAnalysisSCALib()
    ttest_scalib.run(container)

    ttest_scared = scared.TTestAnalysis(precision='float64')
    ttest_scared.run(container)

    np.testing.assert_allclose(ttest_scalib.result, ttest_scared.result, rtol=1e-10, atol=1e-10)


def test_scalib_ttest_with_frame(ths_1, ths_2):
    """Test t-test with frame parameter."""
    from scared.scalib import TTestAnalysisSCALib

    container = scared.TTestContainer(ths_1, ths_2, frame=slice(0, 100))
    ttest = TTestAnalysisSCALib()
    ttest.run(container)

    assert ttest.result is not None
    assert ttest.result.shape == (100,)

    traces_1 = ths_1.samples[:, :100].astype(np.int16)
    traces_2 = ths_2.samples[:, :100].astype(np.int16)

    mean_1 = np.mean(traces_1, axis=0, dtype=np.float64)
    mean_2 = np.mean(traces_2, axis=0, dtype=np.float64)
    var_1 = np.var(traces_1, axis=0, dtype=np.float64, ddof=0)
    var_2 = np.var(traces_2, axis=0, dtype=np.float64, ddof=0)
    n_1 = traces_1.shape[0]
    n_2 = traces_2.shape[0]
    t_expected = (mean_1 - mean_2) / np.sqrt(var_1 / n_1 + var_2 / n_2)

    np.testing.assert_allclose(ttest.result, t_expected, rtol=1e-10, atol=1e-10)


def test_scalib_ttest_with_preprocesses(ths_1, ths_2):
    """Test t-test with preprocessing."""
    from scared.scalib import TTestAnalysisSCALib

    container = scared.TTestContainer(ths_1, ths_2, preprocesses=[scared.preprocesses.square])
    ttest = TTestAnalysisSCALib()
    ttest.run(container)

    assert ttest.result is not None
    assert ttest.result.shape == (1000,)

    traces_1 = scared.preprocesses.square(ths_1.samples[:]).astype(np.int16)
    traces_2 = scared.preprocesses.square(ths_2.samples[:]).astype(np.int16)

    mean_1 = np.mean(traces_1, axis=0, dtype=np.float64)
    mean_2 = np.mean(traces_2, axis=0, dtype=np.float64)
    var_1 = np.var(traces_1, axis=0, dtype=np.float64, ddof=0)
    var_2 = np.var(traces_2, axis=0, dtype=np.float64, ddof=0)
    n_1 = traces_1.shape[0]
    n_2 = traces_2.shape[0]
    t_expected = (mean_1 - mean_2) / np.sqrt(var_1 / n_1 + var_2 / n_2)

    np.testing.assert_allclose(ttest.result, t_expected, rtol=1e-10, atol=1e-10)


def test_scalib_ttest_with_different_ths_lengths_1():
    from scared.scalib import TTestAnalysisSCALib

    trace_length = 1002
    samples_1 = np.random.randint(0, 256, (100, trace_length), dtype='uint8')
    samples_2 = np.random.randint(0, 256, (6000, trace_length), dtype='uint8')  # high to ensure more than 1 batch

    ths_1_small = scared.traces.formats.read_ths_from_ram(samples=samples_1, plaintext=np.random.randint(0, 256, (len(samples_1), 16), dtype='uint8'))
    ths_2_small = scared.traces.formats.read_ths_from_ram(samples=samples_2, plaintext=np.random.randint(0, 256, (len(samples_2), 16), dtype='uint8'))

    container = scared.TTestContainer(ths_1_small, ths_2_small)
    ttest = TTestAnalysisSCALib()
    ttest.run(container)

    assert ttest.result is not None
    assert ttest.result.shape == (trace_length,)


def test_scalib_ttest_with_different_ths_lengths_2():
    from scared.scalib import TTestAnalysisSCALib

    trace_length = 1002
    samples_1 = np.random.randint(0, 256, (6000, trace_length), dtype='uint8')  # high to ensure more than 1 batch
    samples_2 = np.random.randint(0, 256, (100, trace_length), dtype='uint8')

    ths_1_small = scared.traces.formats.read_ths_from_ram(samples=samples_1, plaintext=np.random.randint(0, 256, (len(samples_1), 16), dtype='uint8'))
    ths_2_small = scared.traces.formats.read_ths_from_ram(samples=samples_2, plaintext=np.random.randint(0, 256, (len(samples_2), 16), dtype='uint8'))

    container = scared.TTestContainer(ths_1_small, ths_2_small)
    ttest = TTestAnalysisSCALib()
    ttest.run(container)

    assert ttest.result is not None
    assert ttest.result.shape == (trace_length,)


def test_scalib_ttest_with_int16_traces():
    """Test t-test with int16 traces (SCALib native format)."""
    from scared.scalib import TTestAnalysisSCALib

    shape = (500, 200)
    samples_1 = np.random.randint(-1000, 1000, shape, dtype='int16')
    samples_2 = np.random.randint(-500, 500, shape, dtype='int16')
    plaintext = np.random.randint(0, 256, (shape[0], 16), dtype='uint8')

    ths_1_int16 = scared.traces.formats.read_ths_from_ram(samples=samples_1, plaintext=plaintext)
    ths_2_int16 = scared.traces.formats.read_ths_from_ram(samples=samples_2, plaintext=plaintext)

    container = scared.TTestContainer(ths_1_int16, ths_2_int16)
    ttest = TTestAnalysisSCALib()
    ttest.run(container)

    assert ttest.result is not None
    assert ttest.result.shape == (200,)


def test_scalib_ttest_order_2_basic():
    """Test second-order t-test."""
    from scared.scalib import TTestAnalysisSCALib

    shape = (500, 100)
    samples_1 = np.random.randint(0, 256, shape, dtype='uint8')
    samples_2 = np.random.randint(0, 256, shape, dtype='uint8')
    plaintext = np.random.randint(0, 256, (shape[0], 16), dtype='uint8')

    ths_1_small = scared.traces.formats.read_ths_from_ram(samples=samples_1, plaintext=plaintext)
    ths_2_small = scared.traces.formats.read_ths_from_ram(samples=samples_2, plaintext=plaintext)

    container = scared.TTestContainer(ths_1_small, ths_2_small)
    ttest = TTestAnalysisSCALib(order=2)
    ttest.run(container)

    assert ttest.result is not None
    assert ttest.result.shape == (100,)
    assert ttest.result.dtype == np.float64


def test_scalib_ttest_order_3_basic():
    """Test third-order t-test."""
    from scared.scalib import TTestAnalysisSCALib

    shape = (500, 100)
    samples_1 = np.random.randint(0, 256, shape, dtype='uint8')
    samples_2 = np.random.randint(0, 256, shape, dtype='uint8')
    plaintext = np.random.randint(0, 256, (shape[0], 16), dtype='uint8')

    ths_1_small = scared.traces.formats.read_ths_from_ram(samples=samples_1, plaintext=plaintext)
    ths_2_small = scared.traces.formats.read_ths_from_ram(samples=samples_2, plaintext=plaintext)

    container = scared.TTestContainer(ths_1_small, ths_2_small)
    ttest = TTestAnalysisSCALib(order=3)
    ttest.run(container)

    assert ttest.result is not None
    assert ttest.result.shape == (100,)
    assert ttest.result.dtype == np.float64


def test_scalib_ttest_empty_result_before_run():
    """Test that result is None before running."""
    from scared.scalib import TTestAnalysisSCALib
    ttest = TTestAnalysisSCALib()
    assert ttest.result_all_orders is None


def test_scalib_ttest_result_all_orders_order_3():
    """Test that result_all_orders contains all orders for order=3."""
    from scared.scalib import TTestAnalysisSCALib

    shape = (500, 100)
    samples_1 = np.random.randint(100, 150, shape, dtype='uint8')
    samples_2 = np.random.randint(50, 100, shape, dtype='uint8')
    plaintext = np.random.randint(0, 256, (shape[0], 16), dtype='uint8')

    ths_1_small = scared.traces.formats.read_ths_from_ram(samples=samples_1, plaintext=plaintext)
    ths_2_small = scared.traces.formats.read_ths_from_ram(samples=samples_2, plaintext=plaintext)

    container = scared.TTestContainer(ths_1_small, ths_2_small)
    ttest = TTestAnalysisSCALib(order=3)
    ttest.run(container)

    assert ttest.result_all_orders is not None
    assert ttest.result_all_orders.shape == (3, 100)
    assert ttest.result_all_orders.dtype == np.float64

    assert np.array_equal(ttest.result, ttest.result_all_orders[2])

    assert ttest.result_all_orders[0].shape == (100,)
    assert ttest.result_all_orders[1].shape == (100,)
    assert ttest.result_all_orders[2].shape == (100,)


def test_scalib_ttest_with_small_traces():
    """Test t-test with very small traces."""
    from scared.scalib import TTestAnalysisSCALib

    shape = (50, 10)
    samples_1 = np.random.randint(0, 256, shape, dtype='uint8')
    samples_2 = np.random.randint(0, 256, shape, dtype='uint8')
    plaintext = np.random.randint(0, 256, (shape[0], 16), dtype='uint8')

    ths_1_tiny = scared.traces.formats.read_ths_from_ram(samples=samples_1, plaintext=plaintext)
    ths_2_tiny = scared.traces.formats.read_ths_from_ram(samples=samples_2, plaintext=plaintext)

    container = scared.TTestContainer(ths_1_tiny, ths_2_tiny)
    ttest = TTestAnalysisSCALib()
    ttest.run(container)

    assert ttest.result is not None
    assert ttest.result.shape == (10,)


@pytest.mark.parametrize('dtype', ['uint8', 'int8', 'int16', 'int32'])
def test_scalib_ttest_with_various_dtypes(dtype):
    """Test t-test with various input dtypes."""
    from scared.scalib import TTestAnalysisSCALib

    shape = (200, 100)
    if dtype == 'uint8':
        samples_1 = np.random.randint(0, 256, shape, dtype=dtype)
        samples_2 = np.random.randint(0, 256, shape, dtype=dtype)
    elif dtype == 'int8':
        samples_1 = np.random.randint(-128, 128, shape, dtype=dtype)
        samples_2 = np.random.randint(-128, 128, shape, dtype=dtype)
    elif dtype == 'int16':
        samples_1 = np.random.randint(-1000, 1000, shape, dtype=dtype)
        samples_2 = np.random.randint(-1000, 1000, shape, dtype=dtype)
    else:
        samples_1 = np.random.randint(-10000, 10000, shape, dtype=dtype)
        samples_2 = np.random.randint(-10000, 10000, shape, dtype=dtype)

    plaintext = np.random.randint(0, 256, (shape[0], 16), dtype='uint8')

    ths_1_typed = scared.traces.formats.read_ths_from_ram(samples=samples_1, plaintext=plaintext)
    ths_2_typed = scared.traces.formats.read_ths_from_ram(samples=samples_2, plaintext=plaintext)

    container = scared.TTestContainer(ths_1_typed, ths_2_typed)
    ttest = TTestAnalysisSCALib()
    ttest.run(container)

    assert ttest.result is not None
    assert ttest.result.shape == (100,)
