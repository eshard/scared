from ..context import scared
import pytest
import numpy as np


class DumbDistinguisher(scared.Distinguisher):
    def _initialize(self, traces, data):
        pass

    def _update(self, traces, data):
        pass

    def _compute(self):
        pass

    @property
    def _distinguisher_str(self):
        return 'DumbStandalone'


def test_subclassing_distinguisher_without_appropriate_functions_raises_exceptions():

    class MyDistinguisher(scared.Distinguisher):
        def _compute(self):
            pass

        def _initialize(self, traces, data):
            return super()._initialize(traces, data)

    with pytest.raises(TypeError):
        MyDistinguisher()

    class MyDistinguisher(scared.Distinguisher):
        def _update(self):
            pass

        def _initialize(self, traces, data):
            return super()._initialize(traces, data)

    with pytest.raises(TypeError):
        MyDistinguisher()

    class MyDistinguisher(scared.Distinguisher):
        def _compute(self):
            pass

        def _initialize(self, traces, data):
            return super()._initialize(traces, data)

    with pytest.raises(TypeError):
        MyDistinguisher()


def test_distinguisher_raises_exception_when_passing_a_non_consistent_precision():

    with pytest.raises(TypeError):
        DumbDistinguisher(precision='foo')

    with pytest.raises(ValueError):
        DumbDistinguisher(precision='uint8')


def test_distinguisher_has_float32_dtype_default_precision():
    d = DumbDistinguisher()
    assert d.precision == np.dtype('float32')


def test_distinguisher_mixin_handles_processed_traces_count():
    d = DumbDistinguisher()
    assert d.processed_traces == 0

    data = np.random.randint(0, 255, (500, 16), dtype='uint8')
    traces = np.random.randint(0, 255, (500, 16), dtype='uint8')
    d.update(data=data, traces=traces)
    assert d.processed_traces == 500


def test_dpa_initialize_raises_exception_f_max_data_value_is_not_in_0_1():
    d = scared.DPADistinguisher()
    data = np.random.randint(0, 255, (500, 16), dtype='uint8')
    traces = np.random.randint(0, 255, (500, 16), dtype='uint8')

    with pytest.raises(ValueError):
        d._initialize(traces, data)


def test_dpa_initialize_raises_exception_if_accumulators_are_too_large():
    d = scared.DPADistinguisher(precision='float64')
    data = np.random.randint(0, 1, (1, 1000000), dtype='uint8')
    traces = np.random.randint(0, 255, (1, 1000000), dtype='uint8')
    with pytest.raises(MemoryError):
        d._initialize(traces, data)


def test_dpa_initializes_accumulators_to_zeros_arrays():
    d = scared.DPADistinguisher()
    data = np.random.randint(0, 1, (1, 4096), dtype='uint8')
    traces = np.random.randint(0, 255, (1, 50000), dtype='uint8')

    d._initialize(traces, data)
    assert d.accumulator_traces.shape == (50000,)
    assert d.accumulator_traces.dtype == 'float32'
    assert d.accumulator_ones.shape == (4096, 50000)
    assert d.accumulator_ones.dtype == 'float32'
    assert d.processed_ones.shape == (4096,)
    assert d.processed_ones.dtype == 'uint32'
    assert d.processed_traces == 0


def test_dpa_update_raises_exception_if_traces_or_data_have_improper_types():
    ts = 1000
    nw = 8
    d = scared.DPADistinguisher()
    with pytest.raises(TypeError):
        d.update(traces='foo', data=np.random.randint(0, 1, (ts, nw), dtype='uint8'))
    with pytest.raises(TypeError):
        d.update(data='foo', traces=np.random.randint(0, 1, (ts, nw), dtype='uint8'))


def test_dpa_update_accumulators():
    traces = np.array([
        [0, 1, 2, 3],
        [1, 2, 3, 4],
        [4, 5, 6, 7],
        [2, 3, 4, 5],
        [3, 4, 5, 6]],
        dtype='uint8')
    data = np.array([
        [1, 0],
        [0, 1],
        [1, 0],
        [1, 0],
        [0, 0]],
        dtype='uint8')

    d = scared.DPADistinguisher()
    d.update(traces, data)
    assert [3, 1] == d.processed_ones.tolist()
    assert [10, 15, 20, 25] == d.accumulator_traces.tolist()
    assert [
        [6, 9, 12, 15],
        [1, 2, 3, 4]
    ] == d.accumulator_ones.tolist()
    assert 5 == d.processed_traces

    traces_batch_2 = np.array([
        [2, 3, 4, 5],
        [1, 2, 3, 4],
        [0, 1, 2, 3],
        [3, 4, 5, 6],
        [4, 5, 6, 7]],
        dtype='uint8')
    data_batch_2 = np.array([
        [1, 0],
        [0, 1],
        [0, 0],
        [0, 1],
        [1, 0]],
        dtype='uint8')
    d.update(traces_batch_2, data_batch_2)
    assert [5, 3] == d.processed_ones.tolist()
    assert [20, 30, 40, 50] == d.accumulator_traces.tolist()
    assert [
        [12, 17, 22, 27],
        [5, 8, 11, 14]
    ] == d.accumulator_ones.tolist()
    assert 10 == d.processed_traces


def test_dpa_compute_accumulators_raise_exception_if_no_trace_are_processed_or_state_not_initialized():
    d = scared.DPADistinguisher()
    with pytest.raises(scared.DistinguisherError):
        d.compute()


def test_dpa_compute_accumulators():
    traces = np.array([
        [0, 1, 2, 3],
        [1, 2, 3, 4],
        [4, 5, 6, 7],
        [2, 3, 4, 5],
        [3, 4, 5, 6]],
        dtype='uint8')
    data = np.array([
        [1, 0],
        [0, 1],
        [1, 0],
        [1, 0],
        [0, 0]],
        dtype='uint8')

    d = scared.DPADistinguisher()
    d.update(traces, data)
    results = d.compute()
    assert [
        [0., 0., 0., 0.],
        [-1.25, -1.25, -1.25, -1.25]
    ] == results.tolist()


def test_cpa_initialize_raises_exception_if_accumulators_are_too_large():
    d = scared.CPADistinguisher(precision='float64')
    data = np.random.randint(0, 1, (1, 100000000), dtype='uint8')
    traces = np.random.randint(0, 255, (1, 10000000), dtype='uint8')
    with pytest.raises(MemoryError):
        d._initialize(traces, data)


def test_cpa_initializes_accumulators_to_zeros_arrays():
    d = scared.CPADistinguisher()
    data = np.random.randint(0, 1, (1, 4096), dtype='uint8')
    traces = np.random.randint(0, 255, (1, 50000), dtype='uint8')
    d._initialize(traces, data)
    assert d.ex.shape == (50000,)
    assert d.ex.dtype == 'float32'
    assert d.ex2.shape == (50000,)
    assert d.ex2.dtype == 'float32'

    assert d.exy.shape == (4096, 50000)
    assert d.exy.dtype == 'float32'
    assert d.ey.shape == (4096,)
    assert d.ey.dtype == 'float32'
    assert d.ey2.shape == (4096,)
    assert d.ey2.dtype == 'float32'

    assert d.processed_traces == 0


def test_cpa_update_raises_exception_if_traces_or_data_have_improper_types():
    ts = 1000
    nw = 8

    d = scared.CPADistinguisher()
    with pytest.raises(TypeError):
        d.update(traces='foo', data=np.random.randint(0, 1, (ts, nw), dtype='uint8'))
    with pytest.raises(TypeError):
        d.update(data='foo', traces=np.random.randint(0, 1, (ts, nw), dtype='uint8'))


def test_cpa_update_accumulators():
    traces = np.array([
        [0, 1, 2, 3],
        [1, 2, 3, 4],
        [4, 5, 6, 7],
        [2, 3, 4, 5],
        [3, 4, 5, 6]],
        dtype='uint8')
    data = np.array([
        [2, 0],
        [0, 1],
        [3, 0],
        [1, 0],
        [0, 4]],
        dtype='uint8')

    d = scared.CPADistinguisher()
    d.update(traces, data)
    assert 5 == d.processed_traces
    assert [6, 5] == d.ey.tolist()
    assert [14, 17] == d.ey2.tolist()
    assert [10, 15, 20, 25] == d.ex.tolist()
    assert [30, 55, 90, 135] == d.ex2.tolist()
    assert [
        [14, 20, 26, 32],
        [13, 18, 23, 28]
    ] == d.exy.tolist()


def test_cpa_compute_accumulators():
    traces = np.array([
        [0, 1, 2, 3],
        [1, 2, 3, 4],
        [4, 5, 6, 7],
        [2, 3, 4, 5],
        [3, 4, 5, 6]],
        dtype='uint8')
    data = np.array([
        [2, 0],
        [0, 1],
        [3, 0],
        [1, 0],
        [0, 4]],
        dtype='uint8')

    d = scared.CPADistinguisher(precision='float64')
    d.update(traces, data)
    assert 5 == d.processed_traces
    assert [6, 5] == d.ey.tolist()
    assert [14, 17] == d.ey2.tolist()
    assert [10, 15, 20, 25] == d.ex.tolist()
    assert [30, 55, 90, 135] == d.ex2.tolist()
    assert [
        [14, 20, 26, 32],
        [13, 18, 23, 28]
    ] == d.exy.tolist()

    results = d.compute()
    val_1 = 0.24253562503633297
    val_2 = 0.27386127875258304
    assert [
        [val_1, val_1, val_1, val_1],
        [val_2, val_2, val_2, val_2]
    ] == results.tolist()

    d = scared.CPAAlternativeDistinguisher(precision='float64')
    d.update(traces, data)
    results = d.compute()
    val_1 = 0.24253562503633297
    val_2 = 0.27386127875258304
    assert [
        [val_1, val_1, val_1, val_1],
        [val_2, val_2, val_2, val_2]
    ] == results.tolist()


def test_cpa_compute_accumulators_raise_exception_if_no_trace_are_processed_or_state_not_initialized():
    d = scared.CPADistinguisher()
    with pytest.raises(scared.DistinguisherError):
        d.compute()


def test_distinguishers_linearise_data_dimensions_internally():

    traces = np.array([
        [0, 1, 2, 3],
        [1, 2, 3, 4],
        [4, 5, 6, 7],
        [2, 3, 4, 5],
        [3, 4, 5, 6]],
        dtype='uint8')
    data = np.array([
        [[2], [0]],
        [[0], [1]],
        [[3], [0]],
        [[1], [0]],
        [[0], [4]]],
        dtype='uint8')
    c = scared.CPADistinguisher()

    c.update(traces, data)
    c_r = c.compute()
    assert c_r.shape == (2, 1, 4)


def test_cpa_update_method_converts_traces_and_data_properly():

    traces = np.random.randint(-10000, 10000, size=(50, 200), dtype='int16')
    data = np.random.randint(0, 255, (50, 256, 16), dtype='uint8')

    c = scared.CPADistinguisher()
    c.update(traces, data)

    assert np.alltrue(c.ex2 > 0)
    assert np.alltrue(c.ey2 > 0)
