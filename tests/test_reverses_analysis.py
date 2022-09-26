from .context import scared  # noqa: F401
import warnings
import pytest
import numpy as np


@pytest.fixture
def ths():
    shape = (200, 33)
    samples = np.random.randint(0, 255, shape, dtype='uint8')
    plaintext = np.random.randint(0, 255, (shape[0], 16), dtype='uint8')
    key = np.array([np.random.randint(0, 255, (16,), dtype='uint8') for i in range(shape[0])])
    return scared.traces.formats.read_ths_from_ram(samples=samples, plaintext=plaintext, key=key)


@pytest.fixture(params=[scared.CPAReverse, scared.DPAReverse, scared.ANOVAReverse, scared.NICVReverse, scared.SNRReverse, scared.MIAReverse])
def reverse_class(request):
    return request.param


@pytest.fixture(params=[scared.ANOVAReverse, scared.NICVReverse, scared.SNRReverse, scared.MIAReverse])
def partitioned_klass(request):
    return request.param


@pytest.fixture
def sf():
    def _sf(key, plaintext):
        result = np.empty((plaintext.shape[0], 16), dtype='uint8')
        for byte in range(16):
            result[:, byte] = np.sum(plaintext, axis=1)
        return result
    return scared.reverse_selection_function(_sf)


@pytest.fixture
def container(ths):
    return scared.Container(ths)


class DumbDistinguisherMixin(scared.DistinguisherMixin):

    def _initialize(self, traces, data):
        self.traces = []
        self.data = []

    def _update(self, traces, data):
        self.traces.append(traces)
        self.data.append(data)

    def _compute(self):
        tr = np.array(self.traces).sum(axis=0)
        da = np.array(self.data).sum(axis=0)
        return np.dot(da.T, tr)

    @property
    def _distinguisher_str(self):
        return 'Dumb'


class DumbReverse(scared.BaseReverse, DumbDistinguisherMixin):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.traces = []
        self.data = []
        self.processed_traces = 0


def test_analysis_run_raises_exceptions_if_ths_container_is_not_a_container(reverse_class, sf):
    with pytest.raises(TypeError):
        a = reverse_class(
            selection_function=sf,
            model=scared.Monobit(4))
        a.run('foo')
        assert isinstance(str(a), str)


def test_analysis_object_raises_exceptions_if_sf_is_not_a_selection_function(reverse_class):
    with pytest.raises(TypeError):
        reverse_class(
            selection_function='foo',
            model=scared.HammingWeight())


def test_analysis_object_raises_exceptions_if_model_is_not_a_proper_model_instance(reverse_class, sf):
    with pytest.raises(TypeError):
        reverse_class(
            selection_function=sf,
            model='foo')


def test_base_analysis_cant_be_used_without_subclassing_a_distinguisher_mixin(sf):
    with pytest.raises(NotImplementedError):
        scared.BaseReverse(
            selection_function=sf,
            model=scared.HammingWeight())


def test_analysis_object_compute_intermediate_values(sf, container):
    analysis = DumbReverse(
        selection_function=sf,
        model=scared.HammingWeight()
    )
    int_val = analysis.compute_intermediate_values(container._ths.metadatas)
    assert (len(container._ths), 16) == int_val.shape
    expected = scared.HammingWeight()(sf(**container._ths.metadatas))
    int_val = int_val.reshape((len(container._ths), 16))
    assert np.array_equal(
        expected, int_val
    )
    assert isinstance(str(analysis), str)


def test_analysis_object_process_traces_batch(sf, container):
    analysis = DumbReverse(
        selection_function=sf,
        model=scared.Monobit(5)
    )
    batches = list(container.batches())
    analysis.process(batches[0])
    assert np.array_equal(analysis.traces[0], batches[0].samples[:])
    assert np.array_equal(
        analysis.data[0].reshape(-1, 16),
        analysis.compute_intermediate_values(batches[0].metadatas)
    )
    assert isinstance(str(analysis), str)


def test_analysis_object_compute_results(sf, container):
    analysis = DumbReverse(
        selection_function=sf,
        model=scared.Monobit(5)
    )
    batches = list(container.batches())
    analysis.process(batches[0])
    analysis.compute_results()

    assert (16, 33) == analysis.results.shape
    assert isinstance(str(analysis), str)


def test_analysis_object_run_method(sf, container):
    analysis = DumbReverse(
        selection_function=sf,
        model=scared.Monobit(5)
    )
    analysis.run(container)
    assert (16, 33) == analysis.results.shape
    assert isinstance(str(analysis), str)


def test_dpa_analysis_raise_exception_if_init_with_not_monobit_model(sf):
    with pytest.raises(scared.distinguishers.DistinguisherError):
        scared.DPAReverse(
            model=scared.Value(),
            selection_function=sf
        )


def test_analysis_object_run_method_with_frame(sf, ths):
    analysis = DumbReverse(
        selection_function=sf,
        model=scared.Monobit(5)
    )

    container = scared.Container(ths, frame=slice(None, 10))

    analysis.run(container)
    assert (1, 200, 10) == np.array(analysis.traces).shape

    container.frame = slice(20, 30)
    analysis.run(container)
    assert (2, 200, 10) == np.array(analysis.traces).shape
    assert isinstance(str(analysis), str)


def test_analysis_run_raise_exceptions_if_inconsistent_traces_size_are_used_between_two_process(sf, container):
    analysis = scared.DPAReverse(
        selection_function=sf,
        model=scared.Monobit(3)
    )

    container.frame = slice(20, 30)
    analysis.run(container)
    container.frame = slice(20, None)
    assert isinstance(str(analysis), str)
    with pytest.raises(scared.DistinguisherError):
        analysis.run(container)

    analysis = scared.CPAReverse(
        selection_function=sf,
        model=scared.HammingWeight()
    )
    assert isinstance(str(analysis), str)
    warnings.simplefilter('ignore', RuntimeWarning)
    container.frame = slice(10, 20)
    analysis.run(container)
    container.frame = slice(20, None)
    with pytest.raises(scared.DistinguisherError):
        analysis.run(container)


def test_partitioned_analysis_raises_exception_if_incorrect_partition(sf, partitioned_klass):
    with pytest.raises(TypeError):
        partitioned_klass(selection_function=sf, model=scared.HammingWeight(), partitions='foo')
    with pytest.raises(TypeError):
        partitioned_klass(selection_function=sf, model=scared.HammingWeight(), partitions={})
    with pytest.raises(ValueError):
        partitioned_klass(selection_function=sf, model=scared.HammingWeight(), partitions=np.array([1.2, 3]))


def test_partitioned_analysis_set_partition(sf, partitioned_klass):
    a = partitioned_klass(selection_function=sf, model=scared.HammingWeight(), partitions=np.arange(9))
    assert np.array_equal(np.arange(9), a.partitions)
    assert isinstance(str(a), str)


def test_mia_analysis_raises_exceptions_if_incorrect_histos_parameters(sf):
    with pytest.raises(TypeError):
        scared.MIAReverse(selection_function=sf, model=scared.HammingWeight(), bins_number='foo')
    with pytest.raises(TypeError):
        scared.MIAReverse(selection_function=sf, model=scared.HammingWeight(), bins_number={})
    with pytest.raises(TypeError):
        scared.MIAReverse(selection_function=sf, model=scared.HammingWeight(), bins_number=[1, 23])
    with pytest.raises(TypeError):
        scared.MIAReverse(selection_function=sf, model=scared.HammingWeight(), bins_number=np.array([1.2, 3]))
    with pytest.raises(TypeError):
        scared.MIAReverse(selection_function=sf, model=scared.HammingWeight(), bins_number=38.45)


_bin_edges_fail = {
    'bin number': 12,
    'np method': 'auto',
    'No edges': None
}


@pytest.fixture(params=_bin_edges_fail.keys())
def bin_edges_fail_key(request):
    return request.param


def test_mia_with_invalid_bin_edges_raises_exception(sf, bin_edges_fail_key):
    bin_edges = _bin_edges_fail[bin_edges_fail_key]
    with pytest.raises(TypeError):
        d = scared.MIAReverse(selection_function=sf, model=scared.HammingWeight())
        d.bin_edges = bin_edges
        assert isinstance(str(d), str)

    with pytest.raises(TypeError):
        d = scared.MIAReverse(bin_edges=bin_edges, selection_function=sf, model=scared.HammingWeight())
        d.bin_edges = bin_edges
        assert isinstance(str(d), str)


def test_mia_bin_edges_init(sf):
    a = scared.MIAReverse(bin_edges=np.arange(258), selection_function=sf, model=scared.HammingWeight())
    assert np.array_equal(a.bin_edges, np.arange(258))
    assert isinstance(str(a), str)
